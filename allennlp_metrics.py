import re
import string
from collections import Counter
from typing import Optional, Union, Tuple, List, Dict

import torch
from allennlp.common.checks import ConfigurationError


class Metric:
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions.
        gold_labels : ``torch.Tensor``, required.
            A tensor corresponding to some gold label to evaluate against.
        mask: ``torch.Tensor``, optional (default = None).
            A mask can be passed, in order to deal with metrics which are
            computed over potentially padded elements, such as sequence labels.
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        """
        Compute and return the metric. Optionally also call :func:`self.reset`.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        """
        raise NotImplementedError

    @staticmethod
    def unwrap_to_tensors(*tensors: torch.Tensor):
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures that you're using tensors directly and that they are on
        the CPU.
        """
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class Average(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    def __call__(self, value):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._total_value += list(self.unwrap_to_tensors(value))[0]
        self._count += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    def reset(self):
        self._total_value = 0.0
        self._count = 0



class CategoricalAccuracy(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """
    def __init__(self, top_k: int = 1, tie_break: bool = False) -> None:
        if top_k > 1 and tie_break:
            raise ConfigurationError("Tie break in Categorical Accuracy "
                                     "can be done only for maximum (top_k = 1)")
        if top_k <= 0:
            raise ConfigurationError("top_k passed to Categorical Accuracy must be > 0")
        self._top_k = top_k
        self._tie_break = tie_break
        self.correct_count = 0.
        self.total_count = 0.

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Some sanity checks.
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))

        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        if not self._tie_break:
            # Top K indexes of the predictions (or fewer, if there aren't K of them).
            # Special case topk == 1, because it's common and .max() is much faster than .topk().
            if self._top_k == 1:
                top_k = predictions.max(-1)[1].unsqueeze(-1)
            else:
                top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

            # This is of shape (batch_size, ..., top_k).
            correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        else:
            # prediction is correct if gold label falls on any of the max scores. distribute score by tie_counts
            max_predictions = predictions.max(-1)[0]
            max_predictions_mask = predictions.eq(max_predictions.unsqueeze(-1))
            # max_predictions_mask is (rows X num_classes) and gold_labels is (batch_size)
            # ith entry in gold_labels points to index (0-num_classes) for ith row in max_predictions
            # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
            correct = max_predictions_mask[torch.arange(gold_labels.numel()).long(), gold_labels].float()
            tie_counts = max_predictions_mask.sum(-1)
            correct /= tie_counts.float()
            correct.unsqueeze_(-1)

        if mask is not None:
            correct *= mask.view(-1, 1).float()
            self.total_count += mask.sum()
        else:
            self.total_count += gold_labels.numel()
        self.correct_count += correct.sum()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


class FBetaMeasure(Metric):
    """Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    ``F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)``

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    Parameters
    ----------
    beta : ``float``, optional (default = 1.0)
        The strength of recall versus precision in the F-score.

    average : string, [None (default), 'micro', 'macro']
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

    labels: list, optional
        The set of labels to include and their order if ``average is None``.
        Labels present in the data can be excluded, for example to calculate a
        multi-class average ignoring a majority negative class. Labels not present
        in the data will result in 0 components in a macro average.

    """
    def __init__(self,
                 beta: float = 1.0,
                 average: str = None,
                 labels: List[int] = None) -> None:
        average_options = (None, 'micro', 'macro')
        if average not in average_options:
            raise ConfigurationError(f"`average` has to be one of {average_options}.")
        if beta <= 0:
            raise ConfigurationError("`beta` should be >0 in the F-beta score.")

        self._beta = beta
        self._average = average
        self._labels = labels

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum: Union[None, torch.Tensor] = None
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum: Union[None, torch.Tensor] = None
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum: Union[None, torch.Tensor] = None

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum
        num_classes = predictions.size(-1)
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to FBetaMeasure contains "
                                     f"an id >= {num_classes}, the number of classes.")

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes)
            self._true_sum = torch.zeros(num_classes)
            self._pred_sum = torch.zeros(num_classes)
            self._total_sum = torch.zeros(num_classes)

        if mask is None:
            mask = torch.ones_like(gold_labels)
        mask = mask.to(dtype=torch.bool)
        gold_labels = gold_labels.float()

        argmax_predictions = predictions.max(dim=-1)[1].float()
        true_positives = (gold_labels == argmax_predictions) * mask
        true_positives_bins = gold_labels[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_predictions[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes)

        gold_labels_bins = gold_labels[mask].long()
        if gold_labels.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)

    def get_metric(self,
                   reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]

        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))
        fscore[tp_sum == 0] = 0.0

        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()

        if reset:
            self.reset()

        if self._labels is not None:
            # Retain only selected labels and order them
            precision = precision[self._labels]
            recall = recall[self._labels]
            fscore = fscore[self._labels]

        if self._average is None:
            return {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "fscore": fscore.tolist()
            }
        else:
            return {
                "precision": precision.item(),
                "recall": recall.item(),
                "fscore": fscore.item()
            }

    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None

    @property
    def _true_negative_sum(self):
        if self._total_sum is None:
            return None
        else:
            true_negative_sum = self._total_sum - self._pred_sum - self._true_sum + self._true_positive_sum
            return true_negative_sum


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result



class F1Measure(FBetaMeasure):
    """
    Computes Precision, Recall and F1 with respect to a given ``positive_label``.
    For example, for a BIO tagging scheme, you would pass the classification index of
    the tag you are interested in, resulting in the Precision, Recall and F1 score being
    calculated for this tag only.
    """
    def __init__(self, positive_label: int) -> None:
        super().__init__(beta=1,
                         labels=[positive_label])
        self._positive_label = positive_label

    def get_metric(self,
                   reset: bool = False) -> Tuple[float, float, float]:
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        metric = super().get_metric(reset=reset)
        # Because we just care about the class `positive_label`
        # there is just one item in `precision`, `recall`, `fscore`
        precision = metric['precision'][0]
        recall = metric['recall'][0]
        fscore = metric['fscore'][0]
        return precision, recall, fscore

    @property
    def _true_positives(self):
        # When this metric is never called, `self._true_positive_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_positive_sum is None:
            return 0.0
        else:
            return self._true_positive_sum[self._positive_label]

    @property
    def _true_negatives(self):
        # When this metric is never called, `self._true_negative_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_negative_sum is None:
            return 0.0
        else:
            return self._true_negative_sum[self._positive_label]

    @property
    def _false_positives(self):
        # When this metric is never called, `self._pred_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._pred_sum is None:
            return 0.0
        else:
            # `self._pred_sum` is the total number of instances under each _predicted_ class,
            # including true positives and false positives.
            return self._pred_sum[self._positive_label] - self._true_positives

    @property
    def _false_negatives(self):
        # When this metric is never called, `self._true_sum` is None,
        # under which case we return 0.0 for backward compatibility.
        if self._true_sum is None:
            return 0.0
        else:
            # `self._true_sum` is the total number of instances under each _true_ class,
            # including true positives and false negatives.
            return self._true_sum[self._positive_label] - self._true_positives


class SquadEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        exact_match = SquadEmAndF1.metric_max_over_ground_truths(
            SquadEmAndF1.exact_match_score,
            best_span_string,
            answer_strings)
        f1_score = SquadEmAndF1.metric_max_over_ground_truths(
            SquadEmAndF1.f1_score,
            best_span_string,
            answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"SquadEmAndF1(em={self._total_em}, f1={self._total_f1})"

    @staticmethod
    def normalize_answer(s: str):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = SquadEmAndF1.normalize_answer(prediction).split()
        ground_truth_tokens = SquadEmAndF1.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        p = SquadEmAndF1.normalize_answer(prediction)
        g = SquadEmAndF1.normalize_answer(ground_truth)
        return p == g

    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

