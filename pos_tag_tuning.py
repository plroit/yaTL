import logging
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertConfig, BertForTokenClassification

from allennlp_metrics import CategoricalAccuracy
from data_utils import pad_and_mask, read_all_labels, read_json_samples
from optim_utils import prepare_optimizer
from training import Trainer


class PartOfSpeechDataset(Dataset):
    def __init__(self, samples, pos_tags, max_length):
        self.logger = logging.getLogger(__class__.__name__)
        self.max_length = max_length
        self.pos_to_idx = {pos: idx for idx, pos in enumerate(pos_tags)}
        self.samples = list(self._yield_samples(samples))

    def __getitem__(self, item):
        sample = self.samples[item]
        return sample

    def __len__(self):
        return len(self.samples)

    def _is_valid(self, sample):
        return len(sample['wp_indices']) <= self.max_length

    def _yield_valid_samples(self, samples):
        for sample in samples:
            if not self._is_valid(sample):
                msg = f"Skipping {sample['doc_id']}:, " \
                      f"{sample['sent_id']}"
                self.logger.info(msg)
                continue
            yield sample

    def _yield_samples(self, samples):
        for sample in self._yield_valid_samples(samples):
            tokens = sample['wp_indices']
            wp_indices, wp_masks = pad_and_mask(tokens, self.max_length)
            pos_tags = [self.pos_to_idx[pos] for pos in sample['wp_pos_tags']]
            pos_tags, _ = pad_and_mask(pos_tags, self.max_length, -1)
            yield {
                "input_ids": wp_indices,
                "attention_mask": wp_masks,
                "labels": pos_tags,
                "word_pieces": " ".join(sample['word_pieces'])
            }


class PartOfSpeechClassifier(nn.Module):
    def __init__(self, bert_model_name: str, n_labels: int):
        super().__init__()
        self.n_labels = n_labels
        config = BertConfig.from_pretrained(bert_model_name)
        config.num_labels = n_labels
        self.net = BertForTokenClassification.from_pretrained(bert_model_name, config=config)

    def forward(self, input_ids: torch.tensor, attention_mask, labels=None, **kwargs):
        outputs = self.net(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
        loss, logits = outputs[:2]
        # Trainer expects a loss tensor for gradient back-propagation.
        # Trainer from time-to-time run evaluation code (the metrics)
        # and they need the final logits tensor.
        return {
            'loss': loss,
            "logits": logits
        }


def main(args):
    LEVEL = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")
    device_id = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_id)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading datasets from: {args.train_path}, "
                f"{args.dev_path}, {args.labels_path}")

    labels = read_all_labels(args.labels_path)

    train_set = tqdm(read_json_samples(args.train_path))
    train_set = PartOfSpeechDataset(train_set, labels, args.max_length)
    dev_set = tqdm(read_json_samples(args.dev_path))
    dev_set = PartOfSpeechDataset(dev_set, labels, args.max_length)

    logger.info(f"Loading model: {args.bert_model}")

    classifier = PartOfSpeechClassifier(args.bert_model, len(labels)).to(device)

    optimizer, scheduler = prepare_optimizer(classifier, **vars(args))
    metrics = {"accuracy": CategoricalAccuracy()}

    trainer = Trainer(device,
                      args.batch_size,
                      args.log_dir,
                      metrics,
                      model_name=args.experiment_name,
                      saved_models_path=args.saved_models_path)

    trainer.train(classifier, train_set, dev_set,
                  args.n_total_steps, args.log_frequency, args.primary_metric,
                  optimizer, scheduler, args.max_grad_norm)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("train_path")
    ap.add_argument("dev_path")
    ap.add_argument("--labels_path", required=False)
    ap.add_argument("--bert_model", default="bert-base-uncased")
    ap.add_argument("--device", default=-1, type=int)
    ap.add_argument("--log_dir", default="runs")
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--n_total_steps", default=30000, type=int,
                    help="Iterate through this number of training steps.")
    ap.add_argument("--opt_type", default="bert_adam")
    ap.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate")
    ap.add_argument("--weight_decay", default=0.0, type=float)
    ap.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    ap.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
    ap.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
    ap.add_argument("--log_frequency", default=5, type=int)
    ap.add_argument("--experiment_name", default="", type=str)
    ap.add_argument("--saved_models_path", default="")
    ap.add_argument("--primary_metric", default="")
    ap.add_argument("--max_length", default=128, type=int)
    main(ap.parse_args())
