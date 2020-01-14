import logging
import os
from typing import Any, Dict

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# to use directly allennlp.metrics I will have to load the entire library and dependencies,
# including outdated copies of pytorch_transformers
from allennlp_metrics import F1Measure, Average


def to_device(inputs: Any, device: torch.device):
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs[key] = to_device(value, device)
    elif isinstance(inputs, (list, tuple)):
        inputs = [to_device(value, device) for value in inputs]
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device)

    return inputs


def sample_eval_set(dataset: Dataset, n_size=None, ratio=None):
    if ratio is not None:
        n_size = max(int(len(dataset)*ratio), 1)
    n_size = min(n_size, len(dataset))
    indices = np.random.randint(n_size, size=n_size)
    return Subset(dataset, indices)


MODE_TRAIN = "train"
MODE_DEV = "dev"
MODES = {
    MODE_TRAIN: 0,
    MODE_DEV: 1}


def set_postfix_str(train_iter, train_loss, eval_loss):
    postfix_str = f"train_loss={train_loss:3.3f} "
    postfix_str += f"dev_loss={eval_loss:3.3f}"
    train_iter.set_postfix_str(postfix_str)


# The number of dev set examples to be used for continuous evaluation
# within the training loop every predefined number of steps.
DEV_SET_FRACTION = 0.05


class Trainer:
    def __init__(self,
                 device: torch.device, batch_size: int,
                 log_dir: str, metrics: Dict,
                 model_name="", saved_models_path="",
                 label_resolver=None, collate_fn=None):
        self.logger = logging.getLogger(__class__.__name__)

        self.logger.addHandler(TqdmLoggingHandler())
        self.device = device
        self.collate_fn = collate_fn

        self.log_modes = list(MODES.keys())
        log_dirs = {mode: os.path.join(log_dir, model_name, mode)
                          for mode in MODES.keys()}
        self.writers = {mode: SummaryWriter(log_dir=log_dir)
                        for mode, log_dir in log_dirs.items()}
        model_name = model_name if model_name else "model"
        self.saved_models_path = os.path.join(saved_models_path, model_name)

        self.step = 1
        self.batch_size = batch_size
        self.metrics = metrics
        self.loss = Average()
        self.label_resolver = label_resolver
        self.best_metric_so_far = -np.inf

    def evaluate(self, model: nn.Module, dev_set: Dataset):
        model.eval()
        dev_loader = DataLoader(dev_set,
                                batch_size=self.batch_size,
                                collate_fn=self.collate_fn)
        epoch_desc = f"Evaluating..."
        dev_iter = tqdm(dev_loader, desc=epoch_desc, leave=False)
        with torch.no_grad():
            for batch in dev_iter:
                batch = to_device(batch, self.device)
                outputs = model(**batch)
                self.update_metrics(batch, outputs)

    def single_step(self, model, batch, optimizer, scheduler, max_grad_norm):
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()

        if not np.isinf(max_grad_norm):
            clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler:
            scheduler.step()
        model.zero_grad()
        self.step += 1

        return outputs

    def should_log(self, log_freq):
        should_log = ((self.step + 1) % log_freq) == 0
        return should_log

    def train(self, model, train_set, dev_set,
              n_total_steps: int,
              log_freq: int,
              primary_metric_name: str,
              optimizer: Optimizer,
              scheduler=None,
              max_grad_norm=np.inf):

        model = model.to(self.device)
        model.train()

        for batch in self.get_train_iter(train_set, n_total_steps):
            batch = to_device(batch, self.device)
            outputs = self.single_step(model, batch,
                                       optimizer,
                                       scheduler,
                                       max_grad_norm)

            self.update_metrics(batch, outputs)
            if self.should_log(log_freq):
                use_full_dev = self.step % (10*log_freq) == 0
                metrics = self.eval_within_train(model, dev_set, use_full_dev)
                curr_metric = metrics[primary_metric_name]

                self.save_on_best_score(model, primary_metric_name, curr_metric)
                model.train()

    def eval_within_train(self, model, dev_set, use_full_dev=False):
        # log train metrics so far and reset counters.
        metrics = self.get_metrics(reset=True)
        self.log_metrics(metrics, MODE_TRAIN)

        # evaluate on a small eval set (or a full dev set) and log metrics
        if use_full_dev:
            eval_set = dev_set
        else:
            eval_set = sample_eval_set(dev_set, ratio=DEV_SET_FRACTION)

        self.evaluate(model, eval_set)
        # reset counters to continue training
        metrics = self.get_metrics(reset=True)
        self.log_metrics(metrics, MODE_DEV)
        return metrics

    def get_train_iter(self, train_set: Dataset, n_total_steps):
        epoch = 0
        while self.step < n_total_steps:
            train_loader = DataLoader(train_set,
                                      batch_size=self.batch_size,
                                      collate_fn=self.collate_fn)
            epoch += 1
            train_iter = tqdm(train_loader, desc=f"{[epoch]}", leave=False)
            for batch in train_iter:
                yield batch
                if self.step >= n_total_steps:
                    break

    def update_metrics(self, batch, outputs):
        # update loss separately.
        self.loss(outputs['loss'].item())
        for name, metric in self.metrics.items():
            use_mask = isinstance(batch['labels'], torch.Tensor) \
                       and batch['labels'].shape == batch['input_ids'].shape
            metric(predictions=outputs['logits'],
                   gold_labels=batch['labels'],
                   mask=batch['attention_mask'] if use_mask else None)

    def get_metrics(self, reset: bool):
        metrics = list(self.metrics.items()) + [('loss', self.loss)]
        out_metrics = {}
        for name, metric in metrics:
            value = metric.get_metric(reset)
            if isinstance(metric, F1Measure):
                out_metrics[f"{name}_prec"] = value[0]
                out_metrics[f"{name}_recall"] = value[1]
                out_metrics[f"{name}_f1"] = value[2]
            else:
                out_metrics[name] = value

        return out_metrics

    def log_metrics(self, metrics: Dict[str, float], mode=MODE_TRAIN):
        for name, metric in metrics.items():
            self.writers[mode].add_scalar(name, metric, self.step)

    def save_on_best_score(self, model: nn.Module, metric_name, curr_metric):
        if self.best_metric_so_far > curr_metric:
            return
        self.logger.info(f"[step={self.step}] "
                         f"Reached best result so far: "
                         f"[{metric_name}={curr_metric}]")
        self.best_metric_so_far = curr_metric

        saved_models_path = self.saved_models_path + f".best.pt"
        self.logger.info(f"Saving to: {saved_models_path}")
        torch.save(model.state_dict(), saved_models_path)
        # Save optimizer state as well.


# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
