import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from transformers import AdamW


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_bert_adam(model, learning_rate, adam_epsilon, weight_decay, n_warmup_steps, num_train_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=n_warmup_steps,
                                                num_training_steps=num_train_steps)
    return optimizer, scheduler


def prepare_optimizer(model: nn.Module, opt_type: str,
                      learning_rate: float,
                      weight_decay: float, **kwargs):
    opt_type = opt_type.lower()
    if opt_type == "sgd":
        return SGD(model.parameters(),
                   lr=learning_rate,
                   weight_decay=weight_decay), None
    elif opt_type == "adam":
        return Adam(model.parameters(),
                    lr=learning_rate,
                    eps=kwargs['adam_epsilon']), None
    elif opt_type == "bert_adam":
        n_total_steps = kwargs['n_total_steps']
        return create_bert_adam(model,
                                learning_rate,
                                kwargs['adam_epsilon'],
                                weight_decay,
                                kwargs['warmup_steps'],
                                n_total_steps)
    else:
        raise NotImplementedError(opt_type)

