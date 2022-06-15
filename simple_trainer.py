from tqdm.autonotebook import tqdm
import torch
from collections import defaultdict
import numpy as np
import gc
from torch.nn.utils import clip_grad_norm_

from torch.cuda.amp

def to(batch, device, **kwargs):
    for key in batch:
        batch[key] = batch[key].to(device, **kwargs)

    return batch


def move_adam_stats(opt, device):
    for vals in opt.state.values():
        vals['exp_avg'] = vals['exp_avg'].to(device)
        vals['exp_avg_sq'] = vals['exp_avg_sq'].to(device)


def forw_backw(model, loader, config, logger):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        if i >= config.get('len_epoch', 1e9):
            break

        logger.set_step(logger.step + 1, mode='train')
        batch = to(batch, config['device'])
        print("batch", batch['input_ids'].shape)

        output = model(**batch)

        loss = output.loss

        loss.backward()

        # cause of simple fowrard backward cause OOM, need to clear memory even with batch size 1
        del batch
        del output
        del loss
        gc.collect()


def train_epoch(model, opts, loader, config, logger, metric=None, scheduler=None):
    model.train()

    for i, batch in enumerate(tqdm(iter(loader))):
        if i >= config.get('len_epoch', 1e9):
            break

        logger.set_step(logger.step + 1, mode='train')
        batch = to(batch, config['device'])
        print("batch", batch['input_ids'].shape)

        with torch.cuda.amp.autocast():
            output = model(**batch)

            loss = output.loss

        loss.backward()
        print("mem alloc", torch.cuda.memory_allocated() / 1024 ** 3)
        print("max mem alloc", torch.cuda.max_memory_allocated() / 1024 ** 3)

        if config.get('grad_accum_steps') is None or (i + 1) % config.get('grad_accum_steps') == 0:
            clip_grad_norm_(
                model.parameters(), config.clip_grad_norm
            )
            for opt in opts:
                move_adam_stats(opt, config['device'])
                opt.step()
                opt.zero_grad(set_to_none=True)
                move_adam_stats(opt, 'cpu')

            torch.cuda.empty_cache()

        np_loss = loss.detach().cpu().numpy()
        print("loss", np_loss)

        logger.add_scalar("classification_loss", np_loss)

        del batch
        del output
        del loss
        gc.collect()

        if scheduler is not None:
            scheduler.step()

            logger.add_scalar(
                "learning rate", scheduler.get_last_lr()[0]
            )


@torch.inference_mode()
def evaluate(model, loader, config, loss_fn, metric, logger=None):
    model.eval()
    metrics = defaultdict(list)

    for i, (x, y) in enumerate(tqdm(iter(loader))):
        if logger is not None:
            logger.set_step(logger.step + 1, mode='val')

        x = x.to(config['device'], non_blocking=True)
        y = y.to(config['device'], non_blocking=True)

        res = model(x)

        loss = loss_fn(y, res)

        metrics['accuracy'].append(metric((x.cpu(), y.cpu(), res.cpu())))
        metrics['classification_loss'].append(loss.detach().cpu().numpy())

    if logger is not None:
        for metric_name, metric_val in metrics.items():
            logger.add_scalar(metric_name, np.mean(metric_val))

            print(metric_name, np.mean(metric_val))

    return metrics


@torch.inference_mode()
def inference(model, loader, config):
    model.eval()
    results = []

    for i, x in enumerate(tqdm(iter(loader))):
        x = x.to(config['device'], non_blocking=True)

        res = model(x)

        results.append((res[:, -1] > 0).detach().cpu().numpy())

    return np.concatenate(results) * 2 - 1

