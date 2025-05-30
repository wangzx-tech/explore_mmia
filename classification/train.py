#!/usr/bin/env python3
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import yaml
from timm.models import safe_model_name, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import *

from utils.metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
from models.ple_fusion import PleTrans

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='LLD-MMRI 2023 Training')
# Dataset parameters
parser.add_argument('--data_dir', default='/home/wangzhixian/project/mia-2023/datas/classification/lld-mmri-2023/images', type=str)
parser.add_argument('--train_anno_file', default='/home/wangzhixian/project/mia-2023/datas/classification/lld-mmri-2023/labels/labels_train.txt', type=str)
parser.add_argument('--val_anno_file', default='/home/wangzhixian/project/mia-2023/datas/classification/lld-mmri-2023/labels/labels_val.txt', type=str)
parser.add_argument('--train_transform_list', default=['random_crop',
                                                       'z_flip',
                                                       'x_flip',
                                                       'y_flip',
                                                       'rotation', ],
                    nargs='+', type=str)
parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
parser.add_argument('--img_size', default=(16, 128, 128), type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(14, 112, 112), type=int, nargs='+', help='cropped image size.')
parser.add_argument('--flip_prob', default=0.5, type=float, help='Random flip prob (default: 0.5)')
parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
parser.add_argument('--rcprob', type=float, default=0.25, help='Random contrast prob (default: 0.25)')
parser.add_argument('--angle', default=45, type=int)
parser.add_argument('--num_phase', default=8, type=int)
parser.add_argument('--tau', default=0., type=float)

# Model parameters
parser.add_argument('--model', default='ple_fusion', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--num-classes', type=int, default=7, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('-b', '--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 2)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Regularization parameters
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 8)')

parser.add_argument('--output', default='output/', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='f1', type=str, metavar='EVAL_METRIC',
                    help='Main metric (default: "f1"')
parser.add_argument('--report-metrics', default=['acc', 'f1', 'recall', 'precision', 'kappa'],
                    nargs='+', choices=['acc', 'f1', 'recall', 'precision', 'kappa'],
                    type=str, help='All evaluation metrics')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--gpu', default='1', type=str)


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.world_size = 1
    amp_autocast = suppress  # do nothing
    _logger.info('Training with a single process on 1 GPUs.')

    args.rank = 0  # global rank
    random_seed(args.seed, args.rank)

    if args.model == "ple_fusion":
        model = PleTrans(
            in_channels=args.crop_size[0],
            embed_dim=64,
            num_phase=args.num_phase,
            num_depth=2,
            num_classes=args.num_classes,
            drop_rate=args.drop
        )
    else:
        _logger.info(f'Invalid model name: {args.model}')

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly
    print(model)

    if args.local_rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # move model to GPU, enable channels last layout if set
    model.cuda()    

    # create optimizer
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    loss_scaler = None

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets/dataloader
    dataset_train = MultiPhaseLiverDataset(args, is_training=True)
    dataset_eval = MultiPhaseLiverDataset(args, is_training=False)
    loader_train = create_loader(dataset_train,
                                 batch_size=args.batch_size,
                                 is_training=True,
                                 num_workers=args.workers)
    loader_eval = create_loader(dataset_eval,
                                batch_size=args.batch_size,
                                is_training=False,
                                num_workers=args.workers)

    train_loss_fn = nn.CrossEntropyLoss().cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None

    if args.rank == 0:
        exp_name = '-'.join([
            datetime.now().strftime("%Y%m%d-%H%M%S"),
            safe_model_name(args.model),
        ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer,
            args=args, model_ema=None,
            amp_scaler=None,
            checkpoint_dir=output_dir, recovery_dir=output_dir,
            decreasing=decreasing, max_history=args.checkpoint_hist)

        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast, loss_scaler=loss_scaler)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, amp_autocast=suppress, loss_scaler=None):
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        # input [batchsize, phase, channel, h, w]
        inputs, target = inputs.cuda(), target.cuda()
        with amp_autocast():
            output = model(inputs)
            loss = model.get_loss(target, loss_fn, args.tau)

        losses_m.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})  '
                    'Accuracy: {accuracy:.3f}'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=inputs.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m,
                        accuracy=ACC(output.detach().cpu().numpy(), target.detach().cpu().numpy())))

        if saver is not None and args.recovery_interval and (last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('lr', lr), ('loss', losses_m.avg)])


@torch.no_grad()
def validate(model, loader, loss_fn, args, amp_autocast=suppress):
    model.eval()
    predictions = []
    labels = []
    last_idx = len(loader) - 1
    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        inputs = inputs.cuda()
        target = target.cuda()

        with amp_autocast():
            output = model(inputs)
        if isinstance(output, (tuple, list)):
            output = output[0]

        predictions.append(output)
        labels.append(target)

    evaluation_metrics = compute_metrics(predictions, labels, loss_fn, args)
    if args.local_rank == 0:
        output_str = 'Test:\n'
        for key, value in evaluation_metrics.items():
            output_str += f'{key}: {value}\n'
        _logger.info(output_str)

    return evaluation_metrics


def compute_metrics(outputs, targets, loss_fn, args):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()

    loss = loss_fn(outputs, targets).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
    ])
    return metrics


def gather_data(input):
    """
    gather data from multi gpus
    """
    output_list = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_list, input)
    output = torch.cat(output_list, dim=0)
    return output


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
