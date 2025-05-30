#!/usr/bin/env python3
"""
generate prediction on unlabeled data
"""
import argparse
import json
import logging
import os
import time
from contextlib import suppress

import numpy as np
from timm.models import load_checkpoint
from timm.utils import setup_default_logging
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.transforms import add_noise, add_random_mask
from datasets.mp_liver_dataset import MultiPhaseLiverDataset
from utils.metrics import *
from models.ple_fusion import PleTrans

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='Prediction on unlabeled data')

parser.add_argument('--img_size', default=(16, 128, 128),
                    type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(14, 112, 112),
                    type=int, nargs='+', help='cropped image size.')
parser.add_argument('--num_phase', default=8, type=int)

parser.add_argument('--data_dir', default='datas/classification/lld-mmri-2023/images', type=str)
parser.add_argument('--val_anno_file', default='datas/classification/lld-mmri-2023/labels/labels_test.txt', type=str)
parser.add_argument('--val_transform_list',
                    default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', default='ple_fusion',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=7,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='classification/output/ple_fusion/model_best.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--results-dir', default='classification/output/ple_fusion/', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')

parser.add_argument('--noise', type=bool, default=False)
parser.add_argument('--noise_s', type=int, default=0)
parser.add_argument('--mask', type=bool, default=False)
parser.add_argument('--mask_ratio', type=float, default=0)


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    amp_autocast = suppress  # do nothing

    # create model
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
        assert hasattr(
            model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = model.cuda()

    dataset = MultiPhaseLiverDataset(args, is_training=False)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        shuffle=False)

    predictions = []
    labels = []
    sne_feature = []
    inputs = None
    times = []

    model.eval()
    pbar = tqdm(total=len(dataset))
    with torch.no_grad():
        for (input_, target) in loader:
            target = target.cuda()
            input_ = input_.cuda()

            # Additional process
            if args.noise:
                input_ = add_noise(input, args.noise_s)
            if args.mask:
                input_ = add_random_mask(input, args.mask_ratio)

            # compute output
            with amp_autocast():
                time_start = time.time()
                output = model(input_)
                time_end = time.time()
            times.append(time_end - time_start)
            predictions.append(output)
            labels.append(target)
            pbar.update(args.batch_size)
        pbar.close()
    return process_prediction(predictions, labels, times)


def process_prediction(outputs, targets, times):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach().cpu().numpy()
    time_all = sum(times)
    time_per_img = time_all / targets.shape[0]
    Accuracy = ACC(outputs.cpu().numpy(), targets)
    f1_score = F1_score(outputs.cpu().numpy(), targets)
    kappa_score = Cohen_Kappa(outputs.cpu().numpy(), targets)
    print(f"Acc: {Accuracy}   f1_score: {f1_score}    kappa: {kappa_score}   time_per_img: {time_per_img}")
    pred_score = torch.softmax(outputs, dim=1)
    return pred_score.cpu().numpy()


def write_score2json(score_info, args):
    score_info = score_info.astype(float)
    score_list = []
    anno_info = np.loadtxt(args.val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'prediction': pred,
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    save_name = os.path.join(args.results_dir, args.model + '.json')
    file = open(save_name, 'w')
    file.write(json_data)
    file.close()
    _logger.info(f"Prediction has been saved to '{save_name}'.")


def main():
    setup_default_logging()
    args = parser.parse_args()

    # args.noise = True
    # for s in range(5, 100, 5):
    #     args.noise_s = s
    #     print(f"The noise strength is {s}.")
    #     score = validate(args)
    # args.noise = False

    # args.mask = True
    # for r in range(5, 100, 5):
    #     args.mask_ratio = r * 0.01
    #     print(f"The mask ratio is {r}.")
    #     score = validate(args)
    # args.mask = False

    score = validate(args)
    os.makedirs(args.results_dir, exist_ok=True)
    write_score2json(score, args)


if __name__ == '__main__':
    main()
