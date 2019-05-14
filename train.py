#!/usr/bin/env python

from collections import OrderedDict
import argparse
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader
from cutmix import CutMixCriterion

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument(
        '--block_type',
        type=str,
        default='basic',
        choices=['basic', 'bottleneck'])
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--base_channels', type=int, default=16)

    # cutmix
    parser.add_argument('--use_cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)

    # run config
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # optim config
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cosine',
        choices=['multistep', 'cosine'])
    parser.add_argument('--milestones', type=str, default='[150, 225]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument(
        '--no-tensorboard', dest='tensorboard', action='store_false')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    model_config = OrderedDict([
        ('arch', 'resnet_preact'),
        ('block_type', args.block_type),
        ('depth', args.depth),
        ('base_channels', args.base_channels),
        ('input_shape', (1, 3, 32, 32)),
        ('n_classes', 10),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('scheduler', args.scheduler),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
        ('use_cutmix', args.use_cutmix),
        ('cutmix_alpha', args.cutmix_alpha),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('device', args.device),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, criterion, train_loader, run_config,
          writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_config['tensorboard'] and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        data = data.to(device)
        if isinstance(targets, (tuple, list)):
            targets1, targets2, lam = targets
            targets = (targets1.to(device), targets2.to(device), lam)
        else:
            targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()

        num = data.size(0)
        if isinstance(targets, (tuple, list)):
            targets1, targets2, lam = targets
            correct1 = preds.eq(targets1).sum().item()
            correct2 = preds.eq(targets2).sum().item()
            accuracy = (lam * correct1 + (1 - lam) * correct2) / num
        else:
            correct_ = preds.eq(targets).sum().item()
            accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    train_log = OrderedDict({
        'epoch':
        epoch,
        'train':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'time': elapsed,
        }),
    })
    return train_log


def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            if run_config['tensorboard'] and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image('Test/Image', image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    test_log = OrderedDict({
        'epoch':
        epoch,
        'test':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'time': elapsed,
        }),
    })
    return test_log


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']
    data_config = config['data_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(
        outdir.as_posix()) if run_config['tensorboard'] else None

    # save config as json file in output directory
    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(
        optim_config['batch_size'], run_config['num_workers'], data_config)

    # model
    model = load_model(config['model_config'])
    model.to(torch.device(run_config['device']))
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    if data_config['use_cutmix']:
        train_criterion = CutMixCriterion(reduction='mean')
    else:
        train_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_criterion = nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])
    if optim_config['scheduler'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=optim_config['milestones'],
            gamma=optim_config['lr_decay'])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, optim_config['epochs'], 0)

    # run test before start training
    test(0, model, test_criterion, test_loader, run_config, writer)

    epoch_logs = []
    for epoch in range(1, optim_config['epochs'] + 1):
        scheduler.step()

        train_log = train(epoch, model, optimizer, train_criterion,
                          train_loader, run_config, writer)
        test_log = test(epoch, model, test_criterion, test_loader, run_config,
                        writer)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        with open(outdir / 'log.json', 'w') as fout:
            json.dump(epoch_logs, fout, indent=2)

        state = OrderedDict([
            ('config', config),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('accuracy', test_log['test']['accuracy']),
        ])
        model_path = outdir / 'model_state.pth'
        torch.save(state, model_path)


if __name__ == '__main__':
    main()
