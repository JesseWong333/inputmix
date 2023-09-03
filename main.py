# 

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datasets.compcar_dataset import CompCarsDataset
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy

import utils
from models.fusion_vis import FusionTransformer
from inputmix import InputMix

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # create fusion_model
    model = FusionTransformer(args)
    model.cuda()
    model.eval()

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_val = CompCarsDataset("./sampled_compcars_test.txt", args.data_path, transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256)),
        pth_transforms.RandomCrop((224, 224)),       
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = CompCarsDataset("./sampled_compcars_train.txt", args.data_path, transform=train_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True) # shuffle默认True
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    model.train()
   
    num_steps = int(args.epochs * len(train_loader))
    warmup_steps = int(args.warm_up_epochs * len(train_loader))

    optimizer = torch.optim.AdamW(
        [{'params': model.parameters()}],
        lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        weight_decay=1e-8)
    
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=1e-6 * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  
            warmup_lr_init=1e-6 * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
  
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, optimizer, train_loader, epoch, args.n_last_blocks, scheduler, args.avgpool_patchtokens, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens, args)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict_model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

            # save best
            if test_stats["acc1"] >= best_acc:
                 torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))


    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, optimizer, loader, epoch, n, lr_scheduler, avgpool, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if args.fusionmix > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        mix_fn = InputMix(args.num_labels, args.fusionmix, args.fusion_mix_lam)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    for i, (inp, target) in enumerate(metric_logger.log_every(loader, 5, header)):
        inp = [inp[view_id] for view_id in args.view_ids ]
        if args.fusionmix > 0.:
            inp, target = mix_fn(inp, target)
        inp = [i.cuda(non_blocking=True) for i in inp]
        target = target.cuda(non_blocking=True)
        output = model( inp, args.fusion_layer, n=1)    
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step_update(epoch * len(loader) + i)

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, n, avgpool, args):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = [inp[view_id] for view_id in args.view_ids ]
        inp = [i.cuda(non_blocking=True) for i in inp]
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model( inp, args.fusion_layer, n=1)
          
        loss = criterion(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        batch_size = inp[0].shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
       
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
   
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Traning with linear classification')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')  # set to 8 for 
    parser.add_argument('--pretrained_weights', default='./exp_RGBD/checkpoint_best.pth.tar', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="state_dict_model", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_linear_key", default="state_dict", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--warm_up_epochs', default=10, type=int, help='Number of warm up epochs of training.')
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/mimer/NOBACKUP/groups/naiss2023-22-19/data/compcars/image', type=str)
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./exp_cross_attention_no_share", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1010, type=int, help='Number of labels for linear classifier')
    # parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--evaluate', default=False, help='evaluate model on validation set')
    parser.add_argument('--fusion_layer', default=10, type=int, help='where start to fuse')
    parser.add_argument('--fusionmix', default=0.2, type=float, help='where start to fuse')
    # parser.add_argument('--fusion_mix_lam', default=[0.5, 0.5], type=list, help='where start to fuse')
    parser.add_argument('--fusion_mix_lam', nargs='+', default=[0.5, 0.5], type=float, help='List of lam')
    parser.add_argument('--weight_loss', default=0.5, type=float, help='where start to fuse')
    parser.add_argument('--num_views', default=2, type=int, help='how many views to use')
    parser.add_argument('--view_ids', nargs='+', default=[2, 3], type=int, help='which view to use')
    args = parser.parse_args()
    eval_linear(args)
