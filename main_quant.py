import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from quantization import *
from synthesis import PAINT
from utils import *
from utils.data_utils import find_non_zero_patches

def get_args_parser():
    parser = argparse.ArgumentParser(description="Model Inversion (Quantization)", add_help=False)
    # Environment
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU to use"
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="seed"
    )

    # Experiments
    parser.add_argument(
        "--eval_teacher",
        action="store_true",
        help="only evaluate the teacher model on the validation set"
    )
    parser.add_argument(
        "--model",
        default="deit_base_16_imagenet",
        choices=["deit_base_16_imagenet", "deit_tiny_16_imagenet"],
        help="model to use"
    )
    parser.add_argument(
        "--dataset",
        default="/path/to/dataset",
        help="path to dataset"
    )
    parser.add_argument(
        "--datapool",
        default="/path/to/datapool",
        help="path to datapool"
    )

    # Data Generation
    parser.add_argument(
        "--iterations",
        type=int,
        default=4000,
        help="total number of iterations for inversion"
    )
    parser.add_argument(
        "--synthetic_bs",
        type=int,
        default=32,
        help="batch size for synthetic data"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="number of runs to generate synthetic data (# of images = num_runs * synthetic_bs)"
    )    

    # Downstream Tasks
    parser.add_argument(
        "--calib_batchsize",
        type=int,
        default=200,
        help="batch size for calibration"
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=8,
        help="bit-precision of weights"
    )
    parser.add_argument(
        "--a_bit",
        type=int,
        default=8,
        help="bit-precision of activation"
    )
    parser.add_argument(
        "--prune_it",
        nargs='+',
        type=int,
        default=[-1],
        help='the iteration indexes for inversion stopping; -1: to densely invert data; t1 t2 ... tn: to sparsely invert data and perform inversion stopping at t1, t2, ..., tn'
    )
    parser.add_argument(
        "--prune_ratio",
        nargs='+',
        type=float,
        default=[0],
        help='the proportion of patches to be pruned relative to the current remaining patches; 0: to densely invert data; r1 r2 ... rn: progressively stopping the inversion of a fraction (r1, r2, ..., rn)$$ of patches at iterations (t1, t2, ..., tn), respectively'
    )

    # Paint
    ### Idea 1. Low-Pass Filtering
    parser.add_argument(
        "--lpf_every",
        type=int,
        default=50,
        help="interval for low-pass filtering"
    )
    parser.add_argument(
        "--cutoff_ratio",
        type=float,
        default=0.5,
        help="cutoff ratio for low-pass filtering"
    )

    ### Idea 2. Saliency-to-anchor alignment and peripheral sparsification
    parser.add_argument(
        "--sc_every",
        type=int,
        default=50,
        help="interval for saliency map centering"
    )
    parser.add_argument(
        "--lambda_anchor",
        type=float,
        default=1.0,
        help="lambda for saliency map centering"
    )

    ### Idea 3. Soft targets
    parser.add_argument(
        "--soft_label_alpha",
        type=float,
        default=0.9,
        help="alpha for soft labels (target class probability)"
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="use wandb to log results"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="name of the project"
    )

    return parser.parse_args()

class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit


def get_teacher(name):
    teacher_name = {
        'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
        'deit_base_16_imagenet': 'deit_base_patch16_224',
    }
    if name.split("_")[-1]=='imagenet':
        teacher=build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return teacher

def get_student(name):
    model_zoo = {
        'deit_tiny_16_imagenet': deit_tiny_patch16_224,
        'deit_base_16_imagenet': deit_base_patch16_224,
    }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]

def set_seed(seed):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    args = get_args_parser()

    if args.wandb:
        run_name = (
            f"Paint;{args.model}-"
            f"W{args.w_bit}A{args.a_bit}-"
            f"{'Pit'.join(map(str, args.prune_it))}-"
            f"{'Pr'.join(map(str, args.prune_ratio))}-"
            f"CR{str(args.cutoff_ratio)}-"
            f"LA{str(args.lambda_anchor)}-"
            f"SA{args.soft_label_alpha}"
        )

        wandb.init(
            project=args.project_name,
            name=run_name,
            config=vars(args)
        )
    
    print(f"Arguments: {args}")

    # Set Environments
    set_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load bit-config
    cfg = Config(args.w_bit, args.a_bit)

    # Build model
    model = get_student(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()

    teacher=get_teacher(args.model)
    teacher=teacher.to(device)
    teacher.eval()

    # Build dataloader
    _, val_loader, num_classes, train_transform, _, normalizer = build_dataset(
        args.model.split("_")[0], args.model.split("_")[-1], args.calib_batchsize,
        train_aug=False, keep_zero=True, train_inverse=True, dataset_path=args.dataset
    )

    # --- Teacher-only evaluation ---
    if args.eval_teacher:
        teacher.eval()
        criterion = nn.CrossEntropyLoss().to(device)

        # ViT patch setting
        patch_size = 16 if '16' in args.model else 32
        patch_num  = 197 if patch_size == 16 else 50

        losses = AverageMeter()
        top1   = AverageMeter()
        top5   = AverageMeter()

        val_start_time = time.time()

        loader = val_loader

        pbar = tqdm(loader, desc="Validation", leave=False)

        for i, (data, target) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.no_grad():
                full_index = torch.arange(patch_num, device=device).repeat(data.size(0), 1)

                output = teacher(
                    data,
                    current_abs_index=full_index,
                    next_relative_index=full_index,
                )

                if isinstance(output, (tuple, list)):
                    output = output[0]

                loss = criterion(output, target)

                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

        elapsed = time.time() - val_start_time
        print(" * [Teacher Eval] Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.4f} Time {time:.3f}".format(
            top1=top1, top5=top5, loss=losses, time=elapsed
        ))
        return

    #########################################################
    # Model Inversion (Quantization)
    #########################################################
    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Model Inversion
    iterations = args.iterations
    lr_g = 0.25

    # Hyperparameters
    adv = 0
    bn = 0.0
    oh = 1
    tv1 = 0
    tv2 = 0.0001
    l2 = 0

    # Parameters for Sparsification
    prune_it = args.prune_it
    prune_ratio = args.prune_ratio

    patch_size = 16 if '16' in args.model else 32
    patch_num = 197 if patch_size==16 else 50

    img_tag = run_name
    datapool_path=os.path.join(args.datapool,'%s/%s'%(args.model,img_tag)) # The path to store inverted data
    if os.path.exists(datapool_path):
        shutil.rmtree(datapool_path)
        print(f"Removed existing data at {datapool_path}")

    synthesizer = PAINT(
        teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
            synthesis_batch_size=args.synthetic_bs, sample_batch_size=args.calib_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1,tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None
    )

    print(f"Generating data to {datapool_path}...")
    total_imgs = 0
    for run_idx in tqdm(range(args.num_runs), desc="Synthesizing"):
        start = time.time()
        results = synthesizer.synthesize(
            num_patches=patch_num, prune_it=prune_it, prune_ratio=prune_ratio,
            lpf_every=args.lpf_every, cutoff_ratio=args.cutoff_ratio,
            sc_every=args.sc_every, lambda_anchor=args.lambda_anchor,
            soft_label_alpha=args.soft_label_alpha
        )
        if args.wandb and 'targets' in results:
            wandb.log({"targets": results['targets']}, step=args.seed)

        elapsed = time.time() - start
        print(f"[Run {run_idx+1}/{args.num_runs}] Time: {elapsed:.2f}s")

    dst = synthesizer.data_pool.get_dataset(transform=train_transform)
    print(f"[Calibration] Total synthetic images in pool: {len(dst)}")

    calib_loader = DataLoader(
        dst,
        batch_size=args.calib_batchsize,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    print(f"Calibrating with generated data (full pool pass)...")
    model.model_unfreeze()

    with torch.no_grad():
        for calibrate_data in calib_loader:
            calibrate_data = calibrate_data.to(device)

            current_abs_index = torch.arange(patch_num, device=calibrate_data.device).repeat(calibrate_data.shape[0], 1)
            next_relative_index = torch.cat(
                [
                    torch.zeros(calibrate_data.shape[0], 1, dtype=torch.long).to(calibrate_data.device),
                    find_non_zero_patches(images=calibrate_data, patch_size=patch_size)
                ], dim=1
            )

            _ = model(
                calibrate_data,
                current_abs_index=current_abs_index,
                next_relative_index=next_relative_index
            )

    model.model_quant()
    model.model_freeze()

    # Validate the quantized model
    print("Validating...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, model, criterion, device
    )

    if args.wandb:
        wandb.log({
            "val_loss": float(val_loss),
            "val_prec1": float(val_prec1),
            "val_prec5": float(val_prec5)
        })

def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    pbar = tqdm(val_loader, desc="Validation", leave=False)

    for i, (data, target) in enumerate(pbar):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
