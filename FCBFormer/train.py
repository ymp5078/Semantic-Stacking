import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
from lsc_loss import BinaryKDLoss, ConsistencyLoss


def train_epoch(args, model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, kd_loss, FCB_consist_loss,TB_consist_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    loss_kd_accumulator = []
    loss_kd = torch.zeros(1)
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        if args.gen_image or args.kd_loss_weight > 0. or args.consist_loss_weight > 0.:
            assert len(batch) == 3, 'need the generated image'
            gen_data = batch[2].to(device)
            output, TB_features, FCB_features, out_featDures = model(data, return_features=True)
            output_gen, TB_features_gen, FCB_features_gen, out_features_gen = model(gen_data, return_features=True)
            loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
            loss_gen = Dice_loss(output_gen, target) + BCE_loss(torch.sigmoid(output_gen), target)
            loss += loss_gen
            if args.kd_loss_weight > 0.:
                # loss_kd = args.kd_loss_weight * FCB_consist_loss(out_features,out_features_gen)
                loss_kd = args.kd_loss_weight * kd_loss(out_features_gen,out_featDures)
                loss += loss_kd
            if args.consist_loss_weight > 0.:
                loss_TB_consist = args.consist_loss_weight * TB_consist_loss(TB_features,TB_features_gen)
                loss_FCB_consist = args.consist_loss_weight * FCB_consist_loss(FCB_features,FCB_features_gen)
                loss += loss_TB_consist
                loss += loss_FCB_consist
            # print(TB_features.shape,FCB_features.shape,out_featDures.shape)
        else:
            output = model(data)
            loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        loss_kd_accumulator.append(loss_kd.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tLoss_consist: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    loss_kd.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tAverage Loss_consist: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    np.mean(loss_kd_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "Kvasir_gen":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC_gen":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "Kvasir_aug":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC_aug":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size, gen_input_path=args.gen_input_path, use_aug=args.kd_loss_weight > 0. or args.consist_loss_weight > 0.
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    
    kd_loss = ConsistencyLoss(loss_factor=1.,feature_dim=1)
    FCB_consist_loss = ConsistencyLoss(loss_factor=1.,feature_dim=1)
    TB_consist_loss = ConsistencyLoss(loss_factor=1.,feature_dim=1)

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    if args.mgpu == "true":
        print('using multi gpu')
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        kd_loss,
        FCB_consist_loss,
        TB_consist_loss,
        perf,
        model,
        optimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
        kd_loss,
        FCB_consist_loss,
        TB_consist_loss,
        perf,
        model,
        optimizer,
    ) = build(args)

    if not os.path.exists(os.path.join(args.save_path,"Trained models")):
        os.makedirs(os.path.join(args.save_path,"Trained models"))

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                args, model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss, kd_loss, FCB_consist_loss,TB_consist_loss,
            )
            test_measure_mean, test_measure_std = test(
                model, device, val_dataloader, epoch, perf
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                os.path.join(args.save_path,"Trained models/FCBFormer_" + args.dataset + ".pt") ,
            )
            prev_best_test = test_measure_mean


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir", "CVC", "Kvasir_gen", "CVC_gen", "Kvasir_aug", "CVC_aug"])
    parser.add_argument("--save-path", type=str, required=True, dest="save_path")
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument('--gen_image', action='store_true',
                        default=False, help='the experience name')
    parser.add_argument('--consist_loss_weight', type=float,  default=0.0,
                        help='weight for consist loss')
    parser.add_argument('--kd_loss_weight', type=float,  default=0.0,
                        help='weight for kd loss')
    parser.add_argument(
        "--gen-input-path", type=str, default=None, help="the path to the folder that store the generated images in npz"
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    train(args)


if __name__ == "__main__":
    main()
