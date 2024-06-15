import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.test_dataset == "Kvasir":
        img_path = args.root + "images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.test_dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1
    )

    _, test_indices, _ = dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    state_dict = torch.load(
        os.path.join(args.save_path,"Trained models/FCBFormer_{}.pt".format(args.train_dataset))
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)

    if not os.path.exists(os.path.join(args.save_path,"Predictions")):
        os.makedirs(os.path.join(args.save_path,"Predictions"))
    if not os.path.exists(os.path.join(args.save_path,"Predictions/Trained on {}".format(args.train_dataset))):
        os.makedirs(os.path.join(args.save_path,"Predictions/Trained on {}".format(args.train_dataset)))
    if not os.path.exists(
        os.path.join(args.save_path,"Predictions/Trained on {}/Tested on {}".format(
            args.train_dataset, args.test_dataset
        ))
    ):
        os.makedirs(
            os.path.join(args.save_path,"Predictions/Trained on {}/Tested on {}".format(
                args.train_dataset, args.test_dataset
            ))
        )

    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite(
            os.path.join(args.save_path,"Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i]).split('.')[0]+'_tg.png'
            )),
            np.array(target.cpu())[0,0] * 255,
        )
        out_image = np.array(data.cpu())[0].transpose(1,2,0)
        out_image_max, out_image_min = out_image.max(), out_image.min()
        out_image = (out_image - out_image_min)/(out_image_max - out_image_min) * 255
        cv2.imwrite(
            os.path.join(args.save_path,"Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i]).split('.')[0]+'_img.png'
            )),
            cv2.cvtColor(out_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(args.save_path,"Predictions/Trained on {}/Tested on {}/{}".format(
                args.train_dataset, args.test_dataset, os.path.basename(target_paths[i]).split('.')[0]+'_pred.png'
            )),
            predicted_map * 255,
        )
        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument("--save-path", type=str, required=True, dest="save_path")
    parser.add_argument(
        "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC","Kvasir_gen", "CVC_gen","Kvasir_aug", "CVC_aug"]
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
    )
    parser.add_argument("--data-root", type=str, required=True, dest="root")

    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()

