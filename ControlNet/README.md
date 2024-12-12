# Setup for ControlNet 2.1
Please refer to the original GitHub [repo](https://github.com/lllyasviel/ControlNet/tree/main) for detail. We will summarize the main steps below

First, create a new conda environment

    conda env create -f environment.yaml
    conda activate control

Then, obtain [v2-1_512-ema-pruned.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/tree/main) and place it under [models](./models/)

Next, use the following code to convert the weight to the correct format.

    python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt

Assume you have the datasets ready for [TransUNet](../TransUNet/README.md), [FCBFormer](../FCBFormer/README.md), and [SLAug](../TransUNet/README.md), change the [data_dir] and [absolute_path] in [train_sd21.py](./train_sd21.py). For SLAug, please change [data_dir] in the datasets files [AbdominalDataset.py](./datasets/AbdominalDataset.py) and [CardiacDataset.py](./datasets/CardiacDataset.py) directly following SLAug. 


## Usage

All the tuning and generation scripts are in [scripts](./scripts/). You can also use the following:

### Training

To fine-tune the ControlNet, run

    python train_sd21.py --dataset [dataset_name]

### Generation

To generate images using the trained ControlNet, run

    python generate_seg2image.py --out [gen_output_path] --dataset [dataset_name] --model_path '[checkpoint_path]'

You will need to record [gen_output_path] for the models.

If you want to use your own dataset, port your dataset.py to [datasets](./datasets/) and modify [train_sd21.py](train_sd21.py) to include your dataset.