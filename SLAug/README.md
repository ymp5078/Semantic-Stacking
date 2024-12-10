# Setup for SLAug

Please refer to the original GitHub repo [SLAug](https://github.com/Kaiseem/SLAug) for details. The basic setup is below.

This code requires PyTorch 1.10 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```


## Data preparation

We conduct datasets preparation following [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization)

<details>
  <summary>
    <b>1) Abdominal MRI</b>
  </summary>

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 SPIR) to `nii` files in 3D for the ease of reading.

run `./data/abdominal/CHAOST2/s1_dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/abdominal/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/abdominal/CHAOST2/s2_image_normalize.ipynb`

run `./data/abdominal/CHAOST2/s3_resize_roi_reindex.ipynb`

The processed dataset is stored in `./data/abdominal/CHAOST2/processed/`

</details>

<details>
  <summary>
    <b>1) Abdominal CT</b>
  </summary>

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABSCT/CT/` directory

1.Pre-processing downloaded images

run `./data/abdominal/SABS/s1_intensity_normalization.ipynb` to apply abdominal window.

run `./data/abdominal/SABS/s2_remove_excessive_boundary.ipynb` to remove excessive blank region. 

run `./data/abdominal/SABS/s3_resample_and_roi.ipynb` to do resampling and roi extraction.
</details>

The details for cardiac datasets will be given later.

The original authors also provide the [processed datasets](https://drive.google.com/file/d/1WlXGt3Nffzu1bn6co-qaidHjqWH51smU/view?usp=share_link). Download and unzip the file where the folder structure should look this:

```none
SLAug
├── ...
├── data
│   ├── abdominal
│   │   ├── CHAOST2
│   │   │   ├── processed
│   │   ├── SABSCT
│   │   │   ├── processed
│   ├── cardiac
│   │   ├── processed
│   │   │   ├── bSSFP
│   │   │   ├── LGE
├── ...
```

Please modify `BASEDIR` and `GEN_BASEDIR` in [AbdominalDataset.py](./dataloaders/AbdominalDataset.py), [CardiacDataset.py](./dataloaders/CardiacDataset.py), and [ColonDataset.py](./dataloaders/ColonDataset.py).

## Usage

You can find the training scripts under [scripts](./scripts/)

### Training the model

<details>
  <summary>
    <b>1) Cross-modality Abdominal Dataset</b>
  </summary>
  
For direction CT -> MRI, run the command 
```bash
python main.py --base configs/efficientUnet_SABSCT_to_CHAOS_LSC.yaml --seed 23
```

For direction MRI -> CT, run the command 
```bash
python main.py --base configs/efficientUnet_CHAOS_to_SABSCT_LSC.yaml --seed 23
```

</details>

<details>
  <summary>
    <b>2)  Cross-sequence Cardiac Dataset</b>
  </summary>
  
For direction bSSFP -> LEG, run the command 
```bash
python main.py --base configs/efficientUnet_bSSFP_to_LEG_LSC.yaml --seed 23
```

For direction LEG -> bSSFP, run the command 
```bash
python main.py --base configs/efficientUnet_LEG_to_bSSFP_LSC.yaml --seed 23
```
</details>

<details>
  <summary>
    <b>2)  Cross-site Colon Dataset</b>
  </summary>

For direction CVC -> Kvasir, run the command 
```bash
python main.py --base configs/efficientUnet_CVC_to_Kvasir_LSC.yaml --seed 23
```

For direction Kvasir -> CVC, run the command 
```bash
python main.py --base configs/efficientUnet_Kvasir_to_CVC_LSC.yaml --seed 23
```

</details>


### Inference

<details>
  <summary>
    <b>1) Cross-modality Abdominal Dataset</b>
  </summary>

For direction CT -> MRI (DICE 88.63), run the command 
```bash
python test.py -r logs/2024-03-04T13-31-45_seed23_efficientUnet_SABSCT_to_CHAOS_LSC --epoch 1099
```

For direction MRI -> CT (DICE 83.05), run the command 
```bash
python test.py -r logs/2024-03-04T13-31-45_seed23_efficientUnet_CHAOS_to_SABSCT_LSC --epoch 1099
```


</details>

<details>
  <summary>
    <b>2)  Cross-sequence Cardiac Dataset</b>
  </summary>
  
For direction bSSFP -> LEG, run the command 
```bash
python test.py -r logs/2024-03-04T11-06-43_seed23_efficientUnet_bSSFP_to_LEG_LSC
```

For direction LEG -> bSSFP, run the command 
```bash
python test.py -r logs/2024-03-04T13-29-38_seed23_efficientUnet_LEG_to_bSSFP_LSC
```
</details>

<details>
  <summary>
    <b>2)  Cross-site Colon Dataset</b>
  </summary>

For direction CVC -> Kvasir, run the command 
```bash
python test.py -r /logs/2024-03-04T13-33-29_seed23_efficientUnet_CVC_to_Kvasir_LSC --use_2d
```

For direction Kvasir -> CVC, run the command 
```bash
python test.py -r logs/2024-03-04T14-20-59_seed23_efficientUnet_Kvasir_to_CVC_LSC --use_2d
```

</details>


