# S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging

Official implementation of [AAAI2025] [S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging](https://arxiv.org/abs/2412.13156).

S2S2 is a training pipeline that improves the robustness of a medical image segmentation model.

## Datasets

To obtain the datasets, please refer to the corresponding directory. You will also need to preprocess the data (generate the semantic stack) using ControlNet. Please refere the the [ControlNet](./ControlNet) directory for detail.

## Experiments

To reproduce the experimental results for each base model, please refer to the corresponding directory.

+ [TransUNet](./TransUNet)
+ [FCBFormer](./FCBFormer)
+ [SLAug](./SLAug) 

## License

The license for our code is in [LICENSE](./LICENSE). The original licenses for each project is in the corresponding folder.

## Acknowledgements

Our codes are built on top of [ControlNet](https://github.com/lllyasviel/ControlNet), [TransUNet](https://github.com/Beckschen/TransUNet/tree/main), [FCBFormer](https://github.com/ESandML/FCBFormer/tree/main), and [SLAug](https://github.com/Kaiseem/SLAug). We thank the commuity for their efforts in developing these open-source projects.

## Citation
Please consider to cite our paper. Thank you!

```
@article{pan2024s2s2,
  title={S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging},
  author={Yimu Pan and Sitao Zhang and Alison D. Gernand and Jeffery A. Goldstein and James Z. Wang},
  journal={arXiv preprint arXiv:2412.13156},
  year={2024}
}
```