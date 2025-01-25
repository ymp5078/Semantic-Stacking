# S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging

Official implementation of [AAAI2025] [S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging](https://arxiv.org/abs/2412.13156).

## Summary

S2S2 is a training pipeline that improves the robustness of a medical image segmentation model.

The S2S2 training pipeline is summarize as follow:

1. **Tune Generative Model**
    - Adapt the generative model to the specific characteristics of the dataset through fine-tuning
2. **Generate Augmented Copies**
    - Leverage the fine-tuned generative model to create multiple augmented variants of each sample in the dataset.
3. **Train Using Semantic Stacking**
    - Employ the **semantic stacking** technique to train the model using the augmented dataset, enhancing its capacity to generalize.

## Semantic Stacking
This technique is inspired by traditional image stacking used in image denoising, where multiple noisy images are combined to estimate a clearer image. Similarly, **semantic stacking** involves combining multiple images from a given semantic segmentation map to estimate a denoised semantic representation. Directly processing a large stack of images in each training iteration can be computationally intensive. Through the analysis presented in our paper, we have simplified the **semantic stacking** technique to the following:
```python
""" 
    At each iteration, process two images for each mask based on Sec 3.2.
    The original image is always used as one of the images.
"""
for image_0, mask in dataset:

    # Traditional segmentation pipeline (baseline approach)
    # ----------------------------------
    # Encode the original image
    enc_feat_0 = seg_encoder(image_0)
    # Decode the encoded features
    dec_feat_0 = seg_decoder(enc_feat_0)
    # Perform pixel-level classification
    logits_0 = linear(dec_feat_0)
    # Compute segmentation loss
    loss = seg_loss(image_0, mask)
    # ----------------------------------

    # Semantic stacking extension (proposed approach)
    # ----------------------------------
    # Step 1: Obtain a new image based on the mask using the fine-tuned generative model
    image_1 = finetuned_gen_model(mask)
    # Step 2: Apply the traditional pipeline to the generated image
    enc_feat_1 = seg_encoder(image_1)
    dec_feat_1 = seg_decoder(enc_feat_1)
    logits_1 = linear(dec_feat_1)
    loss += seg_loss(image_1, mask)

    # Step 3: Add consistency losses
    # Encourage similarity in encoder outputs 
    loss += alpha_enc * consist_loss(enc_feat_0, enc_feat_1)
    # Encourage similarity in decoder outputs
    loss += alpha_dec * consist_loss(dec_feat_0, dec_feat_1)
    # ----------------------------------

    # Update model parameters
    loss.backward()
    optimizer.step()
```

## Datasets

To obtain the datasets, please refer to [TransUNet](./TransUNet), [FCBFormer](./FCBFormer), and [SLAug](./SLAug). You will also need to preprocess the data (generate the semantic stack) using ControlNet. Please refere the the [ControlNet](./ControlNet) directory for detail.

For your convenience and reproducibility, we provide the generated images under [gen_images](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/ymp5078_psu_edu/EiswWUefCMNNkcKyL7GtdckB8Vr1LAu5_aPaSOYPhkN24A?e=IE5BTo).

## Experiments

To reproduce the experimental results for each base model, please refer to the corresponding directory.

+ [TransUNet](./TransUNet)
+ [FCBFormer](./FCBFormer)
+ [SLAug](./SLAug) 

The model checkpoints for the experiments are available under [model_weights](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/ymp5078_psu_edu/EiswWUefCMNNkcKyL7GtdckB8Vr1LAu5_aPaSOYPhkN24A?e=IE5BTo).

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