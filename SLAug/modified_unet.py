from segmentation_models_pytorch import Unet

class ModifiedUnet(Unet):
    def forward(self, x, return_features=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if return_features:
            return masks, features[-1],decoder_output
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks