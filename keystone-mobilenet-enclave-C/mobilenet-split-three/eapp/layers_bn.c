/* layers_bn.c */
#include "layers_bn.h"
#include <stdlib.h>

Layers_Bn* Layers_Bn_create(int nInputNum,
                            int nOutputNum,
                            int nInputWidth,
                            int nStride,
                            int fileNum) {
    Layers_Bn* layer = (Layers_Bn*)malloc(sizeof(Layers_Bn));
    if (!layer) return NULL;

    /* Depthwise conv: weightName=fileNum, pad=1, kernel=3, group=1 */
    layer->convDw = ConvLayer_create(fileNum,
                                     nInputNum,
                                     nOutputNum,
                                     nInputWidth,
                                     3, /* kernelWidth */
                                     1, /* pad */
                                     nStride,
                                     1, /* group */
                                     -1 /* biasName */);
    if (!layer->convDw) {
        free(layer);
        return NULL;
    }

    /* BatchNorm */
    layer->bnDw = BatchNormalLayer_create(fileNum,
                                          nOutputNum,
                                          nInputWidth / nStride);
    if (!layer->bnDw) {
        ConvLayer_destroy(layer->convDw);
        free(layer);
        return NULL;
    }

    layer->ReluDw = ReluLayer_create(BatchNormalLayer_get_output_size(layer->bnDw));
    if (!layer->ReluDw) {
        BatchNormalLayer_destroy(layer->bnDw);
        ConvLayer_destroy(layer->convDw);
        free(layer);
        return NULL;
    }

    return layer;
}

void Layers_Bn_destroy(Layers_Bn* layer) {
    if (!layer) return;
    ReluLayer_destroy(layer->ReluDw);
    BatchNormalLayer_destroy(layer->bnDw);
    ConvLayer_destroy(layer->convDw);
    free(layer);
}

void Layers_Bn_forward(Layers_Bn* layer, const float *input) {
    ConvLayer_forward(layer->convDw, input);
    BatchNormalLayer_forward(layer->bnDw, ConvLayer_get_output(layer->convDw));
    ReluLayer_forward(layer->ReluDw, BatchNormalLayer_get_output(layer->bnDw));
}

float* Layers_Bn_get_output(Layers_Bn* layer) {
    return layer ? ReluLayer_get_output(layer->ReluDw) : NULL;
}

int Layers_Bn_get_output_size(Layers_Bn* layer) {
    return layer ? BatchNormalLayer_get_output_size(layer->bnDw) : 0;
}
