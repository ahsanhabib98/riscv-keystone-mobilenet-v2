#include "batchnormalLayer.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ---- tiny fast sqrt() so we don't need -lm ----
   Quake-style fast inverse sqrt + 2 Newton steps. */
static inline float inv_sqrt_fast(float x) {
    if (x <= 0.0f) return 0.0f;
    float xhalf = 0.5f * x;
    union { float f; uint32_t i; } u = { x };
    u.i = 0x5f3759df - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - xhalf * y * y);
    y = y * (1.5f - xhalf * y * y);
    return y;
}
static inline float sqrtf_fast(float x) {
    if (x <= 0.0f) return 0.0f;
    return x * inv_sqrt_fast(x);
}

BatchNormalLayer* BatchNormalLayer_create(int fileNum, int nInputNum, int nInputWidth) {
    BatchNormalLayer* layer = (BatchNormalLayer*)malloc(sizeof(BatchNormalLayer));
    if (!layer) return NULL;

    layer->nInputNum   = nInputNum;
    layer->nInputWidth = nInputWidth;
    layer->nInputSize  = nInputWidth * nInputWidth;
    layer->pfOutput    = (float*)malloc((size_t)nInputNum * layer->nInputSize * sizeof(float));
    layer->pfMean      = NULL;
    layer->pfVar       = NULL;
    layer->pfFiller    = NULL;
    layer->pfBias      = NULL;

    BatchNormalLayer_read_param(layer, fileNum);
    return layer;
}

void BatchNormalLayer_destroy(BatchNormalLayer* layer) {
    if (!layer) return;
    free(layer->pfOutput);
    free(layer->pfMean);
    free(layer->pfVar);
    free(layer->pfFiller);
    free(layer->pfBias);
    free(layer);
}

void BatchNormalLayer_forward(BatchNormalLayer* layer, const float *pfInput) {
    int N = layer->nInputNum;
    int S = layer->nInputSize;
    for (int i = 0; i < N; i++) {
        float mean   = layer->pfMean[i];
        float var    = layer->pfVar[i];
        float filler = layer->pfFiller[i];
        float bias   = layer->pfBias[i];

        /* epsilon to avoid div-by-zero */
        float denom = sqrtf_fast(var + 1e-5f);

        for (int j = 0; j < S; j++) {
            int idx = i * S + j;
            layer->pfOutput[idx] = filler * ((pfInput[idx] - mean) / denom) + bias;
        }
    }
}

float* BatchNormalLayer_get_output(BatchNormalLayer* layer) {
    return layer ? layer->pfOutput : NULL;
}

int BatchNormalLayer_get_output_size(BatchNormalLayer* layer) {
    return layer ? layer->nInputNum * layer->nInputSize : 0;
}

void BatchNormalLayer_read_param(BatchNormalLayer* layer, int fileNum) {
    int N = layer->nInputNum;
    if (fileNum == 1) {
        static const float mean_vals[] = {
            -5.15257e-07, -0.0405053, 0.00492582, 0.0669159, -0.136017, 0.000818472, -0.0590174, 0.0439983, -9.37478e-08, 5.7905e-08, 
            -1.47252e-08, 0.13997, 0.0448315, -1.06377e-07, 1.41654e-07, 0.00141437, 0.00469195, -5.83113e-05, -0.0483438, 0.0512685, 
            -0.0107327, 0.119756
        };
        static const float var_vals[] = {
            7.22756e-07, 13515.8, 845.159, 2829.88, 10725.1, 621.258, 18912.9, 9609.89, 3.94632e-07, 9.67993e-09, 
            2.32901e-08, 9920.35, 12564.3, 5.23604e-09, 1.55803e-07, 644.579, 726.55, 500.854, 6289.15, 10729.1, 
            1881.94, 9839.35
        };
        static const float filler_vals[] = {
            -5.15257e-07, -0.0405053, 0.00492582, 0.0669159, -0.136017, 0.000818472, -0.0590174, 0.0439983, -9.37478e-08, 5.7905e-08, 
            -1.47252e-08, 0.13997, 0.0448315, -1.06377e-07, 1.41654e-07, 0.00141437, 0.00469195, -5.83113e-05, -0.0483438, 0.0512685, 
            -0.0107327, 0.119756
        };
        static const float bias_vals[] = {
            -3.64502e-06, 0.682912, 0.416179, 0.443006, 0.273728, 0.432146, 0.753429, 0.47652, 1.19725e-07, -1.21712e-08, 
            1.29549e-08, 0.661347, 0.469142, 1.5963e-08, 1.12549e-08, 0.38565, 0.477055, 0.400039, 0.510819, 0.370828, 
            -0.283554, 0.270707
        };

        layer->pfMean   = (float*)malloc((size_t)N * sizeof(float));
        layer->pfVar    = (float*)malloc((size_t)N * sizeof(float));
        layer->pfFiller = (float*)malloc((size_t)N * sizeof(float));
        layer->pfBias   = (float*)malloc((size_t)N * sizeof(float));
        memcpy(layer->pfMean,   mean_vals,   (size_t)N * sizeof(float));
        memcpy(layer->pfVar,    var_vals,    (size_t)N * sizeof(float));
        memcpy(layer->pfFiller, filler_vals, (size_t)N * sizeof(float));
        memcpy(layer->pfBias,   bias_vals,   (size_t)N * sizeof(float));
    }
}
