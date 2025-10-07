// Network3.c
#include "Network3.h"
#include "sigmoidLayer.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "edge_call.h"
#include "syscall.h"
#include "crypto.h"   // AES_BLOCKLEN, extern iv[], extern key[], AES_* APIs

#ifndef OCALL_PRINT_TIME
#define OCALL_PRINT_TIME 3
#endif

#define OCALL_PRINT_BUFFER 6
#define OCALL_PRINT_OUTPUT 10

/* Use the key declared in crypto.h (avoid multiple definitions) */
extern uint8_t key[AES_BLOCKLEN];
extern uint8_t iv[AES_BLOCKLEN];

/* OCall wrappers */
static unsigned long ocall_print_output(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_OUTPUT, str, strlen(str) + 1, &ret, sizeof(ret));
    return ret;
}
static unsigned long ocall_print_time(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_TIME, str, strlen(str) + 1, &ret, sizeof(ret));
    return ret;
}
static unsigned long ocall_print_buffer(char *str) {
    unsigned long ret;
    ocall(OCALL_PRINT_BUFFER, str, strlen(str) + 1, &ret, sizeof(ret));
    return ret;
}

/* Our own const-safe strlen to match SDK's non-const strlen(char*) */
static size_t cstrlen(const char *s) {
    const char *p = s;
    while (*p) ++p;
    return (size_t)(p - s);
}

static void concatStrings(char *dest, const char *src) {
    size_t dlen = cstrlen(dest);
    size_t slen = cstrlen(src);
    if (dlen + slen + 1 >= 2048) return;
    memcpy(dest + dlen, src, slen + 1);
}

/* List of 1000 class names (truncated for brevity) */
static const char* g_class_names[] = {
    "Indoor",
    "Human Photo",
    "LDR",
    "Plant",
    "Shopping Mall",
    "Beach",
    "Reverse Light",
    "Sunset",
    "Blue Sky",
    "Snow",
    "Night",
    "Text"
};

Network3* Network3_create(void) {
    Network3 *net = (Network3*)malloc(sizeof(Network3));
    if (!net) return NULL;

    ocall_print_time("Network Init 3 Start\n");
    ocall_print_buffer("Initializing Network 3...\n");

    net->fc = FcLayer_create(7, 720, 12);
    if (!net->fc) { free(net); return NULL; }

    net->sigmoid = SigmoidLayer_create(12);  // same output size as FC
    if (!net->sigmoid) { 
        FcLayer_destroy(net->fc); 
        free(net); 
        return NULL; 
    }

    net->class_names = g_class_names;
    net->class_count = sizeof(g_class_names) / sizeof(g_class_names[0]);

    ocall_print_buffer("Initializing Network 3 Done...\n");
    ocall_print_time("Network Init 3 End");
    return net;
}

void Network3_destroy(Network3* net) {
    if (!net) return;
    SigmoidLayer_destroy(net->sigmoid);
    FcLayer_destroy(net->fc);
    free(net);
}

float* Network3_forward(Network3* net, float* input) {
    ocall_print_time("Inference 3 Start");
    ocall_print_buffer("Getting output...\n");

    FcLayer_forward(net->fc, input);

    SigmoidLayer_forward(net->sigmoid, FcLayer_get_output(net->fc));
    float *pfOutput = SigmoidLayer_get_output(net->sigmoid);

    int nOut = FcLayer_get_output_size(net->fc);

    /* Collect activated outputs > 0.5 */
    int ids[12];
    float vals[12];
    size_t count = 0;
    for (int i = 0; i < nOut; ++i) {
        if (pfOutput[i] > 0.3f && count < 12) {
            ids[count] = i;
            vals[count] = pfOutput[i];
            count++;
        }
    }

    ocall_print_time("Inference 3 End");
    ocall_print_time("Communication 3 Start");

    /* Prepare AES-CBC context */
    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);

    /* Build output string (plaintext) */
    char outbuf[2048] = {0};
    for (size_t i = 0; i < count; ++i) {
        char tmp[64];
        /* vals, ids, and class name */
        snprintf(tmp, sizeof(tmp), "%f: %d: %s\n", vals[i], ids[i], net->class_names[ids[i]]);
        concatStrings(outbuf, tmp);
    }

    /* Pad and encrypt */
    size_t len = cstrlen(outbuf);
    size_t padded = len;
    uint8_t *buffer = (uint8_t*)malloc(padded + AES_BLOCKLEN);
    if (!buffer) {
        ocall_print_buffer("Allocation failed in Network3_forward\n");
        return pfOutput;
    }
    memcpy(buffer, outbuf, len);
    pad_buffer(buffer, &padded);
    AES_CBC_encrypt_buffer(&ctx, buffer, padded);

    /* NOTE: ocall_print_output uses strlen, which is unsafe for binary.
             If your OCALL expects binary, switch to an OCALL that takes an explicit length.
             For now we just send the plaintext for visibility OR base64-encode before sending. */
    ocall_print_output(outbuf);  /* safer for current OCALL; change if you want encrypted bytes sent */

    free(buffer);
    ocall_print_time("Communication 3 End");
    return pfOutput;
}