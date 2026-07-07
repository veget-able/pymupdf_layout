#ifndef TGF_MODEL_H
#define TGF_MODEL_H

#include "tgif_image.h"

#include <stddef.h>
#include <stdint.h>

#define TGF_MAGIC_V1 0x31464754u
#define TGF_MAGIC_V2 0x32464754u
#define TGF_MAX_NAME 48

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum tgif_tensor_type {
	TGF_TENSOR_F32 = 1
} tgif_tensor_type;

typedef enum tgif_layer_type {
	TGF_LAYER_CONV2D = 1,
	TGF_LAYER_DEPTHWISE_CONV2D = 2,
	TGF_LAYER_CONV1D = 3,
	TGF_LAYER_RELU = 4,
	TGF_LAYER_ADD = 5,
	TGF_LAYER_REDUCE_MEAN = 6,
	TGF_LAYER_RESIZE1D = 7,
	TGF_LAYER_SIGMOID = 8,
	TGF_LAYER_EXPAND_H = 9,
	TGF_LAYER_EXPAND_V = 10,
	TGF_LAYER_CONCAT = 11
} tgif_layer_type;

typedef struct tgif_tensor_desc {
	char name[TGF_MAX_NAME];
	uint32_t type;
	uint32_t rank;
	uint32_t dims[4];
	uint64_t data_offset;
	uint64_t element_count;
} tgif_tensor_desc;

typedef struct tgif_model_header {
	uint32_t magic;
	uint32_t version;
	uint32_t input_w;
	uint32_t input_h;
	uint32_t out_w;
	uint32_t out_h;
	uint32_t aux_w;
	uint32_t aux_h;
	uint32_t tensor_count;
	uint32_t layer_count;
} tgif_model_header;

typedef struct tgif_layer_desc {
	uint8_t op_type;
	uint8_t input0;
	uint8_t input1;
	uint8_t input2;
	uint8_t output;
	//uint32_t reserved;
	int32_t params_i[8];
	//float params_f[4];
} tgif_layer_desc;

typedef struct tgif_model {
	tgif_model_header header;
	tgif_tensor_desc *tensors;
	tgif_layer_desc *layers;
	uint8_t *blob;
	size_t blob_size;
} tgif_model;

extern tgif_model table_grid_model;

#endif

#ifdef __cplusplus
}
#endif
