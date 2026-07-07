#ifndef TGF_RUNTIME_H
#define TGF_RUNTIME_H

#include "tgif_image.h"
#include "tgif_model.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct tgif_runtime_output {
	float *h_heatmap;
	size_t h_len;
	float *v_heatmap;
	size_t v_len;
	float *reg;
	size_t reg_count;
	float *ctr;
	size_t ctr_count;
} tgif_runtime_output;

int tgif_run_model(tgif_context *ctx, const tgif_model *model, const tgif_image *image, tgif_runtime_output *out);
void tgif_runtime_output_destroy(tgif_context *ctx, tgif_runtime_output *out);

typedef struct
{
	int h_len;
	float *h;
	int v_len;
	float *v;
} tgif_grid;

tgif_grid *tgif_get_grid(tgif_context *ctx, tgif_runtime_output *rt);
void tgif_grid_destroy(tgif_context *ctx, tgif_grid *grid);

#endif

#ifdef __cplusplus
}
#endif
