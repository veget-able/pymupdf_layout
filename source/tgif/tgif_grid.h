#ifndef TGF_GRID_H
#define TGF_GRID_H

#include "tgif_image.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct tgif_grid_lines {
	float *h_lines;
	size_t h_count;
	float *v_lines;
	size_t v_count;
} tgif_grid_lines;

void tgif_grid_lines_destroy(tgif_context *ctx, tgif_grid_lines *grid);

#endif

#ifdef __cplusplus
}
#endif
