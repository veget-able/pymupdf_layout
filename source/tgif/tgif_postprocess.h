#ifndef TGF_POSTPROCESS_H
#define TGF_POSTPROCESS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

size_t tgif_ccl_1d_centroids(
	const float *heatmap,
	size_t length,
	float threshold,
	float *out_centroids,
	size_t out_capacity
);

#endif

#ifdef __cplusplus
}
#endif
