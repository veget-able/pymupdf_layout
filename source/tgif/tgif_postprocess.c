#include "tgif_image.h"
#include "tgif_postprocess.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int cmp_float_asc(const void *a, const void *b) {
	float fa = *(const float *)a;
	float fb = *(const float *)b;
	return (fa > fb) - (fa < fb);
}

size_t tgif_ccl_1d_centroids(
	const float *heatmap,
	size_t length,
	float threshold,
	float *out_centroids,
	size_t out_capacity
) {
	size_t i;
	size_t count = 0;
	int in_group = 0;
	float weighted_sum = 0.0f;
	float total = 0.0f;
	size_t idx_sum = 0;
	size_t idx_count = 0;

	for (i = 0; i < length; ++i) {
		float val = heatmap[i];
		if (val >= threshold) {
			in_group = 1;
			weighted_sum += val * (float)i;
			total += val;
			idx_sum += i;
			idx_count += 1;
		} else if (in_group) {
			if (count < out_capacity) {
				out_centroids[count] = total > 0.0f
					? (weighted_sum / total)
					: ((float)idx_sum / (float)idx_count);
			}
			count += 1;
			in_group = 0;
			weighted_sum = 0.0f;
			total = 0.0f;
			idx_sum = 0;
			idx_count = 0;
		}
	}

	if (in_group) {
		if (count < out_capacity) {
			out_centroids[count] = total > 0.0f
				? (weighted_sum / total)
				: ((float)idx_sum / (float)idx_count);
		}
		count += 1;
	}

	if (count > 1 && out_capacity > 1) {
		qsort(out_centroids, count < out_capacity ? count : out_capacity, sizeof(float), cmp_float_asc);
	}
	return count;
}
