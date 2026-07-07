#include "tgif_grid.h"

#include <stdlib.h>
#include <string.h>

static void smooth_profile(const float *src, size_t n, int radius, float *dst) {
	size_t i;
	for (i = 0; i < n; ++i) {
		size_t start = i > (size_t)radius ? i - (size_t)radius : 0;
		size_t end = i + (size_t)radius + 1;
		size_t j;
		float sum = 0.0f;
		if (end > n) {
			end = n;
		}
		for (j = start; j < end; ++j) {
			sum += src[j];
		}
		dst[i] = sum / (float)(end - start);
	}
}

static size_t extract_peaks(
	const float *profile,
	size_t n,
	float threshold,
	float *out_lines
) {
	size_t i = 0;
	size_t count = 0;

	while (i < n) {
		if (profile[i] >= threshold) {
			size_t start = i;
			size_t end = i + 1;
			float weighted = profile[i] * (float)i;
			float total = profile[i];
			while (end < n && profile[end] >= threshold) {
				weighted += profile[end] * (float)end;
				total += profile[end];
				end += 1;
			}
			out_lines[count++] = total > 0.0f
				? (weighted / total)
				: (float)(start + end - 1) * 0.5f;
			i = end;
		} else {
			i += 1;
		}
	}

	return count;
}

static unsigned char rgb_to_gray(unsigned char r, unsigned char g, unsigned char b) {
	return (unsigned char)((77u * (unsigned int)r + 150u * (unsigned int)g + 29u * (unsigned int)b) >> 8);
}

void tgif_grid_lines_destroy(tgif_context *ctx, tgif_grid_lines *grid) {
	if (grid == NULL)
		return;
	tgif_free(ctx, grid->h_lines);
	tgif_free(ctx, grid->v_lines);
	memset(grid, 0, sizeof(*grid));
}
