#include "tgif_runtime.h"
#include "tgif_postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct tgif_exec_tensor {
	float *data;
	uint32_t rank;
	uint32_t dims[4];
	int owned;
} tgif_exec_tensor;

static size_t tensor_count(const tgif_exec_tensor *t) {
	size_t n = 1;
	uint32_t i;
	for (i = 0; i < t->rank; ++i) {
		n *= t->dims[i];
	}
	return n;
}

static int alloc_tensor(tgif_context *ctx, tgif_exec_tensor *t, uint32_t rank, const uint32_t *dims) {
	size_t n = 1;
	uint32_t i;
	if (t->owned && t->data != NULL) {
		tgif_free(ctx, t->data);
	}
	memset(t, 0, sizeof(*t));
	t->rank = rank;
	for (i = 0; i < rank; ++i) {
		t->dims[i] = dims[i];
		n *= dims[i];
	}
	t->data = (float *)tgif_calloc(ctx, n, sizeof(float));
	if (t->data == NULL) {
		return -1;
	}
	t->owned = 1;
	return 0;
}

static int clone_or_alias_output(tgif_context *ctx, tgif_exec_tensor *dst, const tgif_exec_tensor *src, int alias) {
	if (alias) {
		if (dst->owned && dst->data != NULL) {
			tgif_free(ctx, dst->data);
		}
		*dst = *src;
		dst->owned = 0;
		return 0;
	}
	if (alloc_tensor(ctx, dst, src->rank, src->dims) != 0) {
		return -1;
	}
	memcpy(dst->data, src->data, tensor_count(src) * sizeof(float));
	return 0;
}

static float sigmoidf_scalar(float x) {
	if (x >= 0.0f) {
		float e = expf(-x);
		return 1.0f / (1.0f + e);
	}
	{
		float e = expf(x);
		return e / (1.0f + e);
	}
}

static int run_relu(tgif_context *ctx, tgif_exec_tensor *out, tgif_exec_tensor *in, int alias) {
	size_t i;
	if (alias) {
		for (i = 0; i < tensor_count(in); ++i) {
			if (in->data[i] < 0.0f) {
				in->data[i] = 0.0f;
			}
		}
		return 0;
	}
	if (clone_or_alias_output(ctx, out, in, 0) != 0) {
		return -1;
	}
	for (i = 0; i < tensor_count(out); ++i) {
		if (out->data[i] < 0.0f) {
			out->data[i] = 0.0f;
		}
	}
	return 0;
}

static int run_sigmoid(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in) {
	size_t i;
	if (alloc_tensor(ctx, out, in->rank, in->dims) != 0) {
		return -1;
	}
	for (i = 0; i < tensor_count(in); ++i) {
		out->data[i] = sigmoidf_scalar(in->data[i]);
	}
	return 0;
}

static int run_add(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *a, const tgif_exec_tensor *b) {
	size_t i;
	if (a->rank != b->rank) {
		return -1;
	}
	if (alloc_tensor(ctx, out, a->rank, a->dims) != 0) {
		return -1;
	}
	for (i = 0; i < tensor_count(a); ++i) {
		out->data[i] = a->data[i] + b->data[i];
	}
	return 0;
}

static int run_reduce_mean(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, int axis4d) {
	uint32_t dims[2];
	uint32_t c = in->dims[0];
	uint32_t h = in->dims[1];
	uint32_t w = in->dims[2];
	uint32_t ch;
	uint32_t i;

	if (in->rank != 3) {
		return -1;
	}
	dims[0] = c;
	dims[1] = (axis4d == 3) ? h : w;
	if (alloc_tensor(ctx, out, 2, dims) != 0) {
		return -1;
	}

	if (axis4d == 3) {
		for (ch = 0; ch < c; ++ch) {
			for (i = 0; i < h; ++i) {
				uint32_t x;
				float sum = 0.0f;
				for (x = 0; x < w; ++x) {
					sum += in->data[(size_t)ch * h * w + (size_t)i * w + x];
				}
				out->data[(size_t)ch * h + i] = sum / (float)w;
			}
		}
	} else {
		for (ch = 0; ch < c; ++ch) {
			for (i = 0; i < w; ++i) {
				uint32_t y;
				float sum = 0.0f;
				for (y = 0; y < h; ++y) {
					sum += in->data[(size_t)ch * h * w + (size_t)y * w + i];
				}
				out->data[(size_t)ch * w + i] = sum / (float)h;
			}
		}
	}
	return 0;
}

static float resize1d_half_pixel(const float *src, uint32_t in_len, uint32_t out_len, uint32_t ch, uint32_t c) {
	float x = (((float)ch + 0.5f) * (float)in_len / (float)out_len) - 0.5f;
	int x0 = (int)floorf(x);
	int x1 = x0 + 1;
	float w1 = x - (float)x0;
	float w0 = 1.0f - w1;
	if (x0 < 0) {
		x0 = 0;
		w0 = 1.0f;
		w1 = 0.0f;
	}
	if (x1 >= (int)in_len) {
		x1 = (int)in_len - 1;
		w1 = 0.0f;
		w0 = 1.0f;
	}
	return w0 * src[(size_t)c * in_len + (uint32_t)x0] + w1 * src[(size_t)c * in_len + (uint32_t)x1];
}

static int run_resize1d(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, uint32_t out_len) {
	uint32_t dims[2];
	uint32_t c;
	uint32_t x;
	if (in->rank != 2) {
		return -1;
	}
	dims[0] = in->dims[0];
	dims[1] = out_len;
	if (alloc_tensor(ctx, out, 2, dims) != 0) {
		return -1;
	}
	for (c = 0; c < in->dims[0]; ++c) {
		for (x = 0; x < out_len; ++x) {
			out->data[(size_t)c * out_len + x] = resize1d_half_pixel(in->data, in->dims[1], out_len, x, c);
		}
	}
	return 0;
}

static int run_expand_h(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, uint32_t out_h, uint32_t out_w) {
	uint32_t dims[3] = {1, out_h, out_w};
	uint32_t y, x;
	if (in->rank != 2 || in->dims[0] != 1 || in->dims[1] != out_h) {
		return -1;
	}
	if (alloc_tensor(ctx, out, 3, dims) != 0) {
		return -1;
	}
	for (y = 0; y < out_h; ++y) {
		for (x = 0; x < out_w; ++x) {
			out->data[(size_t)y * out_w + x] = in->data[y];
		}
	}
	return 0;
}

static int run_expand_v(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, uint32_t out_h, uint32_t out_w) {
	uint32_t dims[3] = {1, out_h, out_w};
	uint32_t y, x;
	if (in->rank != 2 || in->dims[0] != 1 || in->dims[1] != out_w) {
		return -1;
	}
	if (alloc_tensor(ctx, out, 3, dims) != 0) {
		return -1;
	}
	for (y = 0; y < out_h; ++y) {
		for (x = 0; x < out_w; ++x) {
			out->data[(size_t)y * out_w + x] = in->data[x];
		}
	}
	return 0;
}

static int run_concat(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *a, const tgif_exec_tensor *b, const tgif_exec_tensor *c) {
	uint32_t dims[3];
	size_t offset = 0;
	if (a->rank != 3 || b->rank != 3 || c->rank != 3) {
		return -1;
	}
	if (a->dims[1] != b->dims[1] || a->dims[1] != c->dims[1] ||
		a->dims[2] != b->dims[2] || a->dims[2] != c->dims[2]) {
		return -1;
	}
	dims[0] = a->dims[0] + b->dims[0] + c->dims[0];
	dims[1] = a->dims[1];
	dims[2] = a->dims[2];
	if (alloc_tensor(ctx, out, 3, dims) != 0) {
		return -1;
	}
	memcpy(out->data + offset, a->data, tensor_count(a) * sizeof(float));
	offset += tensor_count(a);
	memcpy(out->data + offset, b->data, tensor_count(b) * sizeof(float));
	offset += tensor_count(b);
	memcpy(out->data + offset, c->data, tensor_count(c) * sizeof(float));
	return 0;
}


static inline void
conv2d_depthwise1_template(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	uint32_t oc, oy, ox;

	for (oc = 0; oc < out_c; ++oc) {
		size_t w_off = (size_t)oc * w_dim1 * k_h * k_w;
		for (oy = 0; oy < out_h; ++oy) {
			for (ox = 0; ox < out_w; ++ox) {
				float sum = b_data[oc];
				uint32_t ky, kx;
				size_t in_off = (size_t)oc * in_h * in_w;
				for (ky = 0; ky < k_h; ++ky) {
					size_t in_off2;
					size_t w_off2;
					int in_y = (int)(oy * stride_h + ky * dil_h) - (int)pad_t;
					if (in_y < 0 || in_y >= (int)in_h) {
						continue;
					}
					in_off2 = in_off + (size_t)in_y * in_w;
					w_off2 = w_off + ((size_t)ky * k_w);
					for (kx = 0; kx < k_w; ++kx) {
						int in_x = (int)(ox * stride_w + kx * dil_w) - (int)pad_l;
						if (in_x < 0 || in_x >= (int)in_w) {
							continue;
						}
						sum += in_data[in_off2  + (uint32_t)in_x] *
							w_data[w_off2 + kx];
					}
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
		}
	}
}

static inline void
conv2d_opt_depthwise1_generic(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise1_template(stride_h, stride_w, pad_t, pad_l, dil_h, dil_w, in_c, in_h, in_w,
			out_c, w_dim1, k_h, k_w, out_h, out_w, b_data, in_data, w_data, out_data);
}

static void
conv2d_opt_depthwise1_111111_3_1(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise1_template(1, 1, 1, 1, 1, 1, in_c, in_h, in_w,
			out_c, w_dim1, 3, 3, out_h, out_w, b_data, in_data, w_data, out_data);
}

static void
conv2d_opt_depthwise1_111111_3_2(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise1_template(2, 2, 1, 1, 1, 1, in_c, in_h, in_w,
			out_c, w_dim1, 3, 3, out_h, out_w, b_data, in_data, w_data, out_data);
}

static void
conv2d_opt_depthwise1_222222_3(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise1_template(stride_h, stride_w, 2, 2, 2, 2, in_c, in_h, in_w,
			out_c, w_dim1, 3, 3, out_h, out_w, b_data, in_data, w_data, out_data);
}

static void
conv2d_opt_depthwise1_444444_3(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise1_template(stride_h, stride_w, 4, 4, 4, 4, in_c, in_h, in_w,
			out_c, w_dim1, 3, 3, out_h, out_w, b_data, in_data, w_data, out_data);
}

static inline void
conv2d_depthwise0_template(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	uint32_t oc, oy, ox;

	for (oc = 0; oc < out_c; ++oc) {
		size_t w_off = (size_t)oc * w_dim1 * k_h * k_w;
		for (oy = 0; oy < out_h; ++oy) {
			for (ox = 0; ox < out_w; ++ox) {
				float sum = b_data[oc];
				uint32_t ic;
				for (ic = 0; ic < in_c; ++ic) {
					uint32_t ky, kx;
					size_t in_off = (size_t)ic * in_h * in_w;
					size_t w_off2 = w_off + ((size_t)ic * k_h * k_w);
					for (ky = 0; ky < k_h; ++ky) {
						size_t in_off2;
						size_t w_off3;
						int in_y = (int)(oy * stride_h + ky * dil_h) - (int)pad_t;
						if (in_y < 0 || in_y >= (int)in_h) {
							continue;
						}
						in_off2 = in_off + (size_t)in_y * in_w;
						w_off3 = w_off2 + ((size_t)ky * k_w);
						for (kx = 0; kx < k_w; ++kx) {
							int in_x = (int)(ox * stride_w + kx * dil_w) - (int)pad_l;
							if (in_x < 0 || in_x >= (int)in_w) {
								continue;
							}
							sum += in_data[in_off2  + (uint32_t)in_x] *
								w_data[w_off3 + kx];
						}
					}
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
		}
	}
}

static inline void
conv2d_opt_depthwise0_generic(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise0_template(stride_h, stride_w, pad_t, pad_l, dil_h, dil_w, in_c, in_h, in_w,
			out_c, w_dim1, k_h, k_w, out_h, out_w, b_data, in_data, w_data, out_data);
}

static void
conv2d_opt_depthwise0_000011_1_match_1(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	uint32_t oc, oy, ox;
	size_t in_planesize = in_h * in_w;

	for (oc = 0; oc < out_c; ++oc) {
		size_t w_off = (size_t)oc * w_dim1;
		for (oy = 0; oy < out_h; ++oy) {
			size_t in_off = (size_t)oy * in_w;
			for (ox = 0; ox < out_w; ++ox) {
				float sum = b_data[oc];
				uint32_t ic;
				size_t in_off2 = in_off + ox;
				for (ic = 0; ic < in_c; ++ic) {
					size_t w_off2 = w_off + ((size_t)ic);
					sum += in_data[in_off2] * w_data[w_off2];
					in_off2 += in_planesize;
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
		}
	}
}

static void
conv2d_opt_depthwise0_111111_3_1(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	uint32_t oc, oy, ox;
	size_t in_planesize = in_h * in_w;

	for (oc = 0; oc < out_c; ++oc) {
		size_t w_off = (size_t)oc * w_dim1 * k_h * k_w;
		/* Slow: oy == 0 */
		for (ox = 0; ox < out_w; ++ox) {
			float sum = b_data[oc];
			uint32_t ic;
			for (ic = 0; ic < in_c; ++ic) {
				uint32_t ky, kx;
				size_t in_off = (size_t)ic * in_planesize;
				size_t w_off2 = w_off + ((size_t)ic * 3 * 3);
				for (ky = 0; ky < 3; ++ky) {
					size_t in_off2;
					size_t w_off3;
					int in_y = (int)ky - 1;
					if (in_y < 0 || in_y >= (int)in_h) {
						continue;
					}
					in_off2 = in_off + (size_t)in_y * in_w;
					w_off3 = w_off2 + ((size_t)ky * 3);
					for (kx = 0; kx < 3; ++kx) {
						int in_x = (int)(ox + kx) - (int)1;
						if (in_x < 0 || in_x >= (int)in_w) {
							continue;
						}
						sum += in_data[in_off2  + (uint32_t)in_x] *
							w_data[w_off3 + kx];
					}
				}
			}
			out_data[(size_t)oc * out_h * out_w + ox] = sum;
		}
		for (oy = 1; oy < out_h-1; ++oy) {
			/* y will always be in range. */
			/* ox == 0 */
			{
				float sum = b_data[oc];
				uint32_t ic;
				for (ic = 0; ic < in_c; ++ic) {
					uint32_t ky, kx;
					size_t in_off = (size_t)ic * in_planesize;
					size_t w_off2 = w_off + ((size_t)ic * 3 * 3);
					for (ky = 0; ky < 3; ++ky) {
						size_t in_off2;
						size_t w_off3;
						int in_y = (int)(oy + ky) - 1;
						in_off2 = in_off + (size_t)in_y * in_w;
						w_off3 = w_off2 + ((size_t)ky * 3);
						for (kx = 0; kx < 3; ++kx) {
							int in_x = (int)(ox + kx) - (int)1;
							if (in_x < 0 || in_x >= (int)in_w) {
								continue;
							}
							sum += in_data[in_off2  + (uint32_t)in_x] *
								w_data[w_off3 + kx];
						}
					}
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
			for (ox = 1; ox < out_w-1; ++ox) {
				/* x will always be in range */
				float sum = b_data[oc];
				uint32_t ic;
				size_t in_off = (size_t)(oy - 1)* in_w + ox - 1;
				for (ic = 0; ic < in_c; ++ic) {
#if 1
					sum += in_data[in_off         ] * w_data[w_off];
					sum += in_data[in_off       +1] * w_data[w_off+1];
					sum += in_data[in_off       +2] * w_data[w_off+2];
					sum += in_data[in_off+in_w    ] * w_data[w_off+3];
					sum += in_data[in_off+in_w  +1] * w_data[w_off+4];
					sum += in_data[in_off+in_w  +2] * w_data[w_off+5];
					sum += in_data[in_off+in_w*2  ] * w_data[w_off+6];
					sum += in_data[in_off+in_w*2+1] * w_data[w_off+7];
					sum += in_data[in_off+in_w*2+2] * w_data[w_off+8];
#else
					uint32_t ky, kx;
					size_t w_off2 = w_off;
					size_t in_off2 = in_off;
					for (ky = 3; ky > 0; --ky) {
						for (kx = 3; kx > 0; --kx) {
							sum += in_data[in_off2++] *
								w_data[w_off2++];
						}
						in_off2 += in_w-3;
					}
#endif
					in_off += in_planesize;
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
			/* Slow: ox = out_w-1 */
			{
				float sum = b_data[oc];
				uint32_t ic;
				for (ic = 0; ic < in_c; ++ic) {
					uint32_t ky, kx;
					size_t in_off = (size_t)ic * in_planesize;
					size_t w_off2 = w_off + ((size_t)ic * 3 * 3);
					for (ky = 0; ky < 3; ++ky) {
						size_t in_off2;
						size_t w_off3;
						int in_y = (int)(oy + ky) - 1;
						in_off2 = in_off + (size_t)in_y * in_w;
						w_off3 = w_off2 + ((size_t)ky * 3);
						for (kx = 0; kx < 3; ++kx) {
							int in_x = (int)(ox + kx) - (int)1;
							if (in_x < 0 || in_x >= (int)in_w) {
								continue;
							}
							sum += in_data[in_off2  + (uint32_t)in_x] *
								w_data[w_off3 + kx];
						}
					}
				}
				out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
			}
		}
		/* Slow: oy = out_h-1 */
		for (ox = 0; ox < out_w; ++ox) {
			float sum = b_data[oc];
			uint32_t ic;
			for (ic = 0; ic < in_c; ++ic) {
				uint32_t ky, kx;
				size_t in_off = (size_t)ic * in_planesize;
				size_t w_off2 = w_off + ((size_t)ic * 3 * 3);
				for (ky = 0; ky < 3; ++ky) {
					size_t in_off2;
					size_t w_off3;
					int in_y = (int)(oy + ky) - 1;
					if (in_y < 0 || in_y >= (int)in_h) {
						continue;
					}
					in_off2 = in_off + (size_t)in_y * in_w;
					w_off3 = w_off2 + ((size_t)ky * 3);
					for (kx = 0; kx < 3; ++kx) {
						int in_x = (int)(ox + kx) - (int)1;
						if (in_x < 0 || in_x >= (int)in_w) {
							continue;
						}
						sum += in_data[in_off2  + (uint32_t)in_x] *
							w_data[w_off3 + kx];
					}
				}
			}
			out_data[((size_t)oc * out_h + (size_t)oy) * out_w + ox] = sum;
		}
	}
}

static void
conv2d_opt_depthwise0_111111_3_2(uint32_t stride_h,
	uint32_t stride_w,
	uint32_t pad_t,
	uint32_t pad_l,
	uint32_t dil_h,
	uint32_t dil_w,
	uint32_t in_c,
	uint32_t in_h,
	uint32_t in_w,
	uint32_t out_c,
	uint32_t w_dim1,
	uint32_t k_h,
	uint32_t k_w,
	uint32_t out_h,
	uint32_t out_w,
	const float *b_data,
	const float *in_data,
	const float *w_data,
	float *out_data)
{
	conv2d_depthwise0_template(2, 2, 1, 1, 1, 1, in_c, in_h, in_w,
			out_c, w_dim1, 3, 3, out_h, out_w, b_data, in_data, w_data, out_data);
}

static int run_conv2d(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, const tgif_exec_tensor *w, const tgif_exec_tensor *b, int depthwise, const int32_t *p) {
	uint32_t stride_h = (uint32_t)p[0];
	uint32_t stride_w = (uint32_t)p[1];
	uint32_t pad_t = (uint32_t)p[2];
	uint32_t pad_l = (uint32_t)p[3];
	uint32_t pad_b = (uint32_t)p[4];
	uint32_t pad_r = (uint32_t)p[5];
	uint32_t dil_h = (uint32_t)p[6];
	uint32_t dil_w = (uint32_t)p[7];
	uint32_t in_c = in->dims[0];
	uint32_t in_h = in->dims[1];
	uint32_t in_w = in->dims[2];
	uint32_t out_c = w->dims[0];
	uint32_t w_dim1 = w->dims[1];
	uint32_t k_h = w->dims[2];
	uint32_t k_w = w->dims[3];
	uint32_t out_h = (in_h + pad_t + pad_b - ((k_h - 1u) * dil_h + 1u)) / stride_h + 1u;
	uint32_t out_w = (in_w + pad_l + pad_r - ((k_w - 1u) * dil_w + 1u)) / stride_w + 1u;
	uint32_t dims[3] = {out_c, out_h, out_w};
	void (*fn)(uint32_t stride_h,
		uint32_t stride_w,
		uint32_t pad_t,
		uint32_t pad_l,
		uint32_t dil_h,
		uint32_t dil_w,
		uint32_t in_c,
		uint32_t in_h,
		uint32_t in_w,
		uint32_t out_c,
		uint32_t w_dim1,
		uint32_t k_h,
		uint32_t k_w,
		uint32_t out_h,
		uint32_t out_w,
		const float *b_data,
		const float *in_data,
		const float *w_data,
		float *out_data);

	if (alloc_tensor(ctx, out, 3, dims) != 0) {
		return -1;
	}

	fn = NULL;

	if (pad_t == pad_l && dil_w == dil_h && k_h == k_w && stride_h == stride_w)
	{
		if (pad_t == 0)
		{
			if (dil_w == 1 && !depthwise && k_h == 1 && in_w == out_w && in_h == out_h && stride_h == 1)
			{
				fn = conv2d_opt_depthwise0_000011_1_match_1;
			}
		}
		else if (pad_t == dil_w && k_h == 3)
		{
			if (pad_t == 1)
			{
				if (stride_h == 1)
					fn = depthwise ? conv2d_opt_depthwise1_111111_3_1 : conv2d_opt_depthwise0_111111_3_1;
				else if (stride_h == 2)
					fn = depthwise ? conv2d_opt_depthwise1_111111_3_2 : conv2d_opt_depthwise0_111111_3_2;
			}
			else if (depthwise)
			{
				if (pad_t == 2)
					fn = conv2d_opt_depthwise1_222222_3;
				else if (pad_t == 4)
					fn = conv2d_opt_depthwise1_444444_3;
			}
		}
	}

#ifndef NDEBUG
	if (fn == NULL)
	{
		printf("Non-optimised route!\n");
		fn = depthwise ? conv2d_opt_depthwise1_generic : conv2d_opt_depthwise0_generic;
	}
#endif

	fn(stride_h, stride_w, pad_t, pad_l, dil_h, dil_w, in_c, in_h, in_w,
			out_c, w_dim1, k_h, k_w, out_h, out_w, b->data, in->data, w->data, out->data);

	return 0;
}

static int run_conv1d(tgif_context *ctx, tgif_exec_tensor *out, const tgif_exec_tensor *in, const tgif_exec_tensor *w, const tgif_exec_tensor *b, const int32_t *p) {
	uint32_t stride = (uint32_t)p[0];
	uint32_t pad_l = (uint32_t)p[1];
	uint32_t pad_r = (uint32_t)p[2];
	uint32_t dil = (uint32_t)p[3];
	uint32_t in_c = in->dims[0];
	uint32_t in_len = in->dims[1];
	uint32_t out_c = w->dims[0];
	uint32_t k = w->dims[2];
	uint32_t out_len = (in_len + pad_l + pad_r - ((k - 1u) * dil + 1u)) / stride + 1u;
	uint32_t dims[2] = {out_c, out_len};
	uint32_t oc, x;

	if (alloc_tensor(ctx, out, 2, dims) != 0) {
		return -1;
	}

	for (oc = 0; oc < out_c; ++oc) {
		for (x = 0; x < out_len; ++x) {
			float sum = b->data[oc];
			uint32_t ic;
			for (ic = 0; ic < in_c; ++ic) {
				uint32_t kx;
				for (kx = 0; kx < k; ++kx) {
					int in_x = (int)(x * stride + kx * dil) - (int)pad_l;
					if (in_x < 0 || in_x >= (int)in_len) {
						continue;
					}
					sum += in->data[(size_t)ic * in_len + (uint32_t)in_x] *
						w->data[((size_t)oc * in_c * k) + ((size_t)ic * k) + kx];
				}
			}
			out->data[(size_t)oc * out_len + x] = sum;
		}
	}
	return 0;
}

static void free_exec_tensors(tgif_context *ctx, tgif_exec_tensor *ts, uint32_t n) {
	uint32_t i;
	if (ts == NULL) {
		return;
	}
	for (i = 0; i < n; ++i) {
		if (ts[i].owned && ts[i].data != NULL) {
			tgif_free(ctx, ts[i].data);
		}
	}
	tgif_free(ctx, ts);
}

int tgif_run_model(tgif_context *ctx, const tgif_model *model, const tgif_image *image, tgif_runtime_output *out) {
	tgif_exec_tensor *ts;
	uint32_t i;
	uint32_t image_idx = 0;
	uint32_t h_idx = UINT32_MAX;
	uint32_t v_idx = UINT32_MAX;
	uint32_t reg_idx = UINT32_MAX;
	uint32_t ctr_idx = UINT32_MAX;

	if (model == NULL || image == NULL || out == NULL || model->header.version != 2u) {
		return -1;
	}
	memset(out, 0, sizeof(*out));

	ts = (tgif_exec_tensor *)tgif_calloc(ctx, model->header.tensor_count, sizeof(tgif_exec_tensor));
	if (ts == NULL) {
		return -1;
	}

	for (i = 0; i < model->header.tensor_count; ++i) {
		const tgif_tensor_desc *desc = &model->tensors[i];
		if (strncmp(desc->name, "image", TGF_MAX_NAME) == 0) image_idx = i;
		if (strncmp(desc->name, "h_heatmap", TGF_MAX_NAME) == 0) h_idx = i;
		if (strncmp(desc->name, "v_heatmap", TGF_MAX_NAME) == 0) v_idx = i;
		if (strncmp(desc->name, "reg", TGF_MAX_NAME) == 0) reg_idx = i;
		if (strncmp(desc->name, "ctr", TGF_MAX_NAME) == 0) ctr_idx = i;
		if (desc->element_count > 0) {
			ts[i].data = (float *)(model->blob + desc->data_offset);
			ts[i].rank = desc->rank;
			memcpy(ts[i].dims, desc->dims, sizeof(ts[i].dims));
			ts[i].owned = 0;
		}
	}

	{
		uint32_t dims[3] = {3u, model->header.input_h, model->header.input_w};
		if (alloc_tensor(ctx, &ts[image_idx], 3, dims) != 0 ||
			tgif_image_resize_to_normalized_chw(image, (int)model->header.input_w, (int)model->header.input_h,
				ts[image_idx].data, tensor_count(&ts[image_idx])) != 0) {
			free_exec_tensors(ctx, ts, model->header.tensor_count);
			return -1;
		}
	}

	for (i = 0; i < model->header.layer_count; ++i) {
		const tgif_layer_desc *layer = &model->layers[i];
		tgif_exec_tensor *dst = &ts[layer->output];
		tgif_exec_tensor *a = &ts[layer->input0];
		tgif_exec_tensor *b = (layer->input1 != UINT8_MAX) ? &ts[layer->input1] : NULL;
		tgif_exec_tensor *c = (layer->input2 != UINT8_MAX) ? &ts[layer->input2] : NULL;
		int rc = -1;
		switch (layer->op_type) {
			case TGF_LAYER_CONV2D:
				rc = run_conv2d(ctx, dst, a, b, c, 0, layer->params_i);
				break;
			case TGF_LAYER_DEPTHWISE_CONV2D:
				rc = run_conv2d(ctx, dst, a, b, c, 1, layer->params_i);
				break;
			case TGF_LAYER_CONV1D:
				rc = run_conv1d(ctx, dst, a, b, c, layer->params_i);
				break;
			case TGF_LAYER_RELU:
				rc = run_relu(ctx, dst, a, layer->output == layer->input0);
				break;
			case TGF_LAYER_ADD:
				rc = run_add(ctx, dst, a, b);
				break;
			case TGF_LAYER_REDUCE_MEAN:
				rc = run_reduce_mean(ctx, dst, a, layer->params_i[0]);
				break;
			case TGF_LAYER_RESIZE1D:
				rc = run_resize1d(ctx, dst, a, (uint32_t)layer->params_i[0]);
				break;
			case TGF_LAYER_SIGMOID:
				rc = run_sigmoid(ctx, dst, a);
				break;
			case TGF_LAYER_EXPAND_H:
				rc = run_expand_h(ctx, dst, a, (uint32_t)layer->params_i[0], (uint32_t)layer->params_i[1]);
				break;
			case TGF_LAYER_EXPAND_V:
				rc = run_expand_v(ctx, dst, a, (uint32_t)layer->params_i[0], (uint32_t)layer->params_i[1]);
				break;
			case TGF_LAYER_CONCAT:
				rc = run_concat(ctx, dst, a, b, c);
				break;
			default:
				rc = -1;
				break;
		}
		if (rc != 0) {
			free_exec_tensors(ctx, ts, model->header.tensor_count);
			return -1;
		}
	}

	if (h_idx == UINT32_MAX || v_idx == UINT32_MAX || reg_idx == UINT32_MAX || ctr_idx == UINT32_MAX) {
		free_exec_tensors(ctx, ts, model->header.tensor_count);
		return -1;
	}

	out->h_len = ts[h_idx].dims[1];
	out->h_heatmap = (float *)tgif_malloc(ctx, out->h_len * sizeof(float));
	out->v_len = ts[v_idx].dims[1];
	out->v_heatmap = (float *)tgif_malloc(ctx, out->v_len * sizeof(float));
	out->reg_count = tensor_count(&ts[reg_idx]);
	out->reg = (float *)tgif_malloc(ctx, out->reg_count * sizeof(float));
	out->ctr_count = tensor_count(&ts[ctr_idx]);
	out->ctr = (float *)tgif_malloc(ctx, out->ctr_count * sizeof(float));
	if (out->h_heatmap == NULL || out->v_heatmap == NULL || out->reg == NULL || out->ctr == NULL) {
		free_exec_tensors(ctx, ts, model->header.tensor_count);
		tgif_runtime_output_destroy(ctx, out);
		return -1;
	}

	memcpy(out->h_heatmap, ts[h_idx].data, out->h_len * sizeof(float));
	memcpy(out->v_heatmap, ts[v_idx].data, out->v_len * sizeof(float));
	memcpy(out->reg, ts[reg_idx].data, out->reg_count * sizeof(float));
	memcpy(out->ctr, ts[ctr_idx].data, out->ctr_count * sizeof(float));

	free_exec_tensors(ctx, ts, model->header.tensor_count);
	return 0;
}

void tgif_runtime_output_destroy(tgif_context *ctx, tgif_runtime_output *out) {
	if (out == NULL) {
		return;
	}
	tgif_free(ctx, out->h_heatmap);
	tgif_free(ctx, out->v_heatmap);
	tgif_free(ctx, out->reg);
	tgif_free(ctx, out->ctr);
	memset(out, 0, sizeof(*out));
}

tgif_grid *tgif_get_grid(tgif_context *ctx, tgif_runtime_output *rt)
{
	tgif_grid *grid;
	int i, h_count, v_count;
	float h_min = 1e30f, h_max = -1e30f, v_min = 1e30f, v_max = -1e30f;

	if (rt == NULL)
		return NULL;

	grid = tgif_calloc(ctx, 1, sizeof(*grid));
	if (grid == NULL)
		return NULL;


	for (i = 0; i < rt->h_len; ++i) {
		if (rt->h_heatmap[i] < h_min) h_min = rt->h_heatmap[i];
		if (rt->h_heatmap[i] > h_max) h_max = rt->h_heatmap[i];
	}
	for (i = 0; i < rt->v_len; ++i) {
		if (rt->v_heatmap[i] < v_min) v_min = rt->v_heatmap[i];
		if (rt->v_heatmap[i] > v_max) v_max = rt->v_heatmap[i];
	}

	h_count = tgif_ccl_1d_centroids(rt->h_heatmap, rt->h_len, 0.3f, NULL, 0);
	v_count = tgif_ccl_1d_centroids(rt->v_heatmap, rt->v_len, 0.4f, NULL, 0);
	grid->h = tgif_malloc(ctx, sizeof(grid->h[0]) * h_count);
	grid->v = tgif_malloc(ctx, sizeof(grid->v[0]) * v_count);
	if (grid->h == NULL || grid->v == NULL)
	{
		tgif_free(ctx, grid->h);
		tgif_free(ctx, grid->v);
		tgif_free(ctx, grid);
		return NULL;
	}
	grid->h_len = h_count;
	grid->v_len = v_count;
	(void)tgif_ccl_1d_centroids(rt->h_heatmap, rt->h_len, 0.3f, grid->h, h_count);
	(void)tgif_ccl_1d_centroids(rt->v_heatmap, rt->v_len, 0.4f, grid->v, v_count);

	return grid;
}

void tgif_grid_destroy(tgif_context *ctx, tgif_grid *grid)
{
	if (grid == NULL)
		return;
	tgif_free(ctx, grid->h);
	tgif_free(ctx, grid->v);
	tgif_free(ctx, grid);
}


void *tgif_malloc(tgif_context *ctx, size_t z)
{
	return ctx->malloc(ctx->opaque, z);
}

void *tgif_calloc(tgif_context *ctx, size_t n, size_t z)
{
	return ctx->calloc(ctx->opaque, n, z);
}

void tgif_free(tgif_context *ctx, void *ptr)
{
	ctx->free(ctx->opaque, ptr);
}
