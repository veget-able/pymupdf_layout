/* Shut up MSVC, I know what I'm doing. */
#define _CRT_SECURE_NO_WARNINGS

#include "tgif_image.h"

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int skip_ws_and_comments(FILE *fp) {
	int ch;

	for (;;) {
		ch = fgetc(fp);
		if (ch == EOF) {
			return -1;
		}
		if (isspace(ch)) {
			continue;
		}
		if (ch == '#') {
			do {
				ch = fgetc(fp);
			} while (ch != '\n' && ch != EOF);
			if (ch == EOF) {
				return -1;
			}
			continue;
		}
		if (ungetc(ch, fp) == EOF) {
			return -1;
		}
		return 0;
	}
}

static int read_int_token(FILE *fp, int *value) {
	int ch;
	int sign = 1;
	int result = 0;
	int saw_digit = 0;

	if (skip_ws_and_comments(fp) != 0) {
		return -1;
	}

	ch = fgetc(fp);
	if (ch == '-') {
		sign = -1;
		ch = fgetc(fp);
	}

	while (ch != EOF && isdigit(ch)) {
		saw_digit = 1;
		result = (result * 10) + (ch - '0');
		ch = fgetc(fp);
	}

	if (ch != EOF) {
		ungetc(ch, fp);
	}
	if (!saw_digit) {
		return -1;
	}

	*value = sign * result;
	return 0;
}

static int read_magic(FILE *fp, char magic[3]) {
	int a = fgetc(fp);
	int b = fgetc(fp);
	if (a == EOF || b == EOF) {
		return -1;
	}
	magic[0] = (char)a;
	magic[1] = (char)b;
	magic[2] = '\0';
	return 0;
}

static int consume_single_whitespace(FILE *fp) {
	int ch = fgetc(fp);
	return (ch != EOF && isspace(ch)) ? 0 : -1;
}

int tgif_image_load_pnm(tgif_context *ctx, const char *path, tgif_image **imagep) {
	FILE *fp;
	char magic[3];
	int width;
	int height;
	int maxval;
	size_t pixel_count;
	size_t byte_count;
	int channels;
	tgif_pixel_format format;
	tgif_image *image;

	if (imagep == NULL || path == NULL) {
		return -1;
	}
	*imagep = NULL;

	fp = fopen(path, "rb");
	if (fp == NULL) {
		return -1;
	}

	if (read_magic(fp, magic) != 0) {
		fclose(fp);
		return -1;
	}

	if (strcmp(magic, "P5") == 0) {
		channels = 1;
		format = TGF_PIXFMT_GRAY8;
	} else if (strcmp(magic, "P6") == 0) {
		channels = 3;
		format = TGF_PIXFMT_RGB8;
	} else {
		fclose(fp);
		return -1;
	}

	if (read_int_token(fp, &width) != 0 ||
		read_int_token(fp, &height) != 0 ||
		read_int_token(fp, &maxval) != 0) {
		fclose(fp);
		return -1;
	}

	if (width <= 0 || height <= 0 || maxval != 255) {
		fclose(fp);
		return -1;
	}

	if (consume_single_whitespace(fp) != 0) {
		fclose(fp);
		return -1;
	}

	image = tgif_new_image(ctx, width, height, channels);
	*imagep = image;
	if (image == NULL) {
		fclose(fp);
		return -1;
	}

	pixel_count = (size_t)width * (size_t)height;
	byte_count = pixel_count * (size_t)channels;

	image->pixels = (unsigned char *)tgif_malloc(ctx, byte_count);
	if (image->pixels == NULL) {
		fclose(fp);
		return -1;
	}

	if (fread(image->pixels, 1, byte_count, fp) != byte_count) {
		tgif_image_destroy(ctx, image);
		fclose(fp);
		return -1;
	}

	image->width = width;
	image->height = height;
	image->channels = channels;
	image->stride = (size_t)width * channels;
	image->format = format;

	fclose(fp);
	return 0;
}

void tgif_image_destroy(tgif_context *ctx, tgif_image *image) {
	if (image == NULL)
		return;
	if (!image->borrowed)
		tgif_free(ctx, image->pixels);
	tgif_free(ctx, image);
}

tgif_image *tgif_new_image_borrowed_data(tgif_context *ctx, int w, int h, int n, size_t stride, unsigned char *data)
{
	tgif_image *image;

	if (n != 1 && n != 3)
		return NULL;
	image = tgif_calloc(ctx, 1, sizeof(*image));
	if (image == NULL)
		return NULL;
	image->borrowed = 1;
	image->width = w;
	image->height = h;
	image->channels = n;
	image->stride = stride;
	image->pixels = data;

	return image;
}

tgif_image *tgif_new_image(tgif_context *ctx, int w, int h, int n)
{
	tgif_image *image;

	if (n != 1 && n != 3)
		return NULL;
	image = tgif_calloc(ctx, 1, sizeof(*image));
	if (image == NULL)
		return NULL;
	image->pixels = tgif_malloc(ctx, (size_t)w * h * n);
	if (image->pixels == NULL)
	{
		tgif_free(ctx, image);
		return NULL;
	}
	if (n == 1)
		image->format = TGF_PIXFMT_GRAY8;
	else
		image->format = TGF_PIXFMT_RGB8;
	image->borrowed = 0;
	image->width = w;
	image->height = h;
	image->channels = n;
	image->stride = (size_t)w * n;

	return image;
}

static void sample_rgb_bilinear(const tgif_image *image, float sx, float sy, float rgb[3]) {
	int x0;
	int y0;
	int x1;
	int y1;
	float fx;
	float fy;
	int c;
	float p00, p01, p10, p11;
	size_t stride = image->stride;

	if (sx < 0.0f) {
		sx = 0.0f;
	}
	if (sy < 0.0f) {
		sy = 0.0f;
	}
	if (sx > (float)(image->width - 1)) {
		sx = (float)(image->width - 1);
	}
	if (sy > (float)(image->height - 1)) {
		sy = (float)(image->height - 1);
	}

	x0 = (int)floorf(sx);
	y0 = (int)floorf(sy);
	x1 = x0 + 1;
	y1 = y0 + 1;
	fx = sx - (float)x0;
	fy = sy - (float)y0;

	if (x1 >= image->width) x1 = image->width - 1;
	if (y1 >= image->height) y1 = image->height - 1;

	if (image->channels == 1) {
		/* Expand grayscale input to three identical channels. */
		p00 = image->pixels[y0 * stride + x0];
		p01 = image->pixels[y0 * stride + x1];
		p10 = image->pixels[y1 * stride + x0];
		p11 = image->pixels[y1 * stride + x1];
		rgb[0] = rgb[1] = rgb[2] = (1.0f - fy) * ((1.0f - fx) * p00 + fx * p01) +
						fy * ((1.0f - fx) * p10 + fx * p11);
	} else {
		size_t idx00 = (size_t)y0 * stride + (size_t)x0 * 3u;
		size_t idx01 = (size_t)y0 * stride + (size_t)x1 * 3u;
		size_t idx10 = (size_t)y1 * stride + (size_t)x0 * 3u;
		size_t idx11 = (size_t)y1 * stride + (size_t)x1 * 3u;
		unsigned char *pix = image->pixels;
		for (c = 0; c < 3; ++c) {
			p00 = pix[idx00];
			p01 = pix[idx01];
			p10 = pix[idx10];
			p11 = pix[idx11];
			rgb[c] = (1.0f - fy) * ((1.0f - fx) * p00 + fx * p01) +
				fy * ((1.0f - fx) * p10 + fx * p11);
			pix++;
		}
	}
}

int tgif_image_resize_to_normalized_chw(
	const tgif_image *image,
	int out_w,
	int out_h,
	float *dst,
	size_t dst_count
) {
	size_t plane;
	int y;
	int x;
	float mn = 255.0f;
	float mx = 0.0f;

	if (image == NULL || dst == NULL || out_w <= 0 || out_h <= 0) {
		return -1;
	}

	plane = (size_t)out_w * (size_t)out_h;
	if (dst_count < plane * 3u) {
		return -1;
	}

	if (out_h == image->height && out_w == image->width) {
		unsigned char *data = image->pixels;
		int imin = data[0];
		int imax = data[0];
		size_t x, y, n = (size_t)out_w * image->channels;
		size_t pad = image->stride - (size_t)out_w * image->channels;
		float scale;

		for (y = out_h; y > 0; y--) {
			for (x = out_w; x > 0; x--) {
				unsigned char s = *data++;
				if (s < imin)
					imin = s;
				else if (s > imax)
					imax = s;
				else
					continue;
				if (imin == 0 && imax == 255)
					break;
			}
			if (imin == 0 && imax == 255)
				break;
			data += pad;
		}

		if (imin == imax) {
			memset(dst, 0, plane * 3u * sizeof(float));
			return 0;
		}

		scale = 1.0f / (imax - imin);
		data = image->pixels;
		if (image->channels == 1) {
			for (y = out_h; y > 0; y--) {
				for (x = out_w; x > 0; x--) {
					dst[0] = dst[plane] = dst[2*plane] = (*data++ - imin) * scale;
					dst++;
				}
				data += pad;
			}
		} else {
			for (y = out_h; y > 0; y--) {
				for (x = out_w; x > 0; x--) {
					dst[0] = (*data++ - imin) * scale;
					dst[plane] = (*data++ - imin) * scale;
					dst[2*plane] = (*data++ - imin) * scale;
					dst++;
				}
				data += pad;
			}
		}

		return 0;
	}

	/*
	Preprocessing intentionally operates in the original uint8 domain:
	bilinear resize first, then global min/max normalization to 0..1 over
	the resized 3-channel tensor. sample_rgb_bilinear therefore reads
	unsigned-byte pixels on purpose.
	*/
	for (y = 0; y < out_h; ++y) {
		for (x = 0; x < out_w; ++x) {
			float sx = (((float)x + 0.5f) * (float)image->width / (float)out_w) - 0.5f;
			float sy = (((float)y + 0.5f) * (float)image->height / (float)out_h) - 0.5f;
			float rgb[3];
			size_t idx = (size_t)y * (size_t)out_w + (size_t)x;

			sample_rgb_bilinear(image, sx, sy, rgb);
			dst[idx] = rgb[0];
			dst[plane + idx] = rgb[1];
			dst[(2u * plane) + idx] = rgb[2];

			if (rgb[0] < mn) mn = rgb[0];
			if (rgb[1] < mn) mn = rgb[1];
			if (rgb[2] < mn) mn = rgb[2];
			if (rgb[0] > mx) mx = rgb[0];
			if (rgb[1] > mx) mx = rgb[1];
			if (rgb[2] > mx) mx = rgb[2];
		}
	}

	if (mx > mn) {
		size_t i;
		float scale = 1.0f / (mx - mn);
		for (i = 0; i < plane * 3u; ++i) {
			dst[i] = (dst[i] - mn) * scale;
		}
	} else {
		memset(dst, 0, plane * 3u * sizeof(float));
	}
	return 0;
}
