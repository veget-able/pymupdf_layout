#ifndef TGF_IMAGE_H
#define TGF_IMAGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
	void *opaque;
	void *(*malloc)(void *opaque, size_t size);
	void *(*calloc)(void *opaque, size_t n, size_t size);
	void (*free)(void *opaque, void *ptr);
} tgif_context;

void *tgif_malloc(tgif_context *ctx, size_t size);
void *tgif_calloc(tgif_context *ctx, size_t n, size_t size);
void tgif_free(tgif_context *ctx, void *ptr);


typedef enum tgif_pixel_format {
	TGF_PIXFMT_GRAY8 = 1,
	TGF_PIXFMT_RGB8 = 2
} tgif_pixel_format;

typedef struct tgif_image {
	int width;
	int height;
	int channels;
	size_t stride;
	tgif_pixel_format format;
	int borrowed;
	unsigned char *pixels;
} tgif_image;

int tgif_image_load_pnm(tgif_context *ctx, const char *path, tgif_image **image);
void tgif_image_destroy(tgif_context *ctx, tgif_image *image);
tgif_image *tgif_new_image(tgif_context *ctx, int w, int h, int n);
tgif_image *tgif_new_image_borrowed_data(tgif_context *ctx, int w, int h, int n, size_t stride, unsigned char *data);
int tgif_image_resize_to_normalized_chw(
	const tgif_image *image,
	int out_w,
	int out_h,
	float *dst,
	size_t dst_count
);

#endif

#ifdef __cplusplus
}
#endif
