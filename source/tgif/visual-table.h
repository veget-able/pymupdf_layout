#ifndef TGF_VISUAL_TABLE_H
#define TGF_VISUAL_TABLE_H

#include "mupdf/fitz.h"

#ifdef __cplusplus
extern "C"
{
#endif

int
fz_visual_table_grid_finder(fz_context *ctx, fz_page *page, fz_rect bounds, fz_stext_grid_positions **xposp, fz_stext_grid_positions **yposp);

fz_stext_block *
fz_find_visual_table_within_bounds(fz_context *ctx, fz_stext_page *stext, fz_page *page, fz_rect bounds);

#ifdef __cplusplus
}
#endif

#endif
