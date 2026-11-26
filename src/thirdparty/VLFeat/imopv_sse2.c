/** @file imopv_sse2.c
 ** @brief Vectorized image operations - SSE2 - Definition
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if ! defined(VL_DISABLE_SSE2) & ! defined(__SSE2__)
#error "Compiling with SSE2 enabled, but no __SSE2__ defined"
#endif

#if ! defined(VL_DISABLE_SSE2)

#ifndef VL_IMOPV_SSE2_INSTANTIATING

#include <emmintrin.h>

#include "imopv.h"
#include "imopv_sse2.h"

#if defined(VL_ARCH_X64)
/* Conservative half-open byte span, including row padding. */
static vl_bool
_vl_imconvcol_sse2_float_span (void const *buffer,
                              vl_size rows, vl_size columns, vl_size stride,
                              vl_uintptr *begin, vl_uintptr *end)
{
  vl_uintptr const maximum = (vl_uintptr)-1 ;
  vl_uintptr count ;
  if (rows == 0 || columns == 0) return VL_FALSE ;
  if (stride != 0 && rows - 1 > maximum / stride) return VL_FALSE ;
  count = (rows - 1) * stride ;
  if (columns > maximum - count) return VL_FALSE ;
  count += columns ;
  if (count > maximum / sizeof(float)) return VL_FALSE ;
  count *= sizeof(float) ;
  *begin = (vl_uintptr)buffer ;
  if (count > maximum - *begin) return VL_FALSE ;
  *end = *begin + count ;
  return VL_TRUE ;
}
#endif

#define FLT VL_TYPE_FLOAT
#define VL_IMOPV_SSE2_INSTANTIATING
#include "imopv_sse2.c"

#define FLT VL_TYPE_DOUBLE
#define VL_IMOPV_SSE2_INSTANTIATING
#include "imopv_sse2.c"

/* ---------------------------------------------------------------- */
/* VL_IMOPV_SSE2_INSTANTIATING */
#else

#include "float.h"

/* ---------------------------------------------------------------- */
void
VL_XCAT3(_vl_imconvcol_v, SFX, _sse2)
(T* dst, vl_size dst_stride,
 T const* src,
 vl_size src_width, vl_size src_height, vl_size src_stride,
 T const* filt, vl_index filt_begin, vl_index filt_end,
 int step, unsigned int flags)
{
  vl_index x = 0 ;
  vl_index y, output_y ;
  vl_bool use_simd  = VALIGNED(src_stride) ;
  vl_bool transp    = flags & VL_TRANSPOSE ;
  vl_bool zeropad   = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;
  double totcol = 0 ;
  double simdcol = 0 ;

#if (FLT == VL_TYPE_FLOAT) && defined(VL_ARCH_X64)
  vl_bool separate_buffers = VL_FALSE ;
  vl_uintptr src_begin, src_end, dst_begin, dst_end, filter_begin, filter_end ;
  if (src_width >= 128 && src_height >= 32 && use_simd && step > 0 &&
      filt_end >= filt_begin) {
    vl_size const output_height = (src_height - 1) / step + 1 ;
    /* Unsigned subtraction also handles short supports at signed limits. */
    vl_uintptr filter_count = (vl_uintptr)filt_end - (vl_uintptr)filt_begin ;
    if (dst_stride >= (transp ? output_height : src_width) &&
        filter_count < (vl_uintptr)-1 &&
        _vl_imconvcol_sse2_float_span(src, src_height, src_width, src_stride,
                                     &src_begin, &src_end) &&
        _vl_imconvcol_sse2_float_span(dst,
                                     transp ? src_width : output_height,
                                     transp ? output_height : src_width,
                                     dst_stride, &dst_begin, &dst_end) &&
        _vl_imconvcol_sse2_float_span(filt, 1, filter_count + 1, 0,
                                     &filter_begin, &filter_end)) {
      separate_buffers = (dst_end <= src_begin || src_end <= dst_begin) &&
                         (dst_end <= filter_begin || filter_end <= dst_begin) ;
    }
  }
#endif

  /* let filt point to the last sample of the filter */
  filt += filt_end - filt_begin ;

  while (x < (signed)src_width) {
#if (FLT == VL_TYPE_FLOAT) && defined(VL_ARCH_X64)
    /* Share filter broadcasts across four vectors without changing tap order.
       Aliased layouts, small inputs and tails keep the original traversal. */
    if (separate_buffers && x + 4 * VSIZE < (signed)src_width &&
        VALIGNED(src + x)) {
      for (y = 0, output_y = 0 ; y < (signed)src_height ; y += step, ++output_y) {
        union {VTYPE v ; T x[VSIZE] ;} acc[4] ;
        VTYPE v[4], c ;
        vl_index tap = 0 ;
        vl_index stop = filt_end > y ? VL_MIN(filt_end - y, filt_end - filt_begin + 1) : 0 ;
        vl_index source_y ;
        int b, lane ;
        for (b = 0 ; b < 4 ; ++b) acc[b].v = v[b] = VSTZ() ;

        if (stop > 0) {
          if (!zeropad) {
            for (b = 0 ; b < 4 ; ++b) v[b] = *(VTYPE*)(src + x + b * VSIZE) ;
          }
          while (tap < stop) {
            c = VLD1(filt - tap++) ;
            for (b = 0 ; b < 4 ; ++b) acc[b].v = VADD(acc[b].v, VMUL(v[b], c)) ;
          }
        }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        source_y = tap < stop ? VL_MAX(0, y - filt_end) : 0 ;
        while (tap < stop) {
          c = VLD1(filt - tap++) ;
          for (b = 0 ; b < 4 ; ++b) {
            v[b] = *(VTYPE*)(src + x + b * VSIZE + source_y * src_stride) ;
            acc[b].v = VADD(acc[b].v, VMUL(v[b], c)) ;
          }
          ++source_y ;
        }

        if (zeropad) {
          for (b = 0 ; b < 4 ; ++b) v[b] = VSTZ() ;
        } else if (tap == 0) {
          for (b = 0 ; b < 4 ; ++b) {
            v[b] = *(VTYPE*)(src + x + b * VSIZE + (src_height - 1) * src_stride) ;
          }
        }
        stop = filt_end - filt_begin + 1 ;
        while (tap < stop) {
          c = VLD1(filt - tap++) ;
          for (b = 0 ; b < 4 ; ++b) acc[b].v = VADD(acc[b].v, VMUL(v[b], c)) ;
        }

        for (b = 0 ; b < 4 ; ++b) {
          for (lane = 0 ; lane < VSIZE ; ++lane) {
            if (transp) dst[output_y + (b * VSIZE + lane) * dst_stride] = acc[b].x[lane] ;
            else dst[output_y * dst_stride + b * VSIZE + lane] = acc[b].x[lane] ;
          }
        }
      }
      dst += transp ? 4 * VSIZE * dst_stride : 4 * VSIZE ;
      x += 4 * VSIZE ;
      simdcol += 4 * VSIZE ;
      totcol += 4 * VSIZE ;
      continue ;
    }
#endif
    /* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
     * where supp(filt) = [filt_begin, filt_end] = [fb,fe].
     *
     * Count taps in descending filter order. Clip the upper padding
     * to the actual filter length, address only in-range source rows, and
     * use the last edge for lower padding even without a central sample.
     */

    vl_index tap ;
    vl_index stop ;

    if ((x + VSIZE < (signed)src_width) &
        VALIGNED(src + x) & use_simd)
    {
      /* ----------------------------------------------  Vectorized */
      for (y = 0, output_y = 0 ; y < (signed)src_height ; y += step, ++output_y)  {
        union {VTYPE v ; T x [VSIZE] ; } acc ;
        VTYPE v, c ;
        vl_index source_y ;
        acc.v = VSTZ () ;
        v = VSTZ() ;

        tap = 0 ;
        stop = filt_end > y ? VL_MIN(filt_end - y, filt_end - filt_begin + 1) : 0 ;

        if (stop > 0) {
          if (zeropad) {
            v = VSTZ () ;
          } else {
            v = * (VTYPE*) (src + x) ;
          }
          while (tap < stop) {
            c = VLD1 (filt - tap) ;
            ++tap ;
            acc.v = VADD (acc.v,  VMUL (v, c)) ;
          }
        }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        source_y = tap < stop ? VL_MAX(0, y - filt_end) : 0 ;
        while (tap < stop) {
          v = * (VTYPE*) (src + x + source_y * src_stride) ;
          c = VLD1 (filt - tap) ;
          ++tap ;
          acc.v = VADD (acc.v, VMUL (v, c)) ;
          ++source_y ;
        }

        if (zeropad) v = VSTZ () ;
        else if (tap == 0) v = * (VTYPE*) (src + x + (src_height - 1) * src_stride) ;

        stop = filt_end - filt_begin + 1;
        while (tap < stop) {
          c = VLD1 (filt - tap) ;
          ++tap ;
          acc.v = VADD (acc.v, VMUL (v, c)) ;
        }

        if (transp) {
          dst[output_y] = acc.x[0] ;
          dst[output_y + dst_stride] = acc.x[1] ;
#if(VSIZE == 4)
          dst[output_y + 2 * dst_stride] = acc.x[2] ;
          dst[output_y + 3 * dst_stride] = acc.x[3] ;
#endif
        } else {
          dst[output_y * dst_stride] = acc.x[0] ;
          dst[output_y * dst_stride + 1] = acc.x[1] ;
#if(VSIZE == 4)
          dst[output_y * dst_stride + 2] = acc.x[2] ;
          dst[output_y * dst_stride + 3] = acc.x[3] ;
#endif
        }
      } /* next y */
      if (x + VSIZE < (signed)src_width) {
        dst += transp ? VSIZE * dst_stride : VSIZE ;
      }
      x       += VSIZE ;
      simdcol += VSIZE ;
      totcol  += VSIZE ;
    } else {
      /* -------------------------------------------------  Vanilla */
      for (y = 0, output_y = 0 ; y < (signed)src_height ; y += step, ++output_y) {
        T acc = 0 ;
        T v = 0, c ;
        vl_index source_y ;

        tap = 0 ;
        stop = filt_end > y ? VL_MIN(filt_end - y, filt_end - filt_begin + 1) : 0 ;

        if (stop > 0) {
          if (zeropad) {
            v = 0 ;
          } else {
            v = *(src + x) ;
          }
          while (tap < stop) {
            c = filt[-tap++] ;
            acc += v * c ;
          }
        }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        source_y = tap < stop ? VL_MAX(0, y - filt_end) : 0 ;
        while (tap < stop) {
          v = *(src + x + source_y * src_stride) ;
          c = filt[-tap++] ;
          acc += v * c ;
          ++source_y ;
        }

        if (zeropad) v = 0 ;
        else if (tap == 0) v = src[x + (src_height - 1) * src_stride] ;

        stop = filt_end - filt_begin + 1 ;
        while (tap < stop) {
          c = filt[-tap++] ;
          acc += v * c ;
        }

        if (transp) {
          dst[output_y] = acc ;
        } else {
          dst[output_y * dst_stride] = acc ;
        }
      } /* next y */
      if (x + 1 < (signed)src_width) {
        dst += transp ? dst_stride : 1 ;
      }
      x      += 1 ;
      totcol += 1 ;
    } /* next x */
  }
}

/* ---------------------------------------------------------------- */
#if 0
void
VL_XCAT(_vl_imconvcoltri_v, SFX, sse2)
(T* dst, int dst_stride,
 T const* src,
 int src_width, int src_height, int src_stride,
 int filt_size,
 int step, unsigned int flags)
{
  int x = 0 ;
  int y ;
  int dheight = (src_height - 1) / step + 1 ;
  vl_bool use_simd  = ((src_stride & ALIGNSTRIDE) == 0) &&
  (! (flags & VL_NO_SIMD)) ;
  vl_bool transp = flags & VL_TRANSPOSE ;
  vl_bool zeropad = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;

  T * buff = vl_malloc(sizeof(T) * (src_height + filt_size)) ;
#define fa (1.0 / (double) (filt_size + 1))
  T scale = fa*fa*fa*fa ;
  buff += filt_size ;

  while (x < src_width) {
    T const *srci ;

    use_simd = 0 ;
    if ((x + VSIZE < src_width) &
        (((vl_ptrint)(src + x) & ALIGNPTR) == 0) &
        use_simd)
    {

    } else {
      int stridex = transp ? dst_stride : 1 ;
      int stridey = transp ? 1 : dst_stride ;
      srci = src + x + src_stride * (src_height - 1) ;

      /* integrate backward the column */
      buff [src_height - 1] = *srci ;
      for (y = src_height-2 ; y >=  0 ; --y) {
        srci -= src_stride ;
        buff [y] = buff [y+1] + *srci ;
      }
      if (zeropad) {
        for ( ; y >= - filt_size ; --y) {
          buff [y] = buff [y+1] ;
        }
      } else {
        for ( ; y >= - filt_size ; --y) {
          buff [y] = buff[y+1] + *srci ;
        }
      }

      /* compute the filter forward */
      for (y = - filt_size ; y < src_height - filt_size ; ++y) {
        buff [y] = buff [y] - buff [y + filt_size] ;
      }
      if (! zeropad) {
        for (y = src_height - filt_size ; y < src_height ; ++y) {
          buff [y] = buff [y] - buff [src_height-1]  *
          (src_height - filt_size - y) ;
        }
      }

      /* integrate forward the column */
      for (y = - filt_size + 1 ; y < src_height ; ++y) {
        buff [y] += buff [y - 1] ;
      }

      /* compute the filter backward */
      for (y = src_height - 1 ; y >= 0 ; --y) {
        dst [x*stridex + y*stridey]
        = scale * (buff [y] - buff [y - filt_size]) ;
      }
    } /* next y */
    x += 1 ;
  }
  vl_free (buff - filt_size) ;
}
#endif

#undef FLT
#undef VL_IMOPV_SSE2_INSTANTIATING
#endif

/* ! VL_DISABLE_SSE2 */
#endif
