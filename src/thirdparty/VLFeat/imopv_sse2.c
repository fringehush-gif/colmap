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
  vl_index y ;
  vl_index dheight = (src_height - 1) / step + 1 ;
  vl_bool use_simd  = VALIGNED(src_stride) ;
  vl_bool transp    = flags & VL_TRANSPOSE ;
  vl_bool zeropad   = (flags & VL_PAD_MASK) == VL_PAD_BY_ZERO ;
  double totcol = 0 ;
  double simdcol = 0 ;

#if (FLT == VL_TYPE_FLOAT) && defined(VL_ARCH_X64)
  vl_bool separate_buffers = VL_FALSE ;
  vl_uintptr src_begin, src_end, dst_begin, dst_end, filter_begin, filter_end ;
  /* Unsigned subtraction avoids signed overflow for the filter support. */
  if (src_width >= 128 && src_height >= 32 && use_simd &&
      filt_end >= filt_begin && dheight > 0 &&
      dst_stride >= (transp ? (vl_size)dheight : src_width)) {
    vl_uintptr filter_count = (vl_uintptr)filt_end - (vl_uintptr)filt_begin ;
    if (filter_count < (vl_uintptr)-1 &&
        _vl_imconvcol_sse2_float_span(src, src_height, src_width, src_stride,
                                     &src_begin, &src_end) &&
        _vl_imconvcol_sse2_float_span(dst,
                                     transp ? src_width : (vl_size)dheight,
                                     transp ? (vl_size)dheight : src_width,
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
    /* Advance adjacent SSE vectors together. Keep each pixel's tap order and
       the original vector/scalar path for small images and the remainder. */
    if (separate_buffers && src_width >= 128 && src_height >= 32 && use_simd &&
        x + 4 * VSIZE < (signed)src_width && VALIGNED(src + x)) {
      for (y = 0 ; y < (signed)src_height ; y += step) {
        union {VTYPE v ; T x[VSIZE] ;} a0, a1, a2, a3 ;
        VTYPE v0 = VSTZ(), v1 = VSTZ(), v2 = VSTZ(), v3 = VSTZ(), c ;
        T const *filti = filt ;
        vl_index stop = filt_end - y ;
        T const *srci = src + x - stop * src_stride ;
        int b, lane ;
        a0.v = a1.v = a2.v = a3.v = VSTZ() ;

        if (stop > 0) {
          if (!zeropad) {
            v0 = *(VTYPE*)(src + x) ;
            v1 = *(VTYPE*)(src + x + VSIZE) ;
            v2 = *(VTYPE*)(src + x + 2 * VSIZE) ;
            v3 = *(VTYPE*)(src + x + 3 * VSIZE) ;
          }
          while (filti > filt - stop) {
            c = VLD1(filti--) ;
            a0.v = VADD(a0.v, VMUL(v0, c)) ;
            a1.v = VADD(a1.v, VMUL(v1, c)) ;
            a2.v = VADD(a2.v, VMUL(v2, c)) ;
            a3.v = VADD(a3.v, VMUL(v3, c)) ;
            srci += src_stride ;
          }
        }
        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        while (filti > filt - stop) {
          v0 = *(VTYPE*)srci ;
          v1 = *(VTYPE*)(srci + VSIZE) ;
          v2 = *(VTYPE*)(srci + 2 * VSIZE) ;
          v3 = *(VTYPE*)(srci + 3 * VSIZE) ;
          c = VLD1(filti--) ;
          a0.v = VADD(a0.v, VMUL(v0, c)) ;
          a1.v = VADD(a1.v, VMUL(v1, c)) ;
          a2.v = VADD(a2.v, VMUL(v2, c)) ;
          a3.v = VADD(a3.v, VMUL(v3, c)) ;
          srci += src_stride ;
        }
        if (zeropad) v0 = v1 = v2 = v3 = VSTZ() ;
        stop = filt_end - filt_begin + 1 ;
        while (filti > filt - stop) {
          c = VLD1(filti--) ;
          a0.v = VADD(a0.v, VMUL(v0, c)) ;
          a1.v = VADD(a1.v, VMUL(v1, c)) ;
          a2.v = VADD(a2.v, VMUL(v2, c)) ;
          a3.v = VADD(a3.v, VMUL(v3, c)) ;
        }
        for (b = 0 ; b < 4 ; ++b) {
          T const *values = b == 0 ? a0.x : b == 1 ? a1.x : b == 2 ? a2.x : a3.x ;
          for (lane = 0 ; lane < VSIZE ; ++lane) {
            if (transp) dst[(b * VSIZE + lane) * dst_stride] = values[lane] ;
            else dst[b * VSIZE + lane] = values[lane] ;
          }
        }
        dst += transp ? 1 : dst_stride ;
      }
      if (transp) dst += 4 * VSIZE * dst_stride - dheight ;
      else dst += 4 * VSIZE - dheight * dst_stride ;
      x += 4 * VSIZE ;
      simdcol += 4 * VSIZE ;
      totcol += 4 * VSIZE ;
      continue ;
    }
#endif
    /* Calculate dest[x,y] = sum_p image[x,p] filt[y - p]
     * where supp(filt) = [filt_begin, filt_end] = [fb,fe].
     *
     * CHUNK_A: y - fe <= p < 0
     *          completes VL_MAX(fe - y, 0) samples
     * CHUNK_B: VL_MAX(y - fe, 0) <= p < VL_MIN(y - fb, height - 1)
     *          completes fe - VL_MAX(fb, height - y) + 1 samples
     * CHUNK_C: completes all samples
     */

    T const *filti ;
    vl_index stop ;

    if ((x + VSIZE < (signed)src_width) &
        VALIGNED(src + x) & use_simd)
    {
      /* ----------------------------------------------  Vectorized */
      for (y = 0 ; y < (signed)src_height ; y += step)  {
        union {VTYPE v ; T x [VSIZE] ; } acc ;
        VTYPE v, c ;
        T const *srci ;
        acc.v = VSTZ () ;
        v = VSTZ() ;

        filti = filt ;
        stop = filt_end - y ;
        srci = src + x - stop * src_stride ;

        if (stop > 0) {
          if (zeropad) {
            v = VSTZ () ;
          } else {
            v = * (VTYPE*) (src + x) ;
          }
          while (filti > filt - stop) {
            c = VLD1 (filti--) ;
            acc.v = VADD (acc.v,  VMUL (v, c)) ;
            srci += src_stride ;
          }
        }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        while (filti > filt - stop) {
          v = * (VTYPE*) srci ;
          c = VLD1 (filti--) ;
          acc.v = VADD (acc.v, VMUL (v, c)) ;
          srci += src_stride ;
        }

        if (zeropad) v = VSTZ () ;

        stop = filt_end - filt_begin + 1;
        while (filti > filt - stop) {
          c = VLD1 (filti--) ;
          acc.v = VADD (acc.v, VMUL (v, c)) ;
        }

        if (transp) {
          *dst = acc.x[0] ; dst += dst_stride ;
          *dst = acc.x[1] ; dst += dst_stride ;
#if(VSIZE == 4)
          *dst = acc.x[2] ; dst += dst_stride ;
          *dst = acc.x[3] ; dst += dst_stride ;
#endif
          dst += 1 * 1 - VSIZE * dst_stride ;
        } else {
          *dst = acc.x[0] ; dst += 1 ;
          *dst = acc.x[1] ; dst += 1 ;
#if(VSIZE == 4)
          *dst = acc.x[2] ; dst += 1 ;
          *dst = acc.x[3] ; dst += 1 ;
#endif
          dst += 1 * dst_stride - VSIZE * 1 ;
        }
      } /* next y */
      if (transp) {
        dst += VSIZE * dst_stride - dheight * 1 ;
      } else {
        dst += VSIZE * 1 - dheight * dst_stride ;
      }
      x       += VSIZE ;
      simdcol += VSIZE ;
      totcol  += VSIZE ;
    } else {
      /* -------------------------------------------------  Vanilla */
      for (y = 0 ; y < (signed)src_height ; y += step) {
        T acc = 0 ;
        T v = 0, c ;
        T const* srci ;

        filti = filt ;
        stop = filt_end - y ;
        srci = src + x - stop * src_stride ;

        if (stop > 0) {
          if (zeropad) {
            v = 0 ;
          } else {
            v = *(src + x) ;
          }
          while (filti > filt - stop) {
            c = *filti-- ;
            acc += v * c ;
            srci += src_stride ;
          }
        }

        stop = filt_end - VL_MAX(filt_begin, y - (signed)src_height + 1) + 1 ;
        while (filti > filt - (signed)stop) {
          v = *srci ;
          c = *filti-- ;
          acc += v * c ;
          srci += src_stride ;
        }

        if (zeropad) v = 0 ;

        stop = filt_end - filt_begin + 1 ;
        while (filti > filt - stop) {
          c = *filti-- ;
          acc += v * c ;
        }

        if (transp) {
          *dst = acc ; dst += 1 ;
        } else {
          *dst = acc ; dst += dst_stride ;
        }
      } /* next y */
      if (transp) {
        dst += 1 * dst_stride - dheight * 1 ;
      } else {
        dst += 1 * 1 - dheight * dst_stride ;
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
