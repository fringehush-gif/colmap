// SPDX-License-Identifier: BSD-3-Clause
// Match the public mat.h normal codecs, without fast math/FMA.
// Validated input only. No padding, aliasing, or alignment assumptions.
struct NormalPackOctPair { int16 x; int16 y; };

// ISPC round() is ties-to-even. C++ std::round() is ties-away.
// Do not use floor(abs(x)+0.5): that addition can round across a boundary.
static inline double round_away(double x) {
    double a = abs(x);
    double integral = floor(a);
    double rounded = integral + ((a - integral >= 0.5) ? 1.0 : 0.0);
    return x < 0.0 ? -rounded : rounded;
}

export uniform int32 normal_pack_validate_normals(const uniform float src[],
                                            uniform int32 count) {
    int32 first_bad = 2147483647;
    foreach (i = 0 ... count) {
        float x = src[i];
        // Bit classification rejects both infinities and every NaN payload.
        bool finite = (intbits(x) & 0x7f800000) != 0x7f800000;
        float safe = finite ? x : 0.0f;
        if (!finite || safe < -1.0f || safe > 1.0f)
            first_bad = min(first_bad, i);
    }
    return reduce_min(first_bad);
}

export void normal_pack_snorm(const uniform float src[], uniform int16 dst[],
                            uniform int32 count) {
    foreach (i = 0 ... count)
        dst[i] = (int16)round_away((double)src[i] * 32767.0);
}

// Planar source pointers permit chunking without changing plane pitch.
// AoS result uses a generated ISPC ABI struct, avoiding type-punned C++ stores.
export void normal_pack_oct(const uniform float sx[],
                          const uniform float sy[],
                          const uniform float sz[],
                          uniform NormalPackOctPair dst[],
                          uniform uint16 magnitude[],
                          uniform int32 count,
                          uniform bool with_magnitude) {
    foreach (i = 0 ... count) {
        double x = (double)sx[i], y = (double)sy[i], z = (double)sz[i];
        double l1 = (abs(x) + abs(y)) + abs(z);
        int16 qx = (int16)-32768, qy = (int16)-32768;
        if (l1 != 0.0) {
            double u = x / l1, v = y / l1;
            if (z < 0.0) {
                double old_u = u;
                u = (1.0 - abs(v)) * ((u >= 0.0) ? 1.0 : -1.0);
                v = (1.0 - abs(old_u)) * ((v >= 0.0) ? 1.0 : -1.0);
            }
            qx = (int16)round_away(u * 32767.0);
            qy = (int16)round_away(v * 32767.0);
        }
        dst[i].x = qx;
        dst[i].y = qy;
        if (with_magnitude) {
            double sum = (x * x + y * y) + z * z;
            // Unsuffixed ISPC constants are float, unlike C++; retain double.
            double scaled = sqrt(sum) * 65535.0d / 1.7320508075688772935d;
            magnitude[i] = (uint16)round_away(scaled);
        }
    }
}
