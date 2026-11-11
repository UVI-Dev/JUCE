/*
  ==============================================================================

   This file is part of the JUCE framework.
   Copyright (c) Raw Material Software Limited

   JUCE is an open source framework subject to commercial or open source
   licensing.

   By downloading, installing, or using the JUCE framework, or combining the
   JUCE framework with any other source code, object code, content or any other
   copyrightable work, you agree to the terms of the JUCE End User Licence
   Agreement, and all incorporated terms including the JUCE Privacy Policy and
   the JUCE Website Terms of Service, as applicable, which will bind you. If you
   do not agree to the terms of these agreements, we will not license the JUCE
   framework to you, and you must discontinue the installation or download
   process and cease use of the JUCE framework.

   JUCE End User Licence Agreement: https://juce.com/legal/juce-9-licence/
   JUCE Privacy Policy: https://juce.com/juce-privacy-policy
   JUCE Website Terms of Service: https://juce.com/juce-website-terms-of-service/

   Or:

   You may also use this code under the terms of the AGPLv3:
   https://www.gnu.org/licenses/agpl-3.0.en.html

   THE JUCE FRAMEWORK IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL
   WARRANTIES, WHETHER EXPRESSED OR IMPLIED, INCLUDING WARRANTY OF
   MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ARE DISCLAIMED.

  ==============================================================================
*/

/*  GPU renderer for NSViewComponentPeer (JUCE_METAL_RENDERER).

    Frame (MetalLayerRenderer):
    - The peer keeps the exact list of dirty rects and hands it over once per display refresh;
      nothing goes through AppKit's dirty-rect coalescing. Each window owns a retained texture and
      only the dirty rects are repainted into it, in one traversal of the component tree.
    - paint() runs on the message thread and is recorded (MetalGraphicsContext -> FrameRecorder,
      one per frame in flight). Encoding and commit happen on a serial queue shared by all windows
      (live resize excepted: it must present within the main thread's CA transaction), and so does
      the CPU side of rasterising: building and flattening path outlines, drawing new glyphs. The
      message thread only records, as with Core Animation (drawsAsynchronously), Chrome and
      Flutter.
    - Presentation: the retained texture is copied into a drawable of the window's CAMetalLayer
      on each frame (see encodePresentCopy, and the alternatives measured there).

    Drawing (instances of one quad, a few pipelines):
    - Draws are batched by lookback, as WebRender's AlphaBatchList does (FrameRecorder::reorder):
      each joins an earlier batch with its state unless a batch in between overlaps it; the
      painter's depth (the recording index) keeps the opaque pass correct. Pipelines are specialised per family (shapes, glyphs, images, path coverage)
      with function constants, so no pixel pays for the others' registers.
    - Opaque interiors (opaque solid boxes) are drawn first with a trivial shader and their depth,
      as WebRender's opaque pass: the GPU rejects what they cover before shading it.
    - Rects, rounded rects, circles and lines: distance to the outline per pixel, turned into the
      exact area of the pixel inside it (halfPlaneCoverage), under any affine transform.
    - Paths by coverage counting: their flattened edges are accumulated as signed areas in an
      R32Float atlas on the GPU and a prefix sum per row gives each pixel its winding number with
      exact area anti-aliasing (as juce::EdgeTable). clipToPath reads the same coverage. A
      stroke's outline (offset edges, joins, caps) is built from its centre line on the GPU, by
      the vertex shader of its accumulation (strokeAccumVs).
    - Glyphs: rasterised once by JUCE's CoreGraphics context into an R8 atlas, and found
      through one table per font size.
    - Gradients (colour stops interpolated per pixel), tiled-image fills, image clips, the image
      blend modes and transparency layers are native too. What is left (colour fonts, ...) is drawn
      by the software renderer into a tile that is then composited; JUCE_METAL_RENDERER_STATS
      reports it, with the CPU and GPU time per frame.

    Pixel target: what the CoreGraphics renderer draws, where CoreGraphics is systematic, and the
    exact area elsewhere.
    - Text matches CoreGraphics to the pixel: same rasteriser, and glyphs placed as CoreGraphics
      places them in a window (x rounded down to 1/16 px, baseline rounded up to a whole pixel).
    - Gradients interpolate their stops as CGGradient does (float components, not premultiplied).
    - Straight edges at any angle match CoreGraphics (both are the exact area).
    - Curves and rotated boxes: CoreGraphics strays from the exact area by ~12/255 RMS on their
      edges (its flattening, measured), JUCE's software renderer by a few levels (0.55 cubics);
      this renderer stays within 1/255 of the exact area.
    Measured by capturing windows drawn by each engine (every GraphicsDemo test, plus scenes of
    text at fractional scales, thin strokes, blend modes, gradients, transformed shapes and images)
    and comparing against CoreGraphics and the software renderer at thresholds of 1, 4 and 16
    levels.
*/

// Metal classes are only used behind @available checks or on systems where the device exists.
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wunguarded-availability", "-Wunguarded-availability-new")

#if JUCE_METAL_RENDERER_DIRTY_RECT
 #import <objc/message.h>
#endif

namespace juce
{

namespace MetalRendering
{

struct Float4 { float x = 0, y = 0, z = 0, w = 0; };

// Must match 'Inst' in the shader source (7 x float4, 16-byte aligned).
struct Instance
{
    Float4 xf0;     // device.x = xf0.x * local.x + xf0.y * local.y + xf0.z
    Float4 xf1;     // device.y = xf1.x * local.x + xf1.y * local.y + xf1.z
    Float4 rect;    // local rectangle: x, y, w, h
    Float4 clip;    // device pixels: x, y, w, h (pixel centres outside are discarded)
    Float4 colour;  // premultiplied
    Float4 params;  // x: corner radius (local; coverage: 1 = even-odd), y: stroke width (local, 0 = fill),
                    // z: kind, w: gradient or image pattern index + 1 (0 = solid colour)
    Float4 uv;      // u0, v0, u1, v1 (coverage: xy = offset of the path's area in its atlas page)
};

enum class Kind : int
{
    box = 0,            // (rounded) rectangle, fill or stroke
    ellipse = 1,        // fill or stroke
    glyph = 2,          // R8 glyph atlas (texture 0) * colour
    rgbaTexture = 3,    // premultiplied BGRA texture (texture 1) * colour.a
    alphaTexture = 4,   // R8 texture (texture 1) * colour
    coverage = 5        // path fill: winding numbers from the coverage atlas (texture 4)
};

constexpr const char* shaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Tested and not kept: computing in half precision (twice the ALU rate of float on Apple GPUs)
// saved 2.5% of the GPU time on a widget grid and took edges further from CoreGraphics.

// xf0.w: the instance's painter's order (its index when recorded), which sets its depth: draws are
// reordered before encoding (see FrameRecorder::reorder), so the buffer position isn't that order.
struct Inst { float4 xf0; float4 xf1; float4 rect; float4 clip; float4 colour; float4 params; float4 uv; };

struct VOut
{
    float4 pos [[position]];
    float2 local;
    float2 uv;
    uint iid [[flat]];
    float4 inv [[flat]]; // device -> local, without translation: rows (x, y) and (z, w)
    float2 axisRates [[flat]]; // lengths of those rows: local units per device pixel across the x and y edges
};

bool isAxisAligned (float4 inv) { return inv.y == 0.0 && inv.z == 0.0; }

// Area of the pixel (the unit square around its sample point) on the side n.x <= t of a straight
// edge, n being the edge's unit normal in device pixels: the box-filtered coverage that JUCE's
// EdgeTable and CoreGraphics compute. A ramp of the distance alone, saturate (0.5 + t), is only
// that area for horizontal and vertical edges; at 45 degrees it's off by up to 0.04 (11/255).
// Using that ramp saved only 7% of the GPU time of a full repaint (2.57 to 2.39 ms on an M1): the
// per-pixel cost is in the distance and its Jacobian, not in this area.
// Piecewise: linear while the edge crosses two opposite sides of the square, quadratic while it
// cuts a corner (a, b: the normal's larger and smaller components).
float halfPlaneCoverage (float t, float2 n)
{
    const float a = max (abs (n.x), abs (n.y)), b = min (abs (n.x), abs (n.y));

    if (b < 1.0e-4)
        return saturate (0.5 + t / max (a, 1.0e-6));

    const float halfA = 0.5 * (a - b), halfB = 0.5 * (a + b);

    if (abs (t) <= halfA)
        return 0.5 + t / a;

    const float e = max (0.0, halfB - abs (t));
    const float corner = e * e / (2.0 * a * b);
    return t > 0.0 ? 1.0 - corner : corner;
}

// Area of the pixel between two parallel edges, 'halfWidth' each side of a line passing at
// 'offset' from the sample point (device pixels, along the unit normal n).
float slabCoverage (float offset, float halfWidth, float2 n)
{
    return halfPlaneCoverage (halfWidth - offset, n) - halfPlaneCoverage (-halfWidth - offset, n);
}

// Coverage of the pixel by the box |p| <= halfSize in local space: the product of its coverage by
// the slabs between the x edges and between the y edges (exact area for a box aligned with the
// pixels, and away from the corners of a rotated one). The device normal of the x edges is the
// device gradient of local x, the first row of 'inv'.
float boxCoverage (float2 p, float2 halfSize, float4 inv, float2 axisRates)
{
    const float lx = max (axisRates.x, 1.0e-6), ly = max (axisRates.y, 1.0e-6);
    return slabCoverage (p.x / lx, halfSize.x / lx, inv.xy / lx) * slabCoverage (p.y / ly, halfSize.y / ly, inv.zw / ly);
}

// Opaque pass and occlusion (as WebRender does): every instance gets a depth from its painter's
// order, later instances nearer (instance index iid -> 1 - (iid + 1) * depthStep). The interiors
// of opaque fills are drawn first, with their depth and a shader that only returns their colour
// (no blending, any order: the nearest wins). Everything else then draws in painter's order,
// testing Less without writing depth: the GPU rejects, before running the fragment shader, the
// pixels a later opaque interior covers, and the opaque fills' own interiors (equal depth), so the
// full shader only runs on their anti-aliased edges. Measured on a full repaint of a widget grid:
// the fragment stage is ALU bound, and most of its pixels are opaque interiors.
float painterDepth (uint iid, float depthStep) { return 1.0 - float (iid + 1) * depthStep; }

vertex VOut vs (uint vid [[vertex_id]], uint iid [[instance_id]],
                const device Inst* inst [[buffer(0)]], constant float4& viewport [[buffer(1)]],
                constant float& depthStep [[buffer(2)]])
{
    const Inst I = inst[iid];
    const uint kind = uint (I.params.z);
    const float det = I.xf0.x * I.xf1.y - I.xf0.y * I.xf1.x;
    const float4 inv = float4 (I.xf1.y, -I.xf0.y, -I.xf1.x, I.xf0.x) / (abs (det) > 1.0e-12 ? det : 1.0e-12);

    // Shapes, and images that aren't aligned with the pixels: grow the quad by one device pixel
    // across each edge (in local units along x and y), for the coverage ramp, and half the stroke.
    const float2 onePixel = float2 (length (inv.xy), length (inv.zw));
    const float2 pad = kind <= 1 ? onePixel + 0.5 * I.params.y
                     : ((kind == 3 || kind == 4) && ! isAxisAligned (inv) ? onePixel : float2 (0.0));
    const float2 lo = I.rect.xy - pad, hi = I.rect.xy + I.rect.zw + pad;

    // Each instance is drawn as up to eight quads (a triangle list of 6 vertices per quad; the
    // draw call asks for one quad, or eight for batches holding strokes: FrameRecorder::Batch).
    // A stroke's middle, where its coverage is zero, is left out, as WebRender splits a border
    // into segments (border.rs, create_border_segments): a 1 px outline around a widget would
    // otherwise shade all of the widget's pixels to draw a few.
    // Box strokes: four corners of max (corner radius, half stroke + one pixel) from the box edge
    // (quads 0-3), and four strips between them, half the stroke plus one pixel deep inside the
    // box edge (quads 4-7, the anti-aliasing ramp included). Away from the corners the outline is
    // straight, so a pixel centre beyond a strip is more than half the stroke plus a pixel inside
    // it: zero coverage. Ellipse strokes (only nearly circular ones get here): four bands around
    // the square inscribed in the inner circle, one pixel inside it.
    // Without a hole, quad 0 is the whole quad and the others collapse.
    const uint quad = vid / 6, k = vid % 6;
    const uint cornerIndex = k < 3 ? k : (k == 3 ? 2 : (k == 4 ? 1 : 3));
    const float2 corner = float2 (cornerIndex & 1, cornerIndex >> 1);
    float2 qlo = lo, qhi = hi;
    bool collapse = quad != 0;
    const float2 halfSize = 0.5 * I.rect.zw;

    if (kind == 1 && I.params.y > 0.0)
    {
        const float2 holeHalf = (halfSize - 0.5 * I.params.y - onePixel) * 0.70710678;

        if (all (holeHalf > 0.0))
        {
            const float2 holeLo = I.rect.xy + halfSize - holeHalf, holeHi = I.rect.xy + halfSize + holeHalf;
            collapse = quad >= 4;

            if (quad == 0)      { qhi.y = holeLo.y; }                                          // top
            else if (quad == 1) { qlo.y = holeHi.y; }                                          // bottom
            else if (quad == 2) { qlo.y = holeLo.y; qhi = float2 (holeLo.x, holeHi.y); }       // left
            else                { qlo = float2 (holeHi.x, holeLo.y); qhi.y = holeHi.y; }       // right
        }
    }
    else if (kind == 0 && I.params.y > 0.0)
    {
        const float2 depth = 0.5 * I.params.y + onePixel;    // strips, from the box edge inwards
        const float2 cornerSize = max (float2 (max (0.0, min (I.params.x, min (halfSize.x, halfSize.y)))), depth);
        const float2 boxLo = I.rect.xy, boxHi = I.rect.xy + I.rect.zw;

        if (all (2.0 * cornerSize < I.rect.zw))
        {
            collapse = false;

            if (quad < 4)
            {
                const bool right = (quad & 1) != 0, bottom = (quad & 2) != 0;
                qlo = float2 (right ? boxHi.x - cornerSize.x : lo.x, bottom ? boxHi.y - cornerSize.y : lo.y);
                qhi = float2 (right ? hi.x : boxLo.x + cornerSize.x, bottom ? hi.y : boxLo.y + cornerSize.y);
            }
            else if (quad == 4) { qlo = float2 (boxLo.x + cornerSize.x, lo.y);                qhi = float2 (boxHi.x - cornerSize.x, boxLo.y + depth.y); } // top
            else if (quad == 5) { qlo = float2 (boxLo.x + cornerSize.x, boxHi.y - depth.y);   qhi = float2 (boxHi.x - cornerSize.x, hi.y); }            // bottom
            else if (quad == 6) { qlo = float2 (lo.x, boxLo.y + cornerSize.y);                qhi = float2 (boxLo.x + depth.x, boxHi.y - cornerSize.y); } // left
            else                { qlo = float2 (boxHi.x - depth.x, boxLo.y + cornerSize.y);   qhi = float2 (hi.x, boxHi.y - cornerSize.y); }            // right
        }
    }

    if (collapse)
        qhi = qlo;

    float2 local = mix (qlo, qhi, corner);
    float2 dev = float2 (dot (I.xf0.xy, local) + I.xf0.z, dot (I.xf1.xy, local) + I.xf1.z);

    // Without rotation or shear the quad is a device rectangle: rasterise only its part inside the
    // clip (the fragment shader would discard the rest). A window background filled under a few
    // small dirty rects would otherwise be shaded over the whole window, once per rect.
    const float2 diagonal = float2 (I.xf0.x, I.xf1.y);

    if (I.xf0.y == 0.0 && I.xf1.x == 0.0 && all (abs (diagonal) > 1.0e-6))
    {
        const float2 a = diagonal * qlo + float2 (I.xf0.z, I.xf1.z);
        const float2 b = diagonal * qhi + float2 (I.xf0.z, I.xf1.z);
        const float2 boxLo = max (min (a, b), I.clip.xy);
        const float2 boxHi = max (boxLo, min (max (a, b), I.clip.xy + I.clip.zw));
        dev = mix (boxLo, boxHi, corner);
        local = (dev - float2 (I.xf0.z, I.xf1.z)) / diagonal;
    }

    VOut o;
    o.pos = float4 ((dev - viewport.zw) / viewport.xy * float2 (2.0, -2.0) + float2 (-1.0, 1.0),
                    painterDepth (uint (I.xf0.w), depthStep), 1.0);
    o.local = local;
    o.uv = mix (I.uv.xy, I.uv.zw, (local - I.rect.xy) / max (I.rect.zw, float2 (1e-6)));
    o.iid = iid;
    o.inv = inv;
    o.axisRates = onePixel; // per instance: computed once here, not for every pixel
    return o;
}

// Opaque pass: the part of an opaque fill (a box with a solid opaque colour, no rotation or
// shear; see FrameRecorder::isOpaqueFill) whose pixels it covers fully, one device pixel inside its
// outline (the anti-aliased edge stays in the main pass).
// For a rounded box, the largest of three candidates: the band between the top and bottom corners,
// the band between the left and right ones, or the box inset by r (1 - sqrt(2) / 2), whose
// corners lie on the corner arcs. For a circle (only nearly circular
// ellipses are instances), the inscribed square; for an ellipse, half sizes a / sqrt(2), b / sqrt(2).
struct OpaqueOut { float4 pos [[position]]; float4 colour [[flat]]; };

// Area of a box of the given size inset by 'margin' on each side.
float insetArea (float2 size, float2 margin)
{
    const float2 inner = max (size - 2.0 * margin, 0.0);
    return inner.x * inner.y;
}

vertex OpaqueOut vsOpaque (uint vid [[vertex_id]], uint iid [[instance_id]],
                        const device Inst* inst [[buffer(0)]], constant float4& viewport [[buffer(1)]],
                        constant float& depthStep [[buffer(2)]], const device uint* opaque [[buffer(3)]])
{
    const uint index = opaque[iid];
    const Inst I = inst[index];
    const float2 diagonal = float2 (I.xf0.x, I.xf1.y);
    const float2 translation = float2 (I.xf0.z, I.xf1.z);
    const float2 halfSize = 0.5 * I.rect.zw;
    const float2 onePixel = 1.0 / abs (diagonal); // local units
    float2 margin;                                // local inset from each side of the box

    if (uint (I.params.z) == 1)
    {
        margin = halfSize * (1.0 - 0.70710678) + onePixel;
    }
    else
    {
        const float r = max (0.0, min (I.params.x, min (halfSize.x, halfSize.y)));
        const float2 horizontalBand = float2 (0.0, r), verticalBand = float2 (r, 0.0);
        const float2 inscribed = float2 (r * (1.0 - 0.70710678));
        margin = horizontalBand;

        if (insetArea (I.rect.zw, verticalBand + onePixel) > insetArea (I.rect.zw, margin + onePixel))
            margin = verticalBand;

        if (insetArea (I.rect.zw, inscribed + onePixel) > insetArea (I.rect.zw, margin + onePixel))
            margin = inscribed;

        margin += onePixel;
    }

    const float2 a = diagonal * (I.rect.xy + margin) + translation;
    const float2 b = diagonal * (I.rect.xy + I.rect.zw - margin) + translation;

    // Within the clip, like the draw itself; collapses to nothing when the box is too small.
    const float2 boxLo = max (min (a, b), I.clip.xy);
    const float2 boxHi = max (boxLo, min (max (a, b), I.clip.xy + I.clip.zw));
    const float2 dev = all (I.rect.zw > 2.0 * margin) ? mix (boxLo, boxHi, float2 (vid & 1, vid >> 1)) : boxLo;

    OpaqueOut o;
    o.pos = float4 ((dev - viewport.zw) / viewport.xy * float2 (2.0, -2.0) + float2 (-1.0, 1.0),
                    painterDepth (uint (I.xf0.w), depthStep), 1.0);
    o.colour = I.colour;
    return o;
}

fragment float4 fsOpaque (OpaqueOut in [[stage_in]]) { return in.colour; }
fragment float4 fsOpaqueAlpha (OpaqueOut in [[stage_in]]) { return float4 (in.colour.a); } // R8 targets

// Linear / radial gradients: t from the device pixel position (row0, row1: an affine map), the
// colour from the gradient's stops in the stop buffer (row0.w: first stop, row1.w: stop count),
// interpolated per pixel as CoreGraphics draws a CGGradient (see stopColour).
// meta: type (0 linear, 1 radial, 2 two-point conical, 3 tiled image), spread, unused, opacity
// conic: radial: start radius / end radius; two-point conical: end radius, f, swapped,
// closeToSingular (see below); tiled image: size
struct GradientInfo { float4 row0; float4 row1; float4 meta; float4 conic; };

// Two-point conical (SVG radial with a focal circle): juce::detail::TwoPointConicalGradient::
// calculateWeight, with the point already in its regularised space (focal point at the origin,
// end centre at (1, 0)). 'valid' is false outside the computable cone (transparent, as the
// software renderer). A valid weight may be negative (before the start circle): spread handles it.
float twoPointConicalWeight (float2 q, float4 conic, thread bool& valid)
{
    valid = false;
    const float r1 = conic.x, f = conic.y;
    const bool swapped = conic.z > 0.5, closeToSingular = conic.w > 0.5;
    const float k = r1 * r1 - 1.0;
    const float xPrime = closeToSingular ? q.x * 0.5 : r1 / k * q.x;
    const float yPrime = closeToSingular ? q.y * 0.5 : sqrt (abs (k)) / k * q.y;
    const float xHat = abs (1.0 - f) * xPrime, yHat = abs (1.0 - f) * yPrime;
    float xHatT;

    if (closeToSingular)
    {
        xHatT = (xHat * xHat + yHat * yHat) / xHat;
    }
    else if (r1 < 1.0)
    {
        const float d = xHat * xHat - yHat * yHat;

        if (d < 0.0)
            return 0.0;

        xHatT = ((swapped || (1.0 - f) < 0.0) ? -sqrt (d) : sqrt (d)) - xHat / r1;
    }
    else
    {
        xHatT = sqrt (xHat * xHat + yHat * yHat) - xHat / r1;
    }

    if (! (xHatT >= 0.0))
        return 0.0;

    valid = true;
    const float t = f + sign (1.0 - f) * xHatT;
    return swapped ? 1.0 - t : t;
}

// Colour stops of a gradient at t (pad): pairs (position, 0, 0, 0), colour (not premultiplied).
// Interpolated per pixel, then premultiplied, as CoreGraphics draws a CGGradient.
float4 stopColour (const device float4* stops, uint first, uint count, float t)
{
    float4 c = stops[first + 1];

    if (t > stops[first].x)
    {
        c = stops[first + 2 * (count - 1) + 1];

        for (uint i = 0; i + 1 < count; ++i)
        {
            const float p0 = stops[first + 2 * i].x, p1 = stops[first + 2 * i + 2].x;

            if (t < p1)
            {
                c = mix (stops[first + 2 * i + 1], stops[first + 2 * i + 3], p1 > p0 ? (t - p0) / (p1 - p0) : 1.0);
                break;
            }
        }
    }

    return float4 (c.rgb * c.a, c.a);
}

float4 gradientColour (const device GradientInfo* gradients, uint index, float2 p, const device float4* stops,
                       texture2d<float> image, sampler imageSampler)
{
    const GradientInfo G = gradients[index];
    const float2 q = float2 (dot (G.row0.xy, p) + G.row0.z, dot (G.row1.xy, p) + G.row1.z);

    if (G.meta.x > 2.5) // tiled image fill: q in image pixels, wrapped by a repeat sampler
        return image.sample (imageSampler, q / G.conic.xy) * G.meta.w;

    float t;

    if (G.meta.x < 0.5)
    {
        t = q.x;
    }
    else if (G.meta.x < 1.5)
    {
        t = (length (q) - G.conic.x) / (1.0 - G.conic.x);
    }
    else
    {
        bool valid;
        t = twoPointConicalWeight (q, G.conic, valid);

        if (! valid)
            return float4 (0.0);
    }

    const uint spread = uint (G.meta.y);

    // Same as juce::detail::SpreadMethods: negative positions clamp to 0 before reflect/repeat.
    if (spread == 1)
        t = 1.0 - abs (fract (max (t, 0.0) * 0.5) * 2.0 - 1.0);   // reflect
    else if (spread == 2)
        t = fract (max (t, 0.0));                                  // repeat

    return stopColour (stops, uint (G.row0.w), uint (G.row1.w), t) * G.meta.w;
}

// Coverage of a winding number from the coverage atlas (area-weighted), by fill rule.
float coverageOfWinding (float winding, bool evenOdd)
{
    const float w = abs (winding);
    return evenOdd ? 1.0 - abs (1.0 - fmod (w, 2.0)) : min (w, 1.0);
}

// Clip mask (clipToPath / clipToImageAlpha), zero outside 'bounds' (device px: left, top, right,
// bottom). params.xy: device px -> texel offset; params.z: 0 no mask, 1 an R8 mask (nested clips,
// image clips), 2 / 3 a path clip's winding numbers read straight from the coverage atlas
// (non-zero / even-odd), which saves resolving it into a mask texture first.
struct MaskInfo { float4 bounds; float4 params; };

float maskAt (texture2d<float> mask, MaskInfo info, float2 p)
{
    const uint mode = uint (info.params.z);

    if (mode == 0)
        return 1.0;

    if (any (p < info.bounds.xy) || any (p >= info.bounds.zw))
        return 0.0;

    const float v = mask.read (uint2 (p + info.params.xy)).r;
    return mode == 1 ? v : coverageOfWinding (v, mode == 3);
}

// One pipeline per family of instances (analytic shapes, text, images, coverage): 'family' is a
// function constant, so each variant only keeps its own branch below. A single shader holding them
// all would make every pixel pay the register footprint of the heaviest branch (measured: the
// fragment stage is ALU bound on full repaints).
// The draws are reordered so that each family's instances share a few batches (FrameRecorder::reorder).
// 0: boxes and ellipses (kinds 0, 1), 1: glyphs (2), 2: images (3, 4), 3: path coverage (5).
constant uint family [[function_constant(0)]];

// Single-channel images are drawn into R8 targets (MetalPixelData): their pipelines write the
// alpha into the red channel, which their blending then treats as the destination's alpha.
constant bool alphaTarget [[function_constant(1)]];

static inline float4 toTarget (float4 premultiplied) { return alphaTarget ? float4 (premultiplied.a) : premultiplied; }

// The main pass only tests depth (opaque pass), so testing it before shading changes nothing but
// guarantees that hidden pixels aren't shaded, also on GPUs that would otherwise test after a
// shader that can discard.
[[early_fragment_tests]]
fragment float4 fs (VOut in [[stage_in]], const device Inst* inst [[buffer(0)]],
                    const device GradientInfo* gradients [[buffer(1)]],
                    constant MaskInfo& maskInfo [[buffer(2)]], constant float2& targetOrigin [[buffer(3)]],
                    texture2d<float> glyphs [[texture(0)]], texture2d<float> image [[texture(1)]],
                    const device float4* gradientStops [[buffer(4)]], texture2d<float> mask [[texture(3)]],
                    texture2d<float> coverage [[texture(4)]], sampler imageSampler [[sampler(0)]])
{
    const Inst I = inst[in.iid];
    const float2 dp = in.pos.xy + targetOrigin; // device pixel

    if (any (dp < I.clip.xy) || any (dp >= I.clip.xy + I.clip.zw))
        discard_fragment();

    const float m = maskAt (mask, maskInfo, dp);

    if (m <= 0.0)
        discard_fragment();

    const uint kind = uint (I.params.z);

    if (family == 1)
    {
        constexpr sampler nearest (filter::nearest, coord::normalized);
        return toTarget (I.colour * (glyphs.sample (nearest, in.uv).r * m));
    }

    if (family == 2)
    {
        // Rotated or sheared images get anti-aliased edges, as the software renderer's edge table
        // gives them; aligned ones are drawn as they are.
        const float2 halfSize = 0.5 * I.rect.zw;
        const float edges = isAxisAligned (in.inv) ? 1.0 : boxCoverage (in.local - (I.rect.xy + halfSize), halfSize, in.inv, in.axisRates);
        const float4 texel = image.sample (imageSampler, in.uv);
        return toTarget (kind == 3 ? texel * (I.colour.a * m * edges) : I.colour * (texel.r * m * edges));
    }

    if (family == 3)
    {
        // Path fill: the pixel's winding number from the coverage atlas (uv.xy: atlas offset of the
        // path's area), turned into coverage by the fill rule (params.x: even-odd).
        const float c = coverageOfWinding (coverage.read (uint2 (dp + I.uv.xy)).r, I.params.x > 0.5);

        if (c < 1.0 / 512.0)
            discard_fragment();

        const float4 fill = I.params.w > 0.5 ? gradientColour (gradients, uint (I.params.w) - 1, dp, gradientStops, image, imageSampler)
                                             : I.colour;
        return toTarget (fill * (c * m));
    }

    const float2 halfSize = 0.5 * I.rect.zw;
    const float2 p = in.local - (I.rect.xy + halfSize);
    // Signed distance to the shape, in local units (rounded box: Inigo Quilez; ellipse: his
    // first-order approximation, exact enough for a 1 px ramp).
    float d;

    if (kind == 1)
    {
        const float2 ab = max (halfSize, float2 (1e-4));
        const float k0 = length (p / ab);
        const float k1 = length (p / (ab * ab));
        d = k0 * (k0 - 1.0) / max (k1, 1e-6);
    }
    else
    {
        const float r = min (I.params.x, min (halfSize.x, halfSize.y));
        const float2 q = abs (p) - halfSize + r;
        d = length (max (q, 0.0)) + min (max (q.x, q.y), 0.0) - r;
    }

    float shapeCoverage;

    if (kind == 0 && I.params.x <= 0.0)
    {
        // Square corners (a stroke's mitred joins included): the pixel's area inside the box is the
        // product of its coverage along each axis; a stroke is the outer box minus the inner one.
        const float halfStroke = 0.5 * I.params.y;
        shapeCoverage = boxCoverage (p, halfSize + halfStroke, in.inv, in.axisRates);

        if (I.params.y > 0.0)
            shapeCoverage -= boxCoverage (p, halfSize - halfStroke, in.inv, in.axisRates);
    }
    else
    {
        // The distance's local gradient direction, to turn it into device pixels.
        float2 g;

        if (kind == 1)
        {
            g = normalize (p / max (halfSize * halfSize, float2 (1e-8)) + float2 (1e-12, 0.0));
        }
        else
        {
            const float r = min (I.params.x, min (halfSize.x, halfSize.y));
            const float2 q = abs (p) - halfSize + r;
            g = (q.x > 0.0 && q.y > 0.0) ? normalize (q) : (q.x > q.y ? float2 (1.0, 0.0) : float2 (0.0, 1.0));
            g *= select (float2 (1.0), float2 (-1.0), p < 0.0);
        }

        // In device pixels: d changes by |J^-T g| local units per pixel (exact under any affine
        // transform, non-uniform scale and shear included), along the device normal n. Treating
        // the outline as straight across the pixel, the coverage is the exact area inside it; a
        // stroke (centred on the outline, like strokePath) is the slab around it.
        const float2 v = float2 (in.inv.x * g.x + in.inv.z * g.y, in.inv.y * g.x + in.inv.w * g.y);
        const float rate = max (length (v), 1.0e-6);
        const float2 n = v / rate;
        const float dDevice = d / rate;

        shapeCoverage = I.params.y > 0.0 ? slabCoverage (dDevice, 0.5 * I.params.y / rate, n)
                                         : halfPlaneCoverage (-dDevice, n);
    }

    const float4 colour = I.params.w > 0.5 ? gradientColour (gradients, uint (I.params.w) - 1, dp, gradientStops, image, imageSampler)
                                           : I.colour;
    return toTarget (colour * (shapeCoverage * m));
}

// ---- Mask passes: render into an R8 mask covering 'origin' + target size, multiplied by the
// parent mask (nested clips), so a draw only ever needs one mask. ----
struct MaskPassInfo { float4 origin; MaskInfo parent; float4 inv0; float4 inv1; float4 image; };
// origin.xy: mask origin (device px); inv: device px -> image px; image: size.xy, z: 1 = use .r

// Path clip: the path's coverage, from the coverage atlas (inv0.xy: atlas offset, inv0.z: even-odd;
// image.xy: the mask's area size, outside which nothing was computed).
fragment float4 maskCoverageFs (float4 pos [[position]], constant MaskPassInfo& info [[buffer(0)]],
                                texture2d<float> parentMask [[texture(3)]], texture2d<float> coverage [[texture(4)]])
{
    const float2 p = pos.xy + info.origin.xy;

    if (any (pos.xy >= info.image.xy))
        return float4 (0.0);

    const float c = coverageOfWinding (coverage.read (uint2 (p + info.inv0.xy)).r, info.inv0.z > 0.5);
    return float4 (c * maskAt (parentMask, info.parent, p), 0.0, 0.0, 0.0);
}

vertex float4 fullQuadVs (uint vid [[vertex_id]])
{
    return float4 (float2 (vid & 1, vid >> 1) * 2.0 - 1.0, 0.0, 1.0);
}

fragment float4 maskImageFs (float4 pos [[position]], constant MaskPassInfo& info [[buffer(0)]],
                             texture2d<float> image [[texture(1)]], texture2d<float> parentMask [[texture(3)]],
                             sampler imageSampler [[sampler(0)]])
{
    const float2 p = pos.xy + info.origin.xy;
    const float2 ip = float2 (dot (info.inv0.xy, p) + info.inv0.z, dot (info.inv1.xy, p) + info.inv1.z);
    const float2 uv = ip / info.image.xy;

    if (any (uv < 0.0) || any (uv > 1.0))
        return float4 (0.0);

    const float4 texel = image.sample (imageSampler, uv);
    const float a = info.image.z > 0.5 ? texel.r : texel.a;
    return float4 (a * maskAt (parentMask, info.parent, p), 0.0, 0.0, 0.0);
}

// ---- Coverage counting (path fills), as font-rs and Pathfinder: every line segment contributes,
// to each pixel of its rows, the signed area of that pixel lying to its right. That contribution is
// stored as deltas along x (the cells the segment crosses, plus the cell after them, from where it
// is constant) and summed with additive blending in an R32Float atlas page, one rectangle per path. (Half floats
// aren't enough: each delta's rounding error is carried to every pixel on its right, and a dense
// waveform has hundreds of edges per row.)
// A prefix sum along each row of a path's rectangle then gives every pixel its winding number
// weighted by exact area coverage, the area anti-aliasing of juce::EdgeTable. ----
struct Seg { float4 line; float4 rect; }; // atlas px. line: x0, y0, x1, y1; rect: left, top, right, bottom
struct AccOut { float4 pos [[position]]; uint sid [[flat]]; };

vertex AccOut accumVs (uint vid [[vertex_id]], uint iid [[instance_id]],
                       const device Seg* segs [[buffer(0)]], constant float2& atlasSize [[buffer(1)]])
{
    // The cells a segment changes, row by row: from the cell under its leftmost point in the row to
    // the one after its rightmost. Segments are clipped to the path's rectangle on the CPU, so these
    // cells lie within it.
    // The quad is the segment's bounding box or, when smaller, a band along the segment: in a row, the
    // segment stays within 0.5 |dx/dy| of its x at the row's centre, and the cells' centres within
    // half a cell more (plus the cell after on the right). A long slanted edge's box is mostly cells
    // it doesn't touch: its cost would grow with the square of its length.
    const Seg s = segs[iid];
    const float y0 = max (floor (min (s.line.y, s.line.w)), s.rect.y);
    const float y1 = max (y0, min (ceil (max (s.line.y, s.line.w)), s.rect.w));
    const float x0 = clamp (floor (min (s.line.x, s.line.z)), s.rect.x, s.rect.z - 1.0);
    const float x1 = clamp (floor (max (s.line.x, s.line.z)) + 2.0, x0 + 1.0, s.rect.z);
    const float dxdy = (s.line.z - s.line.x) / (s.line.w - s.line.y);
    const float2 corner = float2 (vid & 1, vid >> 1);
    float2 p = mix (float2 (x0, y0), float2 (x1, y1), corner);

    if (abs (dxdy) + 3.0 < x1 - x0)
    {
        const float halfRun = 0.5 * abs (dxdy);
        const float x = s.line.x + (p.y - s.line.y) * dxdy;
        p.x = corner.x == 0.0 ? x - halfRun - 1.0 : x + halfRun + 2.0;
    }

    AccOut o;
    o.pos = float4 (p / atlasSize * float2 (2.0, -2.0) + float2 (-1.0, 1.0), 0.0, 1.0);
    o.sid = iid;
    return o;
}

// Fraction of the width of the pixel whose right side is at x = right that lies right of the edge,
// averaged over the edge's span in the row: u = right - x(y) is linear in y, so the average of
// clamp (u, 0, 1) is exact (integral over [lo, hi] / (hi - lo), split so that large |u| cancel).
float coveredFraction (float right, float xa, float xb)
{
    const float ua = right - xa, ub = right - xb;
    const float lo = min (ua, ub), hi = max (ua, ub);
    const float a = saturate (lo), b = saturate (hi);
    return hi - lo < 1.0e-4 ? saturate (0.5 * (lo + hi))
                            : (0.5 * (b * b - a * a) + (max (hi, 1.0) - max (lo, 1.0))) / (hi - lo);
}

// A segment's delta in a cell of its path's rectangle, whose left side is at x = left.
float edgeDelta (float4 line, float2 cell, float left)
{
    // The part of the segment within this pixel row (signed: the edge direction gives the winding).
    const float ya = clamp (line.y, cell.y, cell.y + 1.0);
    const float yb = clamp (line.w, cell.y, cell.y + 1.0);
    const float dy = yb - ya;

    if (dy == 0.0)
        return 0.0;

    const float dxdy = (line.z - line.x) / (line.w - line.y);
    const float xa = line.x + (ya - line.y) * dxdy;
    const float xb = line.x + (yb - line.y) * dxdy;

    // Delta from the previous cell; the rectangle's first column holds the full value.
    const float here = coveredFraction (cell.x + 1.0, xa, xb);
    const float before = cell.x > left ? coveredFraction (cell.x, xa, xb) : 0.0;
    return dy * (here - before);
}

fragment float4 accumFs (AccOut in [[stage_in]], const device Seg* segs [[buffer(0)]])
{
    const Seg s = segs[in.sid];
    const float2 cell = floor (in.pos.xy);

    // A band can reach past the rectangle's sides, into a neighbour's.
    if (cell.x < s.rect.x || cell.x >= s.rect.z)
        discard_fragment();

    const float d = edgeDelta (s.line, cell, s.rect.x);

    if (d == 0.0)
        discard_fragment();

    return float4 (d, 0.0, 0.0, 0.0);
}

// ---- Strokes: their outline is built on the GPU from the centre line, each segment on its own
// (Levien & Uguray, "GPU-friendly Stroke Expansion", 2024). The outline of a stroke is a
// closed loop of edges (the left offsets forwards, the end cap, the right offsets backwards, the
// start cap) and coverage counting only needs its edges, in any order. So each point of the centre
// line owns the edges around it: the two offset edges of the segment starting there and its join
// with the next (the outer side beveled, mitered or rounded; the inner side a bevel, whose overlap
// the non-zero rule absorbs, as in that paper); an open sub-path's first point also owns the
// start cap, its last point the end cap. They are built in the stroke's space (where its width
// is), then transformed to the atlas. A quad covers up to eight of a point's edges (usually all of
// them) and sums their deltas in each cell: a quad per edge would be about twice as many as the
// CPU-built outline has, and the vertex stage of this pass is already as costly as its fragments.
// The CPU lists the quads (StrokeQuad): a point has as many as its edges need (round joins: as
// many arc steps as their turn takes).
// Tested and not kept, measured on an M1 (Metal System Trace) against the CPU-built outline:
// - A compute pass writing the edges into a buffer that accumVs then draws: about 0.4 ms more per
//   frame whatever the stroke (the accumulation waits for it).
// - One instance per edge, in a fixed number of slots per point: about five quads per point, twice
//   the CPU outline's, and the vertex stage costs per quad, empty or not (4 ms per frame for 50k
//   points against 1.9). Larger varyings per vertex overflowed the tiler's parameter buffer.
// - A quad's fragments summing all eight edge slots rather than those in use: +50% at 1 px.
// - Each edge recomputing its point's joints, rather than once per vertex: +28% vertex time.
// - Before this, the CPU outline written straight into segments (no Path, no second flattening):
//   only 15-20% less encoding, the joints' maths dominating. ----
struct StrokeJob
{
    float4 xf0, xf1;    // stroke space -> atlas px: x' = dot (xf0.xy, p) + xf0.z, y' = dot (xf1.xy, p) + xf1.z
    float4 rect;        // the path's atlas rectangle
    float4 style;       // half width, miter limit squared, arc step (radians), unused
    uint4 layout;       // joint (PathStrokeType::JointStyle), cap (EndCapStyle), edges per cap, arc steps per half turn
};
struct StrokeSub { uint first, count, closed, job; };
struct StrokePoint { float2 p; };
struct StrokeQuad { uint point, subAndEdge; }; // the point (index in the points), its sub-path << 8 | its first edge

constant uint strokeEdgesPerQuad = 8;

float2 rotated (float2 v, float angle)
{
    const float c = cos (angle), s = sin (angle);
    return float2 (v.x * c - v.y * s, v.x * s + v.y * c);
}

// Edge m of the arc from c + v0 to c + v1, turning by 'angle' in steps of at most 'step'.
bool arcEdge (float2 c, float2 v0, float2 v1, float angle, float step, uint maxSteps, uint m, thread float2& a, thread float2& b)
{
    const uint n = clamp (uint (ceil (abs (angle) / step)), 1u, maxSteps);

    if (m >= n)
        return false;

    a = c + (m == 0 ? v0 : rotated (v0, angle * float (m) / float (n)));
    b = c + (m + 1 == n ? v1 : rotated (v0, angle * float (m + 1) / float (n)));
    return true;
}

// A joint of the centre line: at p, from the segment along t0 (length len0; offset n0: t0 turned a
// quarter turn, times the half width) to the one along t1. The outer side is the right (-n) when
// turning towards +n (cr > 0). On the inner side, the two offset lines cross at
// p +- (n0 + n1) / (1 + d), which lies on both offset edges when half the width times
// tan (half the turn) is at most both lengths: they then end there, as in PathStrokeType's outline
// (its lineIntersection). Otherwise an inner bevel joins them, and the overlap it leaves is
// absorbed by the non-zero rule, as in that paper and in PathStrokeType for short segments.
// Built by one function wherever it is needed, so that the two segments meeting there agree.
struct Joint
{
    float2 p, t0, t1, n0, n1;
    float cr, d;
    bool innerMeets;
    float2 inner; // where the inner offset edges end when they meet
};

Joint strokeJoint (const device StrokePoint* points, StrokeSub sub, uint j, float hw)
{
    const uint n = sub.count;
    const float2 before = points[sub.first + (j + n - 1) % n].p, p = points[sub.first + j % n].p;
    const float2 after = points[sub.first + (j + 1) % n].p;
    const float len0 = length (p - before), len1 = length (after - p);

    Joint J;
    J.p = p;
    J.t0 = (p - before) / len0;
    J.t1 = (after - p) / len1;
    J.n0 = hw * float2 (-J.t0.y, J.t0.x);
    J.n1 = hw * float2 (-J.t1.y, J.t1.x);
    J.cr = J.t0.x * J.t1.y - J.t0.y * J.t1.x;
    J.d = dot (J.t0, J.t1);
    J.innerMeets = J.cr != 0.0 && 1.0 + J.d > 0.0 && hw * abs (J.cr) <= (1.0 + J.d) * min (len0, len1);
    const float2 m = (J.n0 + J.n1) / (1.0 + J.d);
    J.inner = J.cr > 0.0 ? p + m : p - m;
    return J;
}

// Edge e of a joint's join: the loop goes p + n0 -> p + n1 on the left, p - n1 -> p - n0 on the
// right. The outer side gets the miter point or the arc; the inner side a bevel unless its offset
// edges meet.
bool joinEdge (const device StrokeJob& job, Joint J, uint e, thread float2& a, thread float2& b)
{
    const float2 front0 = J.p + J.n0, front1 = J.p + J.n1, back0 = J.p - J.n1, back1 = J.p - J.n0;
    const bool backIsOuter = J.cr > 0.0;
    const float2 inner0 = backIsOuter ? front0 : back0, inner1 = backIsOuter ? front1 : back1;

    // Mitered, within the limit (PathStrokeType: the miter point at most miterLimit half widths
    // from p, i.e. 1 / cos (half the turn) <= miterLimit): the outer side through the miter point.
    if (job.layout.x == 0 && J.cr != 0.0 && 2.0 <= (1.0 + J.d) * job.style.y)
    {
        const float2 m = (J.n0 + J.n1) / (1.0 + J.d);
        const float2 outer0 = backIsOuter ? back0 : front0, outer1 = backIsOuter ? back1 : front1;
        const float2 tip = backIsOuter ? J.p - m : J.p + m;

        switch (e)
        {
            case 0:  a = outer0; b = tip; return true;
            case 1:  a = tip; b = outer1; return true;
            case 2:  a = inner0; b = inner1; return ! J.innerMeets;
            default: return false;
        }
    }

    // Curved: the inner bevel, then an arc on the outer side; a full reversal turns round the end,
    // on the left.
    if (job.layout.x == 1 && (J.cr != 0.0 || J.d < 0.0))
    {
        const float angle = J.cr == 0.0 ? -M_PI_F : atan2 (J.cr, J.d);

        if (e == 0)
        {
            a = angle > 0.0 ? front0 : back0;
            b = angle > 0.0 ? front1 : back1;
            return ! J.innerMeets;
        }

        return angle > 0.0 ? arcEdge (J.p, -J.n1, -J.n0, -angle, job.style.z, job.layout.w, e - 1, a, b)
                           : arcEdge (J.p, J.n0, J.n1, angle, job.style.z, job.layout.w, e - 1, a, b);
    }

    // Beveled (or straight on): both sides, the inner one unless its edges meet.
    switch (e)
    {
        case 0:  a = front0; b = front1; return ! (J.innerMeets && backIsOuter);
        case 1:  a = back0; b = back1; return ! (J.innerMeets && ! backIsOuter);
        default: return false;
    }
}

// Edge e of a cap at p, from p + v (v: the offset, a quarter turn from the outward direction
// 'out') round to p - v.
bool capEdge (const device StrokeJob& job, float2 p, float2 out, float2 v, uint e, thread float2& a, thread float2& b)
{
    if (job.layout.y == 2) // rounded: half a turn through p + out
        return arcEdge (p, v, -v, -M_PI_F, job.style.z, job.layout.w, e, a, b);

    if (job.layout.y == 1) // square
    {
        const float2 x = out * job.style.x;

        switch (e)
        {
            case 0:  a = p + v; b = p + v + x; return true;
            case 1:  a = p + v + x; b = p - v + x; return true;
            case 2:  a = p - v + x; b = p - v; return true;
            default: return false;
        }
    }

    a = p + v;
    b = p - v;
    return e == 0;
}

// Point k's part of a sub-path's outline, in stroke space: (for an open sub-path's first point,
// the start cap,) the two offset edges of the segment starting at it and its join with the next;
// an open sub-path's last point has the end cap. Built once per vertex, then read edge by edge
// (pointEdge).
struct PointOutline
{
    bool capsOnly;                          // an open sub-path's last point
    float2 p0, t0, n0;
    float2 leftStart, leftEnd, rightStart, rightEnd;
    bool hasJoin;
    Joint join;
};

PointOutline pointOutline (const device StrokePoint* points, StrokeSub sub, const device StrokeJob& job, uint k)
{
    const uint n = sub.count;
    const bool closed = sub.closed != 0;
    const uint segments = closed ? n : n - 1;
    const float hw = job.style.x;

    PointOutline o;
    o.capsOnly = k >= segments;
    o.p0 = points[sub.first + k].p;

    if (o.capsOnly)
        return o;

    const float2 p1 = points[sub.first + (k + 1) % n].p;
    o.t0 = normalize (p1 - o.p0);
    o.n0 = hw * float2 (-o.t0.y, o.t0.x);

    // The offset edges, ending where they meet the neighbours' on an inner side.
    o.leftStart = o.p0 + o.n0;
    o.rightStart = o.p0 - o.n0;
    o.leftEnd = p1 + o.n0;
    o.rightEnd = p1 - o.n0;

    if (k > 0 || closed)
    {
        const Joint J = strokeJoint (points, sub, k, hw);

        if (J.innerMeets)
            (J.cr > 0.0 ? o.leftStart : o.rightStart) = J.inner;
    }

    o.hasJoin = k + 1 < segments || closed;

    if (o.hasJoin)
    {
        o.join = strokeJoint (points, sub, k + 1, hw);

        if (o.join.innerMeets)
            (o.join.cr > 0.0 ? o.leftEnd : o.rightEnd) = o.join.inner;
    }

    return o;
}

bool pointEdge (thread const PointOutline& o, const device StrokePoint* points, StrokeSub sub, const device StrokeJob& job,
                uint k, uint e, thread float2& a, thread float2& b)
{
    const float hw = job.style.x;
    const uint capEdges = job.layout.z;

    if (o.capsOnly)
    {
        const float2 t = normalize (o.p0 - points[sub.first + k - 1].p);
        return capEdge (job, o.p0, t, hw * float2 (-t.y, t.x), e, a, b);
    }

    if (k == 0 && sub.closed == 0)
    {
        if (e < capEdges)
            return capEdge (job, o.p0, -o.t0, -o.n0, e, a, b);

        e -= capEdges;
    }

    switch (e)
    {
        case 0:  a = o.leftStart; b = o.leftEnd; return true;
        case 1:  a = o.rightEnd; b = o.rightStart; return true;
        default: return o.hasJoin && joinEdge (job, o.join, e - 2, a, b);
    }
}

struct StrokeOut
{
    float4 pos [[position]];
    float4 rect [[flat]];
    uint count [[flat]]; // the edges in use, packed from e0
    float4 e0 [[flat]], e1 [[flat]], e2 [[flat]], e3 [[flat]], e4 [[flat]], e5 [[flat]], e6 [[flat]], e7 [[flat]];
};

vertex StrokeOut strokeAccumVs (uint vid [[vertex_id]], uint iid [[instance_id]],
                                const device StrokePoint* points [[buffer(0)]], constant float2& atlasSize [[buffer(1)]],
                                constant uint& firstQuad [[buffer(2)]], const device StrokeSub* subs [[buffer(3)]],
                                const device StrokeJob* jobs [[buffer(4)]], const device StrokeQuad* quads [[buffer(5)]])
{
    const StrokeQuad quad = quads[firstQuad + iid];
    const StrokeSub sub = subs[quad.subAndEdge >> 8];
    const device StrokeJob& job = jobs[sub.job];
    const uint k = quad.point - sub.first, firstOfPoint = quad.subAndEdge & 255;

    float4 edges[strokeEdgesPerQuad];
    float2 lowest = float2 (FLT_MAX), highest = float2 (-FLT_MAX); // (fast math: no infinities)
    uint count = 0;
    const float2 s0 = points[sub.first + k].p, s1 = points[sub.first + (k + 1) % sub.count].p;
    const float2 c0 = float2 (dot (job.xf0.xy, s0) + job.xf0.z, dot (job.xf1.xy, s0) + job.xf1.z);
    const float2 c1 = float2 (dot (job.xf0.xy, s1) + job.xf0.z, dot (job.xf1.xy, s1) + job.xf1.z);
    const float dxdy = c0.y != c1.y ? (c1.x - c0.x) / (c1.y - c0.y) : 0.0;
    float margin = 0.0; // the furthest an edge gets from the segment's line, along x

    const PointOutline outline = pointOutline (points, sub, job, k);

    for (uint i = 0; i < strokeEdgesPerQuad; ++i)
        edges[i] = float4 (0.0);

    for (uint i = 0; i < strokeEdgesPerQuad; ++i)
    {
        float2 a, b;

        if (pointEdge (outline, points, sub, job, k, firstOfPoint + i, a, b))
        {
            a = float2 (dot (job.xf0.xy, a) + job.xf0.z, dot (job.xf1.xy, a) + job.xf1.z);
            b = float2 (dot (job.xf0.xy, b) + job.xf0.z, dot (job.xf1.xy, b) + job.xf1.z);

            // Horizontal edges add no area.
            if (a.y != b.y)
            {
                edges[count++] = float4 (a, b);
                lowest = min (lowest, min (a, b));
                highest = max (highest, max (a, b));
                margin = max (margin, max (abs (a.x - c0.x - (a.y - c0.y) * dxdy), abs (b.x - c0.x - (b.y - c0.y) * dxdy)));
            }
        }
    }

    StrokeOut o;
    o.rect = job.rect;
    o.count = count;
    o.e0 = edges[0]; o.e1 = edges[1]; o.e2 = edges[2]; o.e3 = edges[3];
    o.e4 = edges[4]; o.e5 = edges[5]; o.e6 = edges[6]; o.e7 = edges[7];

    if (count == 0)
    {
        o.pos = float4 (-2.0, -2.0, 0.0, 1.0);
        return o;
    }

    // The cells the edges change, as accumVs: their rows, from the cell under their leftmost point
    // to the one after their rightmost, within the rectangle. An edge reaching left of the
    // rectangle (the area cuts the stroke there) adds what lies left of it in its first column,
    // which the box then starts from.
    const float4 r = job.rect;
    const bool crossesLeft = lowest.x < r.x;
    const float y0 = max (floor (lowest.y), r.y);
    const float y1 = max (y0, min (ceil (highest.y), r.w));
    const float x0 = clamp (floor (lowest.x), r.x, r.z - 1.0);
    const float x1 = clamp (floor (highest.x) + 2.0, x0 + 1.0, r.z);
    const float2 corner = float2 (vid & 1, vid >> 1);
    float2 p = mix (float2 (x0, y0), float2 (x1, y1), corner);

    // Or a band along the point's segment, when narrower: an edge's distance along x from the
    // segment's line is linear along it, so at most 'margin' (from its ends), and the line moves by
    // up to 0.5 |dx/dy| across a row (see accumVs). An open sub-path's last point (its end cap)
    // keeps the box.
    const uint segments = sub.closed != 0 ? sub.count : sub.count - 1;

    if (! crossesLeft && k < segments && c0.y != c1.y)
    {
        const float halfWidth = 0.5 * abs (dxdy) + margin;

        if (2.0 * halfWidth + 3.0 < x1 - x0)
        {
            const float x = c0.x + (p.y - c0.y) * dxdy;
            p.x = corner.x == 0.0 ? x - halfWidth - 1.0 : x + halfWidth + 2.0;
        }
    }

    o.pos = float4 (p / atlasSize * float2 (2.0, -2.0) + float2 (-1.0, 1.0), 0.0, 1.0);
    return o;
}

fragment float4 strokeAccumFs (StrokeOut in [[stage_in]])
{
    const float2 cell = floor (in.pos.xy);

    if (cell.x < in.rect.x || cell.x >= in.rect.z)
        discard_fragment();

    // Only the edges in use (the same count over the quad, so these branches don't diverge).
    const float left = in.rect.x;
    float d = edgeDelta (in.e0, cell, left);

    if (in.count > 1) d += edgeDelta (in.e1, cell, left);
    if (in.count > 2) d += edgeDelta (in.e2, cell, left);
    if (in.count > 3) d += edgeDelta (in.e3, cell, left);
    if (in.count > 4) d += edgeDelta (in.e4, cell, left);
    if (in.count > 5) d += edgeDelta (in.e5, cell, left);
    if (in.count > 6) d += edgeDelta (in.e6, cell, left);
    if (in.count > 7) d += edgeDelta (in.e7, cell, left);

    if (d == 0.0)
        discard_fragment();

    return float4 (d, 0.0, 0.0, 0.0);
}

// One SIMD group per row of a path's rectangle: running sum of the deltas along the row, a SIMD
// width of cells per step (simd_prefix_inclusive_sum), the carry passed on from the last lane.
struct CovEntry { uint4 box; uint4 rows; }; // box: x, y, width, height (atlas px); rows.x: first row index

kernel void coveragePrefixSum (texture2d<float, access::read> deltas [[texture(0)]],
                               texture2d<float, access::write> coverage [[texture(1)]],
                               const device CovEntry* entries [[buffer(0)]],
                               constant uint2& counts [[buffer(1)]], // entries, rows
                               uint group [[threadgroup_position_in_grid]],
                               uint simdIndex [[simdgroup_index_in_threadgroup]],
                               uint simdsPerGroup [[simdgroups_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdWidth [[threads_per_simdgroup]])
{
    const uint row = group * simdsPerGroup + simdIndex;

    if (row >= counts.y)
        return;

    uint lo = 0, hi = counts.x - 1;

    while (lo < hi)
    {
        const uint mid = (lo + hi + 1) / 2;

        if (entries[mid].rows.x <= row)
            lo = mid;
        else
            hi = mid - 1;
    }

    const CovEntry e = entries[lo];
    const uint y = e.box.y + (row - e.rows.x);
    const uint end = e.box.x + e.box.z;
    float carry = 0.0;

    for (uint x0 = e.box.x; x0 < end; x0 += simdWidth)
    {
        const uint x = x0 + lane;
        const float d = x < end ? deltas.read (uint2 (x, y)).r : 0.0;
        const float sum = simd_prefix_inclusive_sum (d) + carry;

        if (x < end)
            coverage.write (float4 (sum, 0.0, 0.0, 0.0), uint2 (x, y));

        carry = simd_broadcast (sum, ushort (simdWidth - 1));
    }
}

fragment float4 fsClear (VOut in [[stage_in]], const device Inst* inst [[buffer(0)]])
{
    const Inst I = inst[in.iid];

    if (any (in.pos.xy < I.clip.xy) || any (in.pos.xy >= I.clip.xy + I.clip.zw))
        discard_fragment();

    return float4 (0.0);
}

// ---- Single-channel box blur (SceneEncoder::encodeBlur): the software renderer's algorithm
// (blurSingleChannelImage in juce_Image.cpp), exactly: 'passes' 3-tap averages along a line,
// (a + b + c + 1) / 3 in integers, the pixels outside the line counting as 0. One threadgroup
// per line (a row, then a column), the line kept in threadgroup memory across its passes. ----
struct BlurParams { uint2 srcOrigin; uint2 dstOrigin; uint length; uint vertical; uint passes; uint lines; };

static uint2 linePixel (uint2 origin, uint vertical, uint line, uint i)
{
    return origin + (vertical != 0 ? uint2 (line, i) : uint2 (i, line));
}

// The line's threadgroup memory is sized to it (a fixed 16 KB would leave room for few
// threadgroups at once): up to 256 threads (SharedResources::blurThreads), 64 pixels each.
kernel void boxBlurLines (texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant BlurParams& p [[buffer(0)]],
                          threadgroup uchar* values [[threadgroup(0)]],
                          uint line [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]])
{
    const uint n = p.length;

    for (uint i = tid; i < n; i += threads)
        values[i] = uchar (src.read (linePixel (p.srcOrigin, p.vertical, line, i)).r * 255.0 + 0.5);

    threadgroup_barrier (mem_flags::mem_threadgroup);

    for (uint pass = 0; pass < p.passes; ++pass)
    {
        uchar next[64];

        for (uint i = tid, k = 0; i < n; i += threads, ++k)
        {
            const uint left = i > 0 ? uint (values[i - 1]) : 0u;
            const uint right = i + 1 < n ? uint (values[i + 1]) : 0u;
            next[k] = uchar ((left + uint (values[i]) + right + 1) / 3);
        }

        threadgroup_barrier (mem_flags::mem_threadgroup);

        for (uint i = tid, k = 0; i < n; i += threads, ++k)
            values[i] = next[k];

        threadgroup_barrier (mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < n; i += threads)
        dst.write (float4 (float (values[i]) / 255.0), linePixel (p.dstOrigin, p.vertical, line, i));
}

// The same for lines of up to 16 pixels per SIMD lane (512 on Apple GPUs): one SIMD group per
// line, each lane holding 16 consecutive pixels in registers and exchanging its end pixels with
// its neighbours by shuffles, so no pass waits on a threadgroup barrier.
kernel void boxBlurLinesSimd (texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              constant BlurParams& p [[buffer(0)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint simdIndex [[simdgroup_index_in_threadgroup]],
                              uint simdsPerGroup [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdWidth [[threads_per_simdgroup]])
{
    const uint line = group * simdsPerGroup + simdIndex;

    if (line >= p.lines) // the whole SIMD group: one line each
        return;

    const uint n = p.length, first = lane * 16;
    uint v[16];

    for (uint j = 0; j < 16; ++j)
        v[j] = first + j < n ? uint (src.read (linePixel (p.srcOrigin, p.vertical, line, first + j)).r * 255.0 + 0.5) : 0u;

    for (uint pass = 0; pass < p.passes; ++pass)
    {
        const uint fromLeft = simd_shuffle_up (v[15], 1u);
        const uint fromRight = simd_shuffle_down (v[0], 1u);
        uint previous = lane > 0 ? fromLeft : 0u;

        for (uint j = 0; j < 16; ++j)
        {
            const uint next = j < 15 ? v[j + 1] : (lane + 1 < simdWidth ? fromRight : 0u);
            const uint current = v[j];
            v[j] = first + j < n ? (previous + current + next + 1) / 3 : 0u; // past the end stays 0
            previous = current;
        }
    }

    for (uint j = 0; j < 16; ++j)
        if (first + j < n)
            dst.write (float4 (float (v[j]) / 255.0), linePixel (p.dstOrigin, p.vertical, line, first + j));
}
)MSL";

//==============================================================================
/*  Textures whose release must wait until the GPU no longer uses them. */
struct DeferredReleases
{
    void add (id object)           { if (object != nil) objects.push_back (object); }
    void releaseAll()              { for (id o : objects) [o release]; objects.clear(); }
    std::vector<id> objects;
};

//==============================================================================
/*  Glyph atlas with eviction: the texture is divided into plots of plotSize x plotSize pixels,
    each filled shelf by shelf and stamped with the last frame that drew one of its glyphs. When no open plot has room, an unused plot is opened, or else the least
    recently used one whose frames have all completed on the GPU is emptied and reused: a UI that
    keeps changing text sizes (zooming, resizing) never fills the atlas for good.
    Frames are numbered across all windows (SharedResources::framesRecorded / framesCompleted).
    A run of glyphs looks up its strike (a typeface at one size and scale) once in a hash table,
    then each glyph by its id and subpixel position in that strike's own table.
    A new glyph gets its place in the atlas when it is recorded, sized from the typeface's glyph
    bounds (its metrics, known before its image is drawn), and is rasterised on the encode thread
    (rasterise (RasterJob)), before the frame that first draws it is encoded:
    as Chrome and Flutter rasterise glyphs on their raster threads, and Core Animation does for
    the CoreGraphics renderer, paint() only records.
*/
class GlyphAtlas
{
public:
    static constexpr int size = 4096, plotSize = 256, plotsPerSide = size / plotSize;
    static constexpr int subpixelSteps = 16; // CoreGraphics' horizontal glyph positioning step

    struct Entry
    {
        Rectangle<int> area;    // in the atlas; empty for blank glyphs (which aren't kept)
        Point<int> offset;      // top-left of 'area' relative to the snapped pen position
        int plot = -1;
    };

    /*  A glyph to draw into its atlas area on the encode thread. */
    struct RasterJob
    {
        Font font;
        Typeface::Ptr typeface;
        uint16_t glyph = 0;
        float scale = 1.0f;         // device pixels per font unit
        Point<float> pen;           // in the glyph's box
        Rectangle<int> area;        // in the atlas
    };

    struct Strike
    {
        Typeface::Ptr typeface;                         // kept alive: its address is in the key
        float scale = 1.0f;                             // device pixels per font unit
        std::unordered_map<uint32_t, Entry> glyphs;     // glyph id << 8 | subpixel position
    };

    /*  'frame': the frame about to be recorded; 'completed': the last one done on the GPU. */
    void beginFrame (uint64_t frame, uint64_t completed)
    {
        currentFrame = frame;
        completedFrame = completed;

        // Strikes left empty by evictions (sizes no longer drawn) release their typefaces. Only
        // between frames: a run being recorded holds its strike.
        for (auto it = strikes.begin(); it != strikes.end();)
            it = it->second->glyphs.empty() ? strikes.erase (it) : std::next (it);
    }

    /*  The strike for glyphs of 'font' drawn at 'scale' device pixels per font unit. The font's
        own size and the scale are both part of it: CoreText draws a 10 pt glyph at 2x differently
        from a 20 pt one (optical sizes).
    */
    Strike& getStrike (const Font& font, const Typeface::Ptr& typeface, float scale)
    {
        const StrikeKey key { typeface.get(), roundToInt (font.getHeightInPoints() * 16.0f), roundToInt (scale * 1024.0f),
                              roundToInt (font.getHorizontalScale() * 1024.0f) };
        auto& strike = strikes[key];

        if (strike == nullptr)
        {
            strike = std::make_unique<Strike>();
            strike->typeface = typeface;
            strike->scale = scale;
        }

        return *strike;
    }

    explicit GlyphAtlas (id<MTLDevice> device)
    {
        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: MTLPixelFormatR8Unorm
                                                                        width: size
                                                                       height: size
                                                                    mipmapped: NO];
        desc.usage = MTLTextureUsageShaderRead;
        texture = [device newTextureWithDescriptor: desc];
    }

    ~GlyphAtlas() { [texture release]; }

    id<MTLTexture> getTexture() const noexcept { return texture; }

    /*  Returns nullptr when no plot can take the glyph this frame (every one is in use by frames
        in flight, or the glyph is larger than a plot): the caller draws it as an outline.
        rasterPoints: font.withHeight (heightPx).getHeightInPoints(), the glyph's size in device
        pixels. A new glyph is added to 'jobs', to be rasterised before the frame is encoded.
    */
    const Entry* get (Strike& strike, const Font& font, uint16_t glyph, float rasterPoints, int subpixel,
                      std::vector<RasterJob>& jobs)
    {
        const auto key = ((uint32_t) glyph << 8) | (uint32_t) subpixel;

        if (auto it = strike.glyphs.find (key); it != strike.glyphs.end())
        {
            plots[(size_t) it->second.plot].lastUsed = currentFrame;
            return &it->second;
        }

        // The glyph's bounds at this size with a margin, at a whole-pixel pen position: the
        // margin holds what font smoothing adds around the outline. A glyph without bounds has no
        // outline (colour glyphs don't come here), so nothing to draw; it isn't kept, so that only
        // glyphs in the atlas hold a strike (see beginFrame).
        auto bounds = strike.typeface->getGlyphBounds (glyph);
        bounds = { bounds.getX() * rasterPoints * font.getHorizontalScale(), bounds.getY() * rasterPoints,
                   bounds.getWidth() * rasterPoints * font.getHorizontalScale(), bounds.getHeight() * rasterPoints };

        static const Entry blank;

        if (bounds.isEmpty() || ! bounds.isFinite())
            return &blank;

        Entry e;

        const auto margin = 4;
        const Point<int> origin (margin - (int) std::floor (bounds.getX()), margin - (int) std::floor (bounds.getY()));
        const auto boxWidth = origin.x + (int) std::ceil (bounds.getRight()) + margin + 1;
        const auto boxHeight = origin.y + (int) std::ceil (bounds.getBottom()) + margin;
        const auto place = allocate (boxWidth, boxHeight);

        if (! place.has_value())
            return nullptr;

        e.area = place->first;
        e.plot = place->second;
        e.offset = -origin;
        plots[(size_t) e.plot].glyphs.push_back ({ &strike, key });

        jobs.push_back ({ font, strike.typeface, glyph, strike.scale,
                          { (float) origin.x + (float) subpixel / (float) subpixelSteps, (float) origin.y }, e.area });

        return &strike.glyphs.emplace (key, e).first->second;
    }

    /*  Encode thread. Drawn by JUCE's CoreGraphics context, as in a window using the CoreGraphics
        renderer (CTFontDrawGlyphs with the same font, text matrix, scale and font smoothing).
        Font smoothing only applies to an opaque RGB context: black on white, the coverage is
        then 255 - luminance. One image is reused for every glyph, grown to the largest box yet.
    */
    void rasterise (const RasterJob& job)
    {
        const auto w = job.area.getWidth(), h = job.area.getHeight();

        if (scratch.getWidth() < w || scratch.getHeight() < h)
            scratch = Image (Image::RGB, jmax (w, scratch.getWidth()), jmax (h, scratch.getHeight()), false, NativeImageType());

        {
            Graphics g (scratch);
            g.setColour (Colours::white);
            g.fillRect (0, 0, w, h);
            auto& context = g.getInternalContext();
            context.setFill (Colours::black);
            context.setFont (job.font);
            const auto penInFontUnits = job.pen / job.scale;
            context.drawGlyphs ({ &job.glyph, 1 }, { &penInFontUnits, 1 }, AffineTransform::scale (job.scale));
        }

        // The native image's pixels may be wider than PixelRGB (CoreGraphics has no 24-bit
        // contexts): read through its own pixel stride and format.
        const Image::BitmapData bd (scratch, 0, 0, w, h, Image::BitmapData::readOnly);
        pixels.resize ((size_t) (w * h));

        const auto copyCoverage = [&] (auto pixelType)
        {
            using Pixel = decltype (pixelType);

            for (int y = 0; y < h; ++y)
            {
                const auto* src = bd.getLinePointer (y);

                for (int x = 0; x < w; ++x, src += bd.pixelStride)
                    pixels[(size_t) (y * w + x)] = (uint8) (255 - reinterpret_cast<const Pixel*> (src)->getRed());
            }
        };

        if (bd.pixelFormat == Image::ARGB)
            copyCoverage (PixelARGB());
        else
            copyCoverage (PixelRGB());

        [texture replaceRegion: MTLRegionMake2D ((NSUInteger) job.area.getX(), (NSUInteger) job.area.getY(), (NSUInteger) w, (NSUInteger) h)
                   mipmapLevel: 0
                     withBytes: pixels.data()
                   bytesPerRow: (NSUInteger) w];
    }

private:
    struct StrikeKey
    {
        const Typeface* typeface;
        int pointHeight16, scale1024, hscale1024;

        bool operator== (const StrikeKey& other) const noexcept
        {
            return typeface == other.typeface && pointHeight16 == other.pointHeight16
                && scale1024 == other.scale1024 && hscale1024 == other.hscale1024;
        }
    };

    struct StrikeKeyHash
    {
        size_t operator() (const StrikeKey& k) const noexcept
        {
            auto h = std::hash<const void*>() (k.typeface);

            for (auto v : { k.pointHeight16, k.scale1024, k.hscale1024 })
                h = h * 1000003u ^ std::hash<int>() (v);

            return h;
        }
    };

    struct Plot
    {
        int shelfX = 0, shelfY = 0, shelfHeight = 0; // shelf packing within the plot, 1 px gutters
        uint64_t lastUsed = 0;
        bool opened = false;
        std::vector<std::pair<Strike*, uint32_t>> glyphs; // to forget them on eviction
    };

    // Room for w x h pixels: its atlas area and plot.
    std::optional<std::pair<Rectangle<int>, int>> allocate (int w, int h)
    {
        if (w + 1 > plotSize || h + 1 > plotSize)
            return {};

        const auto tryPlot = [&] (int index) -> std::optional<std::pair<Rectangle<int>, int>>
        {
            auto& p = plots[(size_t) index];
            auto x = p.shelfX, y = p.shelfY, shelfHeight = p.shelfHeight;

            if (x + w + 1 > plotSize)
            {
                x = 0;
                y += shelfHeight + 1;
                shelfHeight = 0;
            }

            // Nothing changes when the glyph doesn't fit: a smaller one may still fit this shelf.
            if (y + h + 1 > plotSize)
                return {};

            const Rectangle<int> area ((index % plotsPerSide) * plotSize + x, (index / plotsPerSide) * plotSize + y, w, h);
            p.shelfX = x + w + 1;
            p.shelfY = y;
            p.shelfHeight = jmax (shelfHeight, h);
            p.lastUsed = currentFrame;
            return std::pair { area, index };
        };

        // The plots being filled, most recent first (a few: older ones are nearly full anyway).
        for (auto it = openPlots.rbegin(); it != openPlots.rend(); ++it)
            if (auto place = tryPlot (*it))
                return place;

        // A plot never used, else the least recently used one no frame in flight still reads.
        auto chosen = -1;

        for (int i = 0; i < (int) plots.size() && chosen < 0; ++i)
            if (! plots[(size_t) i].opened)
                chosen = i;

        if (chosen < 0)
        {
            for (int i = 0; i < (int) plots.size(); ++i)
                if (plots[(size_t) i].lastUsed <= completedFrame
                     && (chosen < 0 || plots[(size_t) i].lastUsed < plots[(size_t) chosen].lastUsed))
                    chosen = i;

            if (chosen < 0)
                return {};

            auto& evicted = plots[(size_t) chosen];

            for (const auto& [strike, key] : evicted.glyphs)
                strike->glyphs.erase (key);

            evicted = {};
        }

        plots[(size_t) chosen].opened = true;
        openPlots.erase (std::remove (openPlots.begin(), openPlots.end(), chosen), openPlots.end());
        openPlots.push_back (chosen);

        if (openPlots.size() > 4)
            openPlots.erase (openPlots.begin());

        return tryPlot (chosen);
    }

    id<MTLTexture> texture = nil;
    std::unordered_map<StrikeKey, std::unique_ptr<Strike>, StrikeKeyHash> strikes;
    Image scratch;                  // where glyphs are rasterised (encode thread)
    std::vector<uint8> pixels;      // a glyph's coverage, uploaded to the atlas (encode thread)
    std::array<Plot, (size_t) (plotsPerSide * plotsPerSide)> plots;
    std::vector<int> openPlots;
    uint64_t currentFrame = 0, completedFrame = 0;
};

//==============================================================================
/*  Image textures, cached per ImagePixelData.
    ImagePixelData notifies changes and deletion on the thread that writes or frees the image
    (often not the message thread), so the entries are shared under a lock: the notifications only
    flag an entry or hand its texture over; uploads and releases happen on the message thread.
*/
class ImageTextureCache final : private ImagePixelData::Listener
{
public:
    explicit ImageTextureCache (id<MTLDevice> d) : device (d) {}

    ~ImageTextureCache() override
    {
        const ScopedLock sl (lock);

        for (auto& [data, entry] : entries)
        {
            data->listeners.remove (this);
            [entry.texture release];

            for (auto& spare : entry.spares)
                [spare.texture release];
        }

        for (auto t : dropped)
            [t release];
    }

    // Message thread.
    id<MTLTexture> get (const Image& image)
    {
        auto* data = image.getPixelData().get();

        if (data == nullptr)
            return nil;

        // Held during the upload: a writer that starts meanwhile waits in its change
        // notification, so the copy can't catch it half way.
        const ScopedLock sl (lock);
        auto it = entries.find (data);

        if (it != entries.end() && ! it->second.changed)
        {
            it->second.lastUsed = currentFrame;
            return it->second.texture;
        }

        if (it != entries.end())
        {
            // Written since it was uploaded (a meter or a spectrum drawn into an Image every frame):
            // the texture may still be read by frames in flight, so it becomes a spare, and a
            // spare that no frame in flight reads any more takes the new contents. That saves
            // creating (and releasing) a texture every frame for such images.
            auto& entry = it->second;

            if (entry.texture != nil)
                entry.spares.push_back ({ entry.texture, entry.lastUsed });

            entry.texture = nil;
            entry.changed = false;
            entry.lastUsed = currentFrame;

            const auto reusable = std::find_if (entry.spares.begin(), entry.spares.end(), [&] (const Spare& spare)
            {
                return spare.lastUsed <= completedFrame && fits (spare.texture, image);
            });

            if (reusable != entry.spares.end())
            {
                entry.texture = reusable->texture;
                entry.spares.erase (reusable);
                write (entry.texture, image);
            }
            else
            {
                entry.texture = upload (image, device);
            }

            // A couple of spares is enough for frames in flight; older ones go.
            while (entry.spares.size() > 2)
            {
                dropped.push_back (entry.spares.front().texture);
                entry.spares.erase (entry.spares.begin());
            }

            // A nil texture stays changed: the next frame tries again. Still registered as a
            // listener either way: removed when the image is deleted.
            entry.changed = entry.texture == nil;
            return entry.texture;
        }

        auto texture = upload (image, device);

        if (texture != nil)
        {
            entries[data] = { texture, false, currentFrame, {} };
            data->listeners.add (this);
        }

        return texture;
    }

    /*  'frame': the frame about to be recorded; 'completed': the last one done on the GPU
        (SharedResources::framesRecorded / framesCompleted).
    */
    void beginFrame (uint64_t frame, uint64_t completed) noexcept
    {
        currentFrame = frame;
        completedFrame = completed;
    }

    /*  Any thread: a texture (not from this cache) that recorded frames may still read, released
        with the textures this cache drops.
    */
    void releaseLater (id<MTLTexture> texture)
    {
        if (texture == nil)
            return;

        const ScopedLock sl (lock);
        dropped.push_back (texture);
    }

    /*  Message thread, at the start of a frame: the textures dropped since the last frame, to be
        released once the frames now in flight have completed.
    */
    void collectReleases (DeferredReleases& releases)
    {
        const ScopedLock sl (lock);

        for (auto t : dropped)
            releases.add (t);

        dropped.clear();
    }

    /*  Uploads an ARGB or single-channel image to a new texture (caller owns it). */
    static id<MTLTexture> upload (const Image& source, id<MTLDevice> device)
    {
        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: source.isSingleChannel() ? MTLPixelFormatR8Unorm
                                                                                                         : MTLPixelFormatBGRA8Unorm
                                                                        width: (NSUInteger) source.getWidth()
                                                                       height: (NSUInteger) source.getHeight()
                                                                    mipmapped: NO];
        desc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [device newTextureWithDescriptor: desc];

        if (texture != nil)
            write (texture, source);

        return texture;
    }

    /*  Whether a texture made by upload() can hold this image. */
    static bool fits (id<MTLTexture> texture, const Image& image)
    {
        return (int) texture.width == image.getWidth() && (int) texture.height == image.getHeight()
            && (texture.pixelFormat == MTLPixelFormatR8Unorm) == image.isSingleChannel();
    }

    /*  Copies an image's pixels into a texture of its size and format. */
    static void write (id<MTLTexture> texture, const Image& source)
    {
        const auto image = (source.isARGB() || source.isSingleChannel()) ? source
                                                                         : source.convertedToFormat (Image::ARGB);
        const auto alpha = image.isSingleChannel();

        // JUCE ARGB is premultiplied, stored as BGRA bytes on little-endian machines.
        const Image::BitmapData bd (image, Image::BitmapData::readOnly);
        const auto pixelBytes = alpha ? 1 : 4;

        if (bd.pixelStride == pixelBytes)
        {
            [texture replaceRegion: MTLRegionMake2D (0, 0, (NSUInteger) image.getWidth(), (NSUInteger) image.getHeight())
                       mipmapLevel: 0
                         withBytes: bd.data
                       bytesPerRow: (NSUInteger) bd.lineStride];
        }
        else
        {
            HeapBlock<uint8> packed ((size_t) (image.getWidth() * image.getHeight() * pixelBytes));

            for (int y = 0; y < image.getHeight(); ++y)
                for (int x = 0; x < image.getWidth(); ++x)
                    memcpy (packed + (y * image.getWidth() + x) * pixelBytes, bd.getPixelPointer (x, y), (size_t) pixelBytes);

            [texture replaceRegion: MTLRegionMake2D (0, 0, (NSUInteger) image.getWidth(), (NSUInteger) image.getHeight())
                       mipmapLevel: 0
                         withBytes: packed.get()
                       bytesPerRow: (NSUInteger) (image.getWidth() * pixelBytes)];
        }
    }

private:
    struct Spare
    {
        id<MTLTexture> texture = nil;
        uint64_t lastUsed = 0;              // the last frame that drew with it
    };

    struct Entry
    {
        id<MTLTexture> texture = nil;
        bool changed = false;
        uint64_t lastUsed = 0;
        std::vector<Spare> spares;          // earlier contents, possibly still read by frames in flight
    };

    // Any thread.
    void imageDataChanged (ImagePixelData* data) override
    {
        const ScopedLock sl (lock);

        if (auto it = entries.find (data); it != entries.end())
            it->second.changed = true;
    }

    void imageDataBeingDeleted (ImagePixelData* data) override
    {
        const ScopedLock sl (lock);

        if (auto it = entries.find (data); it != entries.end())
        {
            if (it->second.texture != nil)
                dropped.push_back (it->second.texture);

            for (auto& spare : it->second.spares)
                dropped.push_back (spare.texture);

            entries.erase (it);
            data->listeners.remove (this);
        }
    }

    id<MTLDevice> device;
    CriticalSection lock;
    std::map<ImagePixelData*, Entry> entries;
    std::vector<id<MTLTexture>> dropped;
    uint64_t currentFrame = 0, completedFrame = 0; // message thread
};

//==============================================================================
struct GradientInfo { Float4 row0, row1, meta, conic; }; // must match the shader


//==============================================================================
/*  Glyph outlines (normalised to 1 pt) for text drawn as paths: rotated, sheared, very large or
    gradient-filled text. Entries hold their typeface so a pointer can't be reused while cached.
*/
class GlyphOutlineCache
{
public:
    const Path& get (const Typeface::Ptr& typeface, uint16_t glyph)
    {
        const auto key = std::make_pair (static_cast<const void*> (typeface.get()), glyph);

        if (auto it = entries.find (key); it != entries.end())
            return it->second.outline;

        if (entries.size() >= 16384)
            entries.clear();

        auto& entry = entries[key];
        entry.typeface = typeface;
        typeface->getOutlineForGlyph (glyph, entry.outline);
        return entry.outline;
    }

private:
    struct Entry { Typeface::Ptr typeface; Path outline; };
    std::map<std::pair<const void*, uint16_t>, Entry> entries;
};

//==============================================================================
/*  Device-wide objects shared by every window: pipelines, glyph atlas, image textures. */
class SharedResources
{
public:
    SharedResources()
    {
        device = MTLCreateSystemDefaultDevice();

        if (device == nil)
            return;

        queue = [device newCommandQueue];

        NSError* error = nil;
        library = [device newLibraryWithSource: (NSString* _Nonnull) [NSString stringWithUTF8String: shaderSource]
                                                      options: nil
                                                        error: &error];

        if (library == nil)
        {
            DBG ("Metal renderer: shader compilation failed: " << nsStringToJuce (error.localizedDescription));
            jassertfalse;
            return;
        }

        vs = [library newFunctionWithName: @"vs"];
        drawPipelines[0] = createDrawPipelines (false);

        auto* depthDesc = [[MTLDepthStencilDescriptor alloc] init];
        depthDesc.depthCompareFunction = MTLCompareFunctionLess;
        depthDesc.depthWriteEnabled = YES;
        depthWriteState = [device newDepthStencilStateWithDescriptor: depthDesc];
        depthDesc.depthCompareFunction = MTLCompareFunctionLess; // an opaque interior isn't drawn twice
        depthDesc.depthWriteEnabled = NO;
        depthTestState = [device newDepthStencilStateWithDescriptor: depthDesc];
        [depthDesc release];

        auto* desc = [[MTLRenderPipelineDescriptor alloc] init];
        auto* attachment = desc.colorAttachments[0];

        // The mask and coverage passes below have no depth attachment.
        // Clip mask passes: one full quad into an R8 target, no blending.
        id<MTLFunction> maskCoverageFs = [library newFunctionWithName: @"maskCoverageFs"];
        id<MTLFunction> fullQuadVs = [library newFunctionWithName: @"fullQuadVs"];
        id<MTLFunction> maskImageFs = [library newFunctionWithName: @"maskImageFs"];
        attachment.pixelFormat = MTLPixelFormatR8Unorm;
        desc.vertexFunction = fullQuadVs;
        desc.fragmentFunction = maskCoverageFs;
        maskCoveragePipeline = [device newRenderPipelineStateWithDescriptor: desc error: &error];
        desc.vertexFunction = fullQuadVs;
        desc.fragmentFunction = maskImageFs;
        maskImagePipeline = [device newRenderPipelineStateWithDescriptor: desc error: &error];

        // Coverage atlas: deltas summed in 32-bit floats, then a prefix sum per row.
        id<MTLFunction> accumVs = [library newFunctionWithName: @"accumVs"];
        id<MTLFunction> accumFs = [library newFunctionWithName: @"accumFs"];
        attachment.pixelFormat = MTLPixelFormatR32Float;
        attachment.blendingEnabled = YES;
        attachment.rgbBlendOperation = attachment.alphaBlendOperation = MTLBlendOperationAdd;
        attachment.sourceRGBBlendFactor = attachment.sourceAlphaBlendFactor = MTLBlendFactorOne;
        attachment.destinationRGBBlendFactor = attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
        desc.vertexFunction = accumVs;
        desc.fragmentFunction = accumFs;
        accumPipeline = [device newRenderPipelineStateWithDescriptor: desc error: &error];
        id<MTLFunction> strokeAccumVs = [library newFunctionWithName: @"strokeAccumVs"];
        id<MTLFunction> strokeAccumFs = [library newFunctionWithName: @"strokeAccumFs"];
        desc.vertexFunction = strokeAccumVs;
        desc.fragmentFunction = strokeAccumFs;
        strokeAccumPipeline = [device newRenderPipelineStateWithDescriptor: desc error: &error];
        [strokeAccumVs release];
        [strokeAccumFs release];
        [accumVs release];
        [accumFs release];

        id<MTLFunction> prefixSum = [library newFunctionWithName: @"coveragePrefixSum"];
        prefixSumPipeline = [device newComputePipelineStateWithFunction: prefixSum error: &error];
        [prefixSum release];
        id<MTLFunction> blur = [library newFunctionWithName: @"boxBlurLines"];
        blurPipeline = [device newComputePipelineStateWithFunction: blur error: &error];
        [blur release];
        id<MTLFunction> blurSimd = [library newFunctionWithName: @"boxBlurLinesSimd"];
        blurSimdPipeline = [device newComputePipelineStateWithFunction: blurSimd error: &error];
        [blurSimd release];
        [maskCoverageFs release];
        [fullQuadVs release];
        [maskImageFs release];

        [desc release];

        auto* samplerDesc = [[MTLSamplerDescriptor alloc] init];
        samplerDesc.sAddressMode = samplerDesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
        samplerDesc.minFilter = samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
        nearestSampler = [device newSamplerStateWithDescriptor: samplerDesc];
        samplerDesc.minFilter = samplerDesc.magFilter = MTLSamplerMinMagFilterLinear;
        linearSampler = [device newSamplerStateWithDescriptor: samplerDesc];
        samplerDesc.sAddressMode = samplerDesc.tAddressMode = MTLSamplerAddressModeRepeat; // tiled image fills
        linearRepeatSampler = [device newSamplerStateWithDescriptor: samplerDesc];
        samplerDesc.minFilter = samplerDesc.magFilter = MTLSamplerMinMagFilterNearest;
        nearestRepeatSampler = [device newSamplerStateWithDescriptor: samplerDesc];
        [samplerDesc release];

        glyphAtlas = std::make_unique<GlyphAtlas> (device);
        imageCache = std::make_unique<ImageTextureCache> (device);

        // Bound as the clip mask when there is none: one white texel, sampled with a huge rect.
        auto* noMaskDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: MTLPixelFormatR8Unorm
                                                                              width: 1 height: 1 mipmapped: NO];
        noMaskDesc.usage = MTLTextureUsageShaderRead;
        noMask = [device newTextureWithDescriptor: noMaskDesc];
        const uint8_t white = 255;
        [noMask replaceRegion: MTLRegionMake2D (0, 0, 1, 1) mipmapLevel: 0 withBytes: &white bytesPerRow: 1];
    }

    ~SharedResources()
    {
        imageCache.reset();
        glyphAtlas.reset();
        [linearSampler release];
        [linearRepeatSampler release];
        [nearestRepeatSampler release];
        [nearestSampler release];

        for (auto& p : drawPipelines)
            p.release();

        [blurPipeline release];
        [blurSimdPipeline release];
        [vs release];
        [library release];
        [maskCoveragePipeline release];
        [maskImagePipeline release];
        [accumPipeline release];
        [prefixSumPipeline release];
        [strokeAccumPipeline release];
        [noMask release];
        [depthWriteState release];
        [depthTestState release];
        [queue release];
        [device release];
        dispatch_release (encodeQueue);
    }

    bool isValid() const noexcept
    {
        return drawPipelines[0].isValid()
            && depthWriteState != nil && depthTestState != nil
            && maskCoveragePipeline != nil && maskImagePipeline != nil && accumPipeline != nil && prefixSumPipeline != nil
            && strokeAccumPipeline != nil && noMask != nil;
    }

    static constexpr int numFamilies = 4;

    /*  The pipelines drawing instances into a target: BGRA8 (windows, layers, ARGB and RGB images)
        or R8 (single-channel images, see 'alphaTarget' in the shader).
    */
    struct DrawPipelines
    {
        id<MTLRenderPipelineState> blend[numFamilies][4] {}; // [instance family][juce::BlendMode]
        id<MTLRenderPipelineState> clear = nil, opaque = nil;

        bool isValid() const noexcept
        {
            return clear != nil && opaque != nil
                && std::all_of (std::begin (blend), std::end (blend),
                                [] (const auto& family) { return std::all_of (std::begin (family), std::end (family), [] (auto p) { return p != nil; }); });
        }

        void release()
        {
            for (auto& family : blend)
                for (auto p : family)
                    [p release];

            [clear release];
            [opaque release];
            *this = {};
        }
    };

    /*  Any thread. The R8 pipelines are only built once a single-channel image is made. */
    const DrawPipelines& getDrawPipelines (bool alphaTarget)
    {
        if (! alphaTarget)
            return drawPipelines[0];

        std::call_once (alphaPipelinesBuilt, [this] { drawPipelines[1] = createDrawPipelines (true); });
        return drawPipelines[1];
    }

    // The box blur's threadgroups: 256 threads, lines of up to 64 pixels per thread (see boxBlurLines).
    static constexpr NSUInteger blurThreads = 256, blurPixelsPerLane = 16; // see boxBlurLinesSimd
    static constexpr int maxBlurLine = 16384;

    bool canBlur() const noexcept
    {
        return blurPipeline != nil && blurPipeline.maxTotalThreadsPerThreadgroup >= blurThreads && blurSimdPipeline != nil;
    }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLDepthStencilState> depthWriteState = nil, depthTestState = nil;
    static constexpr MTLPixelFormat depthFormat = MTLPixelFormatDepth32Float;
    id<MTLComputePipelineState> blurPipeline = nil, blurSimdPipeline = nil;
    id<MTLRenderPipelineState> maskCoveragePipeline = nil, maskImagePipeline = nil, accumPipeline = nil;
    id<MTLRenderPipelineState> strokeAccumPipeline = nil;
    id<MTLComputePipelineState> prefixSumPipeline = nil;
    id<MTLTexture> noMask = nil;
    id<MTLSamplerState> nearestSampler = nil, linearSampler = nil, nearestRepeatSampler = nil, linearRepeatSampler = nil;

    id<MTLSamplerState> getSampler (bool nearest, bool repeat) const noexcept
    {
        return repeat ? (nearest ? nearestRepeatSampler : linearRepeatSampler)
                      : (nearest ? nearestSampler : linearSampler);
    }
    std::unique_ptr<GlyphAtlas> glyphAtlas;
    std::unique_ptr<ImageTextureCache> imageCache;
    GlyphOutlineCache glyphOutlines;

    // Frames of all windows, numbered in recording order (= encoding and GPU order: one encode
    // queue, one command queue), and the last one whose GPU work completed (glyph atlas eviction).
    std::atomic<uint64_t> framesRecorded { 0 }; // incremented on the message thread
    std::atomic<uint64_t> framesCompleted { 0 };

    // Frames are encoded and committed off the message thread, in the order they were painted.
    // One queue for all windows: a texture the caches drop while painting a window stays alive
    // until frames painted after it complete, which then includes every window's earlier frames.
    dispatch_queue_t encodeQueue = dispatch_queue_create ("juce.MetalRenderer.encode",
                                                          dispatch_queue_attr_make_with_qos_class (DISPATCH_QUEUE_SERIAL,
                                                                                                   QOS_CLASS_USER_INTERACTIVE, 0));

private:
    DrawPipelines createDrawPipelines (bool alphaTarget) const
    {
        DrawPipelines result;
        NSError* error = nil;
        auto* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vs;
        desc.depthAttachmentPixelFormat = depthFormat; // window and layer passes test painter's depth
        auto* attachment = desc.colorAttachments[0];
        attachment.pixelFormat = alphaTarget ? MTLPixelFormatR8Unorm : MTLPixelFormatBGRA8Unorm;
        attachment.blendingEnabled = YES; // factors per blend mode below

        // juce::BlendMode's other Porter-Duff operators (the colour is premultiplied and already
        // scaled by coverage, as the software renderer scales the source by its alpha level):
        // source R = S, destinationIn R = D * Sa, destinationOut R = D * (1 - Sa).
        const MTLBlendFactor blendFactors[4][2] { { MTLBlendFactorOne,  MTLBlendFactorOneMinusSourceAlpha },   // sourceOver
                                                  { MTLBlendFactorOne,  MTLBlendFactorZero },                  // source
                                                  { MTLBlendFactorZero, MTLBlendFactorSourceAlpha },           // destinationIn
                                                  { MTLBlendFactorZero, MTLBlendFactorOneMinusSourceAlpha } }; // destinationOut

        // One fragment function per instance family (see 'family' in the shader). A family whose
        // function fails to specialise keeps nil pipelines: isValid() then refuses them (a
        // pipeline without a fragment function would draw nothing).
        for (uint32_t f = 0; f < (uint32_t) numFamilies; ++f)
        {
            auto* constants = [[MTLFunctionConstantValues alloc] init];
            const bool alpha = alphaTarget;
            [constants setConstantValue: &f type: MTLDataTypeUInt atIndex: 0];
            [constants setConstantValue: &alpha type: MTLDataTypeBool atIndex: 1];
            id<MTLFunction> fs = [library newFunctionWithName: @"fs" constantValues: constants error: &error];
            [constants release];

            if (fs == nil)
                continue;

            desc.fragmentFunction = fs;

            for (int b = 0; b < 4; ++b)
            {
                attachment.sourceRGBBlendFactor = attachment.sourceAlphaBlendFactor = blendFactors[b][0];
                attachment.destinationRGBBlendFactor = attachment.destinationAlphaBlendFactor = blendFactors[b][1];
                result.blend[f][b] = [device newRenderPipelineStateWithDescriptor: desc error: &error];
            }

            [fs release];
        }

        id<MTLFunction> fsClear = [library newFunctionWithName: @"fsClear"];
        desc.fragmentFunction = fsClear;
        attachment.blendingEnabled = NO;
        result.clear = [device newRenderPipelineStateWithDescriptor: desc error: &error];
        [fsClear release];

        // Opaque interiors: their colour as is, no blending.
        id<MTLFunction> vsOpaque = [library newFunctionWithName: @"vsOpaque"];
        id<MTLFunction> fsOpaque = [library newFunctionWithName: alphaTarget ? @"fsOpaqueAlpha" : @"fsOpaque"];
        desc.vertexFunction = vsOpaque;
        desc.fragmentFunction = fsOpaque;
        result.opaque = [device newRenderPipelineStateWithDescriptor: desc error: &error];
        [vsOpaque release];
        [fsOpaque release];
        [desc release];
        return result;
    }

    id<MTLLibrary> library = nil;
    id<MTLFunction> vs = nil;
    DrawPipelines drawPipelines[2]; // BGRA8, R8
    std::once_flag alphaPipelinesBuilt;
};

//==============================================================================
/*  A 64-bit hash of recorded content (FrameRecorder::contentKey). */
struct ContentHasher
{
    void bytes (const void* data, size_t size)
    {
        auto* p = static_cast<const uint8*> (data);

        for (; size >= 8; p += 8, size -= 8)
        {
            uint64_t w;
            memcpy (&w, p, 8);
            mix (w);
        }

        uint64_t tail = 0;
        memcpy (&tail, p, size);
        mix (tail ^ ((uint64_t) size << 56));
    }

    template <typename... T>
    void values (T... v) { (bytes (&v, sizeof (v)), ...); }

    template <typename T>
    void all (const std::vector<T>& v)
    {
        static_assert (std::is_trivially_copyable_v<T>);
        values (v.size());
        bytes (v.data(), v.size() * sizeof (T));
    }

    void rect (Rectangle<int> r) { values (r.getX(), r.getY(), r.getWidth(), r.getHeight()); }
    void transform (const AffineTransform& t) { values (t.mat00, t.mat01, t.mat02, t.mat10, t.mat11, t.mat12); }
    void mix (uint64_t w) { h = (h ^ w) * 0x9e3779b97f4a7c15ull; h ^= h >> 32; }

    uint64_t h = 0xcbf29ce484222325ull;
};

//==============================================================================
/*  What MetalGraphicsContext writes into: instances, split into draw batches by texture state. */
struct FrameRecorder
{
    /*  A run of quad instances sharing their textures and mask: recorded in submission order,
        regrouped by reorder() at encode time.
    */
    struct Batch
    {
        id<MTLTexture> image = nil;
        bool nearest = false;
        uint32_t first = 0, count = 0;      // instances
        int mask = -1;                      // index in masks, -1: none
        int layer = -1;                     // texture = that layer target's texture
        bool repeat = false;                // sample 'image' with wrapping (tiled image fills)
        int coveragePage = -1;              // coverage atlas page of path fills
        BlendMode blend = BlendMode::sourceOver;
        int family = 0;                     // pipeline: see 'family' in the shader
        bool segmented = false;             // holds strokes: eight quads per instance, else one (see 'vs')

        NSUInteger verticesPerInstance() const noexcept { return segmented ? 48 : 6; }
    };

    /*  A render target: the window's retained texture (index 0), or a transparency layer drawn
        offscreen then composited into its parent. Layers are encoded in the order they end
        (inner first), the main target last.
    */
    struct Target
    {
        Rectangle<int> area;                // device pixels
        int texWidth = 0, texHeight = 0;    // layers
        std::vector<Batch> batches;
        id<MTLTexture> texture = nil;       // layers: allocated at encode time
        Rectangle<float> drawn;             // device pixels touched by what was drawn into it
        std::vector<uint32_t> opaque;       // instances whose interior occludes earlier draws
    };

    /*  Mask and layer textures come from a pool: their size is the area rounded up to 64 px so
        that a clip or layer whose size changes slightly between frames reuses its texture.
    */
    static int pooledSize (int v) noexcept { return ((jmax (1, v) + 63) / 64) * 64; }

    /*  A clip mask rendered before the main pass: a path's coverage or an image's alpha,
        multiplied by the parent mask. 'area' is in device pixels; the texture is allocated at
        encode time.
    */
    struct Mask
    {
        Rectangle<int> area;
        int texWidth = 0, texHeight = 0;
        int parent = -1;
        int coveragePage = -1;                      // path: its coverage atlas page; inv0: atlas offset,
                                                    // even-odd; imageWidth/Height: the area's size
        id<MTLTexture> image = nil;                 // image: texture and device px -> image px
        Float4 inv0, inv1;
        float imageWidth = 0, imageHeight = 0;
        bool imageIsAlphaOnly = false, nearest = false;
        id<MTLTexture> texture = nil;

        // A path clip that isn't nested in another clip is read from the coverage atlas as it is.
        [[nodiscard]] bool readsCoverage() const noexcept { return coveragePage >= 0 && parent < 0; }
    };

    /*  A key for what was recorded, so that what it renders can be cached (see ImageQueue's blurred
        contents), or 0 when it draws something a key can't identify: images (their pixels can
        change under the same texture) and glyphs (their place in the atlas can be reused).
    */
    uint64_t contentKey() const
    {
        if (! glyphJobs.empty())
            return 0;

        ContentHasher hash;

        for (const auto& target : targets)
        {
            for (const auto& b : target.batches)
            {
                if (b.image != nil || b.family == 1)
                    return 0;

                hash.values (b.nearest, b.first, b.count, b.mask, b.layer, b.repeat, b.coveragePage, (int) b.blend, b.family, b.segmented);
            }

            hash.rect (target.area);
            hash.values (target.texWidth, target.texHeight);
            hash.all (target.opaque);
        }

        for (const auto& m : masks)
        {
            if (m.image != nil)
                return 0;

            hash.rect (m.area);
            hash.values (m.texWidth, m.texHeight, m.parent, m.coveragePage, m.inv0, m.inv1, m.imageWidth, m.imageHeight,
                         m.imageIsAlphaOnly, m.nearest);
        }

        for (const auto& job : coverageJobs)
        {
            hash.values (job.path.isUsingNonZeroWinding());

            for (Path::Iterator it (job.path); it.next();)
                hash.values (it.elementType, it.x1, it.y1, it.x2, it.y2, it.x3, it.y3);

            hash.transform (job.toDevice);
            hash.values (job.stroke.getStrokeThickness(), job.stroke.getJointStyle(), job.stroke.getEndStyle(),
                         job.stroke.getMiterLimit(), job.strokeAccuracy, job.atlasOrigin, job.page);
            hash.transform (job.strokeTransform);
            hash.rect (job.area);
        }

        for (const auto& page : coveragePages)
            hash.values (page.width, page.height, page.evenOdd);

        hash.all (instances);
        hash.all (clears);
        hash.all (gradients);
        hash.all (gradientStops);
        hash.all (encodeOrder);
        hash.values (clearMain);
        return hash.h != 0 ? hash.h : 1;
    }

    void clear (Rectangle<int> mainArea)
    {
        clearMain = false;
        instances.clear();
        clears.clear();
        targets.clear();
        targets.push_back ({ mainArea, 0, 0, {}, nil, {}, {} });
        openTargets.assign (1, 0);
        encodeOrder.clear();
        gradients.clear();
        gradientStops.clear();
        masks.clear();
        coverageJobs.clear();
        glyphJobs.clear();
        coverageSegments.clear();
        coverageEntries.clear();
        coveragePages.clear();
        strokePoints.clear();
        strokeSubs.clear();
        strokeJobs.clear();
        strokeQuads.clear();
        shelfX = shelfY = shelfHeight = 0;
    }

    /*  Path fills by coverage counting (see the shader). Each path gets its device rectangle in an
        R32Float atlas page (shelf packed); its flattened outline becomes segments in that page.
    */
    struct CoverageSegment { Float4 line, rect; };  // must match 'Seg' in the shader
    struct CoverageEntry { uint32_t x, y, width, height, firstRow, unused[3]; }; // must match 'CovEntry'
    struct CoveragePage
    {
        int width = 0, height = 0;
        uint32_t firstSegment = 0, segmentCount = 0, firstEntry = 0, entryCount = 0, rowCount = 0;
        id<MTLTexture> deltas = nil, coverage = nil; // allocated at encode time
        bool evenOdd = false;                        // holds an even-odd path (see the coverage format)
        uint32_t firstStrokeQuad = 0, strokeQuadCount = 0; // its strokes' quads (see strokeAccumVs)
    };

    // Stroke expansion's input (see strokeAccumVs in the shader).
    static constexpr uint32_t strokeEdgesPerQuad = 8;                                                        // must match 'strokeEdgesPerQuad'
    struct StrokeJobData { Float4 xf0, xf1, rect, style; uint32_t joint, cap, capEdges, maxArcSteps; }; // must match 'StrokeJob'
    struct StrokeSub { uint32_t first, count, closed, job; };                                          // must match 'StrokeSub'
    struct StrokeQuad { uint32_t point, subAndEdge; };                                                 // must match 'StrokeQuad'
    struct StrokePoint { float x, y; };                                                                      // must match 'StrokePoint'

    static constexpr int coveragePageWidth = 2048, coveragePageMaxHeight = 4096;

    /*  Tested and not kept, measured on an M1 (Metal System Trace) and on a busy plug-in page:
        - Triangulating paths into anti-aliased meshes (an earlier version of this renderer): on
          GraphicsDemo's stroked paths, rebuilt every frame so that no mesh cache helps, 87% of
          the time went into triangulation (sorting vertices, resolving intersections), and some
          SVG paths failed to triangulate (about 6 per frame in its SVG demo).
        - Sparse coverage, as tile-based renderers do: a path's area cut into 32 x 16 tiles, only
          the runs of tiles its edges touch summed and composited, the others drawn solid or not at
          all from a winding number counted on the CPU (from the edges crossing each tile row's
          top; horizontal edges must mark their tiles too). 5% off the composite and 7-33% off the
          prefix sums, about 0.05 ms per frame (some 4% of that page's GPU time), for ~400 lines
          and a CPU winding count that must match the GPU's sums exactly. The composite's empty
          pixels were already cheap: one read, then discarded.
        - Caching the coverage of paths unchanged from the previous frame: on that page (animated
          modulations and histories) no path repeated.
        - Flattening fills on the GPU: it must place vertices as PathFlatteningIterator does to keep
          EdgeTable's pixels, and the whole encoding was 0.2 ms per frame on that page.
    */

    /*  A path whose coverage the frame needs. paint() only records it (a copy of the path, its
        place in the atlas); its outline is built and flattened into segments on the encode thread
        (resolveCoverage), as Core Animation (drawsAsynchronously), Chrome (cc's display lists,
        rasterised on worker threads) and Flutter (DisplayList, rasterised on its raster thread)
        keep the UI thread to recording and do the CPU side of rasterising elsewhere.
    */
    struct CoverageJob
    {
        Path path;                          // the outline to fill, or the path to stroke
        AffineTransform toDevice;           // outline -> device pixels
        PathStrokeType stroke { 0.0f };     // thickness > 0: 'path' is stroked first
        AffineTransform strokeTransform;    // applied to 'path' before stroking
        float strokeAccuracy = 1.0f;
        Rectangle<int> area;                // device pixels
        Point<int> atlasOrigin;
        int page = -1;
    };

    /*  Returns the atlas page that will hold the path's coverage over 'area' (device pixels) and,
        in 'atlasOrigin', where area's top-left lands.
    */
    int addCoveragePath (const Path& path, const AffineTransform& toDevice, Rectangle<int> area, Point<int>& atlasOrigin)
    {
        auto& job = addCoverageJob (area, ! path.isUsingNonZeroWinding());
        job.path = path;
        job.toDevice = toDevice;
        atlasOrigin = job.atlasOrigin;
        return job.page;
    }

    /*  The same for a stroke of 'path' (transformed by 'strokeTransform', flattened with
        'accuracy', as PathStrokeType::createStrokedPath), drawn through 'toDevice'. Its outline
        is built on the GPU (strokeAccumVs).
    */
    int addCoverageStroke (const Path& path, const PathStrokeType& stroke, const AffineTransform& strokeTransform,
                           float accuracy, const AffineTransform& toDevice, Rectangle<int> area, Point<int>& atlasOrigin)
    {
        auto& job = addCoverageJob (area, false);
        job.path = path;
        job.stroke = stroke;
        job.strokeTransform = strokeTransform;
        job.strokeAccuracy = accuracy;
        job.toDevice = toDevice;
        atlasOrigin = job.atlasOrigin;
        return job.page;
    }

    /*  Encode thread, before the segments are uploaded: flattens each recorded path into its
        page's segments, and each stroke's centre line into the stroke expansion's input (pages
        hold consecutive jobs).
    */
    void resolveCoverage()
    {
        coverageSegments.clear();
        strokePoints.clear();
        strokeSubs.clear();
        strokeJobs.clear();
        strokeQuads.clear();
        auto lastPage = -1;

        for (const auto& job : coverageJobs)
        {
            auto& page = coveragePages[(size_t) job.page];

            if (job.page != lastPage)
            {
                page.firstSegment = (uint32_t) coverageSegments.size();
                page.segmentCount = 0;
                page.firstStrokeQuad = (uint32_t) strokeQuads.size();
                page.strokeQuadCount = 0;
                lastPage = job.page;
            }

            const auto first = coverageSegments.size();
            const auto firstQuad = strokeQuads.size();

            if (job.stroke.getStrokeThickness() > 0.0f)
                addStroke (job);
            else
                addSegments (job.path, job);

            page.segmentCount += (uint32_t) (coverageSegments.size() - first);
            page.strokeQuadCount += (uint32_t) (strokeQuads.size() - firstQuad);
        }
    }

    /*  A stroke's centre line, flattened as PathStrokeType::createStrokedPath does it (in the
        stroke's space, with its tolerance, merging segments shorter than its minimum), and the
        parameters with which strokeAccumVs builds its outline.
    */
    void addStroke (const CoverageJob& job)
    {
        const auto& stroke = job.stroke;
        const auto& xf = job.toDevice;
        const auto halfWidth = 0.5f * stroke.getStrokeThickness();

        // Arcs (round joins and caps) in steps whose sagitta is at most 0.1 device pixel, at the
        // transform's largest scale (its largest singular value), and at most 64 per half turn.
        const auto sumSquares = xf.mat00 * xf.mat00 + xf.mat01 * xf.mat01 + xf.mat10 * xf.mat10 + xf.mat11 * xf.mat11;
        const auto rowDifference = xf.mat00 * xf.mat00 + xf.mat01 * xf.mat01 - xf.mat10 * xf.mat10 - xf.mat11 * xf.mat11;
        const auto rowDot = xf.mat00 * xf.mat10 + xf.mat01 * xf.mat11;
        const auto maxScale = std::sqrt (0.5f * (sumSquares + std::sqrt (rowDifference * rowDifference + 4.0f * rowDot * rowDot)));
        const auto radius = halfWidth * maxScale;
        constexpr auto tolerance = 0.1f, pi = MathConstants<float>::pi;
        const auto step = jmax (pi / 64.0f, radius > tolerance ? 2.0f * std::acos (1.0f - tolerance / radius) : pi);
        const auto maxArcSteps = (uint32_t) jlimit (1.0f, 64.0f, std::ceil (pi / step));

        // The edges each point owns (see strokeAccumVs): two offset edges and a join (miter: an
        // edge to the miter point and two bevels; bevel: two; round: a bevel and the arc's steps,
        // counted here when they could take more than a quad, with one to spare for rounding, else
        // the most a half turn takes); an open sub-path's first and last points have a cap each
        // (butt: one edge; square: three; round: an arc).
        const auto joint = stroke.getJointStyle();
        const auto cap = stroke.getEndStyle();
        const auto capEdges = cap == PathStrokeType::square ? 3u : (cap == PathStrokeType::rounded ? maxArcSteps : 1u);
        const auto countArcSteps = joint == PathStrokeType::curved && 2 + 1 + maxArcSteps > strokeEdgesPerQuad;
        const auto miterLimit = jmax (1.0f, stroke.getMiterLimit());

        const auto joinEdgesAt = [&] (size_t first, size_t n, size_t j) -> uint32_t
        {
            if (joint == PathStrokeType::mitered)
                return 3;

            if (joint == PathStrokeType::beveled)
                return 2;

            if (! countArcSteps)
                return 1 + maxArcSteps;

            const auto& b = strokePoints[first + (j + n - 1) % n];
            const auto& p = strokePoints[first + j % n];
            const auto& a = strokePoints[first + (j + 1) % n];
            const auto t0 = Point<float> (p.x - b.x, p.y - b.y), t1 = Point<float> (a.x - p.x, a.y - p.y);
            const auto turn = std::abs (std::atan2 (t0.x * t1.y - t0.y * t1.x, t0.x * t1.x + t0.y * t1.y));
            return jmax (2u, 1 + jmin (maxArcSteps, (uint32_t) std::ceil (turn / step) + 1));
        };

        const auto dx = (float) (job.atlasOrigin.x - job.area.getX()), dy = (float) (job.atlasOrigin.y - job.area.getY());
        const auto jobIndex = (uint32_t) strokeJobs.size();
        strokeJobs.push_back ({ { xf.mat00, xf.mat01, xf.mat02 + dx, 0.0f },
                                { xf.mat10, xf.mat11, xf.mat12 + dy, 0.0f },
                                { (float) job.atlasOrigin.x, (float) job.atlasOrigin.y,
                                  (float) (job.atlasOrigin.x + job.area.getWidth()), (float) (job.atlasOrigin.y + job.area.getHeight()) },
                                { halfWidth, square (miterLimit), step, 0.0f },
                                (uint32_t) joint, (uint32_t) cap, capEdges, maxArcSteps });

        // Only the part of the centre line that can reach the area is kept: segments beyond
        // where the outline reaches (half the width, times the miter limit or sqrt 2 for square
        // caps, plus a pixel) are dropped and those crossing that margin are cut there, so that a
        // long path mostly outside the area costs nothing, and no edge starts so far away that its
        // coordinates lose precision. The ends this makes, and their caps, lie outside the area.
        Rectangle<float> keep;
        const auto rowDifferenceMin = sumSquares - std::sqrt (rowDifference * rowDifference + 4.0f * rowDot * rowDot);
        const auto minScale = std::sqrt (jmax (0.0f, 0.5f * rowDifferenceMin));
        const auto cull = minScale > 1.0e-6f;

        if (cull)
        {
            const auto reach = halfWidth * jmax (joint == PathStrokeType::mitered ? miterLimit : 1.0f,
                                                 cap == PathStrokeType::square ? MathConstants<float>::sqrt2 : 1.0f);
            keep = job.area.toFloat().transformedBy (xf.inverted()).expanded (reach + 1.0f / minScale);
        }

        auto subFirst = strokePoints.size();
        auto closed = false, cut = false;

        const auto addPoint = [&] (Point<float> p)
        {
            if (strokePoints.size() > subFirst && exactlyEqual (strokePoints.back().x, p.x) && exactlyEqual (strokePoints.back().y, p.y))
                return;

            strokePoints.push_back ({ p.x, p.y });
        };

        const auto endSubPath = [&]
        {
            const auto isClosed = closed && ! cut;
            auto count = strokePoints.size() - subFirst;

            // A closed sub-path's last point is its first.
            if (isClosed && count > 1)
            {
                strokePoints.pop_back();
                --count;
            }

            if (count >= 2)
            {
                // Its points' quads, eight edges each.
                const auto subIndex = (uint32_t) strokeSubs.size();
                jassert (subIndex < (1u << 24));
                strokeSubs.push_back ({ (uint32_t) subFirst, (uint32_t) count, isClosed ? 1u : 0u, jobIndex });
                const auto segments = isClosed ? count : count - 1;

                for (size_t k = 0; k < count; ++k)
                {
                    uint32_t edges = capEdges;

                    if (k < segments)
                    {
                        edges = 2 + (k == 0 && ! isClosed ? capEdges : 0u);

                        if (k + 1 < segments || isClosed)
                            edges += joinEdgesAt (subFirst, count, k + 1);
                    }

                    for (uint32_t e = 0; e < edges; e += strokeEdgesPerQuad)
                        strokeQuads.push_back ({ (uint32_t) (subFirst + k), subIndex << 8 | e });
                }
            }
            else
            {
                strokePoints.resize (subFirst);
            }

            subFirst = strokePoints.size();
            closed = false;
        };

        const auto addSegment = [&] (Point<float> from, Point<float> to)
        {
            auto a = from, b = to;

            if (cull && ! clipToRectangle (a, b, keep))
            {
                cut = true;
                endSubPath();
                return;
            }

            if (a != from)
            {
                cut = true;
                endSubPath();
            }

            if (strokePoints.size() == subFirst)
                addPoint (a);

            addPoint (b);

            if (b != to)
            {
                cut = true;
                endSubPath();
            }
        };

        // PathStrokeHelpers::createStroke's iteration: a segment is kept if it is long enough,
        // closes its sub-path or ends it; a short one is merged into the next.
        constexpr auto minSegmentLengthSquared = 0.0001f;
        Point<float> start;

        for (PathFlatteningIterator it (job.path, job.strokeTransform, Path::defaultToleranceForMeasurement / job.strokeAccuracy); it.next();)
        {
            if (it.subPathIndex == 0)
            {
                endSubPath();
                cut = false;
                start = { it.x1, it.y1 };
            }

            const auto segmentX = it.x2 - start.x, segmentY = it.y2 - start.y;

            if (it.closesSubPath || segmentX * segmentX + segmentY * segmentY > minSegmentLengthSquared || it.isLastInSubpath())
            {
                addSegment (start, { it.x2, it.y2 });

                if (it.closesSubPath)
                {
                    closed = true;
                    endSubPath();
                    cut = false;
                }

                start = { it.x2, it.y2 };
            }
        }

        endSubPath();
    }

    // Liang-Barsky: clips the segment a-b to r, false if none of it is inside.
    static bool clipToRectangle (Point<float>& a, Point<float>& b, Rectangle<float> r) noexcept
    {
        if (r.contains (a) && r.contains (b))
            return true;

        const auto d = b - a;
        auto t0 = 0.0f, t1 = 1.0f;

        const auto clipAgainst = [&] (float p, float q)
        {
            if (exactlyEqual (p, 0.0f))
                return q >= 0.0f;

            const auto t = q / p;

            if (p < 0.0f)
            {
                if (t > t1)
                    return false;

                t0 = jmax (t0, t);
            }
            else
            {
                if (t < t0)
                    return false;

                t1 = jmin (t1, t);
            }

            return true;
        };

        if (! (clipAgainst (-d.x, a.x - r.getX()) && clipAgainst (d.x, r.getRight() - a.x)
               && clipAgainst (-d.y, a.y - r.getY()) && clipAgainst (d.y, r.getBottom() - a.y)))
            return false;

        const auto origin = a;

        if (t1 < 1.0f)
            b = origin + d * t1;

        if (t0 > 0.0f)
            a = origin + d * t0;

        return true;
    }

    CoverageJob& addCoverageJob (Rectangle<int> area, bool evenOdd)
    {
        const auto w = area.getWidth(), h = area.getHeight();
        auto newPage = coveragePages.empty();

        if (! newPage)
        {
            if (shelfX + w > coveragePages.back().width)
            {
                shelfY += shelfHeight;
                shelfX = shelfHeight = 0;
            }

            newPage = w > coveragePages.back().width || (shelfY > 0 && shelfY + h > coveragePageMaxHeight);
        }

        if (newPage)
        {
            coveragePages.push_back ({ jmax (coveragePageWidth, w), 0, 0, 0,
                                       (uint32_t) coverageEntries.size(), 0, 0, nil, nil });
            shelfX = shelfY = shelfHeight = 0;
        }

        auto& page = coveragePages.back();
        const Point<int> atlasOrigin { shelfX, shelfY };
        shelfX += w;
        shelfHeight = jmax (shelfHeight, h);
        page.height = jmax (page.height, shelfY + h);
        page.evenOdd = page.evenOdd || evenOdd;
        coverageEntries.push_back ({ (uint32_t) atlasOrigin.x, (uint32_t) atlasOrigin.y, (uint32_t) w, (uint32_t) h,
                                     page.rowCount, {} });
        ++page.entryCount;
        page.rowCount += (uint32_t) h;

        auto& job = coverageJobs.emplace_back();
        job.area = area;
        job.atlasOrigin = atlasOrigin;
        job.page = (int) coveragePages.size() - 1;
        return job;
    }

    // The outline's edges inside the job's area, moved to its place in the atlas page.
    void addSegments (const Path& outline, const CoverageJob& job)
    {
        const auto area = job.area;
        const auto dx = (float) (job.atlasOrigin.x - area.getX()), dy = (float) (job.atlasOrigin.y - area.getY());
        const Float4 rect { (float) job.atlasOrigin.x, (float) job.atlasOrigin.y,
                            (float) (job.atlasOrigin.x + area.getWidth()), (float) (job.atlasOrigin.y + area.getHeight()) };
        const auto top = (float) area.getY(), bottom = (float) area.getBottom(), right = (float) area.getRight();
        const auto left = (float) area.getX();

        const auto push = [&] (float x1, float y1, float x2, float y2)
        {
            if (! exactlyEqual (y1, y2))
                coverageSegments.push_back ({ { x1 + dx, y1 + dy, x2 + dx, y2 + dy }, rect });
        };

        const auto addSegment = [&] (float x1, float y1, float x2, float y2)
        {
            // Horizontal edges add no area; edges above, below or right of the area add none in it.
            if (exactlyEqual (y1, y2) || jmax (y1, y2) <= top || jmin (y1, y2) >= bottom || jmin (x1, x2) >= right)
                return;

            // Clipped to the area's sides, so that the GPU only visits cells inside it: the part right
            // of it adds nothing there, the part left of it adds its full height to every pixel of its
            // rows, which is what the same height on the area's left edge does.
            const auto yAt = [&] (float x) { return y1 + (x - x1) * (y2 - y1) / (x2 - x1); };

            if (x1 > right)
            {
                y1 = yAt (right);
                x1 = right;
            }
            else if (x2 > right)
            {
                y2 = yAt (right);
                x2 = right;
            }

            if (jmax (x1, x2) <= left)
                return push (left, y1, left, y2);

            if (jmin (x1, x2) < left)
            {
                const auto y = yAt (left);

                if (x1 < left)
                {
                    push (left, y1, left, y);
                    push (left, y, x2, y2);
                }
                else
                {
                    push (x1, y1, left, y);
                    push (left, y, left, y2);
                }

                return;
            }

            push (x1, y1, x2, y2);
        };

        // Same flattening as the software renderer's EdgeTable (default tolerance).
        float startX = 0.0f, startY = 0.0f;

        for (PathFlatteningIterator it (outline, job.toDevice); it.next();)
        {
            if (it.subPathIndex == 0)
            {
                startX = it.x1;
                startY = it.y1;
            }

            addSegment (it.x1, it.y1, it.x2, it.y2);

            // A fill closes open sub-paths.
            if (it.isLastInSubpath() && ! it.closesSubPath)
                addSegment (it.x2, it.y2, startX, startY);
        }
    }

    std::vector<Batch>& currentBatches() noexcept { return targets[(size_t) openTargets.back()].batches; }

    /*  Draw reordering by lookback, as WebRender's AlphaBatchList (batch.rs,
        set_params_and_get_batch) does it: each draw, in painter's order, walks back from the last
        batch and joins the first one with its state, provided none of the batches it walks past
        overlaps it (all of ours blend: anti-aliased coverage), so it can be drawn before them. The
        walk stops at the first batch it overlaps, and after 8 batches (searching further costs CPU
        for few merges). A grid of widgets thus becomes a few batches, one per kind of element,
        and a draw over earlier ones stays after them. Within a batch, instances keep their
        recording order (a draw call rasterises its instances in order), so overlap there is fine.
        The painter's depth (xf0.w) keeps the opaque pass correct whatever the order.
        Each batch's area is WebRender's BatchRects: the union of its draws' bounds, plus the
        bounds themselves once the union is a poor fit (larger than their sum); a long list of
        them reverts to the union alone, which only stops searches earlier.
        Tested and not kept: sorting the whole frame's draws (by group of non-overlapping draws in
        painter's order, then by pipeline) gave the same batches on a widget grid for 0.7 ms more
        encoding per full repaint.
        Rebuilds the target's batches over the new sequence, which starts at 'first' (in batch
        units) and returns the target's instance indices in that sequence. Encode thread.
    */
    std::vector<uint32_t> reorder (Target& target, uint32_t first) const
    {
        const auto sameState = [] (const Batch& a, const Batch& b)
        {
            return a.image == b.image && a.nearest == b.nearest && a.mask == b.mask && a.layer == b.layer
                && a.repeat == b.repeat && a.coveragePage == b.coveragePage && a.blend == b.blend && a.family == b.family;
        };

        // The distinct batch states, in order of first use.
        std::vector<Batch> states;
        std::vector<uint32_t> stateOfBatch;
        size_t total = 0;

        for (const auto& batch : target.batches)
        {
            auto it = std::find_if (states.begin(), states.end(), [&] (const Batch& s) { return sameState (s, batch); });

            if (it == states.end())
                it = states.insert (states.end(), batch);

            stateOfBatch.push_back ((uint32_t) std::distance (states.begin(), it));
            total += batch.count;
        }

        std::vector<uint32_t> sequence;
        sequence.reserve (total);

        // A single state: nothing to group, one batch in recording order.
        if (states.size() <= 1)
        {
            for (const auto& batch : target.batches)
                for (auto i = batch.first; i < batch.first + batch.count; ++i)
                    sequence.push_back (i);

            const auto segmented = std::any_of (target.batches.begin(), target.batches.end(), [] (const Batch& b) { return b.segmented; });
            target.batches = states;

            if (! states.empty())
            {
                target.batches.front().first = first;
                target.batches.front().count = (uint32_t) total;
                target.batches.front().segmented = segmented;
            }

            return sequence;
        }

        static constexpr size_t maxLookback = 8;
        static constexpr size_t maxAreaRects = 64;

        struct Merged
        {
            uint32_t state;
            bool segmented;                         // holds strokes (Batch::segmented)
            std::vector<uint32_t> instances;
            Rectangle<int> bounds;                  // union of the draws' bounds
            std::vector<Rectangle<int>> rects;      // the bounds themselves, when the union is a poor fit
            bool unionOnly = false;                 // too many rects: the union stands for them

            bool overlaps (Rectangle<int> r) const noexcept
            {
                if (! bounds.intersects (r))
                    return false;

                if (rects.empty())
                    return true;

                return std::any_of (rects.begin(), rects.end(), [r] (const auto& item) { return item.intersects (r); });
            }

            void addArea (Rectangle<int> r)
            {
                if (r.isEmpty())
                    return;

                const auto joined = bounds.getUnion (r);

                if (! unionOnly)
                {
                    const auto area = [] (Rectangle<int> a) { return (int64_t) a.getWidth() * a.getHeight(); };

                    if (! rects.empty())
                        rects.push_back (r);
                    else if (! bounds.isEmpty() && area (bounds) + area (r) < area (joined))
                        rects = { bounds, r };

                    if (rects.size() > maxAreaRects)
                    {
                        rects.clear();
                        unionOnly = true;
                    }
                }

                bounds = joined;
            }
        };

        std::vector<Merged> merged;

        for (size_t b = 0; b < target.batches.size(); ++b)
        {
            const auto& batch = target.batches[b];
            const auto state = stateOfBatch[b];

            for (auto i = batch.first; i < batch.first + batch.count; ++i)
            {
                const auto bounds = instanceBounds (instances[i]).getSmallestIntegerContainer().getIntersection (target.area);
                Merged* selected = nullptr;

                // The last batch with this state takes it without any overlap test (same as
                // drawing it right after).
                for (size_t k = merged.size(), steps = 0; k > 0 && steps < maxLookback; --k, ++steps)
                {
                    auto& candidate = merged[k - 1];

                    if (candidate.state == state)
                    {
                        selected = &candidate;
                        break;
                    }

                    if (candidate.overlaps (bounds))
                        break;
                }

                if (selected == nullptr)
                    selected = &merged.emplace_back (Merged { state, false, {}, {}, {} });

                selected->segmented |= batch.family == 0 && instances[i].params.y > 0.0f;
                selected->instances.push_back (i);
                selected->addArea (bounds);
            }
        }

        target.batches.clear();

        for (const auto& m : merged)
        {
            target.batches.push_back (states[m.state]);
            target.batches.back().first = first + (uint32_t) sequence.size();
            target.batches.back().count = (uint32_t) m.instances.size();
            target.batches.back().segmented = m.segmented;
            sequence.insert (sequence.end(), m.instances.begin(), m.instances.end());
        }

        return sequence;
    }

    void add (const Instance& instance, id<MTLTexture> image, bool nearest, int mask, int layer = -1, bool repeat = false,
              int coveragePage = -1, BlendMode blend = BlendMode::sourceOver)
    {
        auto& batches = currentBatches();
        const auto family = familyOf (instance);

        if (batches.empty() || batches.back().image != image || batches.back().nearest != nearest
             || batches.back().mask != mask || batches.back().layer != layer || batches.back().repeat != repeat
             || batches.back().coveragePage != coveragePage || batches.back().blend != blend || batches.back().family != family)
            batches.push_back ({ image, nearest, (uint32_t) instances.size(), 0, mask, layer, repeat, coveragePage, blend, family });

        batches.back().segmented |= family == 0 && instance.params.y > 0.0f;

        if (isOpaqueFill (instance, image, mask, layer, blend))
            targets[(size_t) openTargets.back()].opaque.push_back ((uint32_t) instances.size());

        instances.push_back (instance);
        instances.back().xf0.w = (float) (instances.size() - 1); // painter's order (exact below 2^24)
        ++batches.back().count;
        noteDrawn (instanceBounds (instance));
    }

    /*  The pipeline family an instance is drawn with (see 'family' in the shader). */
    static int familyOf (const Instance& i) noexcept
    {
        switch ((Kind) (int) i.params.z)
        {
            case Kind::glyph:         return 1;
            case Kind::rgbaTexture:
            case Kind::alphaTexture:  return 2;
            case Kind::coverage:      return 3;
            case Kind::box:
            case Kind::ellipse:       break;
        }

        return 0;
    }

    /*  Whether an instance's interior hides whatever was drawn under it: then that interior goes
        into the opaque pass (vsOpaque) and the GPU skips the earlier pixels it covers. Only what's
        simple to bound exactly: an unstroked box, rounded box or circle with an opaque solid
        colour, drawn source-over without a clip mask, with no rotation or shear. Its anti-aliased
        edge stays in the main pass (see vsOpaque).
    */
    static bool isOpaqueFill (const Instance& i, id<MTLTexture> image, int mask, int layer, BlendMode blend) noexcept
    {
        return ((int) i.params.z == (int) Kind::box || (int) i.params.z == (int) Kind::ellipse)
            && i.params.y == 0.0f              // not a stroke
            && i.params.w == 0.0f              // solid colour, no gradient or pattern
            && i.colour.w >= 1.0f              // opaque (premultiplied, opacity included)
            && image == nil && mask < 0 && layer < 0 && blend == BlendMode::sourceOver
            && i.xf0.y == 0.0f && i.xf1.x == 0.0f
            && i.xf0.x != 0.0f && i.xf1.y != 0.0f;
    }

    /*  Conservative device bounds of a quad instance: its shape (with the anti-aliasing ramp and
        half the stroke, as the vertex shader grows it), clipped to its clip rect.
    */
    static Rectangle<float> instanceBounds (const Instance& i) noexcept
    {
        const AffineTransform t (i.xf0.x, i.xf0.y, i.xf0.z, i.xf1.x, i.xf1.y, i.xf1.z);
        const auto maxScale = jmax (std::hypot (i.xf0.x, i.xf1.x), std::hypot (i.xf0.y, i.xf1.y));
        const auto isShape = i.params.z < 1.5f;
        const auto pad = (isShape ? 0.5f * i.params.y : 0.0f) * maxScale + 2.0f;
        return Rectangle<float> (i.rect.x, i.rect.y, i.rect.z, i.rect.w).transformedBy (t).expanded (pad)
                   .getIntersection ({ i.clip.x, i.clip.y, i.clip.z, i.clip.w });
    }

    void noteDrawn (Rectangle<float> r) noexcept
    {
        auto& drawn = targets[(size_t) openTargets.back()].drawn;
        drawn = drawn.isEmpty() ? r : (r.isEmpty() ? drawn : drawn.getUnion (r));
    }

    int beginLayer (Rectangle<int> area)
    {
        targets.push_back ({ area, pooledSize (area.getWidth()), pooledSize (area.getHeight()), {}, nil, {}, {} });
        openTargets.push_back ((int) targets.size() - 1);
        return openTargets.back();
    }

    int endLayer()
    {
        const auto index = openTargets.back();
        openTargets.pop_back();

        // A layer opens over the whole clip but often holds one small shape (an SVG group with an
        // opacity, say): keep only what was drawn. Nothing is encoded yet, so the draws follow.
        auto& target = targets[(size_t) index];
        target.area = target.area.getIntersection (target.drawn.getSmallestIntegerContainer());
        target.texWidth = pooledSize (target.area.getWidth());
        target.texHeight = pooledSize (target.area.getHeight());

        encodeOrder.push_back (index);
        return index;
    }

    std::vector<Target> targets;
    std::vector<int> openTargets, encodeOrder;
    std::vector<Instance> instances, clears;
    std::vector<GradientInfo> gradients;
    std::vector<Float4> gradientStops; // per stop: (position, 0, 0, 0), then its colour (not premultiplied)
    std::vector<Mask> masks;
    std::vector<CoverageJob> coverageJobs;
    std::vector<CoverageSegment> coverageSegments; // filled by resolveCoverage
    std::vector<StrokePoint> strokePoints;          // the same, for strokeAccumVs
    std::vector<StrokeSub> strokeSubs;
    std::vector<StrokeJobData> strokeJobs;
    std::vector<StrokeQuad> strokeQuads;            // the quads of strokeAccumVs
    std::vector<CoverageEntry> coverageEntries;
    std::vector<CoveragePage> coveragePages;
    int shelfX = 0, shelfY = 0, shelfHeight = 0;
    GlyphOutlineCache* glyphOutlines = nullptr;
    DeferredReleases* frameReleases = nullptr; // per-frame textures (software tiles)
    id<MTLDevice> device = nil;
    GlyphAtlas* glyphAtlas = nullptr;
    std::vector<GlyphAtlas::RasterJob> glyphJobs;  // glyphs this frame adds to the atlas
    ImageTextureCache* imageCache = nullptr;
    std::map<std::string, int> fallbackReasons; // only filled when statistics are enabled
    bool collectReasons = false;
    id<MTLTexture> destination = nil;   // an image's texture, drawn into (nil: a window's)
    bool clearMain = false;             // the main target starts cleared (else it keeps its pixels)
};

//==============================================================================
/*  The shared buffers a recorded scene is uploaded to for its encoding (SceneEncoder::encode),
    reused frame after frame once the GPU has read them.
*/
struct EncodeBuffers
{
    id<MTLBuffer> instances = nil, gradients = nil, coverageSegments = nil, coverageEntries = nil, opaqueIndices = nil, gradientStops = nil;
    id<MTLBuffer> strokePoints = nil, strokeSubs = nil, strokeJobs = nil, strokeQuads = nil;

    void release()
    {
        for (auto b : { instances, gradients, coverageSegments, coverageEntries, opaqueIndices, gradientStops,
                        strokePoints, strokeSubs, strokeJobs, strokeQuads })
            [b release];

        *this = {};
    }
};

/*  What a window's statistics (JUCE_METAL_RENDERER_STATS) take from the encoding. */
struct EncodeStatistics
{
    std::atomic<int64> phaseMicros[3] {};
    std::atomic<double> frameGpuStart { 0.0 };
    int64 drawCallsTotal = 0; // instance batches after reordering, encode queue
};

/*  GPU work on a MetalPixelData's texture, queued in order (ImageQueue) and encoded before the
    passes of the next window frame, in its command buffer, or earlier when the image's pixels
    are read (ImageQueue::readBack).
*/
struct ImageJob
{
    enum class Kind { render, blur, copy, upload, clear };

    ImageJob() = default;
    ~ImageJob() { reset(); buffers.release(); }

    // Once its GPU work has completed: ready to be reused.
    void reset()
    {
        [texture release];
        [source release];
        [staging release];
        texture = source = nil;
        staging = nil;
        releases.releaseAll();
    }

    Kind kind = Kind::render;
    id<MTLTexture> texture = nil;     // the image's texture, written (retained)
    id<MTLTexture> source = nil;      // copy: the texture read (retained)
    id<MTLBuffer> staging = nil;      // upload: the pixels, packed rows (retained)
    NSUInteger stagingBytesPerRow = 0;
    Rectangle<int> area;              // blur
    int passes = 0;                   // blur: 3-tap passes along each line
    FrameRecorder recorder;           // render: what a MetalGraphicsContext drew
    EncodeBuffers buffers;            // render
    DeferredReleases releases;        // render: its software tiles (see endDelegate)

    JUCE_DECLARE_NON_COPYABLE (ImageJob)
};

//==============================================================================
/*  Encodes recorded scenes and image jobs into command buffers (encode queue only). */
class SceneEncoder
{
public:
    explicit SceneEncoder (SharedResources& s) : shared (s) {}

    ~SceneEncoder()
    {
        for (auto& p : texturePool)
            [p.texture release];
    }

    /*  Mask and layer textures, reused across frames (as the OpenGL renderer keeps its
        framebuffers). A texture goes back to the pool once the scene that used it has completed;
        pooled textures unused for 120 scenes of the encoder's owner are released.
        The smallest pooled texture at least as large as asked is reused (up to four times the
        area): layers and clips change size from frame to frame as their content moves, and
        creating a Metal texture is costly. Passes are limited to the size asked
        (renderTargetWidth / Height), so the rest of a larger texture costs no GPU time.
    */
    struct PooledTexture { id<MTLTexture> texture; uint64_t lastUsed; };

    id<MTLTexture> acquireTexture (std::vector<id<MTLTexture>>& pooled, int width, int height, MTLPixelFormat format,
                                   MTLTextureUsage usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
                                   MTLStorageMode storage = MTLStorageModePrivate, bool exactSize = false)
    {
        const auto areaOf = [] (id<MTLTexture> t) { return (int64) t.width * (int64) t.height; };
        const auto maxArea = 4 * (int64) width * (int64) height;
        auto best = texturePool.end();

        for (auto it = texturePool.begin(); it != texturePool.end(); ++it)
        {
            const auto t = it->texture;

            const auto sizeFits = exactSize ? ((int) t.width == width && (int) t.height == height)
                                            : ((int) t.width >= width && (int) t.height >= height);

            if (sizeFits && t.pixelFormat == format && t.usage == usage && t.storageMode == storage
                 && areaOf (t) <= maxArea && (best == texturePool.end() || areaOf (t) < areaOf (best->texture)))
            {
                best = it;
            }
        }

        if (best != texturePool.end())
        {
            auto texture = best->texture;
            texturePool.erase (best);
            pooled.push_back (texture);
            return texture;
        }

        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: format
                                                                        width: (NSUInteger) width
                                                                       height: (NSUInteger) height
                                                                    mipmapped: NO];
        desc.usage = usage;
        desc.storageMode = storage;
        auto texture = [shared.device newTextureWithDescriptor: desc];
        pooled.push_back (texture);
        return texture;
    }

    /*  'pooled' (acquired by a scene whose GPU work has completed) back to the pool, at 'now' (a
        count of the scenes of the pool's owner).
    */
    void recycle (std::vector<id<MTLTexture>>& pooled, uint64_t now)
    {
        for (auto t : pooled)
            texturePool.push_back ({ t, now });

        pooled.clear();

        for (auto it = texturePool.begin(); it != texturePool.end();)
        {
            if (now - it->lastUsed > 120)
            {
                [it->texture release];
                it = texturePool.erase (it);
            }
            else
            {
                ++it;
            }
        }
    }

    void encodeJob (ImageJob& job, std::vector<id<MTLTexture>>& pooled, id<MTLCommandBuffer>& commandBuffer)
    {
        id<MTLTexture> texture = job.texture;
        const auto origin = MTLOriginMake (0, 0, 0);
        const auto size = MTLSizeMake (texture.width, texture.height, 1);

        switch (job.kind)
        {
            case ImageJob::Kind::render:
                encode (job.buffers, pooled, job.recorder, commandBuffer, texture, nullptr);
                return;

            case ImageJob::Kind::blur:
                encodeBlur (job, pooled, commandBuffer);
                return;

            case ImageJob::Kind::clear:
            {
                auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = texture;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].clearColor = MTLClearColorMake (0, 0, 0, 0);
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor: pass];
                encoder.label = @"JUCE image clear";
                [encoder endEncoding];
                return;
            }

            case ImageJob::Kind::copy:
            {
                id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
                blit.label = @"JUCE image copy";
                [blit copyFromTexture: job.source sourceSlice: 0 sourceLevel: 0 sourceOrigin: origin sourceSize: size
                            toTexture: texture destinationSlice: 0 destinationLevel: 0 destinationOrigin: origin];
                [blit endEncoding];
                return;
            }

            case ImageJob::Kind::upload:
            {
                id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
                blit.label = @"JUCE image upload";
                [blit copyFromBuffer: job.staging sourceOffset: 0 sourceBytesPerRow: job.stagingBytesPerRow
                 sourceBytesPerImage: job.stagingBytesPerRow * texture.height sourceSize: size
                           toTexture: texture destinationSlice: 0 destinationLevel: 0 destinationOrigin: origin];
                [blit endEncoding];
                return;
            }
        }
    }

    // Makes a scene's shared buffer hold at least 'bytes' (it grows to twice that, so that a
    // growing scene doesn't reallocate it every frame).
    void reserveBuffer (id<MTLBuffer>& buffer, size_t bytes)
    {
        bytes = jmax ((size_t) 256, bytes);

        if (buffer == nil || buffer.length < bytes)
        {
            [buffer release];
            buffer = [shared.device newBufferWithLength: bytes * 2 options: MTLResourceStorageModeShared];
        }
    }

    template <typename Items>
    void upload (id<MTLBuffer>& buffer, const Items& items)
    {
        using Item = typename Items::value_type;
        reserveBuffer (buffer, items.size() * sizeof (Item));
        std::copy (items.begin(), items.end(), static_cast<Item*> (buffer.contents));
    }

    /*  Encodes a recorded scene into 'commandBuffer' (which statistics may commit and replace, to
        time the phases apart): its coverage, masks and targets, the main one drawn over
        'destination'. The textures it takes from the pool are added to 'pooled', its uploads go
        into 'buffers': both must stay until its GPU work completes.
    */
    void encode (EncodeBuffers& buffers, std::vector<id<MTLTexture>>& pooled, FrameRecorder& recorder,
                 id<MTLCommandBuffer>& commandBuffer, id<MTLTexture> destination, const std::shared_ptr<EncodeStatistics>& stats)
    {
        const auto& clears = recorder.clears;
        const auto& instances = recorder.instances;
        reserveBuffer (buffers.instances, (clears.size() + instances.size()) * sizeof (MetalRendering::Instance));

        // The clears, then each target's instances in their reordered sequence (see
        // FrameRecorder::reorder); 'positionOf' maps a recorded instance to its buffer position.
        auto* dst = static_cast<MetalRendering::Instance*> (buffers.instances.contents);
        std::copy (clears.begin(), clears.end(), dst);
        std::vector<uint32_t> positionOf (instances.size(), 0);
        uint32_t next = 0;

        for (auto& target : recorder.targets)
        {
            for (auto index : recorder.reorder (target, next))
            {
                dst[clears.size() + next] = instances[index];
                positionOf[index] = (uint32_t) (clears.size() + next);
                ++next;
            }

            if (stats != nullptr)
                stats->drawCallsTotal += (int64) target.batches.size();
        }

        upload (buffers.gradients, recorder.gradients);
        upload (buffers.gradientStops, recorder.gradientStops);

        // Coverage atlas pages first: path fills read them.
        using FR = MetalRendering::FrameRecorder;
        upload (buffers.coverageSegments, recorder.coverageSegments);
        upload (buffers.coverageEntries, recorder.coverageEntries);

        // Stroke expansion's input.
        if (! recorder.strokePoints.empty())
        {
            upload (buffers.strokePoints, recorder.strokePoints);
            upload (buffers.strokeSubs, recorder.strokeSubs);
            upload (buffers.strokeJobs, recorder.strokeJobs);
            upload (buffers.strokeQuads, recorder.strokeQuads);
        }

        // Statistics: the coverage and mask phases go in their own command buffers, timed apart.
        const auto endPhase = [&] (int phase)
        {
            if (auto phaseStats = stats)
            {
                [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> done)
                {
                    phaseStats->phaseMicros[phase] += (int64) ((done.GPUEndTime - done.GPUStartTime) * 1.0e6);

                    if (phase == 0)
                        phaseStats->frameGpuStart = done.GPUStartTime;
                }];
                [commandBuffer commit];
                commandBuffer = [shared.queue commandBuffer];
            }
        };

        for (auto& page : recorder.coveragePages)
        {
            const auto texWidth = FR::pooledSize (page.width), texHeight = FR::pooledSize (page.height);
            page.deltas = acquireTexture (pooled, texWidth, texHeight, MTLPixelFormatR32Float);
            // The summed winding numbers fit half floats (2^-10 below 2, under 1/255) for the
            // non-zero rule, clamped at 1: only the deltas need 32 bits. The even-odd rule reads
            // the winding modulo 2, which half floats lose with many overlapping layers (a step of
            // 2^-6 from 16 on): pages holding even-odd paths keep 32 bits.
            page.coverage = acquireTexture (pooled, texWidth, texHeight, page.evenOdd ? MTLPixelFormatR32Float : MTLPixelFormatR16Float,
                                            MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead);

            auto* accumPass = [MTLRenderPassDescriptor renderPassDescriptor];
            accumPass.colorAttachments[0].texture = page.deltas;
            accumPass.colorAttachments[0].loadAction = MTLLoadActionClear;
            accumPass.colorAttachments[0].clearColor = MTLClearColorMake (0, 0, 0, 0);
            accumPass.colorAttachments[0].storeAction = MTLStoreActionStore;
            accumPass.renderTargetWidth = (NSUInteger) texWidth;
            accumPass.renderTargetHeight = (NSUInteger) texHeight;

            id<MTLRenderCommandEncoder> accumEncoder = [commandBuffer renderCommandEncoderWithDescriptor: accumPass];
            accumEncoder.label = @"JUCE path coverage accumulation";
            const float atlasSize[2] { (float) texWidth, (float) texHeight };
            [accumEncoder setRenderPipelineState: shared.accumPipeline];
            [accumEncoder setVertexBuffer: buffers.coverageSegments offset: 0 atIndex: 0];
            [accumEncoder setFragmentBuffer: buffers.coverageSegments offset: 0 atIndex: 0];
            [accumEncoder setVertexBytes: atlasSize length: sizeof (atlasSize) atIndex: 1];

            // A page whose paths have no edge inside their areas reads as zero coverage.
            if (page.segmentCount > 0)
                [accumEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 4
                               instanceCount: page.segmentCount baseInstance: page.firstSegment];

            // Strokes: their quads (up to eight edges each), built by the vertex shader (strokeAccumVs).
            if (page.strokeQuadCount > 0)
            {
                [accumEncoder setRenderPipelineState: shared.strokeAccumPipeline];
                [accumEncoder setVertexBuffer: buffers.strokePoints offset: 0 atIndex: 0];
                [accumEncoder setVertexBytes: &page.firstStrokeQuad length: sizeof (page.firstStrokeQuad) atIndex: 2];
                [accumEncoder setVertexBuffer: buffers.strokeSubs offset: 0 atIndex: 3];
                [accumEncoder setVertexBuffer: buffers.strokeJobs offset: 0 atIndex: 4];
                [accumEncoder setVertexBuffer: buffers.strokeQuads offset: 0 atIndex: 5];
                [accumEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 4
                               instanceCount: page.strokeQuadCount];
            }
            [accumEncoder endEncoding];
        }

        endPhase (0);

        if (! recorder.coveragePages.empty())
        {
            // One SIMD group per row, a few rows per threadgroup.
            id<MTLComputeCommandEncoder> sumEncoder = [commandBuffer computeCommandEncoder];
            sumEncoder.label = @"JUCE path coverage prefix sums";
            const auto simdWidth = jmax ((NSUInteger) 1, shared.prefixSumPipeline.threadExecutionWidth);
            const auto rowsPerGroup = jmax ((NSUInteger) 1, (NSUInteger) 128 / simdWidth);
            [sumEncoder setComputePipelineState: shared.prefixSumPipeline];

            for (const auto& page : recorder.coveragePages)
            {
                const uint32_t counts[2] { page.entryCount, page.rowCount };
                [sumEncoder setTexture: page.deltas atIndex: 0];
                [sumEncoder setTexture: page.coverage atIndex: 1];
                [sumEncoder setBuffer: buffers.coverageEntries offset: page.firstEntry * sizeof (FR::CoverageEntry) atIndex: 0];
                [sumEncoder setBytes: counts length: sizeof (counts) atIndex: 1];
                [sumEncoder dispatchThreadgroups: MTLSizeMake ((page.rowCount + rowsPerGroup - 1) / rowsPerGroup, 1, 1)
                           threadsPerThreadgroup: MTLSizeMake (rowsPerGroup * simdWidth, 1, 1)];
            }

            [sumEncoder endEncoding];
        }

        endPhase (1);

        // Clip masks first: they only depend on geometry, and a mask's parent precedes it.
        auto& masks = recorder.masks;
        struct MaskInfo { MetalRendering::Float4 bounds, params; };

        const auto maskInfo = [&masks] (int index) -> MaskInfo
        {
            if (index < 0)
                return {};

            const auto& m = masks[(size_t) index];
            const MetalRendering::Float4 bounds { (float) m.area.getX(), (float) m.area.getY(),
                                                  (float) m.area.getRight(), (float) m.area.getBottom() };

            if (m.readsCoverage())
                return { bounds, { m.inv0.x, m.inv0.y, m.inv0.z > 0.5f ? 3.0f : 2.0f, 0.0f } };

            return { bounds, { (float) -m.area.getX(), (float) -m.area.getY(), 1.0f, 0.0f } };
        };

        const auto maskTexture = [&] (int index)
        {
            if (index < 0)
                return shared.noMask;

            const auto& m = masks[(size_t) index];
            return m.readsCoverage() ? recorder.coveragePages[(size_t) m.coveragePage].coverage : m.texture;
        };

        for (size_t i = 0; i < masks.size(); ++i)
        {
            auto& m = masks[i];

            if (m.readsCoverage())
                continue;

            m.texture = acquireTexture (pooled, m.texWidth, m.texHeight, MTLPixelFormatR8Unorm);

            auto* maskPass = [MTLRenderPassDescriptor renderPassDescriptor];
            maskPass.colorAttachments[0].texture = m.texture;
            maskPass.colorAttachments[0].loadAction = MTLLoadActionClear;
            maskPass.colorAttachments[0].clearColor = MTLClearColorMake (0, 0, 0, 0);
            maskPass.colorAttachments[0].storeAction = MTLStoreActionStore;
            maskPass.renderTargetWidth = (NSUInteger) m.texWidth;
            maskPass.renderTargetHeight = (NSUInteger) m.texHeight;

            id<MTLRenderCommandEncoder> maskEncoder = [commandBuffer renderCommandEncoderWithDescriptor: maskPass];
            maskEncoder.label = @"JUCE clip mask";
            const auto origin = m.area.getPosition().toFloat();
            const auto parent = maskInfo (m.parent);
            const MetalRendering::Float4 info[6] { { origin.x, origin.y, 0.0f, 0.0f }, parent.bounds, parent.params, m.inv0, m.inv1,
                                                   { m.imageWidth, m.imageHeight, m.imageIsAlphaOnly ? 1.0f : 0.0f, 0.0f } };
            [maskEncoder setFragmentBytes: info length: sizeof (info) atIndex: 0];
            [maskEncoder setFragmentTexture: maskTexture (m.parent) atIndex: 3];

            if (m.coveragePage >= 0)
            {
                [maskEncoder setRenderPipelineState: shared.maskCoveragePipeline];
                [maskEncoder setFragmentTexture: recorder.coveragePages[(size_t) m.coveragePage].coverage atIndex: 4];
                [maskEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 4];
            }
            else
            {
                [maskEncoder setRenderPipelineState: shared.maskImagePipeline];
                [maskEncoder setFragmentTexture: m.image atIndex: 1];
                [maskEncoder setFragmentSamplerState: m.nearest ? shared.nearestSampler : shared.linearSampler atIndex: 0];
                [maskEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 4];
            }

            [maskEncoder endEncoding];
        }

        endPhase (2);

        // Opaque interiors of every target, as indices into the instance buffer (after the clears).
        std::vector<size_t> opaqueOffsets;
        {
            size_t total = 0;

            for (const auto& target : recorder.targets)
                total += target.opaque.size();

            reserveBuffer (buffers.opaqueIndices, total * sizeof (uint32_t));

            auto* opaqueDst = static_cast<uint32_t*> (buffers.opaqueIndices.contents);
            size_t offset = 0;

            for (const auto& target : recorder.targets)
            {
                opaqueOffsets.push_back (offset);

                // Front to back (the latest draw first), as WebRender orders its opaque batches: GPUs
                // without hidden-surface removal then reject the earlier ones by depth.
                for (auto it = target.opaque.rbegin(); it != target.opaque.rend(); ++it)
                    opaqueDst[offset++] = positionOf[*it];
            }
        }

        // Painter's depth: one step per instance in the buffer (see painterDepth in the shader).
        const float depthStep = 1.0f / (float) (clears.size() + instances.size() + 2);

        // The depth buffer only lives during each pass: in tile memory where the GPU has it.
        const auto depthStorage = [this]
        {
            if (@available (macOS 11.0, *))
                if ([shared.device supportsFamily: MTLGPUFamilyApple1])
                    return MTLStorageModeMemoryless;

            return MTLStorageModePrivate;
        }();

        for (const auto targetIndex : recorder.encodeOrder)
        {
            auto& target = recorder.targets[(size_t) targetIndex];
            const auto isMain = targetIndex == 0;

            if (target.area.isEmpty() || (target.batches.empty() && ! (isMain && (! clears.empty() || recorder.clearMain))))
                continue;

            const auto& pipelines = shared.getDrawPipelines (isMain && destination.pixelFormat == MTLPixelFormatR8Unorm);

            auto* pass = [MTLRenderPassDescriptor renderPassDescriptor];

            if (! isMain)
            {
                target.texture = acquireTexture (pooled, target.texWidth, target.texHeight, MTLPixelFormatBGRA8Unorm);
                pass.renderTargetWidth = (NSUInteger) target.texWidth;
                pass.renderTargetHeight = (NSUInteger) target.texHeight;

                // The composite samples the whole texture: its coordinates follow the size acquired.
                if ((int) target.texture.width != target.texWidth || (int) target.texture.height != target.texHeight)
                {
                    const MetalRendering::Float4 uv { 0.0f, 0.0f, (float) target.area.getWidth() / (float) target.texture.width,
                                                      (float) target.area.getHeight() / (float) target.texture.height };
                    auto* uploaded = static_cast<MetalRendering::Instance*> (buffers.instances.contents) + clears.size();

                    for (const auto& other : recorder.targets)
                        for (const auto& batch : other.batches)
                            if (batch.layer == targetIndex)
                                for (auto i = batch.first; i < batch.first + batch.count; ++i)
                                    uploaded[i].uv = uv;
                }
            }

            pass.colorAttachments[0].texture = isMain ? destination : target.texture;
            pass.colorAttachments[0].loadAction = isMain && ! recorder.clearMain ? MTLLoadActionLoad : MTLLoadActionClear;
            pass.colorAttachments[0].clearColor = MTLClearColorMake (0, 0, 0, 0);
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;

            // Same size as the colour attachment; cleared to the far plane, never stored.
            id<MTLTexture> colour = pass.colorAttachments[0].texture;
            pass.depthAttachment.texture = acquireTexture (pooled, (int) colour.width, (int) colour.height,
                                                           SharedResources::depthFormat,
                                                           MTLTextureUsageRenderTarget, depthStorage, true);
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.clearDepth = 1.0;
            pass.depthAttachment.storeAction = MTLStoreActionDontCare;

            id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor: pass];
            encoder.label = ! isMain ? @"JUCE transparency layer" : (recorder.destination != nil ? @"JUCE image" : @"JUCE window");
            const auto origin = target.area.getPosition().toFloat();
            const float viewport[4] { (float) (isMain ? target.area.getWidth() : target.texWidth),
                                      (float) (isMain ? target.area.getHeight() : target.texHeight), origin.x, origin.y };
            const float targetOrigin[2] { origin.x, origin.y };
            [encoder setVertexBuffer: buffers.instances offset: 0 atIndex: 0];
            [encoder setFragmentBuffer: buffers.instances offset: 0 atIndex: 0];
            [encoder setVertexBytes: viewport length: sizeof (viewport) atIndex: 1];
            [encoder setVertexBytes: &depthStep length: sizeof (depthStep) atIndex: 2];
            [encoder setFragmentBytes: targetOrigin length: sizeof (targetOrigin) atIndex: 3];

            // The opaque interiors first (colour and depth): what they cover, and their own
            // interiors, are then rejected before shading.
            if (! target.opaque.empty())
            {
                [encoder setRenderPipelineState: pipelines.opaque];
                [encoder setDepthStencilState: shared.depthWriteState];
                [encoder setVertexBuffer: buffers.opaqueIndices offset: opaqueOffsets[(size_t) targetIndex] * sizeof (uint32_t) atIndex: 3];
                [encoder drawPrimitives: MTLPrimitiveTypeTriangleStrip vertexStart: 0 vertexCount: 4
                          instanceCount: target.opaque.size()];
            }

            [encoder setDepthStencilState: shared.depthTestState];

            if (isMain && ! clears.empty())
            {
                [encoder setRenderPipelineState: pipelines.clear];
                // Boxes without strokes: one quad each (see the vertex shader 'vs').
                [encoder drawPrimitives: MTLPrimitiveTypeTriangle vertexStart: 0 vertexCount: 6
                          instanceCount: clears.size() baseInstance: 0];
            }

            [encoder setFragmentTexture: shared.glyphAtlas->getTexture() atIndex: 0];
            [encoder setFragmentBuffer: buffers.gradientStops offset: 0 atIndex: 4];
            [encoder setFragmentBuffer: buffers.gradients offset: 0 atIndex: 1];
            id<MTLRenderPipelineState> boundPipeline = nil;
            int boundMask = -2, boundCoveragePage = -2;

            for (const auto& batch : target.batches)
            {
                if (batch.mask != boundMask)
                {
                    const auto info = maskInfo (batch.mask);
                    [encoder setFragmentTexture: maskTexture (batch.mask) atIndex: 3];
                    [encoder setFragmentBytes: &info length: sizeof (info) atIndex: 2];
                    boundMask = batch.mask;
                }

                if (batch.coveragePage != boundCoveragePage)
                {
                    [encoder setFragmentTexture: batch.coveragePage >= 0 ? recorder.coveragePages[(size_t) batch.coveragePage].coverage
                                                                         : shared.noMask
                                        atIndex: 4];
                    boundCoveragePage = batch.coveragePage;
                }

                auto pipeline = pipelines.blend[batch.family][(size_t) batch.blend];

                if (pipeline != boundPipeline)
                {
                    [encoder setRenderPipelineState: pipeline];

                    if (boundPipeline == nil)
                    {
                        [encoder setVertexBuffer: buffers.instances offset: 0 atIndex: 0];
                        [encoder setFragmentBuffer: buffers.instances offset: 0 atIndex: 0];
                    }

                    boundPipeline = pipeline;
                }

                const auto image = batch.layer >= 0 ? recorder.targets[(size_t) batch.layer].texture : batch.image;
                [encoder setFragmentTexture: image atIndex: 1];
                [encoder setFragmentSamplerState: shared.getSampler (batch.nearest, batch.repeat) atIndex: 0];
                [encoder drawPrimitives: MTLPrimitiveTypeTriangle vertexStart: 0 vertexCount: batch.verticesPerInstance()
                          instanceCount: batch.count baseInstance: clears.size() + batch.first];
            }

            [encoder endEncoding];
        }
    }

private:
    /*  The software renderer's single-channel box blur (see boxBlurLines in the shader) over
        job.area: along its rows into a scratch texture, then along its columns back.
        Measured on an M1, 36 drop shadows of radius 12 (device pixels) per frame: lines held in
        threadgroup memory, two barriers per pass, took ~4 ms of GPU per frame, whether that
        memory was a fixed 16 KB or sized to the line; most lines now go through the SIMD-group
        version, and shadows drawn again come from the cache (ImageQueue::findBlurred).
    */
    void encodeBlur (const ImageJob& job, std::vector<id<MTLTexture>>& pooled, id<MTLCommandBuffer> commandBuffer)
    {
        const auto& area = job.area;
        auto scratch = acquireTexture (pooled, area.getWidth(), area.getHeight(), MTLPixelFormatR8Unorm,
                                       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

        const auto blurLines = [&] (id<MTLTexture> src, id<MTLTexture> dst, Point<int> srcOrigin, Point<int> dstOrigin, bool vertical)
        {
            const auto length = (NSUInteger) (vertical ? area.getHeight() : area.getWidth());
            const auto lines = (NSUInteger) (vertical ? area.getWidth() : area.getHeight());
            const uint32_t params[8] { (uint32_t) srcOrigin.x, (uint32_t) srcOrigin.y, (uint32_t) dstOrigin.x, (uint32_t) dstOrigin.y,
                                       (uint32_t) length, vertical ? 1u : 0u, (uint32_t) job.passes, (uint32_t) lines };
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            encoder.label = vertical ? @"JUCE box blur (columns)" : @"JUCE box blur (rows)";
            [encoder setTexture: src atIndex: 0];
            [encoder setTexture: dst atIndex: 1];
            [encoder setBytes: params length: sizeof (params) atIndex: 0];

            if (const auto simdWidth = shared.blurSimdPipeline.threadExecutionWidth;
                length <= simdWidth * SharedResources::blurPixelsPerLane
                 && shared.blurSimdPipeline.maxTotalThreadsPerThreadgroup >= 4 * simdWidth)
            {
                // Four lines (SIMD groups) per threadgroup.
                [encoder setComputePipelineState: shared.blurSimdPipeline];
                [encoder dispatchThreadgroups: MTLSizeMake ((lines + 3) / 4, 1, 1)
                        threadsPerThreadgroup: MTLSizeMake (4 * simdWidth, 1, 1)];
            }
            else
            {
                const auto threads = jmin (SharedResources::blurThreads, (length + 31) / 32 * 32);
                [encoder setComputePipelineState: shared.blurPipeline];
                [encoder setThreadgroupMemoryLength: (length + 15) / 16 * 16 atIndex: 0];
                [encoder dispatchThreadgroups: MTLSizeMake (lines, 1, 1)
                        threadsPerThreadgroup: MTLSizeMake (threads, 1, 1)];
            }

            [encoder endEncoding];
        };

        // Two encoders: the columns read what the rows wrote.
        blurLines (job.texture, scratch, area.getPosition(), {}, false);
        blurLines (scratch, job.texture, {}, area.getPosition(), true);
    }

    SharedResources& shared;
    std::vector<PooledTexture> texturePool;
};

//==============================================================================
/*  The textures of GPU images (MetalPixelData) and the work queued on them.
    Work on an image's texture (a MetalGraphicsContext drawing into it, a blur, an upload of
    pixels the CPU wrote) is queued on the message thread while a window paints, then encoded at
    the start of that window frame's command buffer, before the passes that draw those images:
    one encode queue and one command queue, so the GPU runs everything in the order it was
    recorded, and the frame's number covers the caches the work used (glyph atlas, image
    textures). Reading an image's pixels (readBack) encodes and commits the queued work on the
    encode queue, after the frames encoded before it, and waits for the GPU.
*/
class ImageQueue
{
public:
    ImageQueue() = default;

    ~ImageQueue()
    {
        // Frames drawing images hold this object, and wait for their GPU work: what is still
        // queued was never drawn.
        dispatch_sync (shared->encodeQueue, ^{});
        pending.clear();
        spare.clear();

        for (auto& [key, entry] : blurred)
            [entry.texture release];

        for (auto& p : texturePool)
            [p.texture release];
    }

    // Message thread: the recording of the window being painted (MetalLayerRenderer).
    FrameRecorder* activeRecorder = nullptr;

    std::unique_ptr<ImageJob> createJob (ImageJob::Kind kind)
    {
        std::unique_ptr<ImageJob> job;

        {
            const ScopedLock sl (lock);

            if (! spare.empty())
            {
                job = std::move (spare.back());
                spare.pop_back();
            }
        }

        if (job == nullptr)
            job = std::make_unique<ImageJob>();

        job->kind = kind;
        return job;
    }

    void submit (std::unique_ptr<ImageJob> job)
    {
        const ScopedLock sl (lock);
        pending.push_back (std::move (job));
    }

    std::vector<std::unique_ptr<ImageJob>> takePending()
    {
        const ScopedLock sl (lock);
        return std::exchange (pending, {});
    }

    // Jobs whose GPU work has completed.
    void recycle (std::vector<std::unique_ptr<ImageJob>>& jobs)
    {
        for (auto& job : jobs)
            job->reset();

        const ScopedLock sl (lock);

        for (auto& job : jobs)
            if (spare.size() < 256)
                spare.push_back (std::move (job));

        jobs.clear();
    }

    /*  Encode queue, before encoding queued jobs: their glyphs rasterised, their paths flattened.
        'window': a frame still being recorded whose glyphs the jobs may use (see readBack).
    */
    static void prepare (SharedResources& shared, std::vector<std::unique_ptr<ImageJob>>& jobs, const FrameRecorder* window)
    {
        if (window != nullptr)
            for (const auto& glyph : window->glyphJobs)
                shared.glyphAtlas->rasterise (glyph);

        for (auto& job : jobs)
        {
            if (job->kind != ImageJob::Kind::render)
                continue;

            for (const auto& glyph : job->recorder.glyphJobs)
                shared.glyphAtlas->rasterise (glyph);

            job->recorder.resolveCoverage();
        }
    }

    /*  Encodes and commits the queued work, then copies 'texture' into a new shared buffer with
        rows of 'bytesPerRow', and returns it (owned by the caller) once the GPU is done.
    */
    id<MTLBuffer> readBack (id<MTLTexture> texture, NSUInteger bytesPerRow)
    {
        auto jobs = takePending();
        auto* jobsToEncode = &jobs;

        // Glyphs the window being painted placed in the atlas are rasterised when its frame is
        // encoded, after this: queued drawing may use them (the same text in a buffered child).
        const FrameRecorder* window = MessageManager::existsAndIsCurrentThread() ? activeRecorder : nullptr;
        __block id<MTLBuffer> result = nil;

        dispatch_sync (shared->encodeQueue, ^
        {
            @autoreleasepool
            {
                prepare (*shared, *jobsToEncode, window);
                id<MTLCommandBuffer> commandBuffer = [shared->queue commandBuffer];

                for (auto& job : *jobsToEncode)
                    encoder.encodeJob (*job, pooled, commandBuffer);

                result = [shared->device newBufferWithLength: bytesPerRow * texture.height options: MTLResourceStorageModeShared];

                if (result != nil)
                {
                    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
                    blit.label = @"JUCE image read-back";
                    [blit copyFromTexture: texture sourceSlice: 0 sourceLevel: 0 sourceOrigin: MTLOriginMake (0, 0, 0)
                               sourceSize: MTLSizeMake (texture.width, texture.height, 1)
                                 toBuffer: result destinationOffset: 0 destinationBytesPerRow: bytesPerRow
                  destinationBytesPerImage: bytesPerRow * texture.height];
                    [blit endEncoding];
                }

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                encoder.recycle (pooled, shared->framesRecorded.load());
            }
        });

        recycle (jobs);
        ++readBacksSinceLastStatistics;
        return result;
    }

    /*  Message thread: takes the job queued last out of the queue if it is a render into
        'texture' (nullptr else). Out of the queue, a read-back can't encode it meanwhile; the
        caller submits it again (it was last, and only the message thread queues jobs), or drops
        it with recycle().
    */
    std::unique_ptr<ImageJob> takeLastRender (id<MTLTexture> texture)
    {
        const ScopedLock sl (lock);

        if (pending.empty() || pending.back()->kind != ImageJob::Kind::render || pending.back()->texture != texture)
            return {};

        auto job = std::move (pending.back());
        pending.pop_back();
        return job;
    }

    void recycle (std::unique_ptr<ImageJob> job)
    {
        std::vector<std::unique_ptr<ImageJob>> jobs;
        jobs.push_back (std::move (job));
        recycle (jobs);
    }

    // Message thread: MetalGraphicsContexts drawing into images, not destroyed yet.
    int openContexts = 0;

    /*  Blurred contents (drop shadows), by the key of what was drawn and blurred: a shadow drawn
        again with the same path, area and radius reuses the texture blurred the first time,
        without drawing or blurring it again (as WebRender caches its blurred box shadows). Its
        pixels are those the first one computed, so it changes none. The
        textures are shared by the images using them (never written, never pooled), and kept
        while drawn: entries unused for 240 frames, or beyond 64 MB, are released.
        Message thread.
    */
    id<MTLTexture> findBlurred (uint64_t key)
    {
        const auto it = blurred.find (key);

        if (it == blurred.end())
            return nil;

        it->second.lastUsed = shared->framesRecorded.load();
        return it->second.texture;
    }

    void addBlurred (uint64_t key, id<MTLTexture> texture)
    {
        auto& entry = blurred[key];
        shared->imageCache->releaseLater (entry.texture);
        entry = { [texture retain], shared->framesRecorded.load() };
    }

    /*  An image texture of exactly that size: one released by an image whose frames have all
        completed, or a new one.
    */
    id<MTLTexture> acquireTexture (int width, int height, MTLPixelFormat format, MTLTextureUsage usage)
    {
        const auto completed = shared->framesCompleted.load();

        {
            const ScopedLock sl (lock);

            for (auto it = texturePool.begin(); it != texturePool.end(); ++it)
            {
                const auto t = it->texture;

                if ((int) t.width == width && (int) t.height == height && t.pixelFormat == format
                     && t.usage == usage && it->released <= completed)
                {
                    texturePool.erase (it);
                    return t;
                }
            }
        }

        auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: format
                                                                        width: (NSUInteger) width
                                                                       height: (NSUInteger) height
                                                                    mipmapped: NO];
        desc.usage = usage;
        desc.storageMode = MTLStorageModePrivate;
        return [shared->device newTextureWithDescriptor: desc];
    }

    /*  Any thread: an image's texture back to the pool, reused once the frames recorded so far
        (which may draw it) have completed. Its queued work holds it until encoded.
    */
    void releaseTexture (id<MTLTexture> texture)
    {
        if (texture == nil)
            return;

        const ScopedLock sl (lock);
        texturePool.push_back ({ texture, shared->framesRecorded.load() });
    }

    /*  Message thread, once per window frame: textures pooled for long (or too many) released,
        and blurred contents unused for long (or too many).
    */
    void trimTextures()
    {
        // The read-backs' pool (see readBack): released once unused for long, read-backs or not.
        if (shared->framesRecorded.load() % 120 == 0)
            dispatch_async (shared->encodeQueue, ^{ encoder.recycle (pooled, shared->framesRecorded.load()); });

        {
            const auto now = shared->framesRecorded.load();
            int64 bytes = 0;

            for (auto& [key, entry] : blurred)
                bytes += (int64) entry.texture.width * (int64) entry.texture.height
                           * (entry.texture.pixelFormat == MTLPixelFormatR8Unorm ? 1 : 4);

            while (! blurred.empty())
            {
                auto oldest = std::min_element (blurred.begin(), blurred.end(),
                                                [] (const auto& a, const auto& b) { return a.second.lastUsed < b.second.lastUsed; });

                if (now - oldest->second.lastUsed <= 240 && bytes <= 64 * 1024 * 1024)
                    break;

                const auto t = oldest->second.texture;
                bytes -= (int64) t.width * (int64) t.height * (t.pixelFormat == MTLPixelFormatR8Unorm ? 1 : 4);
                shared->imageCache->releaseLater (t); // frames in flight may still draw it
                blurred.erase (oldest);
            }
        }

        const auto completed = shared->framesCompleted.load();
        const auto recorded = shared->framesRecorded.load();
        const ScopedLock sl (lock);
        auto excess = (int) texturePool.size() - 64;

        for (auto it = texturePool.begin(); it != texturePool.end();)
        {
            if (it->released <= completed && (excess > 0 || recorded - it->released > 240))
            {
                [it->texture release];
                it = texturePool.erase (it);
                --excess;
            }
            else
            {
                ++it;
            }
        }
    }

    SharedResourcePointer<SharedResources> shared;
    std::atomic<int> readBacksSinceLastStatistics { 0 };

private:
    struct PooledImageTexture { id<MTLTexture> texture; uint64_t released; };
    struct BlurredContent { id<MTLTexture> texture = nil; uint64_t lastUsed = 0; };

    std::unordered_map<uint64_t, BlurredContent> blurred; // message thread

    CriticalSection lock;
    std::vector<std::unique_ptr<ImageJob>> pending, spare;
    std::vector<PooledImageTexture> texturePool;

    // Encode queue (readBack).
    SceneEncoder encoder { *shared };
    std::vector<id<MTLTexture>> pooled;
};

} // namespace MetalRendering

class MetalGraphicsContext;

//==============================================================================
/*  An image whose pixels live in a Metal texture, as Direct2DPixelData keeps them in Direct2D
    bitmaps: the type MetalGraphicsContext gives for temporary images (setBufferedToImage, drop
    shadows, component effects), drawn into, blurred and drawn on the GPU.
    - Its pixels are in the texture, or in a software copy (made when the CPU reads or writes
      them through BitmapData), or both; or in neither, while the image is clear. An image made
      without clearing is cleared all the same.
    - Work on the texture is queued (ImageQueue) and runs before the window frame that draws the
      image. If the frame being recorded already draws the texture, the image takes another one
      for its new pixels (the old ones copied first when kept): the earlier draws keep the earlier
      pixels, as with software images.
    - Contexts made outside a window's paint, or off the message thread, draw into the software
      copy with the software renderer. So do those of an image whose pixels the CPU has read
      after the GPU drew them: reading them back waits for the GPU to finish all the work queued
      before, every time (DropShadowEffect reads its component's image on each paint, through
      Image::convertedToFormat): drawn in software, they are read for free, as before.
    - RGB images keep their pixels in a BGRA texture and are drawn through a view of it whose
      alpha is one.
    - A blur of what was just drawn into a cleared image is cached (ImageQueue::findBlurred):
      shadows drawn again with the same path take the texture blurred the first time.
    - The software copy is only made when its pixels are read (a Mac's GPU isn't lost, except an
      external one unplugged), so the backup setting changes nothing.
*/
class MetalPixelData final : public ImagePixelData,
                             private ImagePixelDataBackupExtensions
{
public:
    MetalPixelData (Image::PixelFormat format, int w, int h) : ImagePixelData (format, w, h) {}

    ~MetalPixelData() override
    {
        releaseTexture();
    }

    /*  Whether the GPU can hold such an image. */
    static bool canHold (Image::PixelFormat format, int w, int h)
    {
        if (w <= 0 || h <= 0 || w > 16384 || h > 16384)
            return false;

        SharedResourcePointer<MetalRendering::SharedResources> shared;

        if (! shared->isValid())
            return false;

        switch (format)
        {
            case Image::ARGB:           return true;
            case Image::SingleChannel:  return shared->getDrawPipelines (true).isValid();
            case Image::RGB:            if (@available (macOS 10.15, *)) return true; return false;
            case Image::UnknownFormat:
            default:                    return false;
        }
    }

    std::unique_ptr<LowLevelGraphicsContext> createLowLevelContext() override;

    ImagePixelData::Ptr clone() override
    {
        jassert (openContexts == 0); // see initialiseBitmapData
        auto* copy = new MetalPixelData (pixelFormat, width, height);
        ImagePixelData::Ptr result (copy);

        if (gpuValid && MessageManager::existsAndIsCurrentThread() && copy->ensureTexture())
        {
            auto job = queue->createJob (MetalRendering::ImageJob::Kind::copy);
            job->texture = [copy->texture retain];
            job->source = [texture retain];
            queue->submit (std::move (job));
            copy->gpuValid = true;
        }
        else if (gpuValid)
        {
            syncBackup (true);
        }

        if (cpuValid)
        {
            copy->backup = backup.createCopy();
            copy->cpuValid = true;
        }

        copy->backupEnabled = backupEnabled;
        return result;
    }

    std::unique_ptr<ImageType> createType() const override;

    void initialiseBitmapData (Image::BitmapData& bitmap, int x, int y, Image::BitmapData::ReadWriteMode mode) override
    {
        // A Graphics drawing into this image must be destroyed before its pixels are accessed:
        // what it draws is only queued then.
        jassert (openContexts == 0);

        const auto wholeImage = x == 0 && y == 0 && bitmap.width == width && bitmap.height == height;
        syncBackup (! (mode == Image::BitmapData::writeOnly && wholeImage));
        backup.getPixelData()->initialiseBitmapData (bitmap, x, y, mode);

        if (mode != Image::BitmapData::readOnly)
        {
            gpuValid = false;
            sendDataChangeMessage();
        }
    }

    void applySingleChannelBoxBlurEffectInArea (Rectangle<int> bounds, int radius) override
    {
        const auto area = bounds.getIntersection ({ width, height });

        if (pixelFormat != Image::SingleChannel || radius <= 0 || area.getWidth() <= 2 || area.getHeight() <= 2
             || jmax (area.getWidth(), area.getHeight()) > MetalRendering::SharedResources::maxBlurLine
             || ! MessageManager::existsAndIsCurrentThread() || ! queue->shared->canBlur())
        {
            ImagePixelData::applySingleChannelBoxBlurEffectInArea (bounds, radius);
            return;
        }

        jassert (openContexts == 0); // see initialiseBitmapData

        if (! gpuValid && ! cpuValid)
            return; // clear: stays clear

        sendDataChangeMessage();

        // Blurring what was just drawn into this cleared image (as DropShadow::drawForPath does),
        // not drawn since: the result can be cached by the key of that recording.
        uint64_t key = 0;

        if (pixelFormat != Image::RGB && texture != nil && drawnInFrame != currentFrame())
        {
            if (auto render = queue->takeLastRender (texture))
            {
                if (const auto drawnKey = render->recorder.clearMain ? render->recorder.contentKey() : 0; drawnKey != 0)
                {
                    const uint64_t blurValues[] { drawnKey, (uint64_t) pixelFormat, (uint64_t) width, (uint64_t) height,
                                                  (uint64_t) area.getX(), (uint64_t) area.getY(), (uint64_t) area.getWidth(),
                                                  (uint64_t) area.getHeight(), (uint64_t) radius };
                    key = 0xcbf29ce484222325ull;

                    for (auto v : blurValues)
                        key = ((key ^ v) * 0x9e3779b97f4a7c15ull) ^ (key >> 29);

                    if (auto cached = queue->findBlurred (key))
                    {
                        queue->recycle (std::move (render));
                        releaseTexture();
                        texture = [cached retain];
                        textureShared = true;
                        gpuValid = true;
                        cpuValid = false;
                        return;
                    }
                }

                queue->submit (std::move (render));
            }
        }

        if (! prepareWrite())
            return;

        auto job = queue->createJob (MetalRendering::ImageJob::Kind::blur);
        job->texture = [texture retain];
        job->area = area;
        job->passes = 2 * radius;
        queue->submit (std::move (job));

        if (key != 0)
        {
            queue->addBlurred (key, texture);
            textureShared = true;
        }
    }

    /*  Message thread, while recording into 'consumer': the texture to draw this image with, its
        pending pixels queued first. nil when 'consumer' draws into this image.
    */
    id<MTLTexture> textureForDrawing (const MetalRendering::FrameRecorder& consumer)
    {
        if (texture != nil && texture == consumer.destination)
            return nil;

        if (! gpuValid)
        {
            if (! ensureTexture())
                return nil;

            renameIfDrawn (false);

            if (texture == nil || ! queuePixels())
                return nil;

            gpuValid = true;
        }

        drawnInFrame = currentFrame();
        return view != nil ? view : texture;
    }

    BackupExtensions* getBackupExtensions() override { return this; }
    const BackupExtensions* getBackupExtensions() const override { return this; }

private:
    friend class MetalGraphicsContext;

    void setBackupEnabled (bool shouldBackup) override  { backupEnabled = shouldBackup; }
    bool isBackupEnabled() const override               { return backupEnabled; }
    bool backupNow() override                           { syncBackup (true); return true; }
    bool needsBackup() const override                   { return gpuValid && ! cpuValid; }
    bool canBackup() const override                     { return gpuValid; }

    uint64_t currentFrame() const { return queue->shared->framesRecorded.load(); }

    MTLPixelFormat textureFormat() const noexcept { return pixelFormat == Image::SingleChannel ? MTLPixelFormatR8Unorm : MTLPixelFormatBGRA8Unorm; }
    int bytesPerPixel() const noexcept            { return pixelFormat == Image::SingleChannel ? 1 : 4; }
    NSUInteger packedBytesPerRow() const noexcept { return (NSUInteger) ((width * bytesPerPixel() + 3) & ~3); }

    bool ensureTexture()
    {
        if (texture != nil)
            return true;

        auto usage = (MTLTextureUsage) (MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

        if (pixelFormat == Image::RGB)
            usage |= MTLTextureUsagePixelFormatView;

        texture = queue->acquireTexture (width, height, textureFormat(), usage);

        if (texture == nil)
            return false;

        if (pixelFormat == Image::RGB)
        {
            if (@available (macOS 10.15, *))
            {
                view = [texture newTextureViewWithPixelFormat: MTLPixelFormatBGRA8Unorm
                                                  textureType: MTLTextureType2D
                                                       levels: NSMakeRange (0, 1)
                                                       slices: NSMakeRange (0, 1)
                                                      swizzle: MTLTextureSwizzleChannelsMake (MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
                                                                                              MTLTextureSwizzleBlue, MTLTextureSwizzleOne)];
            }
        }

        return true;
    }

    void releaseTexture()
    {
        // A texture shared with the blurred contents cache isn't pooled: the cache and other images
        // hold it too.
        if (std::exchange (textureShared, false))
            queue->shared->imageCache->releaseLater (std::exchange (texture, nil));
        else
            queue->releaseTexture (std::exchange (texture, nil));

        queue->shared->imageCache->releaseLater (std::exchange (view, nil));
    }

    /*  The frame being recorded draws the texture, or it is shared (blurred contents): the pixels
        about to be written go into another one (the current ones copied over when 'keepPixels').
    */
    void renameIfDrawn (bool keepPixels)
    {
        if (texture == nil || (drawnInFrame != currentFrame() && ! textureShared))
            return;

        id<MTLTexture> old = [texture retain];
        releaseTexture();

        if (ensureTexture() && keepPixels && gpuValid)
        {
            auto job = queue->createJob (MetalRendering::ImageJob::Kind::copy);
            job->texture = [texture retain];
            job->source = [old retain];
            queue->submit (std::move (job));
        }

        [old release];
    }

    /*  Queues the image's pixels into its texture: the software copy's, or clear ones. */
    [[nodiscard]] bool queuePixels()
    {
        if (! cpuValid)
        {
            auto job = queue->createJob (MetalRendering::ImageJob::Kind::clear);
            job->texture = [texture retain];
            queue->submit (std::move (job));
            return true;
        }

        const auto bytesPerRow = packedBytesPerRow();
        id<MTLBuffer> staging = [queue->shared->device newBufferWithLength: bytesPerRow * (NSUInteger) height
                                                                    options: MTLResourceStorageModeShared];

        if (staging == nil)
            return false;

        // JUCE's ARGB is premultiplied BGRA in memory, RGB is BGR (little-endian): as the texture.
        const Image::BitmapData src (backup, Image::BitmapData::readOnly);
        auto* dst = static_cast<uint8*> (staging.contents);

        for (int y = 0; y < height; ++y)
        {
            const auto* line = src.getLinePointer (y);
            auto* out = dst + (size_t) y * bytesPerRow;

            if (pixelFormat == Image::RGB)
            {
                for (int x = 0; x < width; ++x)
                {
                    const auto* px = line + x * src.pixelStride;
                    out[4 * x] = px[0];
                    out[4 * x + 1] = px[1];
                    out[4 * x + 2] = px[2];
                    out[4 * x + 3] = 255;
                }
            }
            else
            {
                memcpy (out, line, (size_t) (width * bytesPerPixel()));
            }
        }

        auto job = queue->createJob (MetalRendering::ImageJob::Kind::upload);
        job->texture = [texture retain];
        job->staging = staging;
        job->stagingBytesPerRow = bytesPerRow;
        queue->submit (std::move (job));
        return true;
    }

    /*  Before GPU work writes into the texture (message thread). Returns false without a texture. */
    bool prepareWrite()
    {
        if (! ensureTexture())
            return false;

        renameIfDrawn (true);

        if (texture == nil || (! gpuValid && cpuValid && ! queuePixels()))
            return false;

        gpuValid = true;
        cpuValid = false;
        return true;
    }

    /*  Makes the software copy current ('needPixels': else it is about to be overwritten). */
    void syncBackup (bool needPixels)
    {
        if (backup.isNull())
        {
            backup = Image (pixelFormat, width, height, true, SoftwareImageType());

            if (! gpuValid)
            {
                cpuValid = true;
                return;
            }
        }

        if (cpuValid)
            return;

        cpuValid = true;

        if (! needPixels)
            return;

        if (! gpuValid || texture == nil)
        {
            backup.clear (backup.getBounds());
            return;
        }

        readBackAfterDrawing = true;
        const auto bytesPerRow = packedBytesPerRow();
        id<MTLBuffer> pixels = queue->readBack (texture, bytesPerRow);

        if (pixels == nil)
        {
            backup.clear (backup.getBounds());
            return;
        }

        const Image::BitmapData dst (backup, Image::BitmapData::writeOnly);
        const auto* src = static_cast<const uint8*> (pixels.contents);

        for (int y = 0; y < height; ++y)
        {
            const auto* in = src + (size_t) y * bytesPerRow;
            auto* line = dst.getLinePointer (y);

            if (pixelFormat == Image::RGB)
            {
                for (int x = 0; x < width; ++x)
                {
                    auto* px = line + x * dst.pixelStride;
                    px[0] = in[4 * x];
                    px[1] = in[4 * x + 1];
                    px[2] = in[4 * x + 2];
                }
            }
            else
            {
                memcpy (line, in, (size_t) (width * bytesPerPixel()));
            }
        }

        [pixels release];
    }

    SharedResourcePointer<MetalRendering::ImageQueue> queue;
    id<MTLTexture> texture = nil, view = nil;   // view: RGB images, alpha one
    Image backup;                               // software copy
    bool gpuValid = false, cpuValid = false, backupEnabled = true;
    bool readBackAfterDrawing = false;          // contexts then draw in software (see above)
    int openContexts = 0;                       // GPU contexts drawing into it, message thread
    bool textureShared = false;                 // with the blurred contents cache: never written
    uint64_t drawnInFrame = 0;                  // the last frame recorded that draws the texture

    JUCE_LEAK_DETECTOR (MetalPixelData)
};

/*  The type of MetalPixelData; images the GPU can't hold (sizes, formats) are software ones. */
class MetalImageType final : public ImageType
{
public:
    ImagePixelData::Ptr create (Image::PixelFormat format, int w, int h, bool clearImage) const override
    {
        if (MetalPixelData::canHold (format, w, h))
            return *new MetalPixelData (format, w, h);

        return SoftwareImageType().create (format, w, h, clearImage);
    }

    int getTypeID() const override { return (int) ByteOrder::makeInt ('m', 't', 'l', 'i'); }
};

inline std::unique_ptr<ImageType> MetalPixelData::createType() const { return std::make_unique<MetalImageType>(); }

//==============================================================================
class MetalGraphicsContext final : public LowLevelGraphicsContext
{
public:
    MetalGraphicsContext (MetalRendering::FrameRecorder& recorderToUse, float scale, const RectangleList<int>& deviceRegion)
        : recorder (recorderToUse)
    {
        State s;
        s.xf = AffineTransform::scale (scale);

        for (const auto& r : deviceRegion)
            s.region.addWithoutMerging (r.toFloat());

        stack.push_back (std::move (s));
    }

    ~MetalGraphicsContext() override
    {
        while (delegate != nullptr)
            endDelegate();

        while (! layers.empty())
            endTransparencyLayer();

        if (whenDone != nullptr)
            whenDone();
    }

    // Called once the context is destroyed: an image context's recording is then complete.
    std::function<void()> whenDone;

    //==============================================================================
    bool isVectorDevice() const override { return false; }

    void setOrigin (Point<int> o) override
    {
        addTransform (AffineTransform::translation (o));
    }

    void addTransform (const AffineTransform& t) override
    {
        if (delegate != nullptr)
            delegate->context->addTransform (t);

        state().xf = t.followedBy (state().xf);
    }

    float getPhysicalPixelScaleFactor() const override
    {
        return std::sqrt (std::abs (state().xf.getDeterminant()));
    }

    //==============================================================================
    bool clipToRectangle (const Rectangle<int>& r) override
    {
        if (delegate != nullptr)
        {
            delegate->context->clipToRectangle (r);
            state().region.clipTo (deviceBounds (r.toFloat()));
            return ! delegate->context->isClipEmpty();
        }

        if (! isAxisAligned (state().xf))
        {
            Path p;
            p.addRectangle (r);
            clipToPath (p, {});
            return ! isClipEmpty();
        }

        state().region.clipTo (deviceBounds (r.toFloat()));
        return ! state().region.isEmpty();
    }

    bool clipToRectangleList (const RectangleList<int>& list) override
    {
        if (delegate != nullptr)
        {
            delegate->context->clipToRectangleList (list);
            state().region.clipTo (deviceBounds (list.getBounds().toFloat()));
            return ! delegate->context->isClipEmpty();
        }

        if (! isAxisAligned (state().xf))
        {
            Path p;

            for (const auto& r : list)
                p.addRectangle (r);

            clipToPath (p, {});
            return ! isClipEmpty();
        }

        RectangleList<float> device;

        for (const auto& r : list)
            device.add (deviceBounds (r.toFloat()));

        state().region.clipTo (device);
        return ! state().region.isEmpty();
    }

    void excludeClipRectangle (const Rectangle<int>& r) override
    {
        if (delegate != nullptr)
        {
            delegate->context->excludeClipRectangle (r);
            return;
        }

        // Only exact for axis-aligned transforms; otherwise keeping the area is still correct
        // (the excluded component paints over it), it only costs overdraw.
        if (isAxisAligned (state().xf))
            state().region.subtract (deviceBounds (r.toFloat()));
    }

    void clipToPath (const Path& path, const AffineTransform& t) override
    {
        if (delegate == nullptr)
        {
            // GPU mask: the path's coverage (coverage counting) resolved into an R8 mask, times the
            // parent mask. An empty path, or one with no edge in the clip, clips everything (its
            // coverage is zero).
            const auto toDevice = t.followedBy (state().xf);
            const auto deviceBounds = path.getBoundsTransformed (toDevice);
            const auto area = deviceBounds.isFinite() ? maskArea (deviceBounds) : Rectangle<int>();
            Point<int> atlasOrigin;
            const auto page = area.isEmpty() ? -1 : recorder.addCoveragePath (path, toDevice, area, atlasOrigin);

            if (page < 0)
            {
                state().region.clear();
                return;
            }

            MetalRendering::FrameRecorder::Mask m;
            m.area = area;
            m.texWidth = MetalRendering::FrameRecorder::pooledSize (area.getWidth());
            m.texHeight = MetalRendering::FrameRecorder::pooledSize (area.getHeight());
            m.parent = state().mask;
            m.coveragePage = page;
            m.inv0 = { (float) (atlasOrigin.x - area.getX()), (float) (atlasOrigin.y - area.getY()),
                       path.isUsingNonZeroWinding() ? 0.0f : 1.0f, 0.0f };
            m.imageWidth = (float) area.getWidth();
            m.imageHeight = (float) area.getHeight();
            recorder.masks.push_back (m);
            state().mask = (int) recorder.masks.size() - 1;
            state().region.clipTo (area.toFloat());
            return;
        }

        delegate->context->clipToPath (path, t);
        state().region.clipTo (deviceBounds (path.getBoundsTransformed (t)));
    }

    void clipToImageAlpha (const Image& image, const AffineTransform& t) override
    {
        if (delegate == nullptr)
        {
            // GPU mask: the image's alpha, transformed, rendered into an R8 mask (times the parent).
            const auto imageToDevice = t.followedBy (state().xf);
            auto texture = image.isValid() && ! imageToDevice.isSingularity() ? textureOf (image) : nil;

            if (texture != nil)
            {
                const auto area = maskArea (image.getBounds().toFloat().transformedBy (imageToDevice));

                if (area.isEmpty())
                {
                    state().region.clear();
                    return;
                }

                const auto inv = imageToDevice.inverted();
                MetalRendering::FrameRecorder::Mask m;
                m.area = area;
                m.texWidth = MetalRendering::FrameRecorder::pooledSize (area.getWidth());
                m.texHeight = MetalRendering::FrameRecorder::pooledSize (area.getHeight());
                m.parent = state().mask;
                m.image = texture;
                m.inv0 = { inv.mat00, inv.mat01, inv.mat02, 0.0f };
                m.inv1 = { inv.mat10, inv.mat11, inv.mat12, 0.0f };
                m.imageWidth = (float) image.getWidth();
                m.imageHeight = (float) image.getHeight();
                m.imageIsAlphaOnly = image.isSingleChannel();
                m.nearest = state().quality == Graphics::lowResamplingQuality;
                recorder.masks.push_back (m);
                state().mask = (int) recorder.masks.size() - 1;
                state().region.clipTo (area.toFloat());
                return;
            }

            beginDelegate (state().region.getBounds(), DelegateEnd::onRestore, "clipToImageAlpha");
        }

        if (delegate != nullptr)
        {
            delegate->context->clipToImageAlpha (image, t);
            state().region.clipTo (deviceBounds (image.getBounds().toFloat().transformedBy (t)));
        }
    }

    /*  Device-pixel area a new mask needs: its shape's bounds within the current clip. */
    Rectangle<int> maskArea (Rectangle<float> deviceShapeBounds) const
    {
        return deviceShapeBounds.expanded (1.0f).getIntersection (state().region.getBounds()).getSmallestIntegerContainer();
    }

    bool clipRegionIntersects (const Rectangle<int>& r) override
    {
        return state().region.intersectsRectangle (deviceBounds (r.toFloat()));
    }

    Rectangle<int> getClipBounds() const override
    {
        // Called for every component painted: without rotation or shear (as almost always), the
        // inverse is a division, as the software renderer's translation-only fast path.
        const auto& t = state().xf;
        const auto device = state().region.getBounds();

        if (t.mat01 == 0.0f && t.mat10 == 0.0f && t.mat00 != 0.0f && t.mat11 != 0.0f)
        {
            const auto x0 = (device.getX() - t.mat02) / t.mat00, x1 = (device.getRight() - t.mat02) / t.mat00;
            const auto y0 = (device.getY() - t.mat12) / t.mat11, y1 = (device.getBottom() - t.mat12) / t.mat11;
            return Rectangle<float>::leftTopRightBottom (jmin (x0, x1), jmin (y0, y1), jmax (x0, x1), jmax (y0, y1))
                       .getSmallestIntegerContainer();
        }

        return device.transformedBy (t.inverted()).getSmallestIntegerContainer();
    }

    bool isClipEmpty() const override
    {
        return state().region.isEmpty();
    }

    //==============================================================================
    void saveState() override
    {
        if (delegate != nullptr)
        {
            delegate->context->saveState();
            ++delegate->saveDepth;
        }

        stack.push_back (stack.back());
    }

    void restoreState() override
    {
        if (delegate != nullptr)
        {
            if (delegate->saveDepth == 0 && delegate->end == DelegateEnd::onRestore)
                endDelegate();
            else
            {
                delegate->context->restoreState();
                --delegate->saveDepth;
            }
        }

        if (stack.size() > 1)
            stack.pop_back();
    }

    void beginTransparencyLayer (float opacity) override
    {
        if (delegate != nullptr)
        {
            delegate->context->beginTransparencyLayer (opacity);
            ++delegate->layerDepth;
            return;
        }

        // GPU layer (as the OpenGL renderer does with a framebuffer): following draws go to an
        // offscreen target covering the current clip, composited with 'opacity' when it ends.
        // Like the software renderer, the layer saves the state and its end restores it.
        const auto area = state().region.getBounds().getSmallestIntegerContainer();
        stack.push_back (stack.back());
        layers.push_back ({ recorder.beginLayer (area), opacity });
    }

    void endTransparencyLayer() override
    {
        // A software fallback started inside this GPU layer and still open (one waiting for a
        // restore): it belongs in the layer, so it ends first.
        if (delegate != nullptr && delegate->layerDepth == 0)
            endDelegate();

        if (delegate != nullptr)
        {
            delegate->context->endTransparencyLayer();

            --delegate->layerDepth;

            return;
        }

        if (layers.empty())
            return;

        const auto layer = layers.back();
        layers.pop_back();
        recorder.endLayer();

        if (stack.size() > 1)
            stack.pop_back();

        const auto& target = recorder.targets[(size_t) layer.target];
        const auto area = target.area.toFloat();

        if (area.isEmpty())
            return;

        // The layer's content was already clipped (and masked) when drawn: composite it as is.
        MetalRendering::Instance instance;
        instance.xf0 = { 1.0f, 0.0f, 0.0f, 0.0f };
        instance.xf1 = { 0.0f, 1.0f, 0.0f, 0.0f };
        instance.rect = { area.getX(), area.getY(), area.getWidth(), area.getHeight() };
        instance.clip = instance.rect;
        instance.colour = { layer.opacity, layer.opacity, layer.opacity, layer.opacity };
        instance.params = { 0.0f, 0.0f, (float) (int) MetalRendering::Kind::rgbaTexture, 0.0f };
        instance.uv = { 0.0f, 0.0f, area.getWidth() / (float) target.texWidth, area.getHeight() / (float) target.texHeight };
        recorder.add (instance, nil, true, -1, layer.target);
    }

    //==============================================================================
    void setFill (const FillType& fill) override
    {
        if (delegate != nullptr)
            delegate->context->setFill (fill);

        state().fill = fill;
    }

    void setOpacity (float opacity) override
    {
        // Same semantics as the software renderer: replaces the fill colour's alpha.
        if (delegate != nullptr)
            delegate->context->setOpacity (opacity);

        state().fill.setOpacity (opacity);
    }

    void setInterpolationQuality (Graphics::ResamplingQuality quality) override
    {
        if (delegate != nullptr)
            delegate->context->setInterpolationQuality (quality);

        state().quality = quality;
    }

    void setImageBlendMode (BlendMode mode) override
    {
        if (delegate != nullptr)
            delegate->context->setImageBlendMode (mode);

        state().imageBlendMode = mode;
    }

    //==============================================================================
    void fillRect (const Rectangle<int>& r, bool replaceExistingContents) override
    {
        if (delegate != nullptr)
            return delegate->context->fillRect (r, replaceExistingContents);

        // Replacing with a translucent fill: the 'source' operator (R = S). A partly covered edge
        // pixel gets S * coverage, as with the software renderer (PixelARGB::blend scales the
        // source by the coverage, then replaces).
        if (replaceExistingContents && ! (state().fill.isColour() && state().fill.colour.isOpaque()))
            return addShape (MetalRendering::Kind::box, r.toFloat(), 0.0f, 0.0f, state().xf,
                             [&] (auto& c) { c.fillRect (r, true); }, BlendMode::source);

        fillRect (r.toFloat());
    }

    void fillRect (const Rectangle<float>& r) override
    {
        if (delegate != nullptr)
            return delegate->context->fillRect (r);

        addShape (MetalRendering::Kind::box, r, 0.0f, 0.0f, state().xf, [&] (auto& c) { c.fillRect (r); });
    }

    void fillRectList (const RectangleList<float>& list) override
    {
        if (delegate != nullptr)
            return delegate->context->fillRectList (list);

        // Rects drawn one by one are anti-aliased separately: where two of them share a fractional
        // edge, the half-covered pixels get blended twice (0.5 over 0.5 gives 0.75, not 1) and the
        // seams show. That's only safe when every edge lands on a device pixel boundary.
        const auto isPixelAligned = [this] (Rectangle<float> r)
        {
            const auto d = deviceBounds (r);
            const auto onGrid = [] (float v) { return std::abs (v - std::round (v)) < 1.0e-3f; };
            return onGrid (d.getX()) && onGrid (d.getY()) && onGrid (d.getRight()) && onGrid (d.getBottom());
        };

        if (state().fill.isColour() && isAxisAligned (state().xf)
             && std::all_of (list.begin(), list.end(), isPixelAligned))
        {
            for (const auto& r : list)
                fillRect (r);

            return;
        }

        // Otherwise fill the union as one path: coverage counting sums the shared edges to zero, so
        // the anti-aliasing only happens on the outline, as with the software renderer's edge table.
        Path united;

        for (const auto& r : list)
            united.addRectangle (r);

        fillPath (united, {});
    }

    void fillRoundedRectangle (const Rectangle<float>& r, float cornerSize) override
    {
        if (delegate != nullptr)
            return delegate->context->fillRoundedRectangle (r, cornerSize);

        addShape (MetalRendering::Kind::box, r, cornerSize, 0.0f, state().xf,
                  [&] (auto& c) { c.fillRoundedRectangle (r, cornerSize); });
    }

    void drawRoundedRectangle (const Rectangle<float>& r, float cornerSize, float lineThickness) override
    {
        if (delegate != nullptr)
            return delegate->context->drawRoundedRectangle (r, cornerSize, lineThickness);

        addShape (MetalRendering::Kind::box, r, cornerSize, lineThickness, state().xf,
                  [&] (auto& c) { c.drawRoundedRectangle (r, cornerSize, lineThickness); });
    }

    void fillEllipse (const Rectangle<float>& r) override
    {
        if (delegate != nullptr)
            return delegate->context->fillEllipse (r);

        // The ellipse distance in the shader is exact for circles only, and drifts by up to a pixel
        // at the ends of elongated ellipses: those go through a path, like the software renderer.
        if (! isNearlyCircular (r))
            return LowLevelGraphicsContext::fillEllipse (r);

        addShape (MetalRendering::Kind::ellipse, r, 0.0f, 0.0f, state().xf, [&] (auto& c) { c.fillEllipse (r); });
    }

    static bool isNearlyCircular (const Rectangle<float>& r) noexcept
    {
        return std::abs (r.getWidth() - r.getHeight()) <= 0.01f * jmax (r.getWidth(), r.getHeight());
    }

    void drawEllipse (const Rectangle<float>& r, float lineThickness) override
    {
        if (delegate != nullptr)
            return delegate->context->drawEllipse (r, lineThickness);

        if (! isNearlyCircular (r))
            return LowLevelGraphicsContext::drawEllipse (r, lineThickness);

        addShape (MetalRendering::Kind::ellipse, r, 0.0f, lineThickness, state().xf,
                  [&] (auto& c) { c.drawEllipse (r, lineThickness); });
    }

    void drawLine (const Line<float>& line) override
    {
        drawLineWithThickness (line, 1.0f);
    }

    void drawLineWithThickness (const Line<float>& line, float lineThickness) override
    {
        if (delegate != nullptr)
            return delegate->context->drawLineWithThickness (line, lineThickness);

        // A butt-capped line is a rotated rectangle: box shape in the line's own frame.
        const auto length = line.getLength();
        const auto lineToUser = AffineTransform::rotation (std::atan2 (line.getEndY() - line.getStartY(),
                                                                       line.getEndX() - line.getStartX()))
                                    .translated (line.getStart());
        addShape (MetalRendering::Kind::box, { 0.0f, -0.5f * lineThickness, length, lineThickness }, 0.0f, 0.0f,
                  lineToUser.followedBy (state().xf),
                  [&] (auto& c) { c.drawLineWithThickness (line, lineThickness); });
    }

    /*  Draws the coverage of a path (fill or stroke outline) whose device bounds are 'bounds':
        'addPath' records it in the coverage atlas over the given area (see FrameRecorder).
    */
    template <typename AddPath>
    void fillCoverage (Rectangle<float> bounds, bool evenOdd, id<MTLTexture> pattern, int gradient, AddPath&& addPath)
    {
        if (! bounds.isFinite())
            return;

        const auto area = maskArea (bounds);

        if (area.isEmpty())
            return;

        Point<int> atlasOrigin;
        const auto page = addPath (area, atlasOrigin);
        const auto offset = (atlasOrigin - area.getPosition()).toFloat();
        emit (MetalRendering::Kind::coverage, area.toFloat(), AffineTransform(), premultiplied (state().fill.colour),
              evenOdd ? 1.0f : 0.0f, 0.0f, { offset.x, offset.y, 0.0f, 0.0f },
              pattern, state().quality == Graphics::lowResamplingQuality, gradient, pattern != nil, page);
    }

    void fillPath (const Path& path, const AffineTransform& t) override
    {
        if (delegate != nullptr)
            return delegate->context->fillPath (path, t);

        if (path.isEmpty())
            return;

        const auto toDevice = t.followedBy (state().xf);

        id<MTLTexture> pattern = nil;
        const auto gradient = state().fill.isColour() ? -1 : currentPaintIndex (pattern);

        if (! state().fill.isColour() && gradient < 0)
            return fallback (state().fill.isGradient() ? "fillPath (gradient fill)" : "fillPath (image fill)",
                             path.getBoundsTransformed (t), [&] (auto& c) { c.fillPath (path, t); });

        // Coverage counting: the outline's edges are accumulated in the coverage atlas on the GPU,
        // then the path's rectangle is drawn as one quad reading its winding numbers.
        fillCoverage (path.getBoundsTransformed (toDevice), ! path.isUsingNonZeroWinding(), pattern, gradient,
                      [&] (Rectangle<int> area, Point<int>& atlasOrigin)
                      {
                          return recorder.addCoveragePath (path, toDevice, area, atlasOrigin);
                      });
    }

    void strokePath (const Path& path, const PathStrokeType& stroke, const AffineTransform& t) override
    {
        if (delegate != nullptr)
            return delegate->context->strokePath (path, stroke, t);

        // Same centre line as LowLevelGraphicsContext's default: flattened in user space (stroke
        // thickness in user units), with curve accuracy matching the physical pixel scale, on the
        // encode thread (see FrameRecorder::CoverageJob); its outline is built on the GPU.
        const auto accuracy = getPhysicalPixelScaleFactor();

        id<MTLTexture> pattern = nil;
        const auto gradient = state().fill.isColour() ? -1 : currentPaintIndex (pattern);

        if (path.isEmpty() || stroke.getStrokeThickness() <= 0.0f || (! state().fill.isColour() && gradient < 0))
        {
            Path outline;
            stroke.createStrokedPath (outline, path, t, accuracy);
            return fillPath (outline, {});
        }

        // Its bounds: the path's, grown by the furthest a joint or cap reaches from it: half the
        // thickness, times the miter limit for mitered joints (PathStrokeHelpers falls back to a
        // bevel beyond it), times sqrt 2 for square caps.
        const auto halfWidth = 0.5f * stroke.getStrokeThickness();
        const auto reach = halfWidth * jmax (1.0f,
                                             stroke.getJointStyle() == PathStrokeType::mitered ? stroke.getMiterLimit() : 1.0f,
                                             stroke.getEndStyle() == PathStrokeType::square ? MathConstants<float>::sqrt2 : 1.0f);
        const auto& toDevice = state().xf;
        const auto bounds = path.getBoundsTransformed (t).expanded (reach).transformedBy (toDevice);

        fillCoverage (bounds, false, pattern, gradient, [&] (Rectangle<int> area, Point<int>& atlasOrigin)
        {
            return recorder.addCoverageStroke (path, stroke, t, accuracy, toDevice, area, atlasOrigin);
        });
    }

    void drawImage (const Image& image, const AffineTransform& t) override
    {
        if (delegate != nullptr)
            return delegate->context->drawImage (image, t);

        const auto bounds = image.getBounds().toFloat();
        const auto canDrawNatively = image.isValid() && image.getWidth() <= 16384 && image.getHeight() <= 16384;

        if (! canDrawNatively)
            return fallback ("drawImage (size)", bounds.transformedBy (t), [&] (auto& c) { c.drawImage (image, t); });

        auto texture = textureOf (image);

        if (texture == nil)
            return fallback ("drawImage (no texture)", bounds.transformedBy (t), [&] (auto& c) { c.drawImage (image, t); });

        // Like the software renderer: images are drawn with the fill colour's alpha only, a
        // single-channel image as white with that alpha (PixelAlpha blended as a pixel whose
        // colour channels equal its alpha). Filling an alpha channel with the current brush goes
        // through clipToImageAlpha instead.
        const auto alpha = image.isSingleChannel();
        const auto colour = premultiplied (Colours::white.withAlpha (state().fill.colour.getFloatAlpha()));
        const auto nearest = state().quality == Graphics::lowResamplingQuality;
        emit (alpha ? MetalRendering::Kind::alphaTexture : MetalRendering::Kind::rgbaTexture,
              bounds, t.followedBy (state().xf), colour, 0.0f, 0.0f, { 0.0f, 0.0f, 1.0f, 1.0f }, texture, nearest,
              -1, false, -1, state().imageBlendMode);
    }

    //==============================================================================
    void setFont (const Font& f) override
    {
        if (delegate != nullptr)
            delegate->context->setFont (f);

        state().font = f;
    }

    const Font& getFont() override
    {
        return state().font;
    }

    void drawGlyphs (Span<const uint16_t> glyphs, Span<const Point<float>> positions, const AffineTransform& t) override
    {
        if (delegate != nullptr)
            return delegate->context->drawGlyphs (glyphs, positions, t);

        const auto glyphToDevice = t.followedBy (state().xf);
        const auto& font = state().font;
        const auto heightPx = font.getHeight() * std::sqrt (std::abs (glyphToDevice.getDeterminant()));

        auto drawWithSoftware = [&]
        {
            Rectangle<float> area;

            for (size_t i = 0; i < glyphs.size(); ++i)
                area = area.getUnion (Rectangle<float> (positions[i].x - font.getHeight(), positions[i].y - font.getHeight() * 1.5f,
                                                        font.getHeight() * 3.0f, font.getHeight() * 2.5f));

            fallback ("drawGlyphs", area.transformedBy (t), [&] (auto& c) { c.drawGlyphs (glyphs, positions, t); });
        };

        const auto typeface = font.getTypefacePtr();

        // Colour glyphs (COLRv0 / bitmap emoji) have layers neither the atlas nor the outlines keep.
        if (typeface == nullptr || typeface->getColourGlyphFormats() != 0)
            return drawWithSoftware();

        // Glyphs as filled outlines (as the CoreGraphics context does for non-native fonts): the
        // outlines are cached per glyph, then filled like any path.
        auto drawAsOutlines = [&] (size_t first)
        {
            const auto pts = font.getHeightInPoints();

            for (auto i = first; i < glyphs.size(); ++i)
            {
                const auto& outline = recorder.glyphOutlines->get (typeface, glyphs[i]);

                if (! outline.isEmpty())
                    fillPath (outline, AffineTransform::scale (pts * font.getHorizontalScale(), pts)
                                           .translated (positions[i]).followedBy (t));
            }
        };

        // The atlas holds upright glyphs only, up to a size where per-glyph textures stop paying off.
        if (! state().fill.isColour() || ! isTranslationAndScale (glyphToDevice) || heightPx > 160.0f)
            return drawAsOutlines (0);

        const auto colour = premultiplied (state().fill.colour);
        const auto& region = state().region;
        const auto atlasScale = 1.0f / (float) MetalRendering::GlyphAtlas::size;
        const auto rasterPoints = font.withHeight (heightPx).getHeightInPoints();
        const auto glyphScale = heightPx / font.getHeight();
        auto& strike = recorder.glyphAtlas->getStrike (font, typeface, glyphScale);

        for (size_t i = 0; i < glyphs.size(); ++i)
        {
            // Placed as CoreGraphics places glyphs in a window (measured on lines of text at every
            // 1/20 px offset): x rounded down to 1/16 px, the baseline rounded up to a whole pixel
            // (CoreGraphics works bottom-up). CoreGraphics first rounds the device x to a fixed grid
            // of about 1/512 px, so an x up to 1/1024 px below a 1/16 px step lands on the step:
            // measured at x = 300 to 1300 px and under 0.5x and 2x transforms (same threshold, so a
            // device-space grid), on a 2x display. Layout's float noise often ends up there (an
            // advance summing to 108.9998 for 109). The y tolerance only absorbs float noise.
            const auto pen = positions[i].transformedBy (glyphToDevice);
            constexpr auto steps = MetalRendering::GlyphAtlas::subpixelSteps;
            const auto qx = (int) std::floor ((pen.x + 1.0f / 1024.0f) * (float) steps);
            const auto x = qx >= 0 ? qx / steps : -((-qx + steps - 1) / steps);
            const Point<int> snapped (x, (int) std::ceil (pen.y - 1.0f / 64.0f));
            const auto subpixel = qx - x * steps;

            const auto* entry = recorder.glyphAtlas->get (strike, font, glyphs[i], rasterPoints, subpixel, recorder.glyphJobs);

            if (entry == nullptr)
                return drawAsOutlines (i); // atlas full: the remaining glyphs as outlines

            if (entry->area.isEmpty())
                continue;

            const auto dst = entry->area.withPosition (snapped + entry->offset).toFloat();

            if (! region.intersectsRectangle (dst))
                continue;

            const auto a = entry->area.toFloat() * atlasScale;
            emit (MetalRendering::Kind::glyph, dst, {}, colour, 0.0f, 0.0f,
                  { a.getX(), a.getY(), a.getRight(), a.getBottom() }, nil, false);
        }
    }

    std::unique_ptr<ImageType> getPreferredImageTypeForTemporaryImages() const override
    {
        // While a software delegate draws, what it draws into is software.
        if (delegate != nullptr)
            return std::make_unique<SoftwareImageType>();

        return std::make_unique<MetalImageType>();
    }

    uint64_t getFrameId() const override { return 0; }

private:
    //==============================================================================
    /*  The clip region in device pixels. Saved with the state for every component painted, so it
        mustn't allocate: usually a single rectangle, kept inline; several rectangles are shared
        between saved states and copied when one changes (the software renderer's clip regions are
        reference counted for the same reason). Same operations as RectangleList<float>.
    */
    class ClipRegion
    {
    public:
        void addWithoutMerging (Rectangle<float> r)
        {
            auto copy = toList();
            copy.addWithoutMerging (r);
            assign (std::move (copy));
        }

        void clipTo (Rectangle<float> r)
        {
            if (list == nullptr)
            {
                single = single.getIntersection (r);
                return;
            }

            auto copy = *list;
            copy.clipTo (r);
            assign (std::move (copy));
        }

        void clipTo (const RectangleList<float>& other)
        {
            auto copy = toList();
            copy.clipTo (other);
            assign (std::move (copy));
        }

        void subtract (Rectangle<float> r)
        {
            auto copy = toList();
            copy.subtract (r);
            assign (std::move (copy));
        }

        void clear() noexcept                                   { single = {}; list = nullptr; }
        bool isEmpty() const noexcept                           { return list != nullptr ? list->isEmpty() : single.isEmpty(); }
        Rectangle<float> getBounds() const noexcept             { return list != nullptr ? list->getBounds() : single; }

        bool intersectsRectangle (Rectangle<float> r) const noexcept
        {
            return list != nullptr ? list->intersectsRectangle (r) : single.intersects (r);
        }

        const Rectangle<float>* begin() const noexcept          { return list != nullptr ? list->begin() : &single; }
        const Rectangle<float>* end() const noexcept            { return list != nullptr ? list->end() : &single + (single.isEmpty() ? 0 : 1); }

    private:
        RectangleList<float> toList() const
        {
            if (list != nullptr)
                return *list;

            RectangleList<float> result;

            if (! single.isEmpty())
                result.add (single);

            return result;
        }

        void assign (RectangleList<float>&& rects)
        {
            if (rects.getNumRectangles() <= 1)
            {
                single = rects.isEmpty() ? Rectangle<float>() : rects.getRectangle (0);
                list = nullptr;
                return;
            }

            list = std::make_shared<const RectangleList<float>> (std::move (rects));
        }

        Rectangle<float> single;
        std::shared_ptr<const RectangleList<float>> list; // more than one rectangle
    };

    struct State
    {
        AffineTransform xf;              // user -> device pixels
        ClipRegion region;               // device pixels, exact
        FillType fill { Colours::black };
        Font font { FontOptions{} };
        Graphics::ResamplingQuality quality = Graphics::mediumResamplingQuality;
        BlendMode imageBlendMode = BlendMode::sourceOver;
        int mask = -1;                   // clip mask in recorder.masks, -1: none
    };

    /*  The texture to draw an image with: a GPU image's own, or the upload of another. */
    id<MTLTexture> textureOf (const Image& image)
    {
        if (auto* gpuImage = dynamic_cast<MetalPixelData*> (image.getPixelData().get()))
            return gpuImage->textureForDrawing (recorder);

        return recorder.imageCache->get (image);
    }

    State& state() noexcept             { return stack.back(); }
    const State& state() const noexcept { return stack.back(); }

    static bool isAxisAligned (const AffineTransform& t) noexcept
    {
        return (t.mat01 == 0.0f && t.mat10 == 0.0f) || (t.mat00 == 0.0f && t.mat11 == 0.0f);
    }

    static bool isTranslationAndScale (const AffineTransform& t) noexcept
    {
        return t.mat01 == 0.0f && t.mat10 == 0.0f && t.mat00 > 0.0f && std::abs (t.mat00 - t.mat11) < 1.0e-4f;
    }

    Rectangle<float> deviceBounds (Rectangle<float> r) const
    {
        return r.transformedBy (state().xf);
    }

    static MetalRendering::Float4 premultiplied (Colour c) noexcept
    {
        const auto a = c.getFloatAlpha();
        return { c.getFloatRed() * a, c.getFloatGreen() * a, c.getFloatBlue() * a, a };
    }

    //==============================================================================
    /*  One instance per clip rect the shape touches, so the clip stays exact. */
    void emit (MetalRendering::Kind kind, Rectangle<float> local, const AffineTransform& localToDevice,
               MetalRendering::Float4 colour, float radius, float stroke, MetalRendering::Float4 uv,
               id<MTLTexture> texture, bool nearest, int gradient = -1, bool repeat = false, int coveragePage = -1,
               BlendMode blend = BlendMode::sourceOver)
    {
        const auto& t = localToDevice;
        // The anti-aliasing reaches up to a device pixel past the outline: grown after the
        // transform (in local units it would be less than a pixel when scaling down).
        const auto bounds = local.expanded (stroke * 0.5f).transformedBy (t).expanded (2.0f);

        MetalRendering::Instance instance;
        instance.xf0 = { t.mat00, t.mat01, t.mat02, 0.0f };
        instance.xf1 = { t.mat10, t.mat11, t.mat12, 0.0f };
        instance.rect = { local.getX(), local.getY(), local.getWidth(), local.getHeight() };
        instance.colour = colour;
        instance.params = { radius, stroke, (float) (int) kind, (float) (gradient + 1) };
        instance.uv = uv;

        for (const auto& clip : state().region)
        {
            if (! clip.intersects (bounds))
                continue;

            instance.clip = { clip.getX(), clip.getY(), clip.getWidth(), clip.getHeight() };
            recorder.add (instance, texture, nearest, state().mask, -1, repeat, coveragePage, blend);
        }
    }

    template <typename SoftwareOp>
    void addShape (MetalRendering::Kind kind, Rectangle<float> r, float cornerSize, float lineThickness,
                   const AffineTransform& localToDevice, SoftwareOp&& op, BlendMode blend = BlendMode::sourceOver)
    {
        id<MTLTexture> pattern = nil;
        const auto gradient = state().fill.isColour() ? -1 : currentPaintIndex (pattern);

        if (! state().fill.isColour() && gradient < 0)
            return fallback (state().fill.isGradient() ? "shape (gradient fill)" : "shape (image fill)",
                             r.expanded (lineThickness).transformedBy (localToDevice.followedBy (state().xf.inverted())),
                             std::forward<SoftwareOp> (op));

        emit (kind, r, localToDevice, premultiplied (state().fill.colour), cornerSize, lineThickness, {}, pattern,
              state().quality == Graphics::lowResamplingQuality, gradient, pattern != nil, -1, blend);
    }

    /*  Gradient or tiled-image fill registered for this frame (see currentGradientIndex);
        'pattern' receives the image texture for a tiled image. -1: not drawable on the GPU.
    */
    int currentPaintIndex (id<MTLTexture>& pattern)
    {
        const auto& fill = state().fill;

        if (fill.isGradient())
            return currentGradientIndex();

        if (! fill.isTiledImage() || ! fill.image.isValid() || fill.image.isSingleChannel())
            return -1;

        const auto imageToDevice = fill.transform.followedBy (state().xf);

        if (imageToDevice.isSingularity())
            return -1;

        pattern = textureOf (fill.image);

        if (pattern == nil)
            return -1;

        const auto D = imageToDevice.inverted(); // device pixel -> image pixel
        MetalRendering::GradientInfo info;
        info.row0 = { D.mat00, D.mat01, D.mat02, 0.0f };
        info.row1 = { D.mat10, D.mat11, D.mat12, 0.0f };
        info.meta = { 3.0f, 0.0f, 0.0f, fill.getOpacity() };
        info.conic = { (float) fill.image.getWidth(), (float) fill.image.getHeight(), 0.0f, 0.0f };
        recorder.gradients.push_back (info);
        return (int) recorder.gradients.size() - 1;
    }

    /*  Registers the current gradient fill for this frame: its device -> gradient map, the
        circles of a radial one, and its colour stops. -1 if the GPU path can't draw it
        (degenerate or singular gradients).
    */
    int currentGradientIndex()
    {
        const auto& fill = state().fill;
        const auto& g = *fill.gradient;

        const auto gradientToDevice = fill.transform.followedBy (state().xf);

        if (gradientToDevice.isSingularity())
            return -1;

        const auto D = gradientToDevice.inverted(); // device pixel -> gradient space

        const auto p1 = g.point1, d = g.point2 - g.point1;
        MetalRendering::GradientInfo info;
        auto type = g.isRadial ? 1.0f : 0.0f;

        const auto conical = g.isRadial && g.endRadius >= 0.0f
                          && ! approximatelyEqual (g.point1.getDistanceFrom (g.point2), 0.0f, relativeTolerance (0.001f))
                          && ! approximatelyEqual (g.startRadius, g.endRadius, relativeTolerance (0.001f));

        if (conical)
        {
            // SVG-style radial with a start (focal) circle: the software renderer uses
            // detail::TwoPointConicalGradient when its circles aren't concentric and their radii
            // differ. Same set-up as its constructor; the shader does calculateWeight.
            struct Circle { Point<float> c; float r; };
            const Circle start { g.point1, g.startRadius }, end { g.point2, g.endRadius };
            const auto inputTolerance = relativeTolerance (0.001f);

            const auto swapped = approximatelyEqual (end.r, 0.0f);
            auto c0 = swapped ? end : start;
            auto c1 = swapped ? start : end;
            auto f = c0.r / (c0.r - c1.r);
            const auto focalPoint = c0.c * (1.0f - f) + c1.c * f;
            const auto segment = c1.c - focalPoint;
            const auto distToFocalPoint = segment.getDistanceFromOrigin();

            if (approximatelyEqual (distToFocalPoint, 0.0f, inputTolerance))
                return -1;

            const auto regularise = AffineTransform::translation (-focalPoint.x, -focalPoint.y)
                                        .rotated (-std::atan2 (segment.y, segment.x))
                                        .scaled (1.0f / distToFocalPoint);
            const auto scale = std::sqrt (std::abs (regularise.getDeterminant()));
            c0.r *= scale;
            c1.r *= scale;
            f = c0.r / (c0.r - c1.r);
            const auto closeToSingular = approximatelyEqual (c1.r, 1.0f, absoluteTolerance (0.001f));

            const auto M = D.followedBy (regularise); // device pixel -> regularised space
            info.row0 = { M.mat00, M.mat01, M.mat02, 0.0f };
            info.row1 = { M.mat10, M.mat11, M.mat12, 0.0f };
            info.conic = { c1.r, f, swapped ? 1.0f : 0.0f, closeToSingular ? 1.0f : 0.0f };
            type = 2.0f;
        }
        else if (g.isRadial)
        {
            // Concentric circles (detail::RadialGradientView): q = (D(p) - p1) / r1, and as
            // CGContextDrawRadialGradient draws them, t = (|q| - r0 / r1) / (1 - r0 / r1).
            const auto r = g.endRadius >= 0.0f ? g.endRadius : d.getDistanceFromOrigin();
            const auto startRatio = jlimit (0.0f, 0.999f, g.startRadius / r);

            if (r <= 0.0f)
                return -1;

            info.row0 = { D.mat00 / r, D.mat01 / r, (D.mat02 - p1.x) / r, 0.0f };
            info.row1 = { D.mat10 / r, D.mat11 / r, (D.mat12 - p1.y) / r, 0.0f };
            info.conic = { startRatio, 0.0f, 0.0f, 0.0f };
        }
        else
        {
            // t = dot (D(p) - p1, d) / |d|^2, an affine function of p
            const auto len2 = d.x * d.x + d.y * d.y;

            if (len2 <= 0.0f)
                return -1;

            info.row0 = { (D.mat00 * d.x + D.mat10 * d.y) / len2,
                          (D.mat01 * d.x + D.mat11 * d.y) / len2,
                          ((D.mat02 - p1.x) * d.x + (D.mat12 - p1.y) * d.y) / len2, 0.0f };
        }

        const auto spread = g.spreadMethod == ColourGradient::SpreadMethod::reflect ? 1.0f
                          : g.spreadMethod == ColourGradient::SpreadMethod::repeat  ? 2.0f : 0.0f;
        // The colour stops, interpolated per pixel in the shader as CoreGraphics does (CGGradient:
        // float components, not premultiplied), where the software renderer quantises them into
        // a lookup table.
        info.row0.w = (float) recorder.gradientStops.size();
        info.row1.w = (float) g.getNumColours();

        for (int i = 0; i < g.getNumColours(); ++i)
        {
            const auto c = g.getColour (i);
            recorder.gradientStops.push_back ({ (float) g.getColourPosition (i), 0.0f, 0.0f, 0.0f });
            recorder.gradientStops.push_back ({ c.getFloatRed(), c.getFloatGreen(), c.getFloatBlue(), c.getFloatAlpha() });
        }

        info.meta = { type, spread, 0.0f, fill.getOpacity() };
        recorder.gradients.push_back (info);
        return (int) recorder.gradients.size() - 1;
    }

    //==============================================================================
    /*  Software delegate: from beginDelegate to endDelegate, every call is forwarded to a software
        renderer drawing into a transparent tile covering 'deviceArea', which is then composited.
        Exact for source-over content; used for everything the GPU path doesn't handle natively.
    */
    enum class DelegateEnd { immediately, onRestore };

    struct Delegate
    {
        Image image;
        Rectangle<int> area;  // device pixels
        std::unique_ptr<LowLevelGraphicsSoftwareRenderer> context;
        DelegateEnd end = DelegateEnd::immediately;
        int saveDepth = 0, layerDepth = 0;
    };

    void beginDelegate (Rectangle<float> deviceArea, DelegateEnd end, const char* reason)
    {
        if (recorder.collectReasons)
            ++recorder.fallbackReasons[reason];

        const auto area = deviceArea.getIntersection (state().region.getBounds()).getSmallestIntegerContainer();

        if (area.isEmpty())
            return;

        auto d = std::make_unique<Delegate>();
        d->area = area;
        d->end = end;
        d->image = Image (Image::ARGB, area.getWidth(), area.getHeight(), true, SoftwareImageType());

        RectangleList<int> clip;

        for (const auto& r : state().region)
            clip.add (r.getSmallestIntegerContainer().translated (-area.getX(), -area.getY()));

        d->context = std::make_unique<LowLevelGraphicsSoftwareRenderer> (d->image, -area.getPosition(), clip);
        d->context->addTransform (state().xf);
        d->context->setFill (state().fill);
        d->context->setFont (state().font);
        d->context->setInterpolationQuality (state().quality);
        d->context->setImageBlendMode (state().imageBlendMode);
        delegate = std::move (d);
    }

    void endDelegate()
    {
        auto d = std::move (delegate);
        d->context.reset();

        auto texture = MetalRendering::ImageTextureCache::upload (d->image, recorder.device);

        if (texture == nil)
            return;

        recorder.frameReleases->add (texture);

        const auto area = d->area.toFloat();

        MetalRendering::Instance instance;
        instance.xf0 = { 1.0f, 0.0f, 0.0f, 0.0f };
        instance.xf1 = { 0.0f, 1.0f, 0.0f, 0.0f };
        instance.rect = { area.getX(), area.getY(), area.getWidth(), area.getHeight() };
        instance.clip = instance.rect;
        instance.colour = { 1.0f, 1.0f, 1.0f, 1.0f };
        instance.params = { 0.0f, 0.0f, (float) (int) MetalRendering::Kind::rgbaTexture, 0.0f };
        instance.uv = { 0.0f, 0.0f, 1.0f, 1.0f };
        recorder.add (instance, texture, true, state().mask);
    }

    template <typename SoftwareOp>
    void fallback (const char* reason, Rectangle<float> userArea, SoftwareOp&& op)
    {
        beginDelegate (deviceBounds (userArea).expanded (1.0f), DelegateEnd::immediately, reason);

        if (delegate == nullptr)
            return;

        op (*delegate->context);
        endDelegate();
    }

    //==============================================================================
    struct OpenLayer { int target; float opacity; };

    MetalRendering::FrameRecorder& recorder;
    std::vector<State> stack;
    std::vector<OpenLayer> layers;
    std::unique_ptr<Delegate> delegate;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MetalGraphicsContext)
};

//==============================================================================
inline std::unique_ptr<LowLevelGraphicsContext> MetalPixelData::createLowLevelContext()
{
    jassert (openContexts == 0); // see initialiseBitmapData
    sendDataChangeMessage();

    // Recorded while a window paints (its frame then encodes it): the caches the recording uses
    // keep what that frame needs.
    if (! readBackAfterDrawing && MessageManager::existsAndIsCurrentThread() && queue->activeRecorder != nullptr)
    {
        const auto clearFirst = ! gpuValid && ! cpuValid;

        if (prepareWrite())
        {
            auto job = queue->createJob (MetalRendering::ImageJob::Kind::render);
            job->texture = [texture retain];

            const Rectangle<int> bounds { width, height };
            auto& shared = *queue->shared;
            auto& recorder = job->recorder;
            recorder.clear (bounds);
            recorder.clearMain = clearFirst;
            recorder.destination = texture;
            recorder.frameReleases = &job->releases;
            recorder.device = shared.device;
            recorder.glyphAtlas = shared.glyphAtlas.get();
            recorder.imageCache = shared.imageCache.get();
            recorder.glyphOutlines = &shared.glyphOutlines;
            recorder.collectReasons = false;

            auto context = std::make_unique<MetalGraphicsContext> (recorder, 1.0f, RectangleList<int> (bounds));
            ++openContexts;
            ++queue->openContexts;

            // Holds the image: the context may outlive the Image it was made from.
            context->whenDone = [self = Ptr (this), recorded = job.release(), frame = currentFrame()]
            {
                auto& image = *static_cast<MetalPixelData*> (self.get());
                std::unique_ptr<MetalRendering::ImageJob> done (recorded);
                --image.openContexts;
                --image.queue->openContexts;

                // Destroyed after the paint it was made in: the textures and glyphs it recorded were
                // only kept for that frame. Its drawing is lost, as if never made.
                if (frame != image.currentFrame() || image.queue->activeRecorder == nullptr)
                {
                    jassertfalse;
                    image.queue->recycle (std::move (done));
                    image.gpuValid = image.cpuValid = false;
                    return;
                }

                done->recorder.encodeOrder.push_back (0); // the image's own target, after its layers
                image.queue->submit (std::move (done));
            };

            return context;
        }
    }

    syncBackup (true);
    gpuValid = false;
    return backup.getPixelData()->createLowLevelContext();
}

//==============================================================================
/*  Per-window GPU renderer: retained target, instance buffers, presentation. */
class MetalLayerRenderer
{
public:
    static std::unique_ptr<MetalLayerRenderer> create()
    {
        auto renderer = rawToUniquePtr (new MetalLayerRenderer());
        return renderer->shared->isValid() ? std::move (renderer) : nullptr;
    }

    ~MetalLayerRenderer()
    {
        *missedPresentRepaint = nullptr; // repaint requests still queued are dropped (message thread)
        // Let frames still being encoded finish (they use this object).
        dispatch_sync (shared->encodeQueue, ^{});

        // Wait for the GPU, then give the tokens back: libdispatch traps when a semaphore is
        // released with a value below its initial one.
        for (int i = 0; i < numFramesInFlight; ++i)
            dispatch_semaphore_wait (inFlight, DISPATCH_TIME_FOREVER);

        for (int i = 0; i < numFramesInFlight; ++i)
            dispatch_semaphore_signal (inFlight);

        for (auto& f : frames)
        {
            [f.layer release];
            f.releases.releaseAll();

            for (auto t : f.pooled)
                [t release];
            f.buffers.release();
        }

        [retained release];
        dispatch_release (inFlight);
    }

    /*  Paints the dirty rects (in points) into the retained texture and presents it.
        Returns the rects that couldn't be drawn this time (to retry on the next refresh).
    */
    template <typename PaintFn>
    RectangleList<float> drawRectangleList (CAMetalLayer* layer, float scale, PaintFn&& paint,
                                            RectangleList<float> dirty, bool renderSync, bool opaque)
    {
        const auto width = (NSUInteger) std::lround (layer.bounds.size.width * scale);
        const auto height = (NSUInteger) std::lround (layer.bounds.size.height * scale);

        if (width == 0 || height == 0)
            return dirty;

        // All frames in flight: try again on the next refresh rather than blocking the message thread.
        // Every token taken from here on becomes a frame: slots are used in turn, so the slot
        // taken now is the one of the frame three frames ago, whose GPU work is complete.
        if (dispatch_semaphore_wait (inFlight, DISPATCH_TIME_NOW) != 0)
        {
            if (statistics != nullptr)
                ++statistics->skippedRefreshes;

            return dirty;
        }

        auto& frame = frames[(size_t) (frameIndex++ % numFramesInFlight)];
        frame.number = frameIndex;
        frame.releases.releaseAll();
        images->recycle (frame.jobs);
        images->trimTextures();
        shared->imageCache->collectReleases (frame.releases); // textures of changed or deleted images

        // Only when they change: setting them every frame would commit a CA transaction.
        if (! exactlyEqual (layer.contentsScale, (CGFloat) scale))
            layer.contentsScale = scale;

        if (layer.presentsWithTransaction != renderSync)
            layer.presentsWithTransaction = renderSync;

        // Also after switching engines, which creates a new layer at the same size.
        if (! CGSizeEqualToSize (layer.drawableSize, CGSizeMake (width, height)))
            layer.drawableSize = CGSizeMake (width, height);

        if (retained == nil || retained.width != width || retained.height != height)
        {
            frame.releases.add (retained);

            auto* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat: MTLPixelFormatBGRA8Unorm
                                                                            width: width
                                                                           height: height
                                                                        mipmapped: NO];
            desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            desc.storageMode = MTLStorageModePrivate;
            retained = [shared->device newTextureWithDescriptor: desc];

            dirty.clear();
            dirty.add (convertToRectFloat (layer.bounds));
        }

        // Dirty rects in device pixels, snapped outwards (repainting a little more is harmless).
        const Rectangle<int> targetBounds ((int) width, (int) height);
        RectangleList<int> region;

        for (const auto& r : dirty)
            region.add ((r * scale).getSmallestIntegerContainer().getIntersection (targetBounds));

        region.consolidate();
        frame.dirtyPixels = 0;

        for (const auto& r : region)
            frame.dirtyPixels += (int64) r.getWidth() * r.getHeight();

       #if JUCE_METAL_RENDERER_DIRTY_RECT
        dirtyRects->recorded (frame.number, region.getBounds());
       #endif

        if (layer.opaque != opaque)
            layer.opaque = opaque;

        [frame.layer release];
        frame.layer = [layer retain];
        frame.scale = scale;

        auto& recorder = frame.recorder;
        recorder.clear (targetBounds);
        recorder.frameReleases = &frame.releases;
        recorder.device = shared->device;
        recorder.glyphAtlas = shared->glyphAtlas.get();
        frame.globalNumber = ++shared->framesRecorded;
        shared->glyphAtlas->beginFrame (frame.globalNumber, shared->framesCompleted.load());
        shared->imageCache->beginFrame (frame.globalNumber, shared->framesCompleted.load());
        recorder.imageCache = shared->imageCache.get();
        recorder.glyphOutlines = &shared->glyphOutlines;

        if (! opaque)
        {
            for (const auto& r : region)
            {
                const auto f = r.toFloat();
                MetalRendering::Instance clear;
                clear.xf0 = { 1.0f, 0.0f, 0.0f, 0.0f };
                clear.xf1 = { 0.0f, 1.0f, 0.0f, 0.0f };
                clear.rect = clear.clip = { f.getX(), f.getY(), f.getWidth(), f.getHeight() };
                recorder.clears.push_back (clear);
            }
        }

        const auto paintStart = Time::getMillisecondCounterHiRes();

        // Images drawn into meanwhile (setBufferedToImage, shadows) queue their work: this frame
        // encodes it first.
        const auto previousRecorder = std::exchange (images->activeRecorder, &recorder);

        {
            MetalGraphicsContext context (recorder, scale, region);
            paint (context);
        }

        images->activeRecorder = previousRecorder;
        jassert (images->openContexts == 0); // a Graphics on a temporary image outlived the paint
        frame.jobs = images->takePending();
        recorder.encodeOrder.push_back (0); // the window's own target, after all its layers

        const auto paintMs = Time::getMillisecondCounterHiRes() - paintStart;
        id<MTLTexture> destination = retained;

        if (renderSync)
        {
            // Live resize: presented within the current CA transaction, so encoded right here,
            // once the frames before it have been.
            dispatch_sync (shared->encodeQueue, ^{});
            encodeFrame (frame, destination, renderSync, paintMs);
        }
        else
        {
            // The block retains the destination texture; the frame stays reserved until its GPU
            // work completes, after encoding.
            auto* framePtr = &frame;
            dispatch_async (shared->encodeQueue, ^{ encodeFrame (*framePtr, destination, false, paintMs); });
        }

        return {};
    }

    /*  Called on the message thread with an area (in points) to repaint: the window, when a
        frame couldn't be presented (see encodePresentCopy).
    */
    void setRepaintCallback (std::function<void (Rectangle<float>)> callback)
    {
        *missedPresentRepaint = std::move (callback);
    }

    /*  Set the environment variable JUCE_METAL_RENDERER_STATS to print, once per second and per
        window: frames, CPU time per frame (paint on the message thread + encode), GPU time per frame and the software
        fallbacks by reason.
    */
    struct Statistics : public MetalRendering::EncodeStatistics
    {
        void addFrame (double paintMs, MetalRendering::FrameRecorder& frameRecorder, int64 dirty, int jobs, int readBacks)
        {
            ++frames;
            paintMsTotal += paintMs;
            dirtyPixels += dirty;
            imageJobs += jobs;
            imageReadBacks += readBacks;

            // Scene size, to tell what the GPU time goes into.
            layers += (int64) frameRecorder.targets.size() - 1;
            masks += (int64) std::count_if (frameRecorder.masks.begin(), frameRecorder.masks.end(),
                                            [] (const auto& m) { return ! m.readsCoverage(); });
            instances += (int64) frameRecorder.instances.size();

            for (const auto& target : frameRecorder.targets)
                opaqueInstances += (int64) target.opaque.size();

            for (size_t i = 1; i < frameRecorder.targets.size(); ++i)
                layerPixels += (int64) frameRecorder.targets[i].area.getWidth() * frameRecorder.targets[i].area.getHeight();

            for (const auto& m : frameRecorder.masks)
                if (! m.readsCoverage())
                    maskPixels += (int64) m.area.getWidth() * m.area.getHeight();

            for (const auto& e : frameRecorder.coverageEntries)
                coveragePixels += (int64) e.width * e.height;

            for (auto& [reason, count] : frameRecorder.fallbackReasons)
                fallbacks[reason] += count;

            frameRecorder.fallbackReasons.clear();

            const auto now = Time::getMillisecondCounterHiRes();

            if (now - lastPrint < 1000.0)
                return;

            // Phase times are summed when their command buffers complete, a frame or two later:
            // averaged over the frames whose main pass completed meanwhile.
            const auto gpuFrameCount = jmax (1, gpuFrames.exchange (0));
            const auto phaseMs = [&] (int phase) { return String ((double) phaseMicros[phase].exchange (0) / 1000.0 / gpuFrameCount, 3); };

            String line;
            line << "[MetalRenderer " << String::toHexString ((pointer_sized_int) this) << "] "
                 << frames << " frames, cpu: paint " << String (paintMsTotal / frames, 3) << " + encode "
                 << String (encodeMsTotal / frames, 3) << " ms/frame, gpu "
                 << String ((double) gpuMicros.exchange (0) / 1000.0 / gpuFrameCount, 3) << " ms/frame"
                 << " (of which coverage accumulation " << phaseMs (0) << ", prefix sums " << phaseMs (1)
                 << ", masks " << phaseMs (2) << "), present copy "
                 << String ((double) presentMicros.exchange (0) / 1000.0 / gpuFrameCount, 3) << " ms/frame; waits for a drawable "
                 << String ((double) drawableWaitMicros.exchange (0) / 1000.0, 1) << " ms, " << missedDrawables.exchange (0) << " none, "
                 << skippedRefreshes.exchange (0) << " refreshes with all frames in flight; per frame: " << String ((double) instances / frames, 0) << " instances (" << String ((double) opaqueInstances / frames, 0) << " opaque) in "
                 << String ((double) drawCallsTotal / frames, 0) << " draw calls, "
                 << String ((double) layers / frames, 1) << " layers (" << String ((double) layerPixels / frames / 1.0e6, 2) << " Mpx), "
                 << String ((double) masks / frames, 1) << " mask textures (" << String ((double) maskPixels / frames / 1.0e6, 2) << " Mpx), "
                 << "path rectangles " << String ((double) coveragePixels / frames / 1.0e6, 2) << " Mpx, "
                 << "dirty " << String ((double) dirtyPixels / frames / 1.0e6, 2) << " Mpx, "
                 << "image jobs " << String ((double) imageJobs / frames, 1) << ", image read-backs " << imageReadBacks;

            if (! fallbacks.empty())
            {
                line << ", fallbacks:";

                for (auto& [reason, count] : fallbacks)
                    line << " " << reason << " " << String ((double) count / frames, 1) << "/frame";
            }

            std::fprintf (stderr, "%s\n", line.toRawUTF8());
            frames = 0;
            paintMsTotal = encodeMsTotal = 0.0;
            fallbacks.clear();
            layers = masks = instances = opaqueInstances = layerPixels = maskPixels = coveragePixels = 0;
            dirtyPixels = 0;
            drawCallsTotal = 0;
            imageJobs = imageReadBacks = 0;
            lastPrint = now;
        }

        std::atomic<int64> gpuMicros { 0 };
        std::atomic<int> gpuFrames { 0 };
        std::atomic<int64> presentMicros { 0 };
        std::atomic<int64> drawableWaitMicros { 0 }, missedDrawables { 0 }, skippedRefreshes { 0 };
        double encodeMsTotal = 0.0; // on the encode queue, like addFrame

    private:
        int frames = 0;
        int64 layers = 0, masks = 0, instances = 0, opaqueInstances = 0, layerPixels = 0, maskPixels = 0, coveragePixels = 0;
        int64 dirtyPixels = 0; // device pixels repainted
        int64 imageJobs = 0, imageReadBacks = 0;
        double paintMsTotal = 0.0, lastPrint = Time::getMillisecondCounterHiRes();
        std::map<std::string, int> fallbacks;
    };

private:
    static constexpr int numFramesInFlight = 3;

    struct Frame
    {
        MetalRendering::EncodeBuffers buffers;
        std::vector<std::unique_ptr<MetalRendering::ImageJob>> jobs; // image work encoded before the frame's passes
        std::vector<id<MTLTexture>> pooled;   // mask/layer textures, back to the pool when this frame completes
        MetalRendering::DeferredReleases releases;
        MetalRendering::FrameRecorder recorder; // painted on the message thread, then encoded
        uint64_t number = 0, globalNumber = 0;
        CAMetalLayer* layer = nil;  // the window's layer, presented to (retained)
        int64 dirtyPixels = 0;      // area of the dirty region (statistics)
        float scale = 1.0f;         // device pixels per point
    };

    MetalLayerRenderer()
        : inFlight (dispatch_semaphore_create (numFramesInFlight))
    {
        if (std::getenv ("JUCE_METAL_RENDERER_STATS") != nullptr)
        {
            statistics = std::make_shared<Statistics>();

            for (auto& f : frames)
                f.recorder.collectReasons = true;
        }
    }

    void encodeFrame (Frame& frame, id<MTLTexture> destination, bool renderSync, double paintMs)
    {
        @autoreleasepool
        {
            // Statistics read the scene first: once committed, the frame can complete and be
            // handed back to the message thread before this returns.
            if (statistics != nullptr)
                statistics->addFrame (paintMs, frame.recorder, frame.dirtyPixels, (int) frame.jobs.size(),
                                      images->readBacksSinceLastStatistics.exchange (0));

            const auto start = Time::getMillisecondCounterHiRes();
            MetalRendering::ImageQueue::prepare (*shared, frame.jobs, nullptr);

            for (const auto& job : frame.recorder.glyphJobs)
                shared->glyphAtlas->rasterise (job);

            frame.recorder.resolveCoverage();
            encoder.recycle (frame.pooled, frame.number);
            encodeAndPresent (frame, destination, renderSync);

            if (statistics != nullptr)
                statistics->encodeMsTotal += Time::getMillisecondCounterHiRes() - start;
        }
    }

    /*  Copies the frame into a new drawable of the window's layer and returns it to present, or
        nil: no drawable, or the layer was resized since (the frame painted at the new size
        follows).
        The window is one CAMetalLayer and every frame copies all of it, as toolkits drawing with
        Metal present (Visage, Qt, Slint, iPlug2, egui) and as CoreGraphicsMetalLayerRenderer does.
        Tested and not kept, measured on an M1 at 60 Hz:
        - A grid of CAMetalLayer tiles (sublayers), a frame copying and presenting only those it
          touches: far cheaper with a few small repaints, but once a frame presents a few dozen
          tiles WindowServer drops most of them and nextDrawable waits for its timeout (2-5 fps,
          seen from 35 tiles of 128 pt, intermittently). Larger tiles avoid that but then cost
          WindowServer more CPU than one layer.
        - Plain CALayer tiles showing IOSurfaces from a small pool per tile (as Firefox's native
          layers): no dropped frames and less GPU, but more WindowServer CPU: ~10% less CPU + GPU
          time than one layer with small repaints, ~10% more with full-window ones.
        - Drawing straight into full-window IOSurfaces set as layer contents (as Chromium, Qt and
          GTK do): saves the copy (~0.2 ms of GPU per frame) but WindowServer spent ~5.6 ms more
          CPU per frame.
        - A render pass drawing a quad instead of the blit, with framebufferOnly = YES (as Godot and
          SDL3 copy into drawables): the copy took 20-40% longer, WindowServer gained nothing.
    */
    static id<CAMetalDrawable> encodePresentCopy (id<MTLCommandBuffer> commandBuffer, id<MTLTexture> source, CAMetalLayer* layer)
    {
        id<CAMetalDrawable> drawable = [layer nextDrawable];

        if (drawable == nil || drawable.texture.width != source.width || drawable.texture.height != source.height)
            return nil;

        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        blit.label = @"JUCE window copy";
        [blit copyFromTexture: source sourceSlice: 0 sourceLevel: 0
                 sourceOrigin: MTLOriginMake (0, 0, 0)
                   sourceSize: MTLSizeMake (source.width, source.height, 1)
                    toTexture: drawable.texture destinationSlice: 0 destinationLevel: 0
            destinationOrigin: MTLOriginMake (0, 0, 0)];
        [blit endEncoding];
        return drawable;
    }

    void encodeAndPresent (Frame& frame, id<MTLTexture> destination, bool renderSync)
    {
        id<MTLCommandBuffer> commandBuffer = [shared->queue commandBuffer];

        for (auto& job : frame.jobs)
            encoder.encodeJob (*job, frame.pooled, commandBuffer);

        encoder.encode (frame.buffers, frame.pooled, frame.recorder, commandBuffer, destination, statistics);

        auto semaphore = inFlight;
        auto stats = statistics;
        auto* sharedResources = shared.operator->();
        const auto globalNumber = frame.globalNumber;
        [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> done)
        {
            sharedResources->framesCompleted.store (globalNumber);

            if (stats != nullptr)
            {
                // From the start of the frame's first command buffer: one queue completes them in order.
                const auto start = stats->frameGpuStart.exchange (0.0);
                stats->gpuMicros += (int64) ((done.GPUEndTime - (start > 0.0 ? start : done.GPUStartTime)) * 1.0e6);
                ++stats->gpuFrames;
            }

            dispatch_semaphore_signal (semaphore);
        }];

        id<MTLTexture> source = destination;
        id<MTLCommandQueue> queue = shared->queue;
        CAMetalLayer* layer = frame.layer; // the blocks below retain what they capture
        const auto windowArea = Rectangle<int> ((int) source.width, (int) source.height).toFloat() / frame.scale;
        auto repaint = missedPresentRepaint;

       #if JUCE_METAL_RENDERER_DIRTY_RECT
        auto dirty = dirtyRects;
        const auto frameNumber = frame.number;
        const auto markDirty = [dirty, frameNumber, layer] (id<CAMetalDrawable> drawable) { dirty->beforePresent (drawable, frameNumber, layer); };
       #else
        const auto markDirty = [] (id<CAMetalDrawable>) {};
       #endif

        // Without a drawable the window keeps showing an older frame until it next repaints.
        const auto requestRepaint = [repaint, windowArea]
        {
            dispatch_async (dispatch_get_main_queue(), ^
            {
                if (*repaint != nullptr)
                    (*repaint) (windowArea);
            });
        };

        if (renderSync)
        {
            // Live resize: the frame must be presented within the current CA transaction.
            @autoreleasepool
            {
                id<CAMetalDrawable> drawable = encodePresentCopy (commandBuffer, source, layer);
                [commandBuffer commit];
                [commandBuffer waitUntilScheduled];

                if (drawable != nil)
                {
                    markDirty (drawable);
                    [drawable present];
                }
                else
                {
                    requestRepaint();
                }
            }

            return;
        }

        // nextDrawable may block until the compositor gives one back: done on a Metal thread,
        // once the paint has been scheduled (same approach as CoreGraphicsMetalLayerRenderer).
        [commandBuffer addScheduledHandler: ^(id<MTLCommandBuffer>)
        {
            @autoreleasepool
            {
                id<MTLCommandBuffer> present = [queue commandBuffer];
                const auto waitStart = Time::getMillisecondCounterHiRes();
                id<CAMetalDrawable> drawable = encodePresentCopy (present, source, layer);

                if (stats != nullptr)
                    stats->drawableWaitMicros += (int64) ((Time::getMillisecondCounterHiRes() - waitStart) * 1000.0);

                if (drawable == nil)
                {
                    if (stats != nullptr)
                        ++stats->missedDrawables;

                    requestRepaint();
                    return;
                }

                if (stats != nullptr)
                    [present addCompletedHandler: ^(id<MTLCommandBuffer> done)
                    {
                        stats->presentMicros += (int64) ((done.GPUEndTime - done.GPUStartTime) * 1.0e6);
                    }];

                markDirty (drawable);
                [present presentDrawable: drawable];
                [present commit];
            }
        }];

        [commandBuffer commit];
    }

    SharedResourcePointer<MetalRendering::SharedResources> shared;
    SharedResourcePointer<MetalRendering::ImageQueue> images;
    MetalRendering::SceneEncoder encoder { *shared };
    std::array<Frame, numFramesInFlight> frames;
    uint64_t frameIndex = 0;
    id<MTLTexture> retained = nil;
    dispatch_semaphore_t inFlight;
    std::shared_ptr<Statistics> statistics; // shared with completion handlers, which can outlive this object
    // Shared with the blocks that request repaints, which can outlive this object.
    std::shared_ptr<std::function<void (Rectangle<float>)>> missedPresentRepaint = std::make_shared<std::function<void (Rectangle<float>)>>();

   #if JUCE_METAL_RENDERER_DIRTY_RECT
    /*  JUCE_METAL_RENDERER_DIRTY_RECT: tells Core Animation which part of each drawable changed,
        through the private -[CAMetalDrawable setDirtyRect:] (QuartzCore hands it to the layer's
        image queue on present; the default is CGRectInfinite), in drawable pixels with the origin
        at the top left. WindowServer then recomposites that rect rather than the whole window
        (measured on an M1, one small area repainting at 60 Hz: WindowServer GPU 98 -> 3 ms/s,
        CPU 358 -> 255 ms/s; ScreenCaptureKit's dirty rects show the same rect). The drawable
        still receives the whole frame, so what it shows never depends on this.
        WindowServer may skip a frame (two presents in one refresh), and the next frame's rect
        must then cover what the skipped one changed: the rect is everything recorded since the
        last frame known to be on screen (its presented handler reports a presentedTime), or the
        whole drawable when that isn't known (a new layer, or too many frames since). One rect per
        frame: separate dirty areas count as their bounding box.
        Shared with the present blocks (Metal threads), which can outlive the renderer.
    */
    class DirtyRects : public std::enable_shared_from_this<DirtyRects>
    {
    public:
        // Message thread: the dirty bounds (device pixels) of a frame about to be encoded.
        void recorded (uint64_t frame, Rectangle<int> bounds)
        {
            const std::scoped_lock sl (mutex);
            history[frame % history.size()] = { frame, bounds };
        }

        // Metal thread, just before the frame is presented.
        void beforePresent (id<CAMetalDrawable> drawable, uint64_t frame, CAMetalLayer* layer)
        {
            static const auto setDirtyRect = NSSelectorFromString (@"setDirtyRect:");
            static const auto available = [(id) drawable respondsToSelector: setDirtyRect];

            if (! available)
                return;

            if (const auto bounds = changedSinceShown (frame, layer))
                ((void (*) (id, SEL, CGRect)) objc_msgSend) (drawable, setDirtyRect, makeCGRect (*bounds));

            auto self = shared_from_this();
            [drawable addPresentedHandler: ^(id<MTLDrawable> presented)
            {
                if (presented.presentedTime > 0)
                    self->onScreen (frame, layer);
            }];
        }

    private:
        // Nothing (the whole drawable) unless every frame since the one on screen is known.
        std::optional<Rectangle<int>> changedSinceShown (uint64_t frame, CAMetalLayer* layer)
        {
            const std::scoped_lock sl (mutex);

            if (layer != shownLayer || shown == 0 || frame <= shown || frame - shown > history.size())
                return {};

            Rectangle<int> bounds;

            for (auto f = shown + 1; f <= frame; ++f)
            {
                const auto& entry = history[f % history.size()];

                if (entry.frame != f)
                    return {};

                bounds = bounds.getUnion (entry.bounds);
            }

            return bounds;
        }

        void onScreen (uint64_t frame, CAMetalLayer* layer)
        {
            const std::scoped_lock sl (mutex);

            if (layer != shownLayer)
            {
                shownLayer = layer;
                shown = frame;
            }
            else
            {
                shown = jmax (shown, frame);
            }
        }

        struct Entry { uint64_t frame = 0; Rectangle<int> bounds; };
        std::mutex mutex;
        std::array<Entry, 8> history;
        uint64_t shown = 0;               // the last frame known to be on screen
        const void* shownLayer = nullptr; // its layer (not retained: only compared)
    };

    std::shared_ptr<DirtyRects> dirtyRects = std::make_shared<DirtyRects>();
   #endif

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MetalLayerRenderer)
    JUCE_DECLARE_NON_MOVEABLE (MetalLayerRenderer)
};

} // namespace juce

JUCE_END_IGNORE_WARNINGS_GCC_LIKE
