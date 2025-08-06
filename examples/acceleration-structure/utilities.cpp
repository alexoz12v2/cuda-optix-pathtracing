#include "utilities.h"

#include "platform/platform-context.h"

#if !defined(DMT_ARCH_X86_64)
    #error "what"
#endif

#include <ImfRgbaFile.h>
#include <ImfArray.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <utility>
#include <random>

#include <immintrin.h>

namespace dmt {
    bool DMT_FASTCALL slabTest(Point3f rayOrigin, Vector3f rayDirection, Bounds3f const& box, float* outTmin, float* outTmax)
    {
        float tmin = -std::numeric_limits<float>::infinity();
        float tmax = std::numeric_limits<float>::infinity();

        for (int i = 0; i < 3; ++i)
        {
            float invD = 1.0f / rayDirection[i];
            float t0   = (box.pMin[i] - rayOrigin[i]) * invD;
            float t1   = (box.pMax[i] - rayOrigin[i]) * invD;

            if (invD < 0.0f)
                std::swap(t0, t1);

            tmin = std::max(tmin, t0);
            tmax = std::min(tmax, t1);

            if (tmax < tmin)
                return false; // No intersection
        }

        if (outTmin)
            *outTmin = tmin;
        if (outTmax)
            *outTmax = tmax;

        return true; // Intersection occurred
    }

    std::pmr::vector<UniqueRef<Primitive>> makeSinglePrimitivesFromTriangles(std::span<TriangleData const> tris,
                                                                             std::pmr::memory_resource*    memory)
    {
        std::pmr::vector<UniqueRef<Primitive>> out(memory);
        for (uint64_t i = 0; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    std::pmr::vector<UniqueRef<Primitive>> makePrimitivesFromTriangles(std::span<TriangleData const> tris,
                                                                       std::pmr::memory_resource*    memory)
    {
        std::pmr::vector<UniqueRef<Primitive>> out(memory);
        size_t                                 i = 0;

        // Pass 1: Triangles8
        for (; i + 8 <= tris.size(); i += 8)
        {
            Triangles8 group{};
            for (int j = 0; j < 8; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
                group.colors[j]     = tris[i + j].color;
            }
            out.push_back(makeUniqueRef<Triangles8>(memory, std::move(group)));
        }

        // Pass 2: Triangles4
        for (; i + 4 <= tris.size(); i += 4)
        {
            Triangles4 group{};
            for (int j = 0; j < 4; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
                group.colors[j]     = tris[i + j].color;
            }
            out.push_back(makeUniqueRef<Triangles4>(memory, std::move(group)));
        }

        // Pass 3: Triangles2
        for (; i + 2 <= tris.size(); i += 2)
        {
            Triangles2 group{};
            for (int j = 0; j < 2; ++j)
            {
                group.xs[3 * j + 0] = tris[i + j].v0.x;
                group.xs[3 * j + 1] = tris[i + j].v1.x;
                group.xs[3 * j + 2] = tris[i + j].v2.x;

                group.ys[3 * j + 0] = tris[i + j].v0.y;
                group.ys[3 * j + 1] = tris[i + j].v1.y;
                group.ys[3 * j + 2] = tris[i + j].v2.y;

                group.zs[3 * j + 0] = tris[i + j].v0.z;
                group.zs[3 * j + 1] = tris[i + j].v1.z;
                group.zs[3 * j + 2] = tris[i + j].v2.z;
                group.colors[j]     = tris[i + j].color;
            }
            out.push_back(makeUniqueRef<Triangles2>(memory, std::move(group)));
        }

        // Pass 4: Individual Triangle
        for (; i < tris.size(); ++i)
        {
            Triangle group{};
            group.tri = tris[i];
            out.push_back(makeUniqueRef<Triangle>(memory, std::move(group)));
        }

        return out;
    }

    uint32_t morton3D(float x, float y, float z)
    {
        constexpr auto expandBits = [](uint32_t v) -> uint32_t {
            // Expands 10 bits into 30 bits by inserting 2 zeros between each bit
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        };
        // Assumes x, y, z are ∈ [0, 1]
        x = std::clamp(x * 1024.0f, 0.0f, 1023.0f);
        y = std::clamp(y * 1024.0f, 0.0f, 1023.0f);
        z = std::clamp(z * 1024.0f, 0.0f, 1023.0f);

        uint32_t xx = expandBits(static_cast<uint32_t>(x));
        uint32_t yy = expandBits(static_cast<uint32_t>(y));
        uint32_t zz = expandBits(static_cast<uint32_t>(z));

        return (xx << 2) | (yy << 1) | zz;
    }

    void reorderByMorton(std::span<TriangleData> tris)
    {
        Bounds3f bounds = bbEmpty();
        for (auto& t : tris)
            bounds = bbUnion(bounds, Bounds3f{min(min(t.v0, t.v1), t.v2), max(max(t.v0, t.v1), t.v2)});

        auto const getMortonIndex = [bounds](TriangleData const& t) -> uint32_t {
            Point3f  c = (t.v0 + t.v1 + t.v2) / 3;
            Point3f  n = bounds.offset(c); // Normalize to [0,1]
            uint32_t m = morton3D(n.x, n.y, n.z);
            return m;
        };

        std::sort(tris.begin(), tris.end(), [&getMortonIndex](auto const& a, auto const& b) {
            return getMortonIndex(a) < getMortonIndex(b);
        });
    }

    ScanlineRange2D::Iterator::Iterator() : m_p({{-1, -1}}), m_res({{-2, -2}}) {}

    ScanlineRange2D::Iterator::Iterator(Point2i p, Point2i res) : m_p(p), m_res(res) {}

    ScanlineRange2D::Iterator::value_type ScanlineRange2D::Iterator::operator*() const { return m_p; }

    ScanlineRange2D::Iterator& ScanlineRange2D::Iterator::operator++()
    {
        ++m_p.x;
        if (m_p.x >= m_res.x)
        {
            m_p.x = 0;
            ++m_p.y;
            if (m_p.y >= m_res.y)
                m_p.x = -1;
        }

        return *this;
    }

    ScanlineRange2D::Iterator ScanlineRange2D::Iterator::operator++(int)
    {
        Iterator temp = *this;
        ++*this;
        return temp;
    }

    bool ScanlineRange2D::Iterator::operator==(End) const { return m_p.x == -1; }
    bool ScanlineRange2D::Iterator::operator==(Iterator const& other) const
    {
        return m_p == other.m_p && m_res == other.m_res;
    }

    ScanlineRange2D::ScanlineRange2D(Point2i resolution) : m_resolution(resolution)
    {
        assert(m_resolution.x > 0 && m_resolution.y > 0 && "Invalid Resolution");
    }

    ScanlineRange2D::Iterator ScanlineRange2D::begin() const { return Iterator({{0, 0}}, m_resolution); }

    ScanlineRange2D::End ScanlineRange2D::end() const { return {}; }

    bool openEXR(os::Path const& imagePath, RGB** buffer, int32_t* xRes, int32_t* yRes, std::pmr::memory_resource* temp)
    {
        if (!imagePath.isValid() || !imagePath.isFile())
            return false;

        try
        {
            auto const         str = imagePath.toUnderlying(temp);
            Imf::RgbaInputFile file(str.c_str());
            Imath::Box2i       dw = file.dataWindow();

            int32_t width  = dw.max.x - dw.min.x + 1;
            int32_t height = dw.max.y - dw.min.y + 1;

            if (!buffer)
            {
                assert(xRes && yRes);
                *xRes = width;
                *yRes = height;
                return true;
            }
            else
            {
                assert(xRes && yRes && *xRes > 0 && *yRes > 0);
                assert(*xRes == width && *yRes == height); // sanity check

                auto pixels = makeUniqueRef<Imf::Array2D<Imf::Rgba>>(temp, height, width); // note: height x width
                file.setFrameBuffer(&(*pixels)[0][0] - dw.min.x - dw.min.y * width, 1, width); // Adjust for dataWindow origin
                file.readPixels(dw.min.y, dw.max.y);

                for (int32_t y = 0; y < height; ++y)
                {
                    for (int32_t x = 0; x < width; ++x)
                    {
                        (*buffer)[x + y * width].r = static_cast<float>((*pixels)[y][x].r);
                        (*buffer)[x + y * width].g = static_cast<float>((*pixels)[y][x].g);
                        (*buffer)[x + y * width].b = static_cast<float>((*pixels)[y][x].b);
                    }
                }

                return true;
            }

        } catch (...)
        {
            return false;
        }
    }

    bool writePNG(os::Path const& imgPath, RGB const* buffer, int32_t xRes, int32_t yRes, std::pmr::memory_resource* temp)
    {
        size_t const numBytes    = static_cast<size_t>(xRes) * yRes * 3;
        int const    rowStride   = xRes * 3;
        auto         imageBuffer = makeUniqueRef<unsigned char[]>(temp, numBytes);
        if (!imageBuffer)
            return false;

        std::memset(imageBuffer.get(), 0, numBytes);
        for (size_t y = 0; y < yRes; ++y)
        {
            for (size_t x = 0; x < xRes; ++x)
            {
                size_t const bufferIdx = x + y * xRes;
                size_t const imageIdx  = (y * xRes + x) * 3;

                imageBuffer[imageIdx + 0] = static_cast<unsigned char>(fl::clamp01(buffer[bufferIdx].r) * 255.f);
                imageBuffer[imageIdx + 1] = static_cast<unsigned char>(fl::clamp01(buffer[bufferIdx].g) * 255.f);
                imageBuffer[imageIdx + 2] = static_cast<unsigned char>(fl::clamp01(buffer[bufferIdx].b) * 255.f);
            }
        }

        return stbi_write_png(imgPath.toUnderlying(temp).c_str(), xRes, yRes, 3, imageBuffer.get(), rowStride);
    }
} // namespace dmt

namespace dmt::test {
    void testGGXconductor(uint32_t numSamples, Vector3f wo)
    {
        Context ctx;
        assert(ctx.isValid());

        ctx.warn("--------- TESTING GGX CONDUCTOR BRDF -------------", {});
        ctx.warn("it's directional-hemispherical albedo should be less than 1", {});

        std::mt19937                          rng(42);
        std::uniform_real_distribution<float> uDist(0.f, 1.f);

        Normal3f const n{0, 0, 1};
        RGB            etaGold{0.155f, 0.424f, 1.345};
        RGB            etakGold{3.911f, 2.345f, 1.770f};

        float sumR = 0.f, sumG = 0.f, sumB = 0.f;
        ctx.log("  Initialized R = 0, G = 0, B = 0", {});

        for (uint32_t sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
        {
            Point2f u{uDist(rng), uDist(rng)}; // sampling for GGX
            float   uc = uDist(rng);           // optional extra random number

            ggx::BSDF bsdf = ggx::makeConductor(wo, n, n, 0.2f, 0.05f, Vector3f::xAxis(), etaGold, etakGold);


            // Sample the BSDF properly
            ggx::BSDFSample sample = ggx::sample(bsdf, wo, n, u, uc);

            float    pdf = sample.pdf;
            RGB      fr  = sample.f * bsdf.closure.sampleWeight;
            Vector3f wi  = sample.wi;

            if (pdf > 0.f && fr.max() > 0.f && wi.z > 0.f)
            {
                float cosThetaWi = dot(wi, bsdf.closure.N);
                sumR += fr.r * cosThetaWi / pdf;
                sumG += fr.g * cosThetaWi / pdf;
                sumB += fr.b * cosThetaWi / pdf;
            }

            ctx.log("  Updated R = {} G = {} B = {}", std::make_tuple(sumR, sumG, sumB));
        }

        // Correct: do NOT multiply again by 2π — the GGX sampling already handles it
        float integralR = sumR / numSamples;
        float integralG = sumG / numSamples;
        float integralB = sumB / numSamples;

        ctx.log("  Integral R = {} G = {} B = {}", std::make_tuple(integralR, integralG, integralB));

        if (!(integralR <= 1.0f + 1e-3f && integralG <= 1.0f + 1e-3f && integralB <= 1.0f + 1e-3f))
        {
            ctx.error("  directional-hemispherical albedo for GGX BSDF is not normalized", {});
            assert(false && "directional-hemispherical albedo for GGX BSDF is not normalized");
        }
    }

    void bvhTestRays(dmt::BVHBuildNode* rootNode)
    {
        Context ctx;
        assert(ctx.isValid() && "Invalid context");
        std::vector<Ray> testRays = {
            // Straight through center of scene
            Ray({{0.5f, 0.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{1.5f, 1.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{2.5f, 2.5f, -1.0f}}, {{0, 0, 1}}),
            Ray({{3.5f, 3.5f, -1.0f}}, {{0, 0, 1}}),

            // Grazing edges
            Ray({{1.0f, 1.0f, -1.0f}}, {{0, 0, 1}}),
            Ray({{4.0f, 4.0f, -1.0f}}, {{0, 0, 1}}),

            // Missing all
            Ray({{5.0f, 5.0f, -1.0f}}, {{0, 0, 1}}),
            Ray({{-1.0f, -1.0f, -1.0f}}, {{0, 0, 1}}),

            // Diagonal through stack
            Ray({{-1.0f, -1.0f, 1.0f}}, {{1, 1, 1}}),
            Ray({{0.5f, 0.5f, 0.5f}}, {{1, 1, 1}}),

            // Through nested box
            Ray({{1.0f, 1.0f, 1.0f}}, {{0, 1, 0}}),
        };

        for (size_t i = 0; i < testRays.size(); ++i)
        {
            BVHBuildNode* hit = bvh::traverseBVHBuild(testRays[i], rootNode);
            if (hit)
            {
                ctx.log("Ray {} hit leaf bounding box: min = ({}, {}, {}), max = ({}, {}, {})",
                        std::make_tuple(i,
                                        hit->bounds.pMin.x,
                                        hit->bounds.pMin.y,
                                        hit->bounds.pMin.z,
                                        hit->bounds.pMax.x,
                                        hit->bounds.pMax.y,
                                        hit->bounds.pMax.z));
            }
            else
            {
                ctx.log("Ray {} missed the scene.", std::make_tuple(i));
            }
        }
    }

    void testDistribution1D()
    {
        static constexpr uint32_t Size = 64;
        Context                   ctx;
        assert(ctx.isValid() && "Invalid Context");

        float const min = 0.f;
        float const max = 12.f;

        auto const printSpan = [&ctx](std::span<float const> fView, uint32_t lineCount = 16) -> void {
            ctx.log("Size = {}", std::make_tuple(fView.size()));
            std::string str;
            str.reserve(128);
            str += "\n{ ";
            uint32_t currentCount = 0;
            uint32_t index        = 0;
            for (float f : fView)
            {
                str += std::to_string(f);
                if (++currentCount >= lineCount && index < fView.size() - 1)
                {
                    currentCount = 0;
                    str += "\n  ";
                }
                else if (index < fView.size() - 1)
                    str += " ";
                ++index;
            }
            ctx.log("{} }}", std::make_tuple(str));
        };

        std::vector<float> linearFunc;
        linearFunc.reserve(Size);

        for (uint32_t i = 0; i < Size; ++i)
            linearFunc.push_back((static_cast<float>(i) - min) / Size * (max - min));

        assert(linearFunc.size() == Size);

        printSpan(linearFunc);
        // Create the distribution object
        PiecewiseConstant1D dist(linearFunc, min, max);

        // Print the absolute value of the input function used internally
        ctx.log("absFunc:", {});
        printSpan(dist.absFunc());

        // Print the CDF
        ctx.log("CDF:", {});
        printSpan(dist.CDF());

        // Check that the CDF is normalized (i.e., last element == 1.0f)
        float lastCDF = dist.CDF().back();
        assert(std::abs(lastCDF - 1.0f) < 1e-4f && "CDF is not normalized");

        // Check that integral() returns the expected total area under the curve
        float expectedIntegral = 0.f;
        float dx               = (max - min) / Size;
        for (float f : linearFunc)
            expectedIntegral += f * dx;

        float computedIntegral = dist.integral();
        ctx.log("Expected integral = {}, computed = {}", std::make_tuple(expectedIntegral, computedIntegral));
        assert(std::abs(expectedIntegral - computedIntegral) < 1e-4f && "Integral mismatch");

        // Sample a few points
        for (float u : {0.1f, 0.5f, 0.9f})
        {
            float   pdf     = 0.f;
            int32_t offset  = -1;
            float   sampleX = dist.sample(u, &pdf, &offset);
            ctx.log("Sample(u = {}) = {}, pdf = {}, offset = {}", std::make_tuple(u, sampleX, pdf, offset));
        }
    }

    void testDistribution2D()
    {
        static constexpr uint32_t X = 8;
        static constexpr uint32_t Y = 8;

        Context ctx;
        assert(ctx.isValid() && "Invalid Context");

        float const min = 0.f;
        float const max = 1.f;

        Bounds2f domain{{{min, min}}, {{max, max}}};

        // Fill the input data with a separable function (e.g., f(x, y) = x + y)
        dstd::Array2D<float> func(X, Y);
        for (uint32_t y = 0; y < Y; ++y)
        {
            for (uint32_t x = 0; x < X; ++x)
            {
                float fx   = static_cast<float>(x) / X;
                float fy   = static_cast<float>(y) / Y;
                func(x, y) = fx + fy;
            }
        }

        // Print input function
        auto printRowMajor2D = [&ctx](dstd::Array2D<float> const& arr) {
            for (uint32_t y = 0; y < arr.ySize(); ++y)
            {
                std::string row = "[ ";
                for (uint32_t x = 0; x < arr.xSize(); ++x)
                    row += std::to_string(arr(x, y)) + " ";
                row += "]";
                ctx.log("{}", std::make_tuple(row));
            }
        };

        ctx.log("Input Function:", {});
        printRowMajor2D(func);

        // Construct the distribution
        PiecewiseConstant2D dist(func, domain);

        // Check integral
        float expectedIntegral = 0.f;
        float dx               = (domain.pMax.x - domain.pMin.x) / X;
        float dy               = (domain.pMax.y - domain.pMin.y) / Y;

        for (uint32_t y = 0; y < Y; ++y)
            for (uint32_t x = 0; x < X; ++x)
                expectedIntegral += func(x, y) * dx * dy;

        float computedIntegral = dist.integral();
        ctx.log("Expected integral = {}, computed = {}", std::make_tuple(expectedIntegral, computedIntegral));
        assert(std::abs(expectedIntegral - computedIntegral) < 1e-4f && "Integral mismatch");

        // Sample some points and validate PDF
        for (Point2f u : {Point2f{{0.1f, 0.1f}}, Point2f{{0.5f, 0.5f}}, Point2f{{0.9f, 0.9f}}})
        {
            float   pdf = -1.f;
            Point2i offset{{-1, -1}};
            Point2f sample  = dist.sample(u, &pdf, &offset);
            float   evalPDF = dist.pdf(sample);

            ctx.log("Sample(u = [{}, {}]) = [{}, {}], PDF = {}, EvalPDF = {}, offset = ({}, {})",
                    std::make_tuple(u.x, u.y, sample.x, sample.y, pdf, evalPDF, offset.x, offset.y));

            // Optional: check PDF value at sampled location is close to reported PDF
            assert(std::abs(evalPDF - pdf) < 1e-3f && "PDF mismatch at sample point");
        }

        // Inversion test (maps sample back to uniform distribution)
        // Strengthened inversion test
        Point2f uOriginal{{0.25f, 0.25f}}; // uniform sample in [0,1]^2
        float   pdf = 0.f;
        Point2i offset{};
        Point2f pSampled  = dist.sample(uOriginal, &pdf, &offset);
        Point2f uInverted = dist.invert(pSampled);

        ctx.log("Sample(uOriginal) = [{}, {}], Invert(pSampled) = [{}, {}]",
                std::make_tuple(uOriginal.x, uOriginal.y, uInverted.x, uInverted.y));

        bool closeEnough = (std::abs(uOriginal.x - uInverted.x) < 1e-4f) && (std::abs(uOriginal.y - uInverted.y) < 1e-4f);
        assert(closeEnough && "Inversion test failed: inverted value not close to original sample");
    }

    void testOctahedralProj()
    {
        Context ctx;
        assert(ctx.isValid() && "invalid context");
        static constexpr float tol = 0.1f;

        ctx.log(" -- Testing Octahedral Projection (tolerance = {} degrees) --", std::make_tuple(tol));

        Normal3f const       orig{1.f, 2.f, -0.3f};
        OctahedralNorm const octa      = octaFromNorm(orig);
        Normal3f const       projected = normFromOcta(octa);

        float angularErrorRad = std::acos(fl::clamp(dot(projected, orig), -1.0f, 1.0f));
        float angularErrorDeg = angularErrorRad * fl::degFromRad();
        if (angularErrorDeg > tol)
        {
            ctx.error("Incorrect octahedral reconstruction", {});
            assert(false && "Incorrect octahedral reconstruction");
        }

        ctx.warn("Projection works on a tolerance of {}", std::make_tuple(tol));
    }

    static bool compareIntersections(Intersection const& a, Intersection const& b)
    {
        if (a.hit != b.hit)
            return false;
        if (!a.hit)
            return true; // both missed
        static constexpr float eps = 1e-4f;
        return normL2(a.p - b.p) < eps && fl::abs(a.t - b.t) < eps;
    }

    static void testRayAgainstAllGroupedVsUngrouped(std::vector<UniqueRef<Primitive>> const& ungrouped,
                                                    std::vector<UniqueRef<Primitive>> const& grouped,
                                                    Ray const&                               ray)
    {
        Context ctx;
        assert(ctx.isValid() && "Invalid context");

        Intersection closestUngrouped{};
        closestUngrouped.t = fl::infinity();
        for (auto const& prim : ungrouped)
        {
            Intersection its = prim->intersect(ray, closestUngrouped.t);
            if (its.hit && its.t < closestUngrouped.t)
                closestUngrouped = its;
        }

        Intersection closestGrouped{};
        closestGrouped.t = fl::infinity();
        for (auto const& prim : grouped)
        {
            Intersection its = prim->intersect(ray, closestGrouped.t);
            if (its.hit && its.t < closestGrouped.t)
                closestGrouped = its;
        }

        if (!compareIntersections(closestUngrouped, closestGrouped))
        {
            ctx.error("Mismatch:", {});
            ctx.error("Ungrouped hit: {}, p = {} {} {}, t = {}",
                      std::make_tuple(closestUngrouped.hit,
                                      closestUngrouped.p.x,
                                      closestUngrouped.p.y,
                                      closestUngrouped.p.z,
                                      closestUngrouped.t));
            ctx.error("Grouped hit: {}, p = {} {} {}, t = {}",
                      std::make_tuple(closestGrouped.hit,
                                      closestGrouped.p.x,
                                      closestGrouped.p.y,
                                      closestGrouped.p.y,
                                      closestGrouped.p.z,
                                      closestGrouped.t));
            assert(false && "Grouped Triangles and Ungrouped triangles Intersection results differ");
        }
    }

    void testIndexedTriangleGrouping()
    {
        Context ctx;
        assert(ctx.isValid() && "invalid context");

        auto* mem = std::pmr::get_default_resource();

        Scene scene;
        {
            scene.geometry.emplace_back(makeUniqueRef<TriangleMesh>(mem));
            auto& cube = *scene.geometry.back();
            TriangleMesh::unitCube(cube);
        }
        {
            scene.instances.emplace_back(makeUniqueRef<Instance>(mem));
            auto& cubeInstance   = *scene.instances.back();
            cubeInstance.meshIdx = 0;

            Transform const t = Transform::translate({0.f, 1.5f, -0.75f}) * Transform::rotate(45.f, Vector3f::zAxis()) *
                                Transform::scale(0.5f);

            extractAffineTransform(t.m, cubeInstance.affineTransform);
            cubeInstance.bounds = scene.geometry[cubeInstance.meshIdx]->transformedBounds(t);
            cubeInstance.color  = color::rgbFromHsv({0.01f, 0.05f, 0.7f});
        }

        std::vector<UniqueRef<Primitive>> ungroupedPrims;
        std::vector<UniqueRef<Primitive>> groupedPrims;

        Instance const&     instance = *scene.instances[0];
        TriangleMesh const& mesh     = *scene.geometry[instance.meshIdx];
        for (size_t tri = 0; tri < mesh.triCount(); ++tri)
        {
            auto prim         = makeUniqueRef<TriangleIndexed>(mem);
            prim->scene       = &scene;
            prim->instanceIdx = 0;
            prim->triIdx      = tri;
            ungroupedPrims.push_back(std::move(prim));
        }

        {
            size_t tri   = 0;
            bool   found = false;
            for (; tri + 7 < mesh.triCount(); tri += 8)
            {
                if (!found)
                {
                    found = true;
                    ctx.warn("Found in test a triangle 8", {});
                }
                auto prim         = makeUniqueRef<TrianglesIndexed8>(mem);
                prim->scene       = &scene;
                prim->instanceIdx = 0;
                for (size_t i = 0; i < 7; ++i)
                    prim->triIdxs[i] = tri + i;
                groupedPrims.push_back(std::move(prim));
            }

            found = false;
            for (; tri + 3 < mesh.triCount(); tri += 4)
            {
                if (!found)
                {
                    found = true;
                    ctx.warn("Found in test a triangle 4", {});
                }
                auto prim         = makeUniqueRef<TrianglesIndexed4>(mem);
                prim->scene       = &scene;
                prim->instanceIdx = 0;
                for (size_t i = 0; i < 4; ++i)
                    prim->triIdxs[i] = tri + i;
                groupedPrims.push_back(std::move(prim));
            }

            found = false;
            for (; tri + 1 < mesh.triCount(); tri += 2)
            {
                if (!found)
                {
                    found = true;
                    ctx.warn("Found in test a triangle 2", {});
                }
                auto prim         = makeUniqueRef<TrianglesIndexed2>(mem);
                prim->scene       = &scene;
                prim->instanceIdx = 0;
                for (size_t i = 0; i < 2; ++i)
                    prim->triIdxs[i] = tri + i;
                groupedPrims.push_back(std::move(prim));
            }

            found = false;
            for (; tri < mesh.triCount(); tri++)
            {
                if (!found)
                {
                    found = true;
                    ctx.warn("Found in test a triangle remainder", {});
                }
                auto prim         = makeUniqueRef<TriangleIndexed>(mem);
                prim->scene       = &scene;
                prim->instanceIdx = 0;
                prim->triIdx      = tri;
                groupedPrims.push_back(std::move(prim));
            }
        }

        ctx.log("BOUNDS TEST", {});
        auto boundsEqual = [](Bounds3f const& a, Bounds3f const& b, float eps = 1e-4f) -> bool {
            return all(abs(a.pMin - b.pMin) < Vector3f::s(eps)) && all(abs(a.pMax - b.pMax) < Vector3f::s(eps));
        };

        size_t ungroupedIndex = 0;
        for (auto const& groupedPrim : groupedPrims)
        {
            Bounds3f groupedBB   = groupedPrim->bounds();
            Bounds3f ungroupedBB = bbEmpty();

            if (auto* p = dynamic_cast<TrianglesIndexed8*>(groupedPrim.get()); p)
            {
                for (size_t i = 0; i < 8; ++i)
                {
                    assert(ungroupedIndex < ungroupedPrims.size());
                    ungroupedBB = bbUnion(ungroupedBB, ungroupedPrims[ungroupedIndex++]->bounds());
                }
            }
            else if (auto* p = dynamic_cast<TrianglesIndexed4*>(groupedPrim.get()); p)
            {
                for (size_t i = 0; i < 4; ++i)
                {
                    assert(ungroupedIndex < ungroupedPrims.size());
                    ungroupedBB = bbUnion(ungroupedBB, ungroupedPrims[ungroupedIndex++]->bounds());
                }
            }
            else if (auto* p = dynamic_cast<TrianglesIndexed2*>(groupedPrim.get()); p)
            {
                for (size_t i = 0; i < 2; ++i)
                {
                    assert(ungroupedIndex < ungroupedPrims.size());
                    ungroupedBB = bbUnion(ungroupedBB, ungroupedPrims[ungroupedIndex++]->bounds());
                }
            }
            else if (auto* p = dynamic_cast<TriangleIndexed*>(groupedPrim.get()); p)
            {
                assert(ungroupedIndex < ungroupedPrims.size());
                ungroupedBB = ungroupedPrims[ungroupedIndex++]->bounds();
            }

            if (!boundsEqual(groupedBB, ungroupedBB))
            {
                ctx.error("Bounds mismatch for grouped primitive:", {});
                ctx.error("Grouped:   [{} {} {}] - [{} {} {}]",
                          std::make_tuple(groupedBB.pMin.x,
                                          groupedBB.pMin.y,
                                          groupedBB.pMin.z,
                                          groupedBB.pMax.x,
                                          groupedBB.pMax.y,
                                          groupedBB.pMax.z));
                ctx.error("Ungrouped: [{} {} {}] - [{} {} {}]",
                          std::make_tuple(ungroupedBB.pMin.x,
                                          ungroupedBB.pMin.y,
                                          ungroupedBB.pMin.z,
                                          ungroupedBB.pMax.x,
                                          ungroupedBB.pMax.y,
                                          ungroupedBB.pMax.z));
                assert(false && "Grouped bounding box does not match union of ungrouped bounds");
            }
        }

        // INTERSECTION TEST
        ctx.log("INTERSECTION TEST", {});
        // this ray is known to intersect triangle
        Ray const testRay{Vector3f::zero(), normalize(Vector3f{-0.211708546, 0.926500857, -0.311087847})};
        testRayAgainstAllGroupedVsUngrouped(ungroupedPrims, groupedPrims, testRay);

        std::vector<Ray> rays = {{{0.f, 0.f, 0.f}, normalize(Vector3f{-0.2f, 0.9f, -0.3f})},
                                 {{1.f, 2.f, -1.f}, normalize(Vector3f{-1.f, -1.f, 1.f})},
                                 {{0.5f, 2.f, 0.f}, normalize(Vector3f{0.f, -1.f, 0.f})},
                                 {{0.f, 1.5f, -2.f}, normalize(Vector3f{0.f, 0.f, 1.f})}};

        for (auto const& ray : rays)
            testRayAgainstAllGroupedVsUngrouped(ungroupedPrims, groupedPrims, ray);
    }

    void testSphereLightPDFAnalyticCheck()
    {
        Context ctx;
        assert(ctx.isValid() && "invalid context");
        static constexpr float tol = 0.05f; // 5%

        ctx.log(" -- Testing sphereLightPDF() against analytic solid angle (tolerance = {}\x25) --",
                std::make_tuple(tol * 100.0f));

        // Setup: unit strength, small spherical light
        Transform lightFromRender = Transform{}.translate({0.f, 0.f, 3.f});
        Light light = dmt::makePointLight(lightFromRender, RGB::fromScalar(1.f), 0.1f /* radius */, 1.f /* factor */);

        // Receiver setup
        Point3f const  origin{0.f, 0.f, 0.f};
        Vector3f const n = normalize(Vector3f{0.001f, 0.f, 1.f});

        LightSampleContext lsCtx;
        lsCtx.p               = origin;
        lsCtx.n               = n;
        lsCtx.hadTransmission = false;
        lsCtx.ray.o           = origin;
        lsCtx.ray.d           = normalize(light.co - origin);

        Point2f u{0.5f, 0.5f};

        LightSample sample{};
        bool        success = dmt::lightSampleFromContext(light, lsCtx, u, &sample);
        assert(success && "lightSampleFromContext failed");

        ctx.log("sample.p  = {} {} {}", std::make_tuple(sample.p.x, sample.p.y, sample.p.z));
        ctx.log("sample.d  = {} {} {}", std::make_tuple(sample.d.x, sample.d.y, sample.d.z));
        ctx.log("sample.ng = {} {} {}", std::make_tuple(sample.ng.x, sample.ng.y, sample.ng.z));

        // Manually compute solid angle
        float dist         = normL2(light.co - origin);
        float distSqr      = dist * dist;
        float r            = light.data.point.radius;
        float sinThetaMax2 = r * r / distSqr;
        float cosThetaMax  = std::sqrt(std::max(0.f, 1.0f - sinThetaMax2));
        float omega        = 2 * fl::pi() * (1.0f - cosThetaMax);
        float pdfRef       = 1.0f / omega;

        float pdfSampled = sample.pdf;

        float err = std::abs(pdfSampled - pdfRef) / pdfRef;
        if (err > tol)
        {
            ctx.error("sphereLightPDF() does not match analytic solid angle", {});
            ctx.warn("sample.pdf = {}, pdfRef = {}, err = {}", std::make_tuple(pdfSampled, pdfRef, err));
            assert(false && "PDF mismatch");
        }

        // Evaluate radiance contribution
        RGB eval = dmt::lightEval(light, &sample);

        float cosThetaI = std::max(dot(n, sample.d), 0.f);
        float cosThetaO = std::max(dot(sample.ng, -sample.d), 0.f);
        float G         = cosThetaI * cosThetaO / distSqr;

        RGB final = eval * (G / pdfSampled);

        ctx.warn("sphereLightPDF() agrees with solid angle model within {}\x25", std::make_tuple(tol * 100.f));
        ctx.log(" \xe2\x84\xa6 (solid angle) = {}", std::make_tuple(omega));
        ctx.log("Distance        = {}", std::make_tuple(dist));
        ctx.log(" cos(\xce\xb8)          = {}", std::make_tuple(cosThetaI));
        ctx.log(" G               = {}", std::make_tuple(G));
        ctx.log(" Eval            = ({}, {}, {})", std::make_tuple(eval.r, eval.g, eval.b));
        ctx.log(" Final Radiance Over single sample = ({}, {}, {}) (the light is small)",
                std::make_tuple(final.r, final.g, final.b));
        if (final.max() == 0.f)
        {
            ctx.error("There should be some light", {});
            assert(false);
        }
    }

    static void checkUniformDist2D(PiecewiseConstant2D const& distrib)
    {
        static constexpr int32_t numSteps   = 1024;
        float const              yStep      = (distrib.domain().pMax[1] - distrib.domain().pMin[1]) / numSteps;
        float const              xStep      = (distrib.domain().pMax[0] - distrib.domain().pMin[0]) / numSteps;
        float const              firstValue = distrib.pdf(distrib.domain().pMin);

        for (float y = distrib.domain().pMin[1]; y < distrib.domain().pMax[1]; y += yStep)
        {
            for (float x = distrib.domain().pMin[0]; x < distrib.domain().pMax[0]; x += xStep)
            {
                float value = distrib.pdf({x, y});
                assert(fl::abs(firstValue - value) < 1e-5f);
            }
        }
    }

    void testEnvironmentalLightConstantValue()
    {
        Context ctx;
        assert(ctx.isValid());
        ctx.warn("Beginning testing of Environmental Lights", {});

        RGB dummy[4 * 2]{};
        for (uint32_t i = 0; i < 4 * 2; ++i)
            dummy[i] = {1, 0, 0}; // red everywhere
        Quaternion qIdentity{0, 0, 0, 1};
        EnvLight   light{dummy, 4, 2, qIdentity, 1.f};
        ctx.warn("  Should manually check that distribution is uniform", {});
        checkUniformDist2D(light.distrib);

        LightSampleContext lsCtx{Ray{}, Point3f::zero(), Vector3f::yAxis(), false};
        LightSample        sample{};
        Point2f            u{0.5, 0.5};

        bool sampled = envLightSampleFromContext(light, lsCtx, u, &sample);
        if (!sampled)
        {
            ctx.error("Sampling procedure of image failed. That's not possible as there are no black pixels", {});
            assert(false);
        }
        ctx.log("  Sampled From Env Direction {} {} {} in UV {} {} with PDF {}",
                std::make_tuple(sample.d.x, sample.d.y, sample.d.z, sample.uv.x, sample.uv.y, sample.pdf));

        assert(sample.pdf > 0.f);
        assert(fl::abs(normL2(sample.d) - 1.f) < 1e-5f);

        RGB const sampledEval = envLightEval(light, &sample);
        ctx.log("  evaluated sample as RGB {} {} {}", std::make_tuple(sampledEval.r, sampledEval.g, sampledEval.b));
        assert((sampledEval - RGB{1, 0, 0}).max() < 1e-5f);

        ctx.log("  testing arbitrary direction equirectangular evaluation", {});
        float          pdf = 0.f;
        Vector3f const wi  = normalize(Vector3f{0, 0, 1});

        RGB const color = envLightEval(light, wi, &pdf);
        assert(pdf > 0.f);
        assert(color.max() > 0.f);

        ctx.warn("Ending testing of Environmental Lights", {});
    }

    void testQuaternionRotation()
    {
        Quaternion _45DegRot{};
        _45DegRot.w = cosf(fl::pi() * 0.125f);
        _45DegRot.z = sinf(fl::pi() * 0.125f);

        Quaternion _45DegRotConj{};
        _45DegRotConj.w = _45DegRot.w;
        _45DegRotConj.z = -_45DegRot.z;

        Vector3f const start    = Vector3f::yAxis();
        Vector3f const expected = {-sqrtf(2) / 2, sqrtf(2) / 2, 0};

        Vector3f actual{};
        {
            Quaternion const pureStart{start.x, start.y, start.z, 0.f};
            Quaternion const pureActual = _45DegRot * pureStart * _45DegRotConj;

            actual.x = pureActual.x;
            actual.y = pureActual.y;
            actual.z = pureActual.z;
        }

        assert(fl::abs(normL2(actual - expected)) < 1e-5f);
    }
} // namespace dmt::test

static void swap_avx256(__m256* a, __m256* b)
{
    __m256 tmp = *a;
    *a         = *b;
    *b         = tmp;
}


static __m256 neg_avx256(__m256 x)
{
    //-0.0 has only the sign bit set
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    //Flip the sign bit
    return _mm256_xor_ps(x, sign_mask);
}

namespace dmt::bvh {

    void traverseCluster(BVHWiVeSoA* bvh, Ray ray, uint32_t index);

    bool intersectLeaf(BVHWiVeSoA* bvh, Ray ray, uint32_t index);

    //8 predetermined orders per node cluster
    void traverseRay(Ray ray, BVHWiVeSoA* bvh, std::pmr::memory_resource* _temp)
    {
        std::pmr::vector<uint32_t> nodeStack{_temp};
        std::pmr::vector<uint32_t> tminStack{_temp};
        nodeStack.reserve(64);
        tminStack.reserve(64);
        uint32_t nodeRootIndex = 0;
        nodeStack.push_back(nodeRootIndex);

        while (true)
        {
            uint32_t currNodeIndex = nodeStack.back();
            nodeStack.pop_back();

            if (!(bvh->leaf[currNodeIndex]))
            {
                traverseCluster(bvh, ray, currNodeIndex);
            }
            else if (intersectLeaf(bvh, ray, currNodeIndex))
            {
                //compress the stack keeping only elements with a node entry
                //distance closer compared to the primitive distance
            }
            if (nodeStack.empty())
                return;
        }
    }

    static void traverseCluster(BVHWiVeSoA* bvh, Ray ray, uint32_t index)
    {
        __m256 txmin, txmax;
        __m256 tymin, tymax;
        __m256 tzmin, tzmax;

        if (ray.d.x < 0)
            swap_avx256(&(bvh->bxmax[index]), &(bvh->bxmax[index]));
        if (ray.d.y < 0)
            swap_avx256(&(bvh->bymax[index]), &(bvh->bymax[index]));
        if (ray.d.z < 0)
            swap_avx256(&(bvh->bzmax[index]), &(bvh->bzmax[index]));

        //set rayx info
        __m256 rorgx    = _mm256_set1_ps(ray.o.x);
        __m256 ridx     = _mm256_set1_ps(ray.d_inv.x);
        __m256 ridx_neg = neg_avx256(ridx);

        //set parametric variable
        txmin = _mm256_sub_ps(bvh->bxmin[index], rorgx);
        txmin = _mm256_mul_ps(txmin, ridx_neg);
        txmax = _mm256_sub_ps(bvh->bxmax[index], rorgx);
        txmax = _mm256_mul_ps(txmax, ridx);

        //set rayy info
        __m256 rorgy    = _mm256_set1_ps(ray.o.y);
        __m256 ridy     = _mm256_set1_ps(ray.d_inv.y);
        __m256 ridy_neg = neg_avx256(ridy);

        //set parametric variable
        tymin = _mm256_sub_ps(bvh->bymin[index], rorgy);
        tymin = _mm256_mul_ps(tymin, ridy_neg);
        tymax = _mm256_sub_ps(bvh->bymax[index], rorgy);
        tymax = _mm256_mul_ps(tymax, ridy);

        //set rayz info
        __m256 rorgz    = _mm256_set1_ps(ray.o.z);
        __m256 ridz     = _mm256_set1_ps(ray.d_inv.z);
        __m256 ridz_neg = neg_avx256(ridz);

        //set parametric variable
        tzmin = _mm256_sub_ps(bvh->bzmin[index], rorgz);
        tzmin = _mm256_mul_ps(tzmin, ridz_neg);
        tzmax = _mm256_sub_ps(bvh->bzmax[index], rorgz);
        tzmax = _mm256_mul_ps(tzmax, ridz);

        //I need to clip the tmin and tmax with the tnear and tfar
        //_mm256_min_ps
    }

    static bool intersectLeaf(BVHWiVeSoA* bvh, Ray ray, uint32_t index) { return false; }

    BVHBuildNode* traverseBVHBuild(Ray ray, BVHBuildNode* bvh, std::pmr::memory_resource* _temp)
    {
        std::pmr::vector<BVHBuildNode*> activeNodeStack{_temp};
        activeNodeStack.reserve(64);
        activeNodeStack.push_back(bvh);

        BVHBuildNode* intersection = nullptr;
        while (!activeNodeStack.empty())
        {
            BVHBuildNode* current = activeNodeStack.back();
            activeNodeStack.pop_back();

            if (current->childCount > 0)
            {
                // children order of traversal: 1) Distance Heuristic: from smallest to highest tmin - ray origin 2) Sign Heuristic
                // start with distance heuristic
                struct
                {
                    uint32_t i = static_cast<uint32_t>(-1);
                    float    d = fl::infinity();
                } tmins[BranchingFactor];
                uint32_t currentIndex = 0;

                for (uint32_t i = 0; i < current->childCount; ++i)
                {
                    float tmin = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[i]->bounds, &tmin))
                    {
                        tmins[currentIndex].d = tmin;
                        tmins[currentIndex].i = i;
                        ++currentIndex;
                    }
                }

                std::sort(std::begin(tmins), std::begin(tmins) + currentIndex, [](auto const& a, auto const& b) {
                    return a.d > b.d;
                });

                for (uint32_t i = 0; i < currentIndex; ++i)
                    activeNodeStack.push_back(current->children[tmins[i].i]);
            }
            else
            {
                // TODO handle any-hit, closest-hit, ...
                // for now, stop at the first leaf intersection
                intersection = current;
                break;
            }
        }

        return intersection;
    }

    Primitive const* intersectBVHBuild(Ray ray, BVHBuildNode* bvh, Intersection* outIsect, std::pmr::memory_resource* _temp)
    {
        std::pmr::vector<BVHBuildNode*> activeNodeStack{_temp};
        activeNodeStack.reserve(64);
        activeNodeStack.push_back(bvh);

        Primitive const* primitive = nullptr;
        float            nearest   = fl::infinity();

        int dirIsNeg[3] = {ray.d.x < 0.f, ray.d.y < 0.f, ray.d.z < 0.f};

        while (!activeNodeStack.empty())
        {
            BVHBuildNode* current = activeNodeStack.back();
            activeNodeStack.pop_back();

            if (current->childCount > 0)
            {
                // children order of traversal: 1) Distance Heuristic: from smallest to highest tmin - ray origin 2) Sign Heuristic
                // start with distance heuristic
                struct
                {
                    uint32_t i = static_cast<uint32_t>(-1);
                    float    d = fl::infinity();
                } tmins[BranchingFactor];
                uint32_t currentIndex = 0;

                for (uint32_t i = 0; i < current->childCount; ++i)
                {
                    float tmin = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[i]->bounds, &tmin))
                    {
                        tmins[currentIndex].d = tmin;
                        tmins[currentIndex].i = i;
                        ++currentIndex;
                    }
                }

                std::sort(std::begin(tmins), std::begin(tmins) + currentIndex, [](auto const& a, auto const& b) {
                    return a.d < b.d;
                });

                for (uint32_t i = 0; i < currentIndex; ++i)
                    activeNodeStack.push_back(current->children[tmins[i].i]);
            }
            else
            {
                // TODO handle any-hit, closest-hit, ...
                // for now, stop at the first leaf intersection
                for (size_t i = 0; i < current->primitiveCount; ++i)
                {
                    assert(current->primitives[i] && "null primitive");
                    if (auto si = current->primitives[i] ? current->primitives[i]->intersect(ray, fl::infinity())
                                                         : Intersection{};
                        si.hit && si.t < nearest)
                    {
                        primitive = current->primitives[i];
                        nearest   = si.t;
                        if (outIsect)
                            *outIsect = si;
                        // TODO handle anyhit
                    }
                }
            }
        }

        // closesthit
        return primitive;
    }

    // TODO move elsewhere
    static uint32_t partitionActiveList(BVHBuildNode** activeListIDs, float* activeListDistances, uint32_t activeListCount, float tMin)
    {
        uint32_t curr = 0;
        uint32_t end  = activeListCount;
        for (;;)
        {
            for (;;)
            {
                if (curr == end)
                    return curr;

                if (activeListDistances[curr] >= tMin)
                    break;

                ++curr;
            }

            do
            {
                --end;
                if (curr == end)
                    return curr;
            } while (activeListDistances[curr] >= tMin);

            float         tmpDist     = activeListDistances[curr];
            BVHBuildNode* tmpID       = activeListIDs[curr];
            activeListDistances[curr] = activeListDistances[end];
            activeListIDs[curr]       = activeListIDs[end];
            activeListDistances[end]  = tmpDist;
            activeListIDs[end]        = tmpID;
            ++curr;
        }
    }


    Primitive const* intersectWideBVHBuild(Ray ray, BVHBuildNode* bvh, Intersection* outIsect)
    {
        Primitive const* intersectedPrim = nullptr;
        BVHBuildNode*    activeListIDs[8]{};
        float            activeListDistances[8]{};
        BVHBuildNode*    current         = bvh;
        uint32_t         activeListCount = 0;
        float            tMinPrims       = fl::infinity();

        assert(outIsect && current);
        while (true)
        {
            assert(current);
            if (current->childCount == 0)
            {
                bool intersected = false;
                for (uint32_t idx = 0; idx < current->primitiveCount; ++idx)
                {
                    auto isect = current->primitives[idx]->intersect(ray, fl::infinity());
                    if (isect.hit)
                    {
                        intersected = true;
                        if (isect.t < tMinPrims)
                        {
                            *outIsect       = isect;
                            tMinPrims       = isect.t;
                            intersectedPrim = current->primitives[idx];
                        }
                    }
                }

                if (intersected)
                    activeListCount = partitionActiveList(activeListIDs, activeListDistances, activeListCount, tMinPrims);
            }
            else
            {
                for (uint32_t idx = 0; idx < current->childCount; ++idx)
                {
                    float distance = fl::infinity();
                    if (slabTest(ray.o, ray.d, current->children[idx]->bounds, &distance))
                    {
                        activeListIDs[activeListCount]       = current->children[idx];
                        activeListDistances[activeListCount] = distance;
                        ++activeListCount;
                        assert(activeListCount <= 8);
                    }
                }
            }

            if (activeListCount == 0)
                return intersectedPrim;
            else
            {
                int32_t minIdx   = -1;
                float   minValue = fl::infinity();
                for (int32_t i = 0; i < activeListCount; ++i)
                {
                    if (minValue > activeListDistances[i])
                    {
                        minValue = activeListDistances[i];
                        minIdx   = i;
                    }
                }

                --activeListCount;
                assert(minIdx >= 0 && minIdx < 8 && activeListCount < 8 && activeListCount >= 0);
                float         tmpDist                = activeListDistances[minIdx];
                BVHBuildNode* tmpID                  = activeListIDs[minIdx];
                activeListDistances[minIdx]          = activeListDistances[activeListCount];
                activeListIDs[minIdx]                = activeListIDs[activeListCount];
                activeListDistances[activeListCount] = tmpDist;
                activeListIDs[activeListCount]       = tmpID;

                current = activeListIDs[activeListCount];
            }
        }

        return intersectedPrim;
    }

    std::pmr::vector<Primitive const*> extractPrimitivesFromBuild(BVHBuildNode* bvh, std::pmr::memory_resource* memory)
    {
        std::pmr::vector<Primitive const*> ret{memory};

        auto const f = []<typename F>
            requires std::is_invocable_v<F, Primitive const*>
        (auto&& _f, BVHBuildNode* _node, F&& doFunc) -> void {
            if (_node->childCount == 0)
            {
                for (uint32_t i = 0; i < _node->primitiveCount; ++i)
                {
                    assert(_node->primitives[i] && "nullptr primitive found");
                    doFunc(_node->primitives[i]);
                }
            }
            else
            {
                for (uint32_t i = 0; i < _node->childCount; ++i)
                {
                    assert(_node->children[i] && "nullptr child BVH build node found");
                    _f(_f, _node->children[i], doFunc);
                }
            }
        };

        size_t requiredCapacity = 0;
        f(f, bvh, [&requiredCapacity](Primitive const* p) { ++requiredCapacity; });
        ret.reserve(requiredCapacity);
        f(f, bvh, [&ret](Primitive const* p) { ret.push_back(p); });

        return ret;
    }
} // namespace dmt::bvh

namespace dmt::numbers {
    uint16_t permutationElement(int32_t i, uint32_t l, uint64_t p)
    {
        uint32_t w = l - 1;
        w |= w >> 1;
        w |= w >> 2;
        w |= w >> 4;
        w |= w >> 8;
        w |= w >> 16;
        do
        {
            i ^= p;
            i *= 0xe170893d;
            i ^= p >> 16;
            i ^= (i & w) >> 4;
            i ^= p >> 8;
            i *= 0x0929eb3f;
            i ^= p >> 23;
            i ^= (i & w) >> 1;
            i *= 1 | p >> 27;
            i *= 0x6935fa69;
            i ^= (i & w) >> 11;
            i *= 0x74dcb303;
            i ^= (i & w) >> 2;
            i *= 0x9e501cc3;
            i ^= (i & w) >> 2;
            i *= 0xc860a3df;
            i &= w;
            i ^= i >> 5;
        } while (i >= l);
        return (i + p) % l;
    }
} // namespace dmt::numbers