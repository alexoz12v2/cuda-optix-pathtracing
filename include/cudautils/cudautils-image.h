#pragma once

#include "cudautils/cudautils-macro.h"
#include <cudautils/cudautils-vecmath.h>
#include <cudautils/cudautils-color.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

/*
namespace dmt {

    // PixelFormat Definition
    enum class DMT_CORE_API PixelFormat
    {
        U256,
        Half,
        Float
    };

    // PixelFormat Inline Functions
    DMT_CPU_GPU inline bool Is8Bit(PixelFormat format) { return format == PixelFormat::U256; }
    DMT_CPU_GPU inline bool Is16Bit(PixelFormat format) { return format == PixelFormat::Half; }
    DMT_CPU_GPU inline bool Is32Bit(PixelFormat format) { return format == PixelFormat::Float; }

    DMT_CORE_API std::string ToString(PixelFormat format);

    DMT_CORE_API DMT_CPU_GPU int TexelBytes(PixelFormat format);

    // ResampleWeight Definition
    struct DMT_CORE_API ResampleWeight
    {
        int   firstPixel;
        float weight[4];
    };

    // WrapMode Definitions
    enum class DMT_CORE_API WrapMode
    {
        Black,
        Clamp,
        Repeat,
        OctahedralSphere
    };

    struct DMT_CORE_API WrapMode2D
    {
        DMT_CPU_GPU WrapMode2D(WrapMode w) : wrap{w, w} {}
        DMT_CPU_GPU WrapMode2D(WrapMode x, WrapMode y) : wrap{x, y} {}

        WrapMode wrap[2];
    };

    inline DMT_CPU_GPU dstd::optional<WrapMode> ParseWrapMode(char const* w)
    {
        if (!strcmp(w, "clamp"))
            return WrapMode::Clamp;
        else if (!strcmp(w, "repeat"))
            return WrapMode::Repeat;
        else if (!strcmp(w, "black"))
            return WrapMode::Black;
        else if (!strcmp(w, "octahedralsphere"))
            return WrapMode::OctahedralSphere;
        else
            return {};
    }

    inline std::string ToString(WrapMode mode)
    {
        switch (mode)
        {
            case WrapMode::Clamp: return "clamp";
            case WrapMode::Repeat: return "repeat";
            case WrapMode::Black: return "black";
            case WrapMode::OctahedralSphere: return "octahedralsphere";
            default: LOG_FATAL("Unhandled wrap mode"); return nullptr;
        }
    }

    // Image Wrapping Inline Functions
    DMT_CPU_GPU inline bool RemapPixelCoords(Point2i* pp, Point2i resolution, WrapMode2D wrapMode)
    {
        Point2i& p = *pp;

        if (wrapMode.wrap[0] == WrapMode::OctahedralSphere)
        {
            //CHECK(wrapMode.wrap[1] == WrapMode::OctahedralSphere); true->ok
            if (p[0] < 0)
            {
                p[0] = -p[0];                    // mirror across u = 0
                p[1] = resolution[1] - 1 - p[1]; // mirror across v = 0.5
            }
            else if (p[0] >= resolution[0])
            {
                p[0] = 2 * resolution[0] - 1 - p[0]; // mirror across u = 1
                p[1] = resolution[1] - 1 - p[1];     // mirror across v = 0.5
            }

            if (p[1] < 0)
            {
                p[0] = resolution[0] - 1 - p[0]; // mirror across u = 0.5
                p[1] = -p[1];                    // mirror across v = 0;
            }
            else if (p[1] >= resolution[1])
            {
                p[0] = resolution[0] - 1 - p[0];     // mirror across u = 0.5
                p[1] = 2 * resolution[1] - 1 - p[1]; // mirror across v = 1
            }

            // Bleh: things don't go as expected for 1x1 images.
            if (resolution[0] == 1)
                p[0] = 0;
            if (resolution[1] == 1)
                p[1] = 0;

            return true;
        }

        for (int c = 0; c < 2; ++c)
        {
            if (p[c] >= 0 && p[c] < resolution[c])
                // in bounds
                continue;

            switch (wrapMode.wrap[c])
            {
                case WrapMode::Repeat: p[c] = Mod(p[c], resolution[c]); break;
                case WrapMode::Clamp: p[c] = Clamp(p[c], 0, resolution[c] - 1); break;
                case WrapMode::Black: return false;
                default:
                    //LOG_FATAL("Unhandled WrapMode mode");
            }
        }
        return true;
    }

    // ImageMetadata Definition
    struct DMT_CORE_API ImageMetadata
    {
        // ImageMetadata Public Methods
        //I don't know about RGBColorSpace
        RGBColorSpace const* GetColorSpace() const;
        std::string          ToString() const;

        // ImageMetadata Public Members
        pstd::optional<float>                           renderTimeSeconds;
        pstd::optional<Matrix4f>                        cameraFromWorld, NDCFromWorld;
        pstd::optional<Bounds2i>                        pixelBounds;
        pstd::optional<Point2i>                         fullResolution;
        pstd::optional<int>                             samplesPerPixel;
        pstd::optional<float>                           MSE;
        pstd::optional<RGBColorSpace const*>            colorSpace;
        std::map<std::string, std::string>              strings;
        std::map<std::string, std::vector<std::string>> stringVectors;
    };

    struct ImageAndMetadata;

    // ImageChannelDesc Definition
    struct ImageChannelDesc
    {
        operator bool() const { return size() > 0; }

        size_t size() const { return offset.size(); }
        bool   IsIdentity() const
        {
            for (size_t i = 0; i < offset.size(); ++i)
                if (offset[i] != i)
                    return false;
            return true;
        }
        std::string ToString() const;

        InlinedVector<int, 4> offset;
    };

    // ImageChannelValues Definition channels RGBA
    struct ImageChannelValues : public InlinedVector<Float, 4>
    {
        // ImageChannelValues() = default;
        explicit ImageChannelValues(size_t sz, Float v = {}) : InlinedVector<Float, 4>(sz, v) {}

        operator Float() const
        {
            CHECK_EQ(1, size());
            return (*this)[0];
        }
        operator pstd::array<Float, 3>() const
        {
            CHECK_EQ(3, size());
            return {(*this)[0], (*this)[1], (*this)[2]};
        }

        Float MaxValue() const
        {
            Float m = (*this)[0];
            for (int i = 1; i < size(); ++i)
                m = std::max(m, (*this)[i]);
            return m;
        }
        Float Average() const
        {
            Float sum = 0;
            for (int i = 0; i < size(); ++i)
                sum += (*this)[i];
            return sum / size();
        }

        std::string ToString() const;
    };

    // Image Definition
    class Image
    {
    public:
        // Image Public Methods
        Image(Allocator alloc = {}) : p8(alloc), p16(alloc), p32(alloc), format(PixelFormat::U256), resolution(0, 0) {}
        Image(pstd::vector<uint8_t> p8, Point2i resolution, pstd::span<std::string const> channels, ColorEncoding encoding);
        Image(pstd::vector<Half> p16, Point2i resolution, pstd::span<std::string const> channels);
        Image(pstd::vector<float> p32, Point2i resolution, pstd::span<std::string const> channels);

        Image(PixelFormat                   format,
              Point2i                       resolution,
              pstd::span<std::string const> channelNames,
              ColorEncoding                 encoding = nullptr,
              Allocator                     alloc    = {});

        PBRT_CPU_GPU
        PixelFormat Format() const { return format; }
        PBRT_CPU_GPU
        Point2i Resolution() const { return resolution; }
        PBRT_CPU_GPU
        int                      NChannels() const { return channelNames.size(); }
        std::vector<std::string> ChannelNames() const;
        ColorEncoding const      Encoding() const { return encoding; }

        PBRT_CPU_GPU
        operator bool() const { return resolution.x > 0 && resolution.y > 0; }

        PBRT_CPU_GPU
        size_t PixelOffset(Point2i p) const
        {
            DCHECK(InsideExclusive(p, Bounds2i({0, 0}, resolution)));
            return NChannels() * (p.y * resolution.x + p.x);
        }

        PBRT_CPU_GPU
        Float GetChannel(Point2i p, int c, WrapMode2D wrapMode = WrapMode::Clamp) const
        {
            // Remap provided pixel coordinates before reading channel
            if (!RemapPixelCoords(&p, resolution, wrapMode))
                return 0;

            switch (format)
            {
                case PixelFormat::U256:
                { // Return _U256_-encoded pixel channel value
                    Float r;
                    encoding.ToLinear({&p8[PixelOffset(p) + c], 1}, {&r, 1});
                    return r;
                }
                case PixelFormat::Half:
                { // Return _Half_-encoded pixel channel value
                    return Float(p16[PixelOffset(p) + c]);
                }
                case PixelFormat::Float:
                { // Return _Float_-encoded pixel channel value
                    return p32[PixelOffset(p) + c];
                }
                default: LOG_FATAL("Unhandled PixelFormat"); return 0;
            }
        }

        PBRT_CPU_GPU
        Float BilerpChannel(Point2f p, int c, WrapMode2D wrapMode = WrapMode::Clamp) const
        {
            // Compute discrete pixel coordinates and offsets for _p_
            Float x = p[0] * resolution.x - 0.5f, y = p[1] * resolution.y - 0.5f;
            int   xi = pstd::floor(x), yi = pstd::floor(y);
            Float dx = x - xi, dy = y - yi;

            // Load pixel channel values and return bilinearly interpolated value
            pstd::array<Float, 4> v = {GetChannel({xi, yi}, c, wrapMode),
                                       GetChannel({xi + 1, yi}, c, wrapMode),
                                       GetChannel({xi, yi + 1}, c, wrapMode),
                                       GetChannel({xi + 1, yi + 1}, c, wrapMode)};
            return ((1 - dx) * (1 - dy) * v[0] + dx * (1 - dy) * v[1] + (1 - dx) * dy * v[2] + dx * dy * v[3]);
        }

        PBRT_CPU_GPU
        void SetChannel(Point2i p, int c, Float value);

        ImageChannelValues GetChannels(Point2i p, WrapMode2D wrapMode = WrapMode::Clamp) const;

        ImageChannelDesc GetChannelDesc(pstd::span<std::string const> channels) const;

        ImageChannelDesc AllChannelsDesc() const
        {
            ImageChannelDesc desc;
            desc.offset.resize(NChannels());
            for (int i = 0; i < NChannels(); ++i)
                desc.offset[i] = i;
            return desc;
        }

        ImageChannelValues GetChannels(Point2i p, ImageChannelDesc const& desc, WrapMode2D wrapMode = WrapMode::Clamp) const;

        Image SelectChannels(ImageChannelDesc const& desc, Allocator alloc = {}) const;
        Image Crop(Bounds2i const& bounds, Allocator alloc = {}) const;

        void CopyRectOut(Bounds2i const& extent, pstd::span<float> buf, WrapMode2D wrapMode = WrapMode::Clamp) const;
        void CopyRectIn(Bounds2i const& extent, pstd::span<float const> buf);

        ImageChannelValues Average(ImageChannelDesc const& desc) const;

        bool HasAnyInfinitePixels() const;
        bool HasAnyNaNPixels() const;

        ImageChannelValues MAE(ImageChannelDesc const& desc, Image const& ref, Image* errorImage = nullptr) const;
        ImageChannelValues MSE(ImageChannelDesc const& desc, Image const& ref, Image* mseImage = nullptr) const;
        ImageChannelValues MRSE(ImageChannelDesc const& desc, Image const& ref, Image* mrseImage = nullptr) const;

        Image GaussianFilter(ImageChannelDesc const& desc, int halfWidth, Float sigma) const;

        template <typename F>
        Array2D<Float> GetSamplingDistribution(F               dxdA,
                                               Bounds2f const& domain = Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                                               Allocator       alloc  = {});
        Array2D<Float> GetSamplingDistribution()
        {
            return GetSamplingDistribution([](Point2f) { return Float(1); });
        }

        static ImageAndMetadata Read(std::string filename, Allocator alloc = {}, ColorEncoding encoding = nullptr);

        bool Write(std::string name, ImageMetadata const& metadata = {}) const;

        Image ConvertToFormat(PixelFormat format, ColorEncoding encoding = nullptr) const;

        // TODO? provide an iterator to iterate over all pixels and channels?

        PBRT_CPU_GPU
        Float LookupNearestChannel(Point2f p, int c, WrapMode2D wrapMode = WrapMode::Clamp) const
        {
            Point2i pi(p.x * resolution.x, p.y * resolution.y);
            return GetChannel(pi, c, wrapMode);
        }

        ImageChannelValues LookupNearest(Point2f p, WrapMode2D wrapMode = WrapMode::Clamp) const;
        ImageChannelValues LookupNearest(Point2f p, ImageChannelDesc const& desc, WrapMode2D wrapMode = WrapMode::Clamp) const;

        ImageChannelValues Bilerp(Point2f p, WrapMode2D wrapMode = WrapMode::Clamp) const;
        ImageChannelValues Bilerp(Point2f p, ImageChannelDesc const& desc, WrapMode2D wrapMode = WrapMode::Clamp) const;

        void SetChannels(Point2i p, ImageChannelValues const& values);
        void SetChannels(Point2i p, pstd::span<Float const> values);
        void SetChannels(Point2i p, ImageChannelDesc const& desc, pstd::span<Float const> values);

        Image                      FloatResizeUp(Point2i newResolution, WrapMode2D wrap) const;
        void                       FlipY();
        static pstd::vector<Image> GeneratePyramid(Image image, WrapMode2D wrapMode, Allocator alloc = {});

        std::vector<std::string> ChannelNames(ImageChannelDesc const&) const;

        PBRT_CPU_GPU
        size_t BytesUsed() const { return p8.size() + 2 * p16.size() + 4 * p32.size(); }

        PBRT_CPU_GPU
        void const* RawPointer(Point2i p) const
        {
            if (Is8Bit(format))
                return p8.data() + PixelOffset(p);
            if (Is16Bit(format))
                return p16.data() + PixelOffset(p);
            else
            {
                CHECK(Is32Bit(format));
                return p32.data() + PixelOffset(p);
            }
        }
        PBRT_CPU_GPU
        void* RawPointer(Point2i p) { return const_cast<void*>(((Image const*)this)->RawPointer(p)); }

        Image JointBilateralFilter(ImageChannelDesc const&   toFilter,
                                   int                       halfWidth,
                                   Float const               xySigma[2],
                                   ImageChannelDesc const&   joint,
                                   ImageChannelValues const& jointSigma) const;

        std::string ToString() const;

    private:
        // Image Private Methods
        static std::vector<ResampleWeight> ResampleWeights(int oldRes, int newRes);
        bool                               WriteEXR(std::string const& name, ImageMetadata const& metadata) const;
        bool                               WritePFM(std::string const& name, ImageMetadata const& metadata) const;
        bool                               WritePNG(std::string const& name, ImageMetadata const& metadata) const;
        bool                               WriteQOI(std::string const& name, ImageMetadata const& metadata) const;

        std::unique_ptr<uint8_t[]> QuantizePixelsToU256(int* nOutOfGamut) const;

        // Image Private Members
        PixelFormat               format;
        Point2i                   resolution;
        pstd::vector<std::string> channelNames;
        ColorEncoding             encoding = nullptr;
        pstd::vector<uint8_t>     p8;
        pstd::vector<Half>        p16;
        pstd::vector<float>       p32;
    };

    // Image Inline Method Definitions
    inline void Image::SetChannel(Point2i p, int c, Float value)
    {
        // CHECK(!IsNaN(value));
        if (IsNaN(value))
        {
#ifndef PBRT_IS_GPU_CODE
            LOG_ERROR("NaN at pixel %d,%d comp %d", p.x, p.y, c);
#endif
            value = 0;
        }

        switch (format)
        {
            case PixelFormat::U256: encoding.FromLinear({&value, 1}, {&p8[PixelOffset(p) + c], 1}); break;
            case PixelFormat::Half: p16[PixelOffset(p) + c] = Half(value); break;
            case PixelFormat::Float: p32[PixelOffset(p) + c] = value; break;
            default: LOG_FATAL("Unhandled PixelFormat in Image::SetChannel()");
        }
    }

    template <typename F>
    inline Array2D<Float> Image::GetSamplingDistribution(F dxdA, Bounds2f const& domain, Allocator alloc)
    {
        Array2D<Float> dist(resolution[0], resolution[1], alloc);
        ParallelFor(0, resolution[1], [&](int64_t y0, int64_t y1) {
            for (int y = y0; y < y1; ++y)
            {
                for (int x = 0; x < resolution[0]; ++x)
                {
                    // This is noticeably better than MaxValue: discuss / show
                    // example..
                    Float value = GetChannels({x, y}).Average();

                    // Assume Jacobian term is basically constant over the
                    // region.
                    Point2f p  = domain.Lerp(Point2f((x + .5f) / resolution[0], (y + .5f) / resolution[1]));
                    dist(x, y) = value * dxdA(p);
                }
            }
        });
        return dist;
    }

    // ImageAndMetadata Definition
    struct ImageAndMetadata
    {
        Image         image;
        ImageMetadata metadata;
    };

} // namespace dmt

*/

#if defined(DMT_CUDAUTILS_IMPL) || defined(DMT_CUDAUTILS_IMAGE_IMPL)
#include "cudautils-image.cu"
#endif
