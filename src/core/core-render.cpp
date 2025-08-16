#include "core-render.h"

#include "core-bvh-builder.h"
#include "core-light-tree-builder.h"

// write image dependencies
#include <ImfRgbaFile.h>
#include <ImfArray.h>

#define STBI_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace dmt::job {
    struct EvalTileData
    {
        dmt::Point2i                    StartPixel;
        dmt::Point2i                    tileResolution;
        int32_t                         SamplesPerPixel;
        dmt::sampling::HaltonOwen*      samplers;
        dmt::filtering::Mitchell const* filter;
        dmt::Transform const*           cameraFromRaster;
        dmt::Transform const*           renderFromCamera;
        dmt::BVHWiVeCluster*            bvhRoot;
        int32_t                         bvhNodeNum;
        dmt::EnvLight const*            envLight;
        dmt::ApproxDifferentialsContext diffCtx;
        TextureCache*                   texCache;
        dmt::film::RGBFilm*             film;
        Scene const*                    scene;
        std::span<LightTreeBuildNode>   lightTreeNodes;
        std::span<LightTrailPair>       lightTreeBittrails;
    };

    static void evalTile(uintptr_t _data, uint32_t tid)
    {
        EvalTileData& data = *std::bit_cast<EvalTileData*>(_data);

        dmt::sampling::HaltonOwen& sampler = data.samplers[tid];

        int32_t xEnd = data.StartPixel.x + data.tileResolution.x;
        int32_t yEnd = data.StartPixel.y + data.tileResolution.y;

        for (int32_t y = data.StartPixel.y; y < yEnd; ++y)
        {
            for (int32_t x = data.StartPixel.x; x < xEnd; ++x)
            {
                std::cout << "[Thread " << tid << "] Pixel " << x << " " << y << std::endl;
                for (int32_t sampleIndex = 0; sampleIndex < data.SamplesPerPixel; ++sampleIndex)
                {
                    std::pmr::monotonic_buffer_resource scratch{4096}; // TODO better
                    sampler.startPixelSample({x, y}, sampleIndex);
                    camera::CameraSample const cs = camera::getCameraSample(sampler, {x, y}, *data.filter);

                    Ray ray{camera::generateRay(cs, *data.cameraFromRaster, *data.renderFromCamera)};

                    RGB  L = RGB::fromScalar(0.f), beta = RGB::fromScalar(1.f);
                    bool hadTransmission = false, specularBounce = false, anyNonSpecularBounces = false;

                    while (true)
                    {
                    }

                    data.film->addSample({x, y}, L, cs.filterWeight);
                }
            }
        }
    }

} // namespace dmt::job

namespace dmt::render_thread {
    struct Data
    {
        Scene*                      scene;
        TextureCache*               texCache;
        Parameters const*           params;
        UniqueRef<BVHWiVeCluster[]> bvh;
        ThreadPoolV2*               workers;
    };

    static UniqueRef<BVHWiVeCluster[]> bvhBuild(Scene const& scene, uint32_t* nodeCount)
    {
        if (scene.instances.size() > 0)
        {
            std::pmr::vector<BVHBuildNode*>        perInstanceBvhNodes{};
            std::pmr::vector<UniqueRef<Primitive>> primitives{};
            std::pmr::memory_resource*             memHeap = std::pmr::get_default_resource();
            perInstanceBvhNodes.reserve(64);
            primitives.reserve(256);

            for (size_t instanceIdx = 0; instanceIdx < scene.instances.size(); ++instanceIdx)
            {
                perInstanceBvhNodes.push_back(bvh::buildForInstance(scene, instanceIdx, primitives, memHeap, memHeap));
            }

            auto* bvhRoot = reinterpret_cast<BVHBuildNode*>(memHeap->allocate(sizeof(BVHBuildNode)));
            std::memset(bvhRoot, 0, sizeof(BVHBuildNode));
            bvh::buildCombined(bvhRoot, perInstanceBvhNodes, memHeap, memHeap);

            std::pmr::monotonic_buffer_resource tmp{4096};

            BVHWiVeCluster* wivebvh = bvh::buildBVHWive(bvhRoot, nodeCount, &tmp, memHeap);

            bvh::cleanup(bvhRoot, memHeap);

            return UniqueRef<BVHWiVeCluster[]>{wivebvh, PmrDeleter::create<BVHWiVeCluster[]>(memHeap, *nodeCount)};
        }
        else
        {
            *nodeCount = 0;
            return nullptr;
        }
    }

    static void mainLoop(void* data)
    {
        static constexpr uint32_t TileWidth          = 32;
        static constexpr uint32_t TileHeight         = 32;
        static constexpr uint32_t ScratchBufferBytes = 4096;

        Data*    rtData       = reinterpret_cast<Data*>(data);
        uint32_t bvhNodeCount = 0;

        // construct BVH
        rtData->bvh = bvhBuild(*rtData->scene, &bvhNodeCount);

        // define film, sampler, camera
        int32_t const Width           = rtData->params->filmResolution.x;
        int32_t const Height          = rtData->params->filmResolution.y;
        int32_t const SamplesPerPixel = rtData->params->samplesPerPixel;

        auto scratchBuffer = makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), ScratchBufferBytes);
        std::pmr::monotonic_buffer_resource    scratch{scratchBuffer.get(), ScratchBufferBytes};
        std::pmr::unsynchronized_pool_resource pool;


        film::RGBFilm       film{{{Width, Height}}, 1e5f, &pool};
        filtering::Mitchell filter{{{2.f, 2.f}}, 1.f / 3.f, 1.f / 3.f, &pool, &scratch};
        resetMonotonicBufferPointer(scratch, scratchBuffer.get(), ScratchBufferBytes);

        // define camera (image plane physical dims, resolution given by image)
        Vector3f const cameraPosition{0.f, 0.f, 0.f};
        Normal3f const cameraDirection = normalFrom(rtData->params->cameraDirection);
        float const    focalLength     = rtData->params->focalLength;
        float const    sensorHeight    = rtData->params->sensorSize;
        float const    aspectRatio     = static_cast<float>(Width) / Height;

        Transform const cameraFromRaster = transforms::cameraFromRaster_Perspective(focalLength, sensorHeight, Width, Height);
        Transform const renderFromCamera = transforms::worldFromCamera(cameraDirection, cameraPosition);
        ApproxDifferentialsContext
            diffCtx = camera::minDifferentialsFromCamera(cameraFromRaster, renderFromCamera, film, SamplesPerPixel);

        // build light tree
        std::pmr::vector<LightTreeBuildNode> lightTreeNodes{&pool};
        std::pmr::vector<LightTrailPair>     lightTreeBittrails{&pool};
        if (rtData->scene->lights.size() > 0)
        {
            lightTreeBuild(rtData->scene->lights, &lightTreeNodes, &lightTreeBittrails, &scratch);
            resetMonotonicBufferPointer(scratch, scratchBuffer.get(), ScratchBufferBytes);
        }

        // allocate all necessary jobs
        uint32_t const NumTileX = ceilDiv(static_cast<uint32_t>(Width), TileWidth);
        uint32_t const NumTileY = ceilDiv(static_cast<uint32_t>(Height), TileHeight);
        uint32_t const NumJobs  = NumTileX * NumTileY;

        UniqueRef<job::EvalTileData[]> jobData = makeUniqueRef<job::EvalTileData[]>(&pool, NumJobs);

        ThreadPoolV2& threadpool = *rtData->workers;

        UniqueRef<sampling::HaltonOwen[]> tlsSamplers = makeUniqueRef<sampling::HaltonOwen[]>(&pool, threadpool.numThreads());
        for (uint32_t tidx = 0; tidx < threadpool.numThreads(); ++tidx)
        {
            int seed = 18123 * sinf(32424 * tidx);
            std::construct_at<sampling::HaltonOwen>(&tlsSamplers[tidx], SamplesPerPixel, Point2i{Width, Height}, seed);
        }

        Point2i startPix{0, 0};
        Point2i tileSize{TileWidth, TileHeight};
        for (uint32_t job = 0; job < NumJobs; ++job)
        {
            jobData[job].StartPixel         = startPix;
            jobData[job].tileResolution     = tileSize;
            jobData[job].SamplesPerPixel    = SamplesPerPixel;
            jobData[job].samplers           = tlsSamplers.get();
            jobData[job].filter             = &filter;
            jobData[job].cameraFromRaster   = &cameraFromRaster;
            jobData[job].renderFromCamera   = &renderFromCamera;
            jobData[job].diffCtx            = diffCtx;
            jobData[job].film               = &film;
            jobData[job].bvhRoot            = rtData->bvh.get();
            jobData[job].bvhNodeNum         = bvhNodeCount;
            jobData[job].envLight           = rtData->params->envLight.get();
            jobData[job].texCache           = rtData->texCache;
            jobData[job].scene              = rtData->scene;
            jobData[job].lightTreeNodes     = lightTreeNodes;
            jobData[job].lightTreeBittrails = lightTreeBittrails;

            if (job != 0 && job % NumTileX == 0)
            {
                startPix.x = 0;
                startPix.y += TileHeight;
                tileSize.x = Width;
                tileSize.y = fminf(TileHeight, Height - startPix.y);
            }
            else
            {
                startPix.x += TileWidth;
                tileSize.x = fminf(TileWidth, Width - startPix.x);
            }

            threadpool.addJob({job::evalTile, std::bit_cast<uintptr_t>(&jobData[job])}, EJobLayer::ePriority0);
        }
        threadpool.kickJobs();
        threadpool.waitForAll();

        std::cout << "[RT] Finished everything" << std::endl;
    }
} // namespace dmt::render_thread

namespace dmt {
    Renderer::Renderer(size_t tmpSize) :
    m_bigBuffer{makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), tmpSize)},
    m_bigBufferSize{tmpSize},
    m_bigTmpMem{m_bigBuffer.get(), m_bigBufferSize},
    m_poolMem{},
    m_renderThread{render_thread::mainLoop, &m_poolMem},
    m_workers{std::thread::hardware_concurrency(), &m_poolMem},
    m_renderThreadData{makeUniqueRef<unsigned char[]>(&m_poolMem, sizeof(render_thread::Data), alignof(render_thread::Data))}
    {
        std::construct_at(reinterpret_cast<render_thread::Data*>(m_renderThreadData.get()));
    }

    Renderer::~Renderer() noexcept
    {
        if (m_renderThread.running())
            m_renderThread.join();
        std::destroy_at(reinterpret_cast<render_thread::Data*>(m_renderThreadData.get()));
    }

    std::pmr::monotonic_buffer_resource Renderer::tmpMem() { return {m_smallBuffer, 256, &m_bigTmpMem}; }

    void Renderer::resetBigTmp()
    {
        std::destroy_at(&m_bigTmpMem);
        std::construct_at(&m_bigTmpMem, m_bigBuffer.get(), m_bigBufferSize);
    }

    void Renderer::startRenderThread()
    {
        auto* rtData     = reinterpret_cast<render_thread::Data*>(m_renderThreadData.get());
        rtData->bvh      = nullptr;
        rtData->scene    = &scene;
        rtData->texCache = &texCache;
        rtData->params   = &params;
        rtData->workers  = &m_workers;
        m_renderThread.start(m_renderThreadData.get());
    }
} // namespace dmt

namespace dmt::sampling {
    DigitPermutations::DigitPermutations(uint32_t maxPrimeIndex, uint32_t seed, std::pmr::memory_resource* _memory) :
    m_memory{_memory},
    m_maxPrimeIndex{maxPrimeIndex}
    {
        assert(maxPrimeIndex <= NumPrimes);

        m_buffer = reinterpret_cast<unsigned char*>(
            m_memory->allocate(m_maxPrimeIndex * (sizeof(UniqueRef<uint16_t[]>) + sizeof(uint8_t))));
        if (!m_buffer)
            return;

        for (uint32_t i = 0; i < maxPrimeIndex; ++i)
        {
            // compute number of digits for the base
            uint32_t const base     = Primes[i];
            float const    invBase  = dmt::fl::rcp(base);
            uint8_t        nDigits  = 0;
            float          invBaseM = 1.f;
            // until floating point subtracton of the current power has an effect
            while (1 - (base - 1) * invBaseM < 1)
            {
                ++nDigits;
                invBaseM *= invBase;
            }

            nDigitsBuffer()[i] = nDigits;
            std::construct_at(&permBuffer()[i], makeUniqueRef<uint16_t[]>(m_memory, nDigits * base));

            // compute permutations for all digits
            uint16_t* permutations = permBuffer()[i].get();
            for (int32_t digitIndex = 0; digitIndex < nDigits; ++digitIndex)
            {
                uint64_t const dseed = dmt::numbers::hashIntegers(base, static_cast<uint32_t>(digitIndex), seed);
                for (int32_t digitValue = 0; digitValue < base; ++digitValue)
                {
                    int32_t const index = digitIndex * base + digitValue;
                    // only one?
                    permutations[index] = dmt::numbers::permutationElement(digitValue, base, dseed);
                }
            }
        }
    }

    DigitPermutations::DigitPermutations(DigitPermutations&& other) noexcept :
    m_memory(other.m_memory),
    m_maxPrimeIndex(other.m_maxPrimeIndex),
    m_buffer(other.m_buffer)
    {
        other.m_memory        = nullptr;
        other.m_maxPrimeIndex = 0;
        other.m_buffer        = nullptr;
    }

    DigitPermutations& DigitPermutations::operator=(DigitPermutations&& other) noexcept
    {
        if (this != &other)
        {
            // Free existing buffer
            destroy();

            // Steal other's resources
            m_memory        = other.m_memory;
            m_maxPrimeIndex = other.m_maxPrimeIndex;
            m_buffer        = other.m_buffer;

            other.m_memory        = nullptr;
            other.m_maxPrimeIndex = 0;
            other.m_buffer        = nullptr;
        }
        return *this;
    }

    void DigitPermutations::destroy() noexcept
    {
        if (m_buffer && m_memory)
        {
            // Destroy all UniqueRefs explicitly
            for (uint32_t i = 0; i < m_maxPrimeIndex; ++i)
            {
                std::destroy_at(&permBuffer()[i]);
            }

            m_memory->deallocate(m_buffer, m_maxPrimeIndex * (sizeof(UniqueRef<uint16_t[]>) + sizeof(uint8_t)));
        }

        m_buffer = nullptr;
    }

    // -- Utility Methods for Halton Owen --
    static float radicalInverse(uint32_t primeIndex, uint64_t num)
    {
        assert(primeIndex < NumPrimes && "Exceeding number of primes for radical inverse computation");

        uint32_t const base    = Primes[primeIndex];
        float const    invBase = dmt::fl::rcp(base);
        float          result  = 0.f;
        float          factor  = invBase;

        while (num > 0)
        {
            uint64_t const digit = num % base;

            result += digit * factor;
            num /= base;
            factor *= invBase;
        }

        return result;
    }

    static uint64_t inverseRadicalInverse(uint64_t inverse, int32_t base, int32_t nDigits)
    {
        uint64_t index = 0;
        for (int32_t i = 0; i < nDigits; ++i)
        {
            uint64_t const digit = inverse % base;
            inverse /= base;
            index *= base;
            index += digit;
        }
        return index;
    }

    static float scrambledRadicalInverse(DigitPermutation const perm, uint64_t num)
    {
        float const invBase       = dmt::fl::rcp(perm.base);
        int32_t     digitIndex    = 0;
        float       invBaseM      = 1.f;
        uint64_t    reverseDigits = 0;
        while (1 - (perm.base - 1) * invBaseM < 1)
        {
            uint64_t const next       = num / perm.base;
            int32_t const  digitValue = num - next * perm.base; // basically num % perm.base
            reverseDigits *= perm.base + perm.permute(digitIndex, digitValue);
            invBaseM *= invBase;
            ++digitIndex;
            num = next;
        }

        return std::min(invBaseM * reverseDigits, dmt::fl::oneMinusEps());
    }

    static float owenScrambledRadicalInverse(int32_t primeIndex, uint64_t num, uint32_t hash)
    {
        assert(primeIndex < NumPrimes && "primeIndex too high");
        namespace dn                 = dmt::numbers;
        uint32_t const base          = Primes[primeIndex];
        float const    invBase       = dmt::fl::rcp(base);
        int32_t        digitIndex    = 0;
        float          invBaseM      = 1.f;
        uint64_t       reverseDigits = 0;
        while (1 - (base - 1) * invBaseM < 1)
        {
            uint64_t const next = num / base;
            // second param ignored. TODO: SIMD
            int32_t const digitValue = dn::permutationElement(num - next * base, base, dn::mixBits(hash ^ reverseDigits));
            reverseDigits = reverseDigits * base + digitValue;
            invBaseM *= invBase;
            ++digitIndex;
            num = next;
        }

        return std::min(invBaseM * reverseDigits, dmt::fl::oneMinusEps());
    }

    static float sampleDim(int dim, int64_t haltonIndex)
    {
        namespace dn = dmt::numbers;
        float const res = owenScrambledRadicalInverse(dim, haltonIndex, dn::mixBits(1 + static_cast<uint64_t>(dim) << 4));
        assert(res >= 0 && res <= 1.f && "Owen scrabled radical inverse broken");
        return res;
    }

    static inline constexpr uint64_t multiplicativeInverse(int64_t a, int64_t n)
    {
        constexpr auto extendedGCD = [](auto&& f, uint64_t _a, uint64_t _b, int64_t* _x, int64_t* _y) -> void {
            if (_b == 0)
            {
                *_x = 1;
                *_y = 0;
            }
            else
            {
                int64_t d = _a / _b, xp, yp;
                f(f, _b, _a % _b, &xp, &yp);
                *_x = yp;
                *_y = xp - (d * yp);
            }
        };
        int64_t x, y;
        extendedGCD(extendedGCD, a, n, &x, &y);
        return x % n;
    }


    // -- halton owen --
    HaltonOwen::HaltonOwen(int32_t samplesPerPixel, Point2i resolution, int32_t seed)
    {
        for (int32_t i = 0; i < NumDimensions; ++i)
        {
            // using smallest primes for the 2 dimensions
            int32_t const base  = DimPrimes[i];
            int32_t       scale = 1, exp = 0;
            while (scale < std::min(resolution[i], MaxHaltonResolution))
            {
                scale *= base;
                ++exp;
            }
            m_baseScales[i]    = scale;
            m_baseExponents[i] = exp;
        }

        m_multInvs[0] = multiplicativeInverse(m_baseScales[1], m_baseScales[0]);
        m_multInvs[1] = multiplicativeInverse(m_baseScales[0], m_baseScales[1]);
    }

    void HaltonOwen::startPixelSample(Point2i p, int32_t sampleIndex, int32_t dim)
    {
        int32_t const sampleStride = m_baseScales[0] * m_baseScales[1];
        m_haltonIndex              = 0;

        if (sampleStride > 1)
        {
            // reproject pixel coordinates onto halton grid
            Point2i pm{{p[0] % MaxHaltonResolution, p[1] % MaxHaltonResolution}};
            for (int32_t i = 0; i < NumDimensions; ++i)
            {
                // scaling by base^exponent makes it so that the first exponent digits to the right of the . in the radical inverse
                // are shifted to the left. Hence, if you scale by base^exponent, the integer part of
                // the result will be in [0, base^exponent - 1] -> call it x (or y, depends on dimension)
                // let x_r be radical inverse of the last j digits of x (pixel coord), j exponent in dimension 0
                // let y_r be radical inverse of the last k digits of y (pixel coord), k exponent in dimension 1
                // x_r = (haltonIndex mod 2^j), y_r = (haltonIndex mod 3^k), solve for haltonIndex
                uint64_t const dimOffset = inverseRadicalInverse(pm[i], DimPrimes[i], m_baseExponents[i]);
                m_haltonIndex += dimOffset * (sampleStride / m_baseScales[i]) * m_multInvs[i];
            }
            m_haltonIndex %= sampleStride;
        }

        m_haltonIndex += sampleIndex * sampleStride;
        m_dimension = std::max(2, dim);
    }

    float HaltonOwen::get1D()
    {
        if (m_dimension >= NumPrimes)
            m_dimension = 2;
        return sampleDim(m_dimension++, m_haltonIndex);
    }

    Point2f HaltonOwen::get2D()
    {
        Point2f p{};
        if (m_dimension + 1 >= NumPrimes)
            m_dimension = 2;
        int const dim = m_dimension;
        m_dimension += 2;

        p.x = sampleDim(dim, m_haltonIndex);
        p.y = sampleDim(dim + 1, m_haltonIndex);
        return p;
    }

    Point2f HaltonOwen::getPixel2D()
    {
        Point2f p{.5f, .5f};
        p.x = radicalInverse(DimPrimeIndices[0], m_haltonIndex >> m_baseExponents[0]);
        p.y = radicalInverse(DimPrimeIndices[1], m_haltonIndex / m_baseScales[1]);
        assert(p.x <= 1.f && p.x >= 0.f && p.y <= 1.f && p.y >= 0.f && "Out of bounds");
        return p;
    }
} // namespace dmt::sampling

namespace dmt::camera {
    Ray generateRay(CameraSample const& cs, Transform const& cameraFromRaster, Transform const& renderFromCamera)
    {
        Point3f const pxImage{{cs.pFilm.x, cs.pFilm.y, 0}};
        Point3f const pCamera{{cameraFromRaster(pxImage)}};
        // TODO add lens?
        // time should use lerp(cs.time, shutterOpen, shutterClose)
        Ray const ray{Point3f{{0, 0, 0}}, normalize(pCamera), cs.time};
        // TODO handle tMax better
        float tMax = 1e5f;
        return renderFromCamera(ray, &tMax);
    }

    ApproxDifferentialsContext minDifferentialsFromCamera(Transform const&     cameraFromRaster,
                                                          Transform const&     renderFromCamera,
                                                          film::RGBFilm const& film,
                                                          uint32_t             samplesPerPixel)
    {
        static constexpr uint32_t CellNum = 512;

        ApproxDifferentialsContext ret{};
        ret.samplesPerPixel     = samplesPerPixel;
        ret.minPosDifferentialX = Vector3f::s(fl::infinity());
        ret.minPosDifferentialY = Vector3f::s(fl::infinity());
        ret.minDirDifferentialX = Vector3f::s(fl::infinity());
        ret.minDirDifferentialY = Vector3f::s(fl::infinity());

        Vector3f const       dxCamera = cameraFromRaster(Point3f{1, 0, 0}) - cameraFromRaster(Point3f{0, 0, 0});
        Vector3f const       dyCamera = cameraFromRaster(Point3f{0, 1, 0}) - cameraFromRaster(Point3f{0, 0, 0});
        camera::CameraSample sample{};
        sample.pLens        = {0.5f, 0.5f};
        sample.time         = 0.5f;
        sample.filterWeight = 1.f;

        for (uint32_t i = 0; i < CellNum; ++i)
        {
            sample.pFilm.x = static_cast<float>(i) / (CellNum - 1) * film.resolution().x;
            sample.pFilm.y = static_cast<float>(i) / (CellNum - 1) * film.resolution().y;

            Ray const      ray      = camera::generateRay(sample, cameraFromRaster, renderFromCamera);
            Point3f const  rxOrigin = ray.o, ryOrigin = ray.o; // TOOD holds only for lensless perspective cameras
            Vector3f const rxDirection = normalize(renderFromCamera(renderFromCamera.applyInverse(ray.d) + dxCamera));
            Vector3f const ryDirection = normalize(renderFromCamera(renderFromCamera.applyInverse(ray.d) + dyCamera));

            Vector3f dox = renderFromCamera.applyInverse(rxOrigin - ray.o);
            Vector3f doy = renderFromCamera.applyInverse(ryOrigin - ray.o);
            if (dotSelf(dox) < dotSelf(ret.minPosDifferentialX))
                ret.minPosDifferentialX = dox;
            if (dotSelf(doy) < dotSelf(ret.minPosDifferentialY))
                ret.minPosDifferentialY = doy;

            Frame const f = Frame::fromZ(ray.d);

            Vector3f const df = normalize(f.toLocal(ray.d)); // should be 0 0 1
            assert(fl::abs(df.z - 1.f) < 1e-5f && fl::nearZero(df.x) && fl::nearZero(df.y));
            Vector3f const dxf = normalize(f.toLocal(rxDirection));
            Vector3f const dyf = normalize(f.toLocal(ryDirection));

            if (dotSelf(dxf - df) < dotSelf(ret.minDirDifferentialX))
                ret.minDirDifferentialX = dxf - df;
            if (dotSelf(dyf - df) < dotSelf(ret.minDirDifferentialY))
                ret.minDirDifferentialY = dyf - df;
        }

        return ret;
    }
} // namespace dmt::camera

namespace dmt::film {
    void RGBFilm::writeImage(os::Path const& imagePath, std::pmr::memory_resource* temp)
    {
        using std::uint8_t;

        int const   width     = m_resolution.x;
        int const   height    = m_resolution.y;
        int const   numPixels = width * height;
        int const   channels  = 3; // RGB, 24-bit
        float const gamma     = 1.0f / 2.2f;

        // Allocate 8-bit per channel RGB image buffer
        UniqueRef<uint8_t[]> image = makeUniqueRef<uint8_t[]>(temp, numPixels * channels);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int const    idx = x + width * y;
                Pixel const& px  = m_pixels[idx];

                float const scale = px.weightSum > 0 ? 1.0f / px.weightSum : 0.0f;

                // basic gamma correction
                float const r = std::pow(float(px.rgbSum[0] * scale), gamma);
                float const g = std::pow(float(px.rgbSum[1] * scale), gamma);
                float const b = std::pow(float(px.rgbSum[2] * scale), gamma);

                // Clamp and convert to 8-bit
                constexpr auto toByte = [](float v) -> uint8_t {
                    return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
                };

                image[static_cast<size_t>(idx) * 3 + 0] = toByte(r);
                image[static_cast<size_t>(idx) * 3 + 1] = toByte(g);
                image[static_cast<size_t>(idx) * 3 + 2] = toByte(b);
            }
        }

        // Write PNG using stb_image_write
        stbi_write_png(imagePath.toUnderlying(temp).c_str(), // path
                       width,
                       height,          // resolution
                       channels,        // number of channels (RGB)
                       image.get(),     // image buffer
                       width * channels // stride in bytes
        );
    }

    void RGBFilm::addSample(Point2i pixel, RGB sample, float weight)
    {
        float const m = [](RGB rgb) {
            if (rgb.r > rgb.g && rgb.r > rgb.b)
                return rgb.r;
            else if (rgb.g > rgb.r && rgb.g > rgb.b)
                return rgb.g;
            else
                return rgb.b;
        }(sample);
        if (m > m_maxComponentValue)
        {
            sample.r *= m_maxComponentValue / m;
            sample.g *= m_maxComponentValue / m;
            sample.b *= m_maxComponentValue / m;
        }
        assert(pixel.x < m_resolution.x && pixel.y < m_resolution.y && "Outside pixel boundaries");
        Pixel& px = m_pixels[pixel.x + m_resolution.x * pixel.y];
        for (int32_t c = 0; c < 3; ++c)
            px.rgbSum[c] += reinterpret_cast<float*>(&sample)[c];

        px.weightSum += weight;
    }
} // namespace dmt::film
