#include "core-render.h"

#include "core-bvh-builder.h"
#include "core-light-tree-builder.h"

// write image dependencies
#include <ImfRgbaFile.h>
#include <ImfArray.h>

#define STBI_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "platform/platform-context.h"

// TODO remove
//#define DMT_SINGLE_THREAD
#define DMT_DBG_PIXEL
#define DMT_DBG_PIXEL_X     69
#define DMT_DBG_PIXEL_Y     71
#define DMT_DBG_SAMPLEINDEX 0x1
#define DMT_DBG_DEPTH       0x0
#define DMT_PBRT_HITERROR

namespace dmt {
    // make absolute-value vector
    static inline Vector3f absVec(Vector3f v) { return Vector3f{fabsf(v.x), fabsf(v.y), fabsf(v.z)}; }

    // estimate hit error from position differentials (conservative)
#if !defined(DMT_PBRT_HITERROR)
    static inline Vector3f estimateHitError(Vector3f dpdx, Vector3f dpdy, Point3f p)
#else
    static inline Vector3f estimateHitError(float b0, float b1, float b2, Point3f p0, Point3f p1, Point3f p2)
#endif
    {
#if !defined(DMT_PBRT_HITERROR)
        // differential-based error (like PBRT)
        Vector3f err = (absVec(dpdx) + absVec(dpdy)) * fl::gamma(5);

        // add a tiny absolute epsilon proportional to the magnitude of p
        // this avoids underflow when dpdx/dpdy are tiny or missing
        float    maxAbsP = fmaxf(fmaxf(fabsf(p.x), fabsf(p.y)), fabsf(p.z)) + 1e-4f;
        float    absEps  = fmaxf(1e-6f, 8.0f * std::numeric_limits<float>::epsilon() * maxAbsP);
        Vector3f absErr  = Vector3f::s(absEps);

        return err + absErr;
#else
        Point3f  pAbsSum = Point3f{abs(b0 * p0)} + abs(b1 * p1) + abs(b2 * p2);
        Vector3f pError  = fl::gamma(7) * Vector3f(pAbsSum);
        return pError;
#endif
    }

    static inline Point3f offsetRayOrigin(Point3f p, Vector3f pError, Normal3f ng, Vector3f w)
    {
        Vector3f n = ng.asVec();
        // conservative offset distance along normal
        float d = dot(absVec(n), pError);

        // add a tiny absolute safety margin to ensure robust separation
        // the constant below can be tuned; 1e-4 is conservative for typical meters-scale scenes
        // constexpr float extraMargin = 2e-3f;
        // d += extraMargin;

        // choose direction away from the surface relative to ray direction w
        Vector3f offset = n * d;

        Point3f po = p + offset;

        // Move each component to next float away from the surface to break ties
        auto bump = [](float x, float o) {
            if (o > 0.f)
                return fl::nextFloatUp(x);
            if (o < 0.f)
                return fl::nextFloatDown(x);
            return x;
        };

        po.x = bump(po.x, offset.x);
        po.y = bump(po.y, offset.y);
        po.z = bump(po.z, offset.z);

        return po;
    }

} // namespace dmt

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
        int32_t                         maxDepth;
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
#if defined(DMT_DBG_PIXEL)
                if (x == DMT_DBG_PIXEL_X && y == DMT_DBG_PIXEL_Y)
                    int i = 0;
#endif
                // TODO remove
                {
                    Context ctx;
                    ctx.log("[Thread {}] Pixel {} {}", std::make_tuple(tid, x, y));
                }
                for (int32_t sampleIndex = 0; sampleIndex < data.SamplesPerPixel; ++sampleIndex)
                {
#if defined(DMT_DBG_PIXEL)
                    if (x == DMT_DBG_PIXEL_X && y == DMT_DBG_PIXEL_Y && sampleIndex == DMT_DBG_SAMPLEINDEX)
                        int i = 0;
#endif

                    std::pmr::monotonic_buffer_resource scratch{4096}; // TODO better
                    sampler.startPixelSample({x, y}, sampleIndex);
                    camera::CameraSample const cs = camera::getCameraSample(sampler, {x, y}, *data.filter);

                    Ray ray{camera::generateRay(cs, *data.cameraFromRaster, *data.renderFromCamera)};

                    RGB  L = RGB::fromScalar(0.f), beta = RGB::fromScalar(1.f);
                    bool hadTransmission = false, specularBounce = false, anyNonSpecularBounces = false;
                    // LightSampleContext prevIntrCtx; needed only for emissive surfaces
                    float    bsdfPdf = 1.f, etascale = 1.f;
                    uint32_t depth = 0;

                    while (true)
                    {
                        // TODO remove
                        //if (data.StartPixel.x != 0 && data.StartPixel.y != 0)
                        //{
                        //    Context ctx;
                        //    ctx.log("[Thread {}]  Ray O {} {} {} Ray D {} {} {}",
                        //            std::make_tuple(tid, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y, ray.d.z));
                        //}

#if defined(DMT_DBG_PIXEL)
                        if (x == DMT_DBG_PIXEL_X && y == DMT_DBG_PIXEL_Y && sampleIndex == DMT_DBG_SAMPLEINDEX &&
                            depth == DMT_DBG_DEPTH)
                            int i = 0;
#endif

                        // 1. Trace Rays towards scene to find an intersection (Main Path)
                        uint32_t           instanceIdx = 0;
                        size_t             triIdx      = 0;
                        triangle::Triisect trisect     = triangle::Triisect::nothing();
                        if (!bvh::traverseRay(ray, data.bvhRoot, data.bvhNodeNum, &instanceIdx, &triIdx, &trisect))
                        {
                            if (data.envLight)
                            {
                                float     pdfLight = 0;
                                RGB const Le       = envLightEval(*data.envLight, ray.d, &pdfLight);
                                // 2. if no intersection, light at infinity, break
                                if (depth == 0 || specularBounce) // specular bounce = PDF BSDF is Dirac Delta
                                    L += beta * Le;
                                else // MIS
                                    L += beta * (bsdfPdf / (bsdfPdf + pdfLight)) * Le;
                            }
                            break;
                        }
                        else
                        {
                            // 3. check if maximum number of bounces has been reached and break
                            if (depth >= data.maxDepth)
                                break;
                            depth++;

                            // 4. if intersection and emissive, account for emission (TODO later maybe)
                            // Bonus. compute intersection information
                            Instance const&        instance = *data.scene->instances[instanceIdx];
                            TriangleMesh const&    mesh     = *data.scene->geometry[instance.meshIdx];
                            IndexedTri const&      triangle = mesh.getIndexedTri(triIdx);
                            Transform const        xform    = transformFromAffine(instance.affineTransform);
                            SurfaceMaterial const& mat      = data.scene->materials[triangle.matIdx];
                            Point2f const          uv0      = mesh.getUV(triangle.v[0].uvIdx);
                            Point2f const          uv1      = mesh.getUV(triangle.v[1].uvIdx);
                            Point2f const          uv2      = mesh.getUV(triangle.v[2].uvIdx);

                            // Perform the barycentric interpolation.
                            float const w0 = 1.0f - trisect.u - trisect.v;
                            float const w1 = trisect.u;
                            float const w2 = trisect.v;

                            Point2f const  uv   = w0 * uv0 + w1 * uv1 + w2 * uv2;
                            Vector2f const duv1 = uv1 - uv0;
                            Vector2f const duv2 = uv2 - uv0;

                            Point3f const p0 = xform(mesh.getPosition(triangle.v[0].positionIdx));
                            Point3f const p1 = xform(mesh.getPosition(triangle.v[1].positionIdx));
                            Point3f const p2 = xform(mesh.getPosition(triangle.v[2].positionIdx));

                            Vector3f const dp1 = p1 - p0;
                            Vector3f const dp2 = p2 - p0;

                            float const det = duv1.x * duv2.y - duv1.y * duv2.x;
                            Vector3f    dpdu, dpdv;
                            if (fl::abs(det) < 1e-8f)
                            {
                                // UVs are degenerate � fallback to geometric basis
                                Vector3f ng = normalize(cross(p2 - p0, p1 - p0));
                                coordinateSystemFallback(ng, &dpdu, &dpdv); // builds arbitrary tangent frame
                                return;
                            }

                            float const invDet = 1.0f / det;
                            dpdu               = (duv2.y * dp1 - duv1.y * dp2) * invDet;
                            dpdv               = (-duv2.x * dp1 + duv1.x * dp2) * invDet;

                            float const   t = trisect.t;
                            Point3f const p = ray.o + ray.d * t;

                            Vector3f const n0    = mesh.getNormal(triangle.v[0].normalIdx);
                            Vector3f const n1    = mesh.getNormal(triangle.v[1].normalIdx);
                            Vector3f const n2    = mesh.getNormal(triangle.v[2].normalIdx);
                            Normal3f const ngObj = normalFrom(n0 + n1 + n2);
                            Normal3f const ng    = xform(ngObj);

// TODO remove
#if 0
                            if (bool cameraRay = depth == 1; cameraRay)
                            {
                                assert(dot(ng, -ray.d) > 0 && "Camera ray cannot intersect backface");
                            }
#endif

                            Vector3f dpdx, dpdy;
                            data.diffCtx.p = p;
                            data.diffCtx.n = ng;
                            approximate_dp_dxy(data.diffCtx, &dpdx, &dpdy);

#if defined(DMT_PBRT_HITERROR)
                            Vector3f const pErr = estimateHitError(trisect.u, trisect.w, trisect.w, p0, p1, p2);
#else
                            Vector3f const pErr = estimateHitError(dpdx, dpdy, p);
#endif

                            UVDifferentialsContext uvCtx{};
                            uvCtx.dpdu = dpdu;
                            uvCtx.dpdv = dpdv;
                            uvCtx.dpdx = dpdx;
                            uvCtx.dpdy = dpdy;

                            TextureEvalContext texCtx{};
                            texCtx.p    = p;
                            texCtx.dpdx = dpdx;
                            texCtx.dpdy = dpdy;
                            texCtx.n    = ng;
                            texCtx.uv   = uv;
                            texCtx.dUV  = duv_From_dp_dxy(uvCtx);

                            Normal3f const ns = xform(materialShadingNormal(mat, *data.texCache, texCtx, ngObj));

                            // 5. (Maybe) BSDF regularization
                            // 6. Sample direct illumination and cast shadow ray
                            // choose between envLight and light tree
                            // TODO: If material is specular skip direct lighting estimation
                            LightSampleContext lightCtx{};
                            lightCtx.ray             = ray;
                            lightCtx.n               = ns;
                            lightCtx.hadTransmission = hadTransmission;
                            lightCtx.p               = p;

                            LightSample  ls[4]{};
                            float        pmfs[4]{};
                            Light const* plights[4]{};
                            uint32_t     lsCount = 0; // number of shadow rays

                            float                  uLightChoice  = sampler.get1D();
                            Point2f                uLight        = sampler.get2D();
                            static constexpr float lightStartPMF = 0.5f;
                            if (uLightChoice < 0.5f && data.envLight) // TODO: light choice/tree vs env not fixed 0.5
                            {
                                lsCount = envLightSampleFromContext(*data.envLight, lightCtx, uLight, &ls[0]);
                                pmfs[0] = lightStartPMF;
                            }
                            else
                            {
                                if (uLightChoice >= 0.5f)
                                    uLightChoice = fl::clamp01((uLightChoice - 0.5f) * 2.f);
                                LightSplit const split = lightTreeAdaptiveSplit(data.lightTreeNodes[0], p);
                                SelectedLights const lights = selectLightsFromSplit(split, p, ng, uLightChoice, lightStartPMF); // TODO use ns

                                for (uint32_t i = 0; i < lights.count; ++i)
                                {
                                    Light const& light = data.scene->lights[lights.indices[i]];
                                    if (lightSampleFromContext(light, lightCtx, uLight, &ls[lsCount]))
                                    {
                                        plights[lsCount] = &light;
                                        pmfs[lsCount]    = lights.pmfs[i];
                                        lsCount++;
                                    }
                                }
                            }

                            for (uint32_t shadowRayIdx = 0; shadowRayIdx < lsCount; ++shadowRayIdx)
                            {
                                float tLight = 0;
                                // nudge towards light to avoid self intersections
                                Point3f            shadowOrigin = offsetRayOrigin(p, pErr, ng, ls[shadowRayIdx].d);
                                Ray const          shadowRay{shadowOrigin, ls[shadowRayIdx].d};
                                uint32_t           idx;
                                size_t             tri;
                                triangle::Triisect sect;

                                // evaluate material BSDF and ensure its value is nonzero for the sampled direction
                                if (ls[shadowRayIdx].pdf == 0)
                                    continue;

                                BSDFEval eval = materialEval(mat, *data.texCache, texCtx, -ray.d, shadowRay.d, ng);
                                if (eval.pdf == 0.f || eval.f.max() < 1e-5f)
                                    continue;

                                // ensure shadow ray doesn't encounter any obstacle in its path towards the source
                                // TODO account for mesh light?
                                if (bvh::traverseRay(shadowRay, data.bvhRoot, data.bvhNodeNum, &idx, &tri, &sect))
                                {
                                    //assert(!(instanceIdx == idx) || hadTransmission); // debugging purposes
                                    continue;
                                }
                                //assert(dot(shadowRay.d, ng) > 0); // todo remove

                                // account for direct lighting contribution
                                RGB Le{};
                                if (lsCount == 1 && !plights[shadowRayIdx]) // if envLight
                                    Le = envLightEval(*data.envLight, &ls[shadowRayIdx]);
                                else if (plights[shadowRayIdx] && lightIntersect(*plights[shadowRayIdx], shadowRay, &tLight))
                                    Le = lightEval(*plights[shadowRayIdx], &ls[shadowRayIdx]);

                                if (Le.max() > 0.f)
                                {
                                    float pdfLight = ls[shadowRayIdx].pdf;
                                    pdfLight *= pmfs[shadowRayIdx];

                                    // TODO account for 0 radiance paths in statistics
                                    if (ls[shadowRayIdx].delta)
                                        L += beta * Le * eval.f / pdfLight;
                                    else
                                        L += beta * (Le * eval.f / (pdfLight + eval.pdf)); // pdfLight / pdfLight simplified from MIS
                                }
                            }

                            // 7. Sample new path direction
                            BSDFSample /*const*/
                                bs = materialSample(mat, *data.texCache, texCtx, -ray.d, ng, sampler.get2D(), sampler.get1D());
                            if (bs.pdf == 0.f) // if doesn't bounce, path dies
                                break;

                            // TODO: remove
                            bs.wi = normalize(bs.wi);

                            // 8. update path state variables and perform russian roulette to terminate the path early
                            beta *= bs.f * absDot(bs.wi, ns) / bs.pdf;
                            assert(!fl::isInfOrNaN(beta.r) && !fl::isInfOrNaN(beta.g) && !fl::isInfOrNaN(beta.b));
                            bsdfPdf        = bs.pdf;
                            specularBounce = false;        // TODO compute if bounce was specular
                            anyNonSpecularBounces |= true; // TODO compute if bounce was specular
                            if (bool transmission = bs.eta != 1.f; transmission)
                            {
                                hadTransmission = true;
                                etascale *= bs.eta * bs.eta; // TODO see if do the reciprocal when going inside-out
                            }
                            // prevIntrCtx = ...
                            Point3f nextOrigin = offsetRayOrigin(p, pErr, ng, bs.wi);
                            ray                = Ray{nextOrigin, bs.wi};

                            // roussian roulette
                            RGB const rrBeta = beta * etascale;
                            if (rrBeta.max() < 1 && depth > 1)
                            {
                                float q = fl::clamp01(rrBeta.max());
                                if (sampler.get1D() < q)
                                    break;     // kill ray
                                beta /= 1 - q; // if survived, make it more significant
                            }
                        }
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

            BVHBuildNode* bvhRoot = nullptr;
            if (perInstanceBvhNodes.size() > 1)
            {
                bvhRoot = reinterpret_cast<BVHBuildNode*>(memHeap->allocate(sizeof(BVHBuildNode)));
                std::memset(bvhRoot, 0, sizeof(BVHBuildNode));
                bvh::buildCombined(bvhRoot, perInstanceBvhNodes, memHeap, memHeap);
            }
            else
            {
                bvhRoot = perInstanceBvhNodes[0];
            }

            std::pmr::monotonic_buffer_resource tmp{4096};

            BVHWiVeCluster* wivebvh = bvh::buildBVHWive(bvhRoot, nodeCount, &tmp, memHeap);

            if (perInstanceBvhNodes.size() > 1)
            {
                bvh::cleanup(bvhRoot, memHeap);
            }

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

        Context ctx;
        ctx.log("[RT] Started Render Thread", {});

        Data*    rtData       = reinterpret_cast<Data*>(data);
        uint32_t bvhNodeCount = 0;

        // construct BVH
        rtData->bvh = bvhBuild(*rtData->scene, &bvhNodeCount);
        ctx.log("[RT] Built BVH", {});

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
        ctx.log("[RT] Constructed Film Buffer and Mitchell Filter Distribution", {});

        // define camera (image plane physical dims, resolution given by image)
        Vector3f const cameraPosition  = rtData->params->cameraPosition;
        Normal3f const cameraDirection = normalFrom(rtData->params->cameraDirection);
        float const    focalLength     = rtData->params->focalLength;
        float const    sensorHeight    = rtData->params->sensorSize;

        Transform const            cameraFromRaster = transforms::cameraFromRaster_Perspective(focalLength,
                                                                                    sensorHeight,
                                                                                    static_cast<uint32_t>(Width),
                                                                                    static_cast<uint32_t>(Height));
        Transform const            renderFromCamera = transforms::worldFromCamera(cameraDirection, cameraPosition);
        ApproxDifferentialsContext diffCtx          = camera::minDifferentialsFromCamera(cameraFromRaster,
                                                                                renderFromCamera,
                                                                                film,
                                                                                static_cast<uint32_t>(SamplesPerPixel));
        ctx.log("[RT] Constructed Camera Parameters", {});

        // build light tree
        std::pmr::vector<LightTreeBuildNode> lightTreeNodes{&pool};
        std::pmr::vector<LightTrailPair>     lightTreeBittrails{&pool};
        if (rtData->scene->lights.size() > 0)
        {
            lightTreeBuild(rtData->scene->lights, &lightTreeNodes, &lightTreeBittrails, &scratch);
            resetMonotonicBufferPointer(scratch, scratchBuffer.get(), ScratchBufferBytes);
        }
        ctx.log("[RT] Constructed Light Tree", {});

        // allocate all necessary jobs
        uint32_t const NumTileX = ceilDiv(static_cast<uint32_t>(Width), TileWidth);
        uint32_t const NumTileY = ceilDiv(static_cast<uint32_t>(Height), TileHeight);
        uint32_t const NumJobs  = NumTileX * NumTileY;

        UniqueRef<job::EvalTileData[]> jobData = makeUniqueRef<job::EvalTileData[]>(&pool, NumJobs);
        ctx.log("[RT] Allocated Job Data", {});

        ThreadPoolV2& threadpool = *rtData->workers;

        UniqueRef<sampling::HaltonOwen[]> tlsSamplers = makeUniqueRef<sampling::HaltonOwen[]>(&pool, threadpool.numThreads());
        ctx.log("[RT] Allocated TLS for Halton Samplers with Owen Scrambling", {});
        for (uint32_t tidx = 0; tidx < threadpool.numThreads(); ++tidx)
        {
            int seed = 18123 * sinf(32424 * tidx);
            std::construct_at<sampling::HaltonOwen>(&tlsSamplers[tidx], SamplesPerPixel, Point2i{Width, Height}, seed);
        }

        threadpool.addJob({[](uintptr_t f, uint32_t tid) {}, 0}, EJobLayer::ePriority0);

        for (uint32_t job = 0; job < NumJobs; ++job)
        {
            uint32_t tileX = job % NumTileX;
            uint32_t tileY = job / NumTileX;

            Point2i startPix{static_cast<int>(tileX * TileWidth), static_cast<int>(tileY * TileHeight)};

            Point2i tileSize{static_cast<int>(std::min<int>(TileWidth, Width - startPix.x)),
                             static_cast<int>(std::min<int>(TileHeight, Height - startPix.y))};

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
            jobData[job].maxDepth           = rtData->params->maxDepth;

            threadpool.addJob({job::evalTile, std::bit_cast<uintptr_t>(&jobData[job])}, EJobLayer::ePriority0);
        }
        threadpool.kickJobs();
        threadpool.waitForAll();

        ctx.log("[RT] Finished everything", {});

        // TODO better
        film.writeImage(os::Path::executableDir() / "render.png");
    }
} // namespace dmt::render_thread

namespace dmt {
    Renderer::Renderer(size_t tmpSize) :
    scene{&heapAligned},
    m_bigBuffer{makeUniqueRef<unsigned char[]>(std::pmr::get_default_resource(), tmpSize)},
    m_bigBufferSize{tmpSize},
    m_bigTmpMem{m_bigBuffer.get(), m_bigBufferSize},
    m_renderThread{render_thread::mainLoop, &m_poolMem},
#if defined(DMT_SINGLE_THREAD)
    m_workers{1, &m_poolMem},
#else
    m_workers{std::thread::hardware_concurrency(), &m_poolMem},
#endif
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
