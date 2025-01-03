module;

#include "dmtmacros.h"

#include <array>
#include <atomic>
#include <bit>
#include <string_view>
#include <type_traits>

#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstring>

module middleware;

template <typename Enum, size_t N>
    requires(std::is_enum_v<Enum>)
static constexpr Enum enumFromStr(char const* str, std::array<std::string_view, N> const& types, Enum defaultEnum)
{
    for (uint8_t i = 0; i < types.size(); ++i)
    {
        if (std::strncmp(str, types[i].data(), types[i].size()) == 0)
        {
            return ::dmt::fromUnderlying<Enum>(i);
        }
    }
    return defaultEnum;
}

namespace dmt {
    // values can also be closed into arrays even if something expects a scalar. parameters are unordered
    namespace dict {
        using namespace std::string_view_literals;

        // Begin of line keywords
        static constexpr SText Option         = "Option"sv;
        static constexpr SText Camera         = "Camera"sv;
        static constexpr SText Sampler        = "Sampler"sv;
        static constexpr SText ColorSpace     = "ColorSpace"sv;
        static constexpr SText Film           = "Film"sv;
        static constexpr SText Filter         = "Filter"sv;
        static constexpr SText Integrator     = "Integrator"sv;
        static constexpr SText Accelerator    = "Accelerator"sv;
        static constexpr SText WorldBegin     = "WorldBegin"sv;
        static constexpr SText AttributeBegin = "AttributeBegin"sv;
        static constexpr SText AttributeEnd   = "AttributeEnd"sv;
        static constexpr SText Include = "Include"sv; // supports another pbrt file, literal or compressed with gzip,

        // transformations on the CTM (Current Transformation Matrix) (reset to Identity at world begin)
        static constexpr SText LookAt    = "LookAt"sv;    // expects a 9 element array
        static constexpr SText Translate = "Translate"sv; // expects x y z
        static constexpr SText Scale     = "Scale"sv;
        static constexpr SText Rotate    = "Rotate"sv;                  // expects 4 eleemnt array
        static constexpr SText CoordinateSystem = "CoordinateSystem"sv; // follows a name, saves a snapshot of the current CTM
        static constexpr SText CoordSysTransform = "CoordSysTransform"sv; // sets a named transform for usage
        static constexpr SText Transform         = "Transform"sv;         // 16 floats
        static constexpr SText ConcatTransform   = "ConcatTransform"sv;   // 16 floats
        static constexpr SText TransformTimes    = "TransformTimes"sv;    // in header, expects 2 floats (not array)
        static constexpr SText ActiveTransform = "ActiveTransform"sv; // expects either "StartTime", "EndTime" or "All"

        // misc
        static constexpr SText ReverseOrientation = "ReverseOrientation"sv; // reverses normals of all sequet shapes (and therefore their area lights) (another is the Option "twosided")

        // data types
        static constexpr SText tBool      = "bool"sv;
        static constexpr SText tFloat     = "float"sv;
        static constexpr SText tInteger   = "integer"sv;
        static constexpr SText tString    = "string"sv;
        static constexpr SText tRGB       = "rgb"sv;
        static constexpr SText tPoint2    = "point2"sv;
        static constexpr SText tPoint3    = "point3"sv;
        static constexpr SText tPoint     = "point"sv; // synonym of point3
        static constexpr SText tNormal    = "normal"sv;
        static constexpr SText tBlackbody = "blackbody"sv;
        static constexpr SText tSpectrum  = "spectrum"sv;
        static constexpr SText tVector3   = "vector3"sv;
        static constexpr SText tVector    = "vector"sv; // synonym of vector3

        // acene wide rendering options: general options
        namespace opts {
            static constexpr SText disablepixeljitter      = "disablepixeljitter"sv;      // bool, false
            static constexpr SText disabletexturefiltering = "disabletexturefiltering"sv; // bool false
            static constexpr SText disablewavelengthjitter = "disablewavelengthjitter"sv; // bool false
            static constexpr SText displacementedgescale   = "displacementedgescale"sv;   // float 1
            static constexpr SText msereferenceimage       = "msereferenceimage"sv;       // string (none)
            static constexpr SText msereferenceout         = "msereferenceout"sv;         // string (none)
            static constexpr SText rendercoordsys          = "rendercoordsys"sv; // precedes a render coord system
            namespace rendercoordsys_literals {
                static constexpr SText cameraworld = "cameraworld"sv; // default
                static constexpr SText camera      = "camera"sv;
                static constexpr SText world       = "world"sv;
            } // namespace rendercoordsys_literals
            static constexpr SText seed         = "seed"sv;         // integer 0
            static constexpr SText forcediffuse = "forcediffuse"sv; // bool false
            static constexpr SText pixelstats   = "pixelstats"sv;   // bool false (images encode per pixel statistics)
            static constexpr SText wavefront    = "wavefront"sv;    //bool false
        } // namespace opts

        // acene wide rendering options: cameras
        namespace camera {
            // camera types
            static constexpr SText perspective  = "perspective"sv;
            static constexpr SText orthographic = "orthographic"sv;
            static constexpr SText realistic    = "realistic"sv;
            static constexpr SText spherical    = "spherical"sv;
            // common params
            static constexpr SText shutteropen  = "shutteropen"sv;  // float 0
            static constexpr SText shutterclose = "shutterclose"sv; // float 1
            // projecting
            static constexpr SText frameaspectratio = "frameaspectratio"sv; // float coomputed from x and y res of the film
            static constexpr SText screenwindow  = "screenwindow"sv;        // float, computed from aspect ratio
            static constexpr SText lensradius    = "lensradius"sv;          // float 0
            static constexpr SText focaldistance = "focaldistance"sv;       // float 10^30
            // perspective
            static constexpr SText fov = "fov"sv; // float 90
            // spherical
            static constexpr SText mapping = "mapping"sv; // precedes a mapping, "equalarea", "equirectangular"
            namespace mapping_literals {
                static constexpr SText equalarea       = "equalarea"sv;
                static constexpr SText equirectangular = "equirectangular"sv;
            } // namespace mapping_literals
            // realistic
            static constexpr SText lensfile         = "lensfile"sv;         // string (none)
            static constexpr SText aperturediameter = "aperturediameter"sv; // float 1.0 mm
            static constexpr SText focusdistance    = "focusdistance"sv;    // float 10.0 m
            static constexpr SText aperture = "aperture"sv; // aperture, "circular" (either built in or filename)
            namespace aperture_builtin {
                static constexpr SText circular = "circular"sv; // default
                static constexpr SText gaussian = "gaussian"sv;
                static constexpr SText square   = "square"sv;
                static constexpr SText pentagon = "pentagon"sv;
                static constexpr SText star     = "star"sv;
            } // namespace aperture_builtin
        } // namespace camera

        // scene wide rendering optinos: samplers
        namespace sampler {
            // sampler types
            static constexpr SText halton      = "halton"sv;
            static constexpr SText independent = "independent"sv;
            static constexpr SText paddedsobol = "paddedsobol"sv;
            static constexpr SText sobol       = "sobol"sv;
            static constexpr SText stratified  = "stratified"sv;
            static constexpr SText zsobol      = "zsobol"sv;
            // samplers which use pseudorandom values
            static constexpr SText seed = "seed"sv; // integer, default took from command line options. file takes precedence
            // every sampler except StratifiedSampler. Furthermore, PaddedSobolSampler, SobolSampler, ZSobolSampler expect a POT
            static constexpr SText pixelsamples = "pixelsamples"sv; // integer, 16
            // halton, paddedSobol, Sobol and ZSobol want a randomization
            static constexpr SText randomization = "randomization"sv; // default = fastowen for everyone except Halton, for halton it's "permutedigits"
            namespace randomization_literals {
                static constexpr SText fastowen      = "fastowen"sv; // not available with Halton
                static constexpr SText permutedigits = "permutedigits"sv;
                static constexpr SText owen          = "owen"sv;
                static constexpr SText none          = "none"sv;
            } // namespace randomization_literals
            // Stratified sampler
            static constexpr SText jitter   = "jitter"sv;   // bool true
            static constexpr SText xsamples = "xsamples"sv; // integer 4
            static constexpr SText ysamples = "ysamples"sv; // integer 4
        } // namespace sampler

        // scene wide rendering optinos: color spaces
        namespace colorspace {
            // colorspace types
            static constexpr SText srgb       = "srgb"sv;
            static constexpr SText aces2065_1 = "aces2065-1"sv;
            static constexpr SText rec2020    = "rec2020"sv;
            static constexpr SText dci_p3     = "dci-p3"sv;
        } // namespace colorspace

        // scene wide rendering optinos: film
        namespace film {
            // film types
            static constexpr SText rgb      = "rgb"sv;
            static constexpr SText gbuffer  = "gbuffer"sv;
            static constexpr SText spectral = "spectral"sv;
            // common
            static constexpr SText xresolution  = "xresolution"sv;  // integer 1280
            static constexpr SText yresolution  = "yresolution"sv;  // integer 720
            static constexpr SText cropwindow   = "cropwindow"sv;   // float[4] [0 1 0 1]
            static constexpr SText pixelbounds  = "pixelbounds"sv;  // integer[4] [0 xres 0 yres]
            static constexpr SText diagonal     = "diagonal"sv;     // float 35
            static constexpr SText filename     = "filename"sv;     // string "pbrt.exr"
            static constexpr SText savefp16     = "savefp16"sv;     // bool true
            static constexpr SText iso          = "iso"sv;          // float 100
            static constexpr SText whitebalance = "whitebalance"sv; // float 0
            static constexpr SText sensor       = "sensor"sv;       // sensor enum, "cie1931"
            namespace sensor_literals {
                static constexpr SText cie1931             = "cie1931"sv;
                static constexpr SText canon_eos_100d      = "canon_eos_100d"sv;
                static constexpr SText canon_eos_1dx_mkii  = "canon_eos_1dx_mkii"sv;
                static constexpr SText canon_eos_200d      = "canon_eos_200d"sv;
                static constexpr SText canon_eos_200d_mkii = "canon_eos_200d_mkii"sv;
                static constexpr SText canon_eos_5d        = "canon_eos_5d"sv;
                static constexpr SText canon_eos_5d_mkii   = "canon_eos_5d_mkii"sv;
                static constexpr SText canon_eos_5d_mkiii  = "canon_eos_5d_mkiii"sv;
                static constexpr SText canon_eos_5d_mkiv   = "canon_eos_5d_mkiv"sv;
                static constexpr SText canon_eos_5ds       = "canon_eos_5ds"sv;
                static constexpr SText canon_eos_m         = "canon_eos_m"sv;
                static constexpr SText hasselblad_l1d_20c  = "hasselblad_l1d_20c"sv;
                static constexpr SText nikon_d810          = "nikon_d810"sv;
                static constexpr SText nikon_d850          = "nikon_d850"sv;
                static constexpr SText sony_ilce_6400      = "sony_ilce_6400"sv;
                static constexpr SText sony_ilce_7m3       = "sony_ilce_7m3"sv;
                static constexpr SText sony_ilce_7rm3      = "sony_ilce_7rm3"sv;
                static constexpr SText sony_ilce_9         = "sony_ilce_9"sv;
            } // namespace sensor_literals
            static constexpr SText maxcomponentvalue = "maxcomponentvalue"sv; // float std::numeric_limits<float>::infinity()
            // gbuffer
            static constexpr SText coordinatesystem = "coordinatesystem"sv;
            namespace coordinatesystem_literals {
                static constexpr SText camera = "camera"sv;
                static constexpr SText world  = "world"sv;
            } // namespace coordinatesystem_literals
            // spectral
            static constexpr SText nbuckets  = "nbuckets"sv;  // integer 16
            static constexpr SText lambdamin = "lambdamin"sv; // float 360.f
            static constexpr SText lambdamax = "lambdamax"sv; // float 830.f
        } // namespace film

        // scene wide rendering optinos: filters
        namespace filter {
            // filter types
            static constexpr SText gaussian = "gaussian"sv;
            static constexpr SText box      = "box"sv;
            static constexpr SText mitchell = "mitchell"sv;
            static constexpr SText sinc     = "sinc"sv;
            static constexpr SText triangle = "triangle"sv;
            // common
            static constexpr SText xradius = "xradius"sv; // float (depends on type)
            static constexpr SText yradius = "yradius"sv; // float (depends on type)
            // gaussian
            static constexpr SText sigma = "sigma"sv; // float 0.5
            // mitchell
            static constexpr SText B = "B"sv; // float 1/3
            static constexpr SText C = "C"sv; // float 1/3
            // sinc
            static constexpr SText tau = "tau"sv; // float 3
        } // namespace filter

        // scene wide rendering optinos: integrators
        namespace integrator {
            // integrator types
            static constexpr SText volpath          = "volpath"sv;
            static constexpr SText ambientocclusion = "ambientocclusion"sv;
            static constexpr SText bdpt             = "bdpt"sv;
            static constexpr SText lightpath        = "lightpath"sv;
            static constexpr SText mlt              = "mlt"sv;
            static constexpr SText path             = "path"sv;
            static constexpr SText randomwalk       = "randomwalk"sv;
            static constexpr SText simplepath       = "simplepath"sv;
            static constexpr SText simplevolpath    = "simplevolpath"sv;
            static constexpr SText sppm             = "sppm"sv;
            // most of the integrators
            static constexpr SText maxdepth     = "maxdepth"sv;     // integer 5 (all but ambientocclusion)
            static constexpr SText lightsampler = "lightsampler"sv; // enum "bvh" (path, volpath, wavefront)
            namespace lightsampler_literals {
                static constexpr SText bvh     = "bvh"sv;
                static constexpr SText uniform = "uniform"sv;
                static constexpr SText power   = "power"sv;
            } // namespace lightsampler_literals
            static constexpr SText regularize = "regularize"sv; // bool false (bdpt, mlt, path, volpath, wavefront)
            // ambientocclusion
            static constexpr SText cossample   = "cossample"sv;   // bool ture
            static constexpr SText maxdistance = "maxdistance"sv; // float std::numeric_limits<float>::infinity()
            // bdpt
            static constexpr SText visualizestrategies = "visualizestrategies"sv; // bool false
            static constexpr SText visualizeweights    = "visualizeweights"sv;    // bool false
            // mlt
            static constexpr SText bootstrapsamples     = "bootstrapsamples"sv;     // integer 100000
            static constexpr SText chains               = "chains"sv;               // integer 1000
            static constexpr SText mutationsperpixel    = "mutationsperpixel"sv;    // integer 100
            static constexpr SText largestepprobability = "largestepprobability"sv; // float 0.3f
            static constexpr SText sigma                = "sigma"sv;                // 0.01
            // simplepath
            static constexpr SText samplebsdf   = "samplebsdf"sv;   // bool true
            static constexpr SText samplelights = "samplelights"sv; // bool true
            // sppm
            static constexpr SText photonsperiteration = "photonsperiteration"sv; // integer -1 (== equal to num pixels)
            static constexpr SText radius              = "radius"sv;              // float 1
            static constexpr SText seed                = "seed"sv;                // integer 0
        } // namespace integrator

        // scene wide rendering optinos: accelerators
        namespace accelerator {
            // accelerator types
            static constexpr SText bvh    = "bvh"sv;
            static constexpr SText kdtree = "kdtree"sv;
            // bvh
            static constexpr SText maxnodeprims = "maxnodeprims"sv; // integer 4
            static constexpr SText splitmethod  = "splitmethod"sv;  // enum "sah"
            namespace splitmethod_literals {
                static constexpr SText sah    = "sah"sv;
                static constexpr SText middle = "middle"sv;
                static constexpr SText equal  = "equal"sv;
                static constexpr SText hlbvh  = "hlbvh"sv;
            } // namespace splitmethod_literals
            // kdtree
            static constexpr SText intersectcost = "intersectcost"sv; // integer 5
            static constexpr SText traversalcost = "traversalcost"sv; // integer 1
            static constexpr SText emptybonus    = "emptybonus"sv;    // float 0.5f
            static constexpr SText maxprims      = "maxprims"sv;      // integer 1
            static constexpr SText maxdepth      = "maxdepth"sv;      // integer -1
        } // namespace accelerator

        // scene wide rendering optinos: pertecipation media (see below)
        // MakeNamedMedium are allowed even before the World Block, and a single call to MediumInterface is also allowed,
        // to specify only the exterior medium, in which the camera starts in. Default is vacuum

        // describing the scene:
        static constexpr SText Attribute = "Attribute"sv; // specify attribute off the line instead of inline. it wants a quoted target, one of "shape" "light", "material", "texture"
        static constexpr SText Shape = "Shape"sv; // expects a name and a parameter lists. name is one of "bilinearmesh", "curve", "cylinder", "disk", "sphere", "trianglemesh",
        static constexpr SText ObjectBegin    = "ObjectBegin"sv; // expects a name
        static constexpr SText ObjectEnd      = "ObjectEnd"sv;
        static constexpr SText ObjectInstance = "ObjectInstance"sv; // expects a ObjectBegin name. Takes the CTM
        static constexpr SText LightSource = "LightSource"sv; // takes a quoted type, "point", "distant", "goniometric", "infinite", "point", "projection", "spot"
        static constexpr SText AreaLightSource = "AreaLightSource"sv;
        static constexpr SText Material = "Material"sv; // expects a type and param list. each param can be either a value, rgb, spectrum, or a texture. sets the current material. if you want to declare it and use it later, use a named material
        static constexpr SText MakeNamedMaterial = "MakeNamedMaterial"sv;
        static constexpr SText NamedMaterial     = "NamedMaterial"sv;
        static constexpr SText Texture           = "Texture"sv;       // expects name, type, class param list
        static constexpr SText MakeNamedMedium = "MakeNamedMedium"sv; // declares a medium, expects name type param list
        static constexpr SText MediumInterface = "MediumInterface"sv;

        // describing the scene: shapes
        namespace shape {
            // shape types
            static constexpr SText bilinearmesh = "bilinearmesh"sv;
            static constexpr SText curve        = "curve"sv;
            static constexpr SText cylinder     = "cylinder"sv;
            static constexpr SText disk         = "disk"sv;
            static constexpr SText sphere       = "sphere"sv;
            static constexpr SText trianglemesh = "trianglemesh"sv;
            static constexpr SText loopsubdiv   = "loopsubdiv"sv;
            static constexpr SText plymesh      = "plymesh"sv;
            // common
            static constexpr SText alpha = "alpha"sv; // either a float or texture (def float 1)
            // spceific (namespaces due to the fact they have clashing options)
            namespace curve {
                static constexpr SText P     = "P"sv;
                static constexpr SText basis = "basis"sv;
                namespace basis_literals {
                    static constexpr SText bezier  = "bezier"sv;  // 2D
                    static constexpr SText bspline = "bspline"sv; // 3D
                } // namespace basis_literals
                static constexpr SText degree = "degree"sv; // integer 3 (default) or 2
                static constexpr SText type   = "type"sv;
                namespace type_literals {
                    static constexpr SText flat = "flat"sv; // face the incident ray
                    static constexpr SText cylinder = "cylinder"sv; // includes a shading normal, hence expects 3 number after it??
                    static constexpr SText ribbon = "ribbon"sv; // curve has a fixed orientation, specified by N
                } // namespace type_literals
                static constexpr SText N          = "N"sv;          // only for "ribbon" curve types
                static constexpr SText width      = "width"sv;      // float 1
                static constexpr SText width0     = "width0"sv;     // float 1
                static constexpr SText width1     = "width1"sv;     // float 1
                static constexpr SText splitdepth = "splitdepth"sv; // integer 3
            } // namespace curve
            namespace cylinder {
                static constexpr SText radius = "radius"sv; // float 1
                static constexpr SText zmin   = "zmin"sv;   // float -1
                static constexpr SText zmax   = "zmax"sv;   // float 1
                static constexpr SText phimax = "phimax"sv; // float 360
            } // namespace cylinder
            namespace disk {
                static constexpr SText height      = "height"sv;      // float 0
                static constexpr SText radius      = "radius"sv;      // radius 1
                static constexpr SText innerradius = "innerradius"sv; // float 0
                static constexpr SText phimax      = "phimax"sv;      // float 360
            } // namespace disk
            namespace sphere {
                static constexpr SText radius = "radius"sv; // float 1
                static constexpr SText zmin   = "zmin"sv;   // float -radius
                static constexpr SText zmax   = "zmax"sv;   // float radius
                static constexpr SText phimax = "phimax"sv; // float 360
            } // namespace sphere
            namespace trianglemesh {
                // P N S uv must be same size. only P is required
                static constexpr SText indices = "indices"sv; // required unless there are only 3 vertices, integer[]
                static constexpr SText P       = "P"sv;       // point3[]
                static constexpr SText N  = "N"sv;  // normal[], if present, shading normals are computed using these
                static constexpr SText S  = "S"sv;  // vector3[], per-vertex tangents
                static constexpr SText uv = "uv"sv; // point2[]
            } // namespace trianglemesh
            namespace plymesh {
                static constexpr SText filename = "filename"sv; // relative path of .ply or .ply.gz (gzip compressed)
                static constexpr SText displacement = "displacement"sv; // displacement texture
                static constexpr SText edgelength = "edgelength"sv; // edges of a triangle are split until this is met, def. 1.f
            } // namespace plymesh
            namespace loopsubdiv {                            // mesh which is subdivided
                static constexpr SText levels  = "levels"sv;  // integer 3
                static constexpr SText indices = "indices"sv; // integer[]
                static constexpr SText P       = "P"sv;       // point[]
            } // namespace loopsubdiv
        } // namespace shape

        // describing the scene: lights
        namespace light {
            // light types
            static constexpr SText distant     = "distant"sv;
            static constexpr SText goniometric = "goniometric"sv;
            static constexpr SText infinite    = "infinite"sv;
            static constexpr SText point       = "point"sv;
            static constexpr SText projection  = "projection"sv;
            static constexpr SText spot        = "spot"sv;
            // common (either power or illuminance, not both)
            static constexpr SText power       = "power"sv;
            static constexpr SText illuminance = "illuminance"sv;
            static constexpr SText scale       = "scale"sv;
            namespace distant {
                static constexpr SText L = "L"sv; // spectrum, spectral radiance, default = current color space illuminant
                static constexpr SText from = "from"sv; // point, 0 0 0
                static constexpr SText to   = "to"sv;   // point, 0 0 1
            } // namespace distant
            namespace goniometric {
                static constexpr SText filename = "filename"sv; // string, no default, required
                static constexpr SText I        = "I"sv;        // current color space's illuminant
            } // namespace goniometric
            namespace infinite {
                // either filename or L
                static constexpr SText filename = "filename"sv;
                static constexpr SText portal   = "portal"sv; // point3[4], window through which the light is visible
                static constexpr SText L        = "L"sv;      // radiance intensity = L * scale * power
            } // namespace infinite
            namespace point {
                static constexpr SText I = "I"sv; // spectrum, default = current color space illuminant. spectrad dist of light emitted radiant intensity
                static constexpr SText from = "from"sv; // point, 0 0 0, light location
            } // namespace point
            namespace projection {
                static constexpr SText I        = "I"sv;        // spectrum
                static constexpr SText fov      = "fov"sv;      // float, 90
                static constexpr SText filename = "filename"sv; // string, required
            } // namespace projection
            namespace spotlight {
                static constexpr SText I              = "I"sv;              // spectrum, spectral intensity
                static constexpr SText from           = "from"sv;           // point, 0 0 0
                static constexpr SText to             = "to"sv;             // point, 0 0 1
                static constexpr SText coneangle      = "coneangle"sv;      // float, 30
                static constexpr SText conedeltaangle = "conedeltaangle"sv; // float, 5
            } // namespace spotlight
        } // namespace light

        // describing the scene: area light
        namespace arealight {
            // arealight types
            static constexpr SText diffuse = "diffuse"sv;
            // diffuse
            namespace diffuse {
                static constexpr SText filename = "filename"sv; // string required no default
                static constexpr SText L        = "L"sv;        // spectrum, emitted spectral radiance distribution
                static constexpr SText twosided = "twosided"sv; // bool, false = emit light only in halfspace pointed by normal of shape
            } // namespace diffuse
        } // namespace arealight

        // describing the scene: material
        namespace material {
            // material types
            static constexpr SText coateddiffuse       = "coateddiffuse"sv;
            static constexpr SText coatedconductor     = "coatedconductor"sv;
            static constexpr SText conductor           = "conductor"sv;
            static constexpr SText dielectric          = "dielectric"sv;
            static constexpr SText diffuse             = "diffuse"sv;
            static constexpr SText diffusetransmission = "diffusetransmission"sv;
            static constexpr SText hair                = "hair"sv;
            static constexpr SText interface           = "interface"sv;
            static constexpr SText measured            = "measured"sv;
            static constexpr SText mix                 = "mix"sv;
            static constexpr SText subsurface          = "subsurface"sv;
            static constexpr SText thindielectric      = "thindielectric"sv;
            // all but "interface" and "mix"
            static constexpr SText displacement = "displacement"sv;
            static constexpr SText normalmap    = "normalmap"sv;
            // common
            static constexpr SText roughness      = "roughness"sv; // float texture, GGX roughness isotropic
            static constexpr SText uroughness     = "uroughness"sv;
            static constexpr SText vroughness     = "vroughness"sv;
            static constexpr SText remaproughness = "remaproughness"sv;
            namespace coated {                              // coateddiffuse and coatedconductor
                static constexpr SText albedo = "albedo"sv; // spectrum texture, scattering albedo between interface and diffuse layers. in [0, 1]
                static constexpr SText g         = "g"sv;         // float texture, in [-1, 1]
                static constexpr SText maxdepth  = "maxdepth"sv;  // integer 10
                static constexpr SText nsamples  = "nsamples"sv;  // integer 1
                static constexpr SText thickness = "thickness"sv; // float 0.01f
                namespace diffuse {
                    static constexpr SText reflectance = "reflectance"sv; // spectrum texture, default 0.5
                }
                namespace conductor {
                    static constexpr SText conductor_eta = "conductor.eta"sv; // spectrum
                    static constexpr SText conductor_k   = "conductor.k"sv;   // spectrum
                    static constexpr SText reflectance   = "reflectance"sv;   // spectrum (NOT texture)
                } // namespace conductor
            } // namespace coated
            namespace conductor {
                namespace builtin_spectrum {
                    static constexpr SText glass_BK7    = "glass-BK7"sv;    // Index of refraction for BK7 glass
                    static constexpr SText glass_BAF10  = "glass-BAF10"sv;  // Index of refraction for BAF10 glass
                    static constexpr SText glass_FK51A  = "glass-FK51A"sv;  // Index of refraction for FK51A glass
                    static constexpr SText glass_LASF9  = "glass-LASF9"sv;  //Index of refraction for LASF9 glass
                    static constexpr SText glass_F5     = "glass-F5"sv;     // Index of refraction for F5 glass
                    static constexpr SText glass_F10    = "glass-F10"sv;    // Index of refraction for F10 glass
                    static constexpr SText glass_F11    = "glass-F11"sv;    // Index of refraction for F11 glass
                    static constexpr SText metal_Ag_eta = "metal-Ag-eta"sv; // Index of refraction for silver.
                    static constexpr SText metal_Ag_k   = "metal-Ag-k"sv;   // Extinction coefficient for silver.
                    static constexpr SText metal_Al_eta = "metal-Al-eta"sv; // Index of refraction for aluminum.
                    static constexpr SText metal_Al_k   = "metal-Al-k"sv;   // Extinction coefficient for aluminum.
                    static constexpr SText metal_Au_eta = "metal-Au-eta"sv; // Index of refraction for gold.
                    static constexpr SText metal_Au_k   = "metal-Au-k"sv;   // Extinction coefficient for gold.
                    static constexpr SText metal_Cu_eta = "metal-Cu-eta"sv; // Index of refraction for copper.
                    static constexpr SText metal_Cu_k   = "metal-Cu-k"sv;   // Extinction coefficient for copper.
                    static constexpr SText metal_CuZn_eta = "metal-CuZn-eta"sv; // Index of refraction for copper zinc alloy.
                    static constexpr SText metal_CuZn_k = "metal-CuZn-k"sv; // Extinction coefficient for copper zinc alloy.
                    static constexpr SText metal_MgO_eta = "metal-MgO-eta"sv; // Index of refraction for magnesium oxide.
                    static constexpr SText metal_MgO_k = "metal-MgO-k"sv; // Extinction coefficient for magnesium oxide.
                    static constexpr SText metal_TiO2_eta = "metal-TiO2-eta"sv; // Index of refraction for titanium dioxide.
                    static constexpr SText metal_TI02_k = "metal-TI02-k"sv; // Extinction coefficient for titanium dioxide.
                    static constexpr SText stdillum_A    = "stdillum-A"sv;    // CIE standard illuminant A.
                    static constexpr SText stdillum_D50  = "stdillum-D50"sv;  // CIE standard illuminant D50.
                    static constexpr SText stdillum_D65  = "stdillum-D65"sv;  // CIE standard illuminant D65.
                    static constexpr SText stdillum_F1   = "stdillum-F1"sv;   // CIE standard illuminants F1
                    static constexpr SText stdillum_F2   = "stdillum-F2"sv;   // CIE standard illuminants F2
                    static constexpr SText stdillum_F3   = "stdillum-F3"sv;   // CIE standard illuminants F3
                    static constexpr SText stdillum_F4   = "stdillum-F4"sv;   // CIE standard illuminants F4
                    static constexpr SText stdillum_F5   = "stdillum-F5"sv;   // CIE standard illuminants F5
                    static constexpr SText stdillum_F6   = "stdillum-F6"sv;   // CIE standard illuminants F6
                    static constexpr SText stdillum_F7   = "stdillum-F7"sv;   // CIE standard illuminants F7
                    static constexpr SText stdillum_F8   = "stdillum-F8"sv;   // CIE standard illuminants F8
                    static constexpr SText stdillum_F9   = "stdillum-F9"sv;   // CIE standard illuminants F9
                    static constexpr SText stdillum_F10  = "stdillum-F10"sv;  // CIE standard illuminants F10
                    static constexpr SText stdillum_F11  = "stdillum-F11"sv;  // CIE standard illuminants F11
                    static constexpr SText stdillum_F12  = "stdillum-F12"sv;  // CIE standard illuminants F12
                    static constexpr SText illum_acesD60 = "illum-acesD60"sv; // D60 illuminant from ACES.
                } // namespace builtin_spectrum
                static constexpr SText eta = "eta"sv; // spectrum texture (or built ix name), default = metal_Cu_eta
                static constexpr SText k   = "k"sv;   // spectrum texture (or built in name), default = metal_Cu_k
                static constexpr SText reflectance = "reflectance"sv; // spectrum texture (computed if absent?)
            } // namespace conductor
            namespace dielectric {
                static constexpr SText eta = "eta"sv; // float texture or spectrum texture. default float texture 1.5 constant
            }
            namespace diffuse {
                static constexpr SText reflectance = "reflectance"sv; // spectrum texture, def 0.5
            }
            namespace diffusetransmission {
                static constexpr SText reflectance   = "reflectance"sv;   // spectrum texture, def 0.25
                static constexpr SText transmittance = "transmittance"sv; // spectrum texture, def 0.25
                static constexpr SText scale         = "scale"sv;         // float texture, def 1
            } // namespace diffusetransmission
            namespace hair {
                // Color related: if sigma_a specified, everything else is ignored. if reflectance specified (and sigma_a is not), then ignore everything else. if nothing
                // is specified, use eumelanin 1.3 and pheomelanin 0
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum texture, absorption coefficient inside hair, normalized with respect to hair diameter
                static constexpr SText reflectance = "reflectance"sv; // spectrum texture, alternative to absorption coefficient
                static constexpr SText eumelanin = "eumelanin"sv; // float texture. 0.3 -> blonde, 1.3 -> brown, 8 -> black
                static constexpr SText pheomelanin = "pheomelanin"sv; // float texture, the higher the more red the hair gets
                // Shape related and other
                static constexpr SText eta    = "eta"sv;    // float texture, def 1.55
                static constexpr SText beta_m = "beta_m"sv; // float texture, def 0.3, [0,1]
                static constexpr SText beta_n = "beta_n"sv; // float texture, def 0.3, [0,1]
                static constexpr SText alpha  = "alpha";    // float texture, def 2 degrees
            } // namespace hair
            namespace measured {
                static constexpr SText filename = "filename"sv; // string filename
            }
            namespace mix {
                static constexpr SText materials = "materials"sv; // string[2], material names
                static constexpr SText amount    = "amount"sv;    // texture float, def 0.5,
            } // namespace mix
            namespace subsurface {
                // specified in one of 3 ways (+ common parameter eta + g)
                // 1. sigma_a + sigma_s (+ scale)
                // 2. reflectance + mean free path (mfp)
                // 3. name of builtin scattering properties
                static constexpr SText eta = "eta"sv; // float texture, 1.33, IOR of the scattering volume
                static constexpr SText g   = "g"sv;   // float texture, Henyey Greenstein asymmetry parameter
                static constexpr SText mfp = "mfp"sv; // float texture mean free path of hte volume in meters (only if reflectance)
                static constexpr SText name = "name"sv; // string, name of measured subsurface scattering coefficients
                static constexpr SText reflectance = "reflectance"sv; // spectrum texture, TODO see https://github.com/mmp/pbrt-v4/blob/cdccb71cb1e153b63e538f624efcc13ab0f9bda2/src/pbrt/media.cpp#L79
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum texture, default = RGB(0.0011, 0.0024, 0.014)
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum texture, default = RGB(2.55, 3.12, 3.77)
                static constexpr SText scale   = "scale"sv;
            } // namespace subsurface
        } // namespace material

        // describing the scene: texture
        namespace texture {
            // texture class
            static constexpr SText bilerp       = "bilerp"sv;       // {Float, Spectrum} BilerpTexture
            static constexpr SText checkerboard = "checkerboard"sv; // {Float, Spectrum} CheckerboardTexture
            static constexpr SText constant     = "constant"sv;     // {Float, Spectrum} ConstantTexture
            static constexpr SText directionmix = "directionmix"sv; // {Float, Spectrum} DirectionMixTexture
            static constexpr SText dots         = "dots"sv;         // {Float, Spectrum} DotsTexture
            static constexpr SText fbm          = "fbm"sv;          // FBmTexture
            static constexpr SText imagemap     = "imagemap"sv;     // Float, Spectrum} ImageTexture
            static constexpr SText marble       = "marble"sv;       // MarbleTexture
            static constexpr SText mix          = "mix"sv;          // {Float, Spectrum} MixTexture
            static constexpr SText ptex         = "ptex"sv;         // {Float, Spectrum} PtexTexture
            static constexpr SText scale        = "scale"sv;        // {Float, Spectrum} ScaledTexture
            static constexpr SText windy        = "windy"sv;        // WindyTexture
            static constexpr SText wrinkled     = "wrinkled"sv;     // WrinkledTexture
            // texture types
            static constexpr SText tSpectrum = "spectrum"sv; // colored texture, either nbuckets spectrum value pairs or rgb (depends on colorspace)
            static constexpr SText tFloat = "float"sv; // black and white texture
            // texture mapping types
            static constexpr SText mapping = "mapping"sv; // enum, def = uv
            namespace mapping_literals {
                static constexpr SText uv          = "uv"sv;
                static constexpr SText spherical   = "spherical"sv;
                static constexpr SText cylindrical = "cylindrical"sv;
                static constexpr SText planar      = "planar"sv;
            } // namespace mapping_literals
            // uv: scale and delta
            // spherical and cylindrical: use the current transformation matrix to orient and center
            // planar: delta, v1, v2
            // parameters incorrectly placed are ignored
            static constexpr SText uscale = "uscale"sv; // float, def = 1
            static constexpr SText vscale = "vscale"sv; // float, def = 1
            static constexpr SText udelta = "udelta"sv; // float, def = 0
            static constexpr SText vdelta = "vdelta"sv; // float, def = 0
            static constexpr SText v1     = "v1"sv;     // vector, def = 1 0 0
            static constexpr SText v2     = "v2"sv;     // vector, def = 0 1 0
            namespace encoding_literals {
                static constexpr SText sRGB   = "sRGB"sv;
                static constexpr SText linear = "linear"sv;
                static constexpr SText gamma  = "gamma"sv; // expects a float following it
            } // namespace encoding_literals

            namespace bilerp {
                static constexpr SText v00 = "v00"sv; // spectrum texture or float texture. def = float, 0
                static constexpr SText v01 = "v01"sv; // spectrum texture or float texture. def = float, 1
                static constexpr SText v10 = "v10"sv; // spectrum texture or float texture. def = float, 0
                static constexpr SText v11 = "v11"sv; // spectrum texture or float texture. def = float, 1
            } // namespace bilerp
            namespace checkerboard {
                static constexpr SText dimension = "dimension"sv; // integer, def = 2 (can be either 2 or 3)
                static constexpr SText tex1      = "tex1"sv;      // spectrum texture or float texture, def = float 1
                static constexpr SText tex2      = "tex2"sv;      // spectrum texture or float texture, def = float 0
            } // namespace checkerboard
            namespace constant {
                static constexpr SText value = "value"sv; // nbuckets values or rgb if spectrum, 1 value if float
            }
            namespace directionmix {
                static constexpr SText tex1 = "tex1"sv; // spectrum texture or float texture, def = float 0
                static constexpr SText tex2 = "tex2"sv; // spectrum texture or float texture, def = float 1
                static constexpr SText dir  = "dir"sv;  // vector, def = 0 1 0
            } // namespace directionmix
            namespace dots {
                static constexpr SText inside  = "inside"sv;  // spectrum texture or float texture, def = float 1
                static constexpr SText outside = "outside"sv; // spectrum texture or float texture, def = float 0
            } // namespace dots
            namespace perlin {                                    // fbm, wrinkled, windy
                static constexpr SText octaves   = "octaves"sv;   // integer, def = 8
                static constexpr SText roughness = "roughness"sv; // float, def = 0.5
            } // namespace perlin
            namespace imagemap {
                static constexpr SText filename = "filename"sv; // string, required, no def, has to end with ".tga", ".pfm", ".exr"
                namespace filename_extensions {
                    static constexpr SText _tga = ".tga"sv;
                    static constexpr SText _pfm = ".pfm"sv;
                    static constexpr SText _exr = ".exr"sv;
                } // namespace filename_extensions
                static constexpr SText wrap = "wrap"sv; // enum, def = repeat
                namespace wrap_literals {
                    static constexpr SText repeat = "repeat"sv;
                    static constexpr SText black  = "black"sv;
                    static constexpr SText clamp  = "clamp"sv;
                } // namespace wrap_literals
                static constexpr SText maxanisotropy = "maxanisotropy"sv; // float, def = 8, max elliptical eccentricity for EWA
                static constexpr SText filter = "filter"sv; // enum, def = bilinear, filter used to sample from the mipmapped texture
                namespace filter_literals {
                    static constexpr SText bilinear  = "bilinear"sv;
                    static constexpr SText ewa       = "ewa"sv;
                    static constexpr SText trilinear = "trilinear"sv;
                    static constexpr SText point     = "point"sv;
                } // namespace filter_literals
                static constexpr SText encoding = "encoding"sv; // enum, def = sRGB, how to convert a 8bit color to float
                static constexpr SText scale = "scale"sv;       // float, def = 1, scale to apply to the looked up value
                static constexpr SText invert = "invert"sv; // bool, def = false. If true, each value is converted with f(x) = 1 - x
            } // namespace imagemap
            namespace marble {                                    // still perlin
                static constexpr SText octaves   = "octaves"sv;   // integer, def = 8
                static constexpr SText roughness = "roughness"sv; // float, def = 0.5
                static constexpr SText scale     = "scale"sv;     // float, def = 1, scaling factor for inpouts
                static constexpr SText variation = "variation"sv; // float, def = 0.2, scaling factor for output
            } // namespace marble
            namespace mix {
                static constexpr SText tex1   = "tex1"sv;   // spectrum texture or float texture, def = float, 0
                static constexpr SText tex2   = "tex2"sv;   // spectrum texture or float texture, def = float, 1
                static constexpr SText amount = "amount"sv; // float texture, def = float 0.5
            } // namespace mix
            namespace ptex {
                static constexpr SText encoding = "encoding"sv; // enum, def = gamma 2.2
                static constexpr SText filename = "filename"sv; // stringfilename, end with ptex, required no def
                namespace filename_extensions {
                    static constexpr SText _ptex = ".ptex"sv;
                }
                static constexpr SText scale = "scale"sv; // float, def = 1
            } // namespace ptex
            namespace scale {
                static constexpr SText tex   = "tex"sv;   // spectrum texture or float texture to be scaled, def float 1
                static constexpr SText scale = "scale"sv; // float texture, def float 1
            } // namespace scale
        } // namespace texture

        // describing the scene: participating media
        namespace media {
            // Syntax: `MakeNamedMedium "name" "rgb sigma_a" [ ... ] "rgb sigma_s" [ ... ] "float scale" <num> "string type" "homogeneous"`
            static constexpr SText type = "type"sv; // type is required, string enum
            namespace type_literals {
                static constexpr SText cloud       = "cloud"sv;
                static constexpr SText homogeneous = "homogeneous"sv;
                static constexpr SText nanovdb     = "nanovdb"sv;
                static constexpr SText rgbgrid     = "rgbgrid"sv;
                static constexpr SText uniformgrid = "uniformgrid"sv;
            } // namespace type_literals
            namespace homogeneous {
                static constexpr SText g       = "g"sv;       // Henyey Greenstein asymmetry, float, def 0, [-1,1]
                static constexpr SText Le      = "Le"sv;      // spectrum, def 0, distribution of emitted radiance
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText preset = "preset"sv; // TODO see https://github.com/mmp/pbrt-v4/blob/cdccb71cb1e153b63e538f624efcc13ab0f9bda2/src/pbrt/media.cpp#L79
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
            } // namespace homogeneous
            namespace uniformgrid { // generalization of homogeneous, so it takes all its parameters plus the following
                static constexpr SText g       = "g"sv;       // Henyey Greenstein asymmetry, float, def 0, [-1,1]
                static constexpr SText Le      = "Le"sv;      // spectrum, def 0, distribution of emitted radiance
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText preset = "preset"sv; // TODO see https://github.com/mmp/pbrt-v4/blob/cdccb71cb1e153b63e538f624efcc13ab0f9bda2/src/pbrt/media.cpp#L79
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
                static constexpr SText nx      = "nx"sv; // integer, def = 1, number of density sample in dimension x
                static constexpr SText ny      = "ny"sv; // integer, def = 1, number of density sample in dimension y
                static constexpr SText nz      = "nz"sv; // integer, def = 1, number of density sample in dimension z
                static constexpr SText density = "density"sv; // nx*ny*nz numbers in row-major order, optional
                static constexpr SText p0 = "p0"sv; // point3, def 0 0 0, min bound of the density grid in medium space
                static constexpr SText p1 = "p1"sv; // point3, def 1 1 1, max bound of the density grid in medium space
                static constexpr SText temperature = "temperature"sv; // float[], nx*ny*nz kelvin values, row-major order, optional, then converted to blackbody emission spectra
                static constexpr SText temperatureoffset = "temperatureoffset"sv; // float, def = 0
                static constexpr SText temperaturescale  = "temperaturescale"sv;  // float, def = 1
            } // namespace uniformgrid
            namespace rgbgrid { // alternative to uniformgrid, so takes all parameters of homogeneous EXCEPT preset
                static constexpr SText g       = "g"sv;       // Henyey Greenstein asymmetry, float, def 0, [-1,1]
                static constexpr SText Le      = "Le"sv;      // spectrum, def 0, distribution of emitted radiance
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
                static constexpr SText p0 = "p0"sv; // point3, def 0 0 0, min bound of the density grid in medium space
                static constexpr SText p1 = "p1"sv; // point3, def 1 1 1, max bound of the density grid in medium space
            } // namespace rgbgrid
            namespace cloud {                       // perlin
                static constexpr SText p0 = "p0"sv; // point3, def 0 0 0, min bound of the density grid in medium space
                static constexpr SText p1 = "p1"sv; // point3, def 1 1 1, max bound of the density grid in medium space
                static constexpr SText density   = "density"sv;   // float, def = 1
                static constexpr SText frequency = "frequency"sv; // float, def = 5
                static constexpr SText g         = "g"sv; // Henyey Greenstein asymmetry parameter, float, def 0, [-1,1]
                static constexpr SText sigma_a   = "sigma_a"sv;  // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s   = "sigma_s"sv;  // spectrum, scattering cross section, def 1
                static constexpr SText wispness  = "wispness"sv; // float, def 1
            } // namespace cloud
            namespace nanovdb {
                static constexpr SText g       = "g"sv; // Henyey Greenstein asymmetry parameter, float, def 0, [-1,1]
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText temperatureoffset = "temperatureoffset"sv; // float, def = 0
                static constexpr SText temperaturescale  = "temperaturescale"sv;  // float, def = 1
                static constexpr SText filename          = "filename"sv;          // string
            } // namespace nanovdb
        } // namespace media
    } // namespace dict

    ERenderCoordSys renderCoordSysFromStr(char const* str)
    { // array needs to follow the order in which the enum values are declared
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ERenderCoordSys> count = toUnderlying(ERenderCoordSys::eCount);
        static constexpr std::array<std::string_view, count>     types{"cameraworld"sv, "camera"sv, "world"sv};

        return ::enumFromStr(str, types, ERenderCoordSys::eCameraWorld);
    }

    ECameraType cameraTypeFromStr(char const* str)
    { // array needs to follow the order in which the enum values are declared
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ECameraType> count = toUnderlying(ECameraType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"orthographic"sv, "perspective"sv, "realistic"sv, "spherical"sv};

        return ::enumFromStr(str, types, ECameraType::ePerspective);
    }

    ESphericalMapping sphericalMappingFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESphericalMapping> count = toUnderlying(ESphericalMapping::eCount);
        static constexpr std::array<std::string_view, count>       types{"equalarea"sv, "equirectangular"sv};

        return ::enumFromStr(str, types, ESphericalMapping::eEqualArea);
    }

    ESamplerType samplerTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESamplerType> count = toUnderlying(ESamplerType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"halton"sv, "independent"sv, "paddedsobol"sv, "sobol"sv, "stratified"sv, "zsobol"sv};

        return ::enumFromStr(str, types, ESamplerType::eZSobol);
    }

    ERandomization randomizationFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ERandomization> count = toUnderlying(ERandomization::eCount);
        static constexpr std::array<std::string_view, count> types{"fastowen"sv, "none"sv, "permutedigits"sv, "owen"sv};

        return ::enumFromStr(str, types, ERandomization::eFastOwen);
    }

    EColorSpaceType colorSpaceTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EColorSpaceType> count = toUnderlying(EColorSpaceType::eCount);
        static constexpr std::array<std::string_view, count> types{"srgb"sv, "rec2020"sv, "aces2065-1"sv, "dci-p3"sv};

        return ::enumFromStr(str, types, EColorSpaceType::eSRGB);
    }

    EFilmType filmTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EFilmType>   count = toUnderlying(EFilmType::eCount);
        static constexpr std::array<std::string_view, count> types{"rgb"sv, "gbuffer"sv, "spectral"sv};

        return ::enumFromStr(str, types, EFilmType::eRGB);
    }

    ESensor sensorFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ESensor> count = toUnderlying(ESensor::eCount);
        static constexpr std::array<std::string_view, count>
            types{"cie1931"sv,
                  "canon_eos_100d"sv,
                  "canon_eos_1dx_mkii"sv,
                  "canon_eos_200d"sv,
                  "canon_eos_200d_mkii"sv,
                  "canon_eos_5d"sv,
                  "canon_eos_5d_mkii"sv,
                  "canon_eos_5d_mkiii"sv,
                  "canon_eos_5d_mkiv"sv,
                  "canon_eos_5ds"sv,
                  "canon_eos_m"sv,
                  "hasselblad_l1d_20c"sv,
                  "nikon_d810"sv,
                  "nikon_d850"sv,
                  "sony_ilce_6400"sv,
                  "sony_ilce_7m3"sv,
                  "sony_ilce_7rm3"sv,
                  "sony_ilce_9"sv};

        return ::enumFromStr(str, types, ESensor::eCIE1931);
    }

    EGVufferCoordSys gBufferCoordSysFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EGVufferCoordSys> count = toUnderlying(EGVufferCoordSys::eCount);
        static constexpr std::array<std::string_view, count>      types{"camera"sv, "world"sv};

        return ::enumFromStr(str, types, EGVufferCoordSys::eCamera);
    }

    EFilterType filterTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EFilterType> count = toUnderlying(EFilterType::eCount);
        static constexpr std::array<std::string_view, count> types{
            "box"sv,
            "gaussian"sv,
            "mitchell"sv,
            "sinc"sv,
            "triangle"sv,
        };

        return ::enumFromStr(str, types, EFilterType::eGaussian);
    }

    float defaultRadiusFromFilterType(EFilterType e)
    {
        switch (e)
        {
            using enum EFilterType;
            case eBox:
                return 0.5f;
            case eMitchell:
                return 2.f;
            case eSinc:
                return 4.f;
            case eTriangle:
                return 2.f;
            case eGaussian:
                [[fallthrough]];
            default:
                return 1.5f;
        }
    }

    EIntegratorType integratorTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EIntegratorType> count = toUnderlying(EIntegratorType::eCount);
        static constexpr std::array<std::string_view, count>
            types{"volpath"sv,
                  "ambientocclusion"sv,
                  "bdpt"sv,
                  "lightpath"sv,
                  "mlt"sv,
                  "path"sv,
                  "randomwalk"sv,
                  "simplepath"sv,
                  "simplevolpath"sv,
                  "sppm"sv};

        return ::enumFromStr(str, types, EIntegratorType::eVolPath);
    }

    ELightSampler lightSamplerFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<ELightSampler> count = toUnderlying(ELightSampler::eCount);
        static constexpr std::array<std::string_view, count>   types{"bvh"sv, "uniform"sv, "power"sv};

        return ::enumFromStr(str, types, ELightSampler::eBVH);
    }

    EAcceletatorType acceleratorTypeFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EAcceletatorType> count = toUnderlying(EAcceletatorType::eCount);
        static constexpr std::array<std::string_view, count>      types{"bvh"sv, "kdtree"sv};

        return ::enumFromStr(str, types, EAcceletatorType::eBVH);
    }

    EBVHSplitMethod bvhSplitMethodFromStr(char const* str)
    {
        using namespace std::string_view_literals;
        static constexpr std::underlying_type_t<EBVHSplitMethod> count = toUnderlying(EBVHSplitMethod::eCount);
        static constexpr std::array<std::string_view, count>     types{"sah"sv, "middle"sv, "equal"sv, "hlbvh"sv};

        return ::enumFromStr(str, types, EBVHSplitMethod::eSAH);
    }

    // CTrie ----------------------------------------------------------------------------------------------------------

    std::atomic<uint32_t>& SNode::refCounterAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        // Compute the base address of the element.
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address.
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + sizeof(std::atomic<uint32_t>) <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        // Calculate the offset for the reference counter within the structure.
        uintptr_t refCounterAddr = alignToAddr(alignedElemAddr + valueSize, alignof(std::atomic<uint32_t>));

        return *reinterpret_cast<std::atomic<uint32_t>*>(refCounterAddr);
    }

    uint32_t SNode::incrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        return refCounterAt(index, valueSize, valueAlign).fetch_add(1, std::memory_order_seq_cst) + 1;
    }

    uint32_t SNode::decrRefCounter(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        return refCounterAt(index, valueSize, valueAlign).fetch_sub(1, std::memory_order_seq_cst) - 1;
    }

    uint32_t SNode::keyCopy(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        return keyRef(index, valueSize, valueAlign);
    }

    uint32_t& SNode::keyRef(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address.
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + sizeof(uint32_t) <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        // Calculate the offset for the key within the structure.
        uintptr_t refCounterAddr = alignToAddr(alignedElemAddr + valueSize, alignof(std::atomic<uint32_t>));
        uintptr_t keyAddr        = refCounterAddr + sizeof(uint32_t);

        return *reinterpret_cast<uint32_t*>(keyAddr);
    }

    bool SNode::tryReadLock(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        bool res = bits.casToReadLocked(index);
        if (res)
            incrRefCounter(index, valueSize, valueAlign);
        return res;
    }

    bool SNode::releaseReadLock(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        uint32_t val = decrRefCounter(index, valueSize, valueAlign);
        if (val == 0)
            return bits.setUnused(index);
        else
            return false;
    }

    uintptr_t SNode::getElementBaseAddress(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const
    {
        uintptr_t elemBaseAddr = reinterpret_cast<uintptr_t>(data.data()) + index * valueSize;

        // Ensure alignment of the base address
        uint64_t  mask            = valueAlign - 1;
        uintptr_t alignedElemAddr = (elemBaseAddr + mask) & ~mask;
        assert(elemBaseAddr == alignedElemAddr);
        assert(alignedElemAddr + valueSize <= reinterpret_cast<uintptr_t>(data.data()) + data.size());

        return alignedElemAddr;
    }

    void const* SNode::valueConstAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign) const
    {
        uintptr_t baseAddr = getElementBaseAddress(index, valueSize, valueAlign);
        return reinterpret_cast<void const*>(baseAddr);
    }

    // Access value as void*
    void* SNode::valueAt(uint32_t index, uint32_t valueSize, uint32_t valueAlign)
    {
        uintptr_t baseAddr = getElementBaseAddress(index, valueSize, valueAlign);
        return reinterpret_cast<void*>(baseAddr);
    }

    TaggedPointer CTrie::newINode(MemoryContext& mctx) const
    {
        TaggedPointer ret   = m_table.allocate(mctx, sizeof(INode), alignof(INode));
        auto*         pNode = reinterpret_cast<SNode*>(m_table.rawPtr(ret));
        std::construct_at(pNode);
        return ret;
    }

    TaggedPointer CTrie::newSNode(MemoryContext& mctx) const
    {
        TaggedPointer ret   = m_table.allocate(mctx, sizeof(INode), alignof(INode));
        auto*         pNode = reinterpret_cast<INode*>(m_table.rawPtr(ret));
        std::construct_at(pNode);
        return ret;
    }

    CTrie::CTrie(MemoryContext& mctx, AllocatorTable const& table, uint32_t valueSize, uint32_t valueAlign) :
    m_table(table),
    m_root(newSNode(mctx)),
    m_valueSize(valueSize),
    m_valueAlign(valueAlign)
    {
        if (m_root == taggedNullptr)
        {
            mctx.pctx.error("Couldn't allocate root node for CTrie");
            std::abort();
        }
    }

    inline constexpr uint32_t elemSize(uint32_t valueSize, uint32_t valueAlign)
    {
        // 2. Padding to ensure valueAlign alignment for the ref counter + key.
        uint32_t alignedOffset = (valueSize + valueAlign - 1) & ~(valueAlign - 1);

        // 3. Atomic ref counter + key (uint32_t + uint32_t = 8 bytes).
        constexpr uint32_t metadataSize = sizeof(std::atomic<uint32_t>) + sizeof(uint32_t);

        // Calculate the total element size.
        uint32_t elementSize = alignedOffset + metadataSize;
        return elementSize;
    }

    inline constexpr uint32_t posINode(uint32_t hash, uint32_t level)
    {
        // Extract the 5-bit packet corresponding to the current level.
        uint32_t shiftAmount = level * 5;
        uint32_t index       = (hash >> shiftAmount) & ((1u << 5) - 1); // Mask to extract 5 bits.
        return index;
    }

    inline constexpr uint32_t posSNode(uint32_t hash, uint32_t level, uint32_t valueSize, uint32_t valueAlign)
    {
        uint32_t elementSize = elemSize(valueSize, valueAlign);

        // Calculate the number of elements that can fit.
        constexpr uint32_t dataSize    = sizeof(SNode::Data); // SNode data size
        uint32_t           numElements = dataSize / elementSize;

        // Calculate the position.
        uint32_t shiftAmount = (level * 5); // Adjust based on 5-bit packet per level.
        uint32_t pos         = (hash >> shiftAmount) & smallestPOTMask(numElements); // Mask to extract relevant bits.

        return clamp(pos, 0u, numElements - 1u);
    }

    void CTrie::cleanupINode(MemoryContext& mctx, INode* inode, void (*dctor)(MemoryContext& mctx, void* value))
    {
        for (uint32_t i = 0; i < cardinalityINode; ++i)
        {
            while (inode->bits.checkOccupied(i))
                std::this_thread::yield();

            if (inode->bits.checkFree(i))
                continue;

            else if (inode->bits.checkINode(i))
            {
                while (!inode->bits.setOccupied(i, true))
                    std::this_thread::yield();

                TaggedPointer pt    = inode->children[i];
                INode*        child = reinterpret_cast<INode*>(m_table.rawPtr(pt));
                cleanupINode(mctx, child, dctor);
                m_table.free(mctx, pt, sizeof(INode), alignof(INode));
            }
            else if (inode->bits.checkSNode(i))
            {
                while (!inode->bits.setOccupied(i, true))
                    std::this_thread::yield();

                TaggedPointer pt    = inode->children[i];
                SNode*        snode = reinterpret_cast<SNode*>(m_table.rawPtr(pt));
                cleanupSNode(mctx, snode, dctor);
                m_table.free(mctx, pt, sizeof(SNode), alignof(SNode));
            }
            else
                assert(false);
        }
    }

    void CTrie::cleanupSNode(MemoryContext& mctx, SNode* snode, void (*dctor)(MemoryContext& mctx, void* value))
    {
        uint32_t elementSize = elemSize(m_valueSize, m_valueAlign);
        uint32_t capacity    = sizeof(SNode::Data) / elementSize;
        for (uint32_t i = 0; i < capacity; ++i)
        {
            if (snode->bits.checkFree(i))
                continue;
            while (snode->bits.checkWriteLocked(i))
                std::this_thread::yield();
            if (!snode->bits.casFreeToWriteLocked(i, SNodeBitmap::unused))
                continue;

            void* elem = snode->valueAt(i, m_valueSize, m_valueAlign);
            dctor(mctx, elem);
            snode->bits.setFree(i);
        }
    }

    void CTrie::cleanup(MemoryContext& mctx, void (*dctor)(MemoryContext& mctx, void* value))
    {
        INode* pRoot = reinterpret_cast<INode*>(m_table.rawPtr(m_root));
        cleanupINode(mctx, pRoot, dctor);
        m_table.free(mctx, m_root, sizeof(INode), alignof(INode));
    }

    bool CTrie::insert(MemoryContext& mctx, uint32_t key, void const* value)
    {
        static constexpr uint32_t dataSize    = sizeof(SNode::Data); // SNode data size
        uint32_t                  elementSize = elemSize(m_valueSize, m_valueAlign);
        uint32_t                  numElements = dataSize / elementSize;

        INode*   inode     = reinterpret_cast<INode*>(m_table.rawPtr(m_root));
        SNode*   snode     = nullptr;
        uint32_t level     = 0;
        uint32_t parentPos = 0;
        while (level < 6)
        {
            if (!snode) // current node is inode
            {
                uint32_t pos = posINode(key, level);
                if (inode->bits.checkFree(pos) && inode->bits.setOccupied(pos))
                { // allocate a new snode
                    TaggedPointer ptr = newSNode(mctx);
                    if (ptr == taggedNullptr)
                    {
                        inode->bits.setFree(pos);
                        return false;
                    }
                    inode->children[pos] = ptr;
                    inode->bits.setSNode(pos);
                }
                else
                {
                    while (inode->bits.checkOccupied(pos))
                        std::this_thread::yield();
                    if (inode->bits.checkFree(pos))
                        return false;
                }

                if (inode->bits.checkINode(pos))
                    inode = reinterpret_cast<INode*>(m_table.rawPtr(inode->children[pos]));
                else if (inode->bits.checkSNode(pos))
                    snode = reinterpret_cast<SNode*>(m_table.rawPtr(inode->children[pos]));
                else
                {
                    assert(false);
                    return false;
                }

                parentPos = pos;
            }
            else
            {
                uint32_t pos = posSNode(key, level, m_valueSize, m_valueAlign);
                while (snode->bits.checkWriteLocked(pos))
                    std::this_thread::yield();
                if (snode->bits.checkFree(pos) && snode->bits.casFreeToWriteLocked(pos))
                { // write the value inside the thing
                    std::memcpy(snode->valueAt(pos, m_valueSize, m_valueAlign), value, m_valueSize);
                    snode->keyRef(pos, m_valueSize, m_valueAlign) = key;
                    snode->refCounterAt(pos, m_valueSize, m_valueAlign).store(0, std::memory_order_seq_cst);
                    snode->bits.setUnused(pos);
                    return true;
                }
                else
                { // TODO collision!
                    if (snode->tryReadLock(pos, m_valueSize, m_valueAlign))
                    { // if the key is already there, don't insert
                        if (uint32_t keyEx = snode->keyCopy(pos, m_valueSize, m_valueAlign); keyEx == key)
                        {
                            snode->releaseReadLock(pos, m_valueSize, m_valueAlign);
                            return false;
                        }
                        else
                            snode->releaseReadLock(pos, m_valueSize, m_valueAlign);
                    }
                    else
                        return false;

                    uint64_t prevState = inode->bits.getValue(parentPos);
                    if (inode->bits.setOccupied(parentPos, true)) // Lock the parent position
                    {
                        TaggedPointer newINodePtr = newINode(mctx);
                        if (newINodePtr == taggedNullptr)
                        {
                            inode->bits.setValue(prevState, parentPos); // Restore state
                            return false;
                        }

                        INode* newINode = reinterpret_cast<INode*>(m_table.rawPtr(newINodePtr));
                        assert(newINode != inode);

                        // Reinsert all elements from the SNode into the new INode
                        for (uint32_t i = 0; i < numElements; ++i)
                        {
                            if (!snode->bits.checkFree(i))
                            {
                                while (!snode->tryReadLock(i, m_valueSize, m_valueAlign))
                                    std::this_thread::yield();

                                uint32_t existingKey   = snode->keyRef(i, m_valueSize, m_valueAlign);
                                void*    existingValue = snode->valueAt(i, m_valueSize, m_valueAlign);

                                // Perform recursive insertion at the next level
                                if (!reinsert(mctx, newINode, existingKey, existingValue, level + 1))
                                {
                                    // Cleanup on failure
                                    inode->bits.setValue(prevState, parentPos);
                                    snode->bits.setUnused(i);
                                    return false;
                                }
                                snode->releaseReadLock(i, m_valueSize, m_valueAlign);
                            }
                        }

                        // Replace SNode with the new INode in the parent
                        TaggedPointer snodePtr     = inode->children[parentPos];
                        inode->children[parentPos] = newINodePtr;
                        inode->bits.setINode(parentPos);

                        // Lock all non-free elements and deallocate the old SNode
                        for (uint32_t i = 0; i < numElements; ++i)
                        {
                            while (!snode->bits.casFreeToWriteLocked(i, SNodeBitmap::unused))
                                std::this_thread::yield();
                        }
                        m_table.free(mctx, snodePtr, sizeof(SNode), alignof(SNode));

                        // Restart the insertion process into the new INode
                        inode = newINode;
                        snode = nullptr;
                    }
                    else
                        return false;

                    --level;
                }
            }

            ++level;
        }

        return false;
    }


    bool CTrie::reinsert(MemoryContext& mctx, INode* inode, uint32_t key, void const* value, uint32_t level)
    {
        uint32_t pos = posINode(key, level);

        if (inode->bits.checkFree(pos) && inode->bits.setOccupied(pos))
        { // Allocate a new SNode
            TaggedPointer ptr = newSNode(mctx);
            if (ptr == taggedNullptr)
            {
                inode->bits.setFree(pos);
                return false;
            }
            inode->children[pos] = ptr;
            inode->bits.setSNode(pos);
        }

        if (inode->bits.checkINode(pos))
        {
            assert(false);
            return false;
        }
        else if (inode->bits.checkSNode(pos))
        {
            SNode*   childSNode = reinterpret_cast<SNode*>(m_table.rawPtr(inode->children[pos]));
            uint32_t sPos       = posSNode(key, level, m_valueSize, m_valueAlign);
            if (childSNode->bits.checkFree(sPos) && childSNode->bits.casFreeToWriteLocked(sPos))
            {
                std::memcpy(childSNode->valueAt(sPos, m_valueSize, m_valueAlign), value, m_valueSize);
                childSNode->keyRef(sPos, m_valueSize, m_valueAlign) = key;
                childSNode->refCounterAt(sPos, m_valueSize, m_valueAlign).store(0, std::memory_order_seq_cst);
                childSNode->bits.setUnused(sPos);
                return true;
            }
        }

        return false;
    }

    bool CTrie::remove(MemoryContext& mctx, uint32_t key)
    {
        static constexpr uint32_t dataSize    = sizeof(SNode::Data); // SNode data size
        uint32_t                  elementSize = elemSize(m_valueSize, m_valueAlign);
        uint32_t                  numElements = dataSize / elementSize;

        INode*   inode     = reinterpret_cast<INode*>(m_table.rawPtr(m_root));
        SNode*   snode     = nullptr;
        uint32_t level     = 0;
        uint32_t parentPos = 0;

        while (level < 6)
        {
            if (!snode) // current node is an INode
            {
                uint32_t pos = posINode(key, level);
                if (inode->bits.checkFree(pos))
                    return false;

                if (inode->bits.checkSNode(pos))
                {
                    snode     = reinterpret_cast<SNode*>(m_table.rawPtr(inode->children[pos]));
                    parentPos = pos;
                }
                else if (inode->bits.checkINode(pos))
                {
                    inode = reinterpret_cast<INode*>(m_table.rawPtr(inode->children[pos]));
                }
                else
                {
                    assert(false);
                    return false;
                }
            }
            else
            {
                uint32_t pos = posSNode(key, level, m_valueSize, m_valueAlign);
                while (snode->bits.checkReadLocked(pos) || snode->bits.checkWriteLocked(pos))
                    std::this_thread::yield();

                if (snode->bits.checkFree(pos))
                    return false; // Key not found in the SNode

                while (!snode->bits.casFreeToWriteLocked(pos, SNodeBitmap::unused))
                    std::this_thread::yield();

                uint32_t keyCopy = snode->keyCopy(pos, m_valueSize, m_valueAlign);
                if (keyCopy == key)
                {
                    snode->bits.setFree(pos);
                    return true;
                }
                else
                {
                    snode->bits.setUnused(pos);
                    return false;
                }
            }

            ++level;
        }

        return false;
    }

    // Parsing --------------------------------------------------------------------------------------------------------
    // https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/parser.cpp
    char WordParser::decodeEscaped(char c)
    {
        switch (c)
        {
            case EOF:
                // You shouldn't be here
                return EOF;
            case 'b':
                return '\b';
            case 'f':
                return '\f';
            case 'n':
                return '\n';
            case 'r':
                return '\r';
            case 't':
                return '\t';
            case '\\':
                return '\\';
            case '\'':
                return '\'';
            case '\"':
                return '\"';
            default:
                assert(false && "invalid escaped character");
                std::abort();
        }
        return 0; // NOTREACHED
    }

    char WordParser::getChar(std::string_view str, size_t idx)
    {
        if (m_needsContinuation && idx < m_bufferLength)
        {
            return m_buffer[idx];
        }
        else if (m_needsContinuation)
        {
            return str[idx - m_bufferLength];
        }
        else
        {
            return str[idx];
        }
    }

    void WordParser::copyToBuffer(std::string_view str)
    {
        assert(m_bufferLength + str.size() < 256);
        std::memcpy(m_buffer + m_bufferLength, str.data(), str.size());
        m_bufferLength += static_cast<uint32_t>(str.size());
    }

    std::string_view WordParser::catResult(std::string_view str, size_t start, size_t end)
    {
        if (m_needsContinuation)
        {
            m_needsContinuation = false;
            if (end > start + m_bufferLength) // if you read past the buffer
            {
                size_t len = end - start - m_bufferLength;
                assert(m_bufferLength + len < 256);
                m_numCharReadLastTime += len;
                std::string_view s = str.substr(0, len);
                std::memcpy(m_buffer + m_bufferLength, s.data(), s.size());
                m_bufferLength += static_cast<uint32_t>(s.size());
            }
            else // you read only the m_buffer (the whole thing, so no need to change buffer langth)
            {
                m_numCharReadLastTime = 0;
            }
        }
        else
        {
            assert(end > start);
            assert(end - start <= 256);
            size_t len = end - start;
            m_numCharReadLastTime += len;
            copyToBuffer(str.substr(start, len));
        }

        return {m_buffer, m_bufferLength};
    }

    bool WordParser::needsContinuation() const
    {
        return m_needsContinuation;
    }

    uint32_t WordParser::numCharReadLast() const
    {
        return m_numCharReadLastTime;
    }

    bool WordParser::endOfStr(std::string_view str, size_t idx) const
    {
        if (m_needsContinuation)
        {
            if (idx >= m_bufferLength)
                idx = idx - m_bufferLength;
            else
                idx = 0ULL;
        }

        return idx >= str.size();
    }

    std::string_view WordParser::nextWord(std::string_view str)
    {
        if (!m_needsContinuation)
        {
            std::memset(m_buffer, 0, sizeof(m_buffer));
            std::memset(m_escapedBuffer, 0, sizeof(m_escapedBuffer));
            m_bufferLength = 0;
        }
        m_numCharReadLastTime = 0;

        size_t i = 0;
        while (i < str.size())
        {
            size_t start = i;
            char   c     = getChar(str, i++);

            if (std::isspace(static_cast<unsigned char>(c)))
            { // nothing
                ++m_numCharReadLastTime;
            }
            else if (c == '"') // parse string, scan to closing quote
            {
                if (!m_needsContinuation)
                {
                    m_haveEscaped = false;
                }

                while (true)
                {
                    if (endOfStr(str, i))
                    { // string parsing was interrupted by the end of the chunk
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    if ((c = getChar(str, i++)) != '"')
                    {
                        if (c == '\n')
                        { // TODO error hendling
                            m_needsContinuation = false;
                            return "";
                        }
                        else if (c == '\\')
                        {
                            m_haveEscaped = true;
                        }
                    }
                    else
                    {
                        break;
                    }
                } // while not end quote

                if (!m_haveEscaped)
                {
                    return catResult(str, start, i);
                }
                else
                { // TODO decude escaped
                    m_haveEscaped   = false;
                    uint32_t escIdx = 0;
                    for (uint32_t j = start; j < i; ++j)
                    {
                        if (getChar(str, j) != '\\')
                        {
                            m_escapedBuffer[escIdx++] = str[j];
                        }
                        else
                        {
                            ++j; // go past '\\'
                            assert(j < i);
                            m_escapedBuffer[escIdx++] = decodeEscaped(str[j]);
                        }
                        m_escapedBuffer[escIdx] = '\0';
                        return catResult({m_escapedBuffer, escIdx}, start, i);
                    }
                }
            } // end parse string
            else if (c == '[' || c == ']') // parse begin/end array
            {
                m_needsContinuation = false;
                return catResult(str, start, start + 1);
            }
            else if (c == '#') // comment. Scan until EOL or EOF
            {
                while (true)
                {
                    if (endOfStr(str, i))
                    {
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    c = getChar(str, i++);
                    if (c == '\n' || c == '\r')
                    {
                        --i;
                        break;
                    }
                }

                return catResult(str, start, i);
            }
            else // regular character. go until end of word/number
            {
                while (true)
                {
                    if (endOfStr(str, i))
                    {
                        copyToBuffer(str.substr(start, i - start));
                        m_needsContinuation = true;
                        return "";
                    }

                    c = getChar(str, i++);
                    if (std::isspace(static_cast<unsigned char>(c)) || c == '"' || c == '[' || c == ']')
                    {
                        --i;
                        break;
                    }
                }

                return catResult(str, start, i);
            }
        }

        // you found only whitespaces
        m_needsContinuation = false;
        return "";
    }

} // namespace dmt

namespace dmt::model {
} // namespace dmt::model

namespace dmt::job {
    void parseSceneHeader(uintptr_t address)
    {
        using namespace dmt;
        char                  buffer[512]{};
        ParseSceneHeaderData& data = *std::bit_cast<ParseSceneHeaderData*>(address);
        AppContext&           actx = *data.actx;
        actx.log("Starting Parse Scene Header Job");
        bool error = false;

        ChunkedFileReader reader{actx.mctx.pctx, data.filePath.data(), 512};
        if (reader)
        {
            for (uint32_t chunkNum = 0; chunkNum < reader.numChunks(); ++chunkNum)
            {
                bool status = reader.requestChunk(actx.mctx.pctx, buffer, chunkNum);
                if (!status)
                {
                    error = true;
                    break;
                }

                status = reader.waitForPendingChunk(actx.mctx.pctx);
                if (!status)
                {
                    error = true;
                    break;
                }

                uint32_t         size = reader.lastNumBytesRead();
                std::string_view chunkView{buffer, size};
                actx.log("Read chunk content:\n{}\n", {chunkView});
            }
        }
        else
        {
            actx.error("Couldn't open file \"{}\"", {data.filePath});
        }

        if (error)
        {
            actx.error("Something went wrong during job execution");
        }

        actx.log("Parse Scene Header Job Finished");
        std::atomic_thread_fence(std::memory_order_release);
        std::atomic_store_explicit(&data.done, 1, std::memory_order_relaxed);
    }
} // namespace dmt::job