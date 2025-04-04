#include "middleware-parser.h"

#include <array>
#include <atomic>
#include <bit>
#include <map>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>
#include <span>

#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstring>

// TODO if this appears after the integrator has bee3n already parsed, you need to modify it somehow
// Option "bool wavefront" true
// TODO if this asppears after the sampler has been alread, you need to modify it somehow
// Option "integer seed" 15

namespace dmt {
    // values can also be closed into arrays even if something expects a scalar. parameters are unordered
    // all filenames are either absolute or relative to the starting pbrt file
    // "Import statement only allowed inside world definition block."
    namespace dict {
        using namespace std::string_view_literals;

        namespace target {
            static constexpr SText shape    = "shape"sv;
            static constexpr SText light    = "light"sv;
            static constexpr SText material = "material"sv;
            static constexpr SText medium   = "medium"sv;
            static constexpr SText texture  = "texture"sv;
        } // namespace target

        // directives (update array at the end of the namnespace if you chnage this)
        namespace directive {
            static constexpr SText Option         = "Option"sv;
            static constexpr SText Camera         = "Camera"sv;
            static constexpr SText Sampler        = "Sampler"sv;
            static constexpr SText ColorSpace     = "ColorSpace"sv;
            static constexpr SText Film           = "Film"sv;
            static constexpr SText PixelFilter    = "PixelFilter"sv;
            static constexpr SText Integrator     = "Integrator"sv;
            static constexpr SText Accelerator    = "Accelerator"sv;
            static constexpr SText WorldBegin     = "WorldBegin"sv;
            static constexpr SText AttributeBegin = "AttributeBegin"sv;
            static constexpr SText AttributeEnd   = "AttributeEnd"sv;
            static constexpr SText Include = "Include"sv; // supports another pbrt file, literal or compressed with gzip,
            static constexpr SText Import = "Import"sv; // supports another pbrt file, literal, imports only named entities. only in world block

            // transformations on the CTM (Current Transformation Matrix) (reset to Identity at world begin)
            static constexpr SText Identity  = "Identity"sv;
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
        } // namespace directive
        namespace activetransform_literals {
            static constexpr SText StartTime = "StartTime"sv;
            static constexpr SText EndTime   = "EndTime"sv;
            static constexpr SText All       = "All"sv;
        } // namespace activetransform_literals

        // data types
        namespace types {
            static constexpr SText tVector    = "vector"sv; // synonym of vector3
            static constexpr SText tPoint     = "point"sv;  // synonym of point3
            static constexpr SText tNormal    = "normal"sv;
            static constexpr SText tBool      = "bool"sv;
            static constexpr SText tFloat     = "float"sv;
            static constexpr SText tInteger   = "integer"sv;
            static constexpr SText tString    = "string"sv;
            static constexpr SText tRGB       = "rgb"sv;
            static constexpr SText tPoint2    = "point2"sv;
            static constexpr SText tPoint3    = "point3"sv;
            static constexpr SText tNormal3   = "normal3"sv;
            static constexpr SText tBlackbody = "blackbody"sv;
            static constexpr SText tSpectrum  = "spectrum"sv;
            static constexpr SText tVector2   = "vector2"sv;
            static constexpr SText tVector3   = "vector3"sv;
            static constexpr SText tTexture   = "texture"sv;
        } // namespace types

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
            namespace curve_params {
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
            } // namespace curve_params
            namespace cylinder_params {
                static constexpr SText radius = "radius"sv; // float 1
                static constexpr SText zmin   = "zmin"sv;   // float -1
                static constexpr SText zmax   = "zmax"sv;   // float 1
                static constexpr SText phimax = "phimax"sv; // float 360
            } // namespace cylinder_params
            namespace disk_params {
                static constexpr SText height      = "height"sv;      // float 0
                static constexpr SText radius      = "radius"sv;      // radius 1
                static constexpr SText innerradius = "innerradius"sv; // float 0
                static constexpr SText phimax      = "phimax"sv;      // float 360
            } // namespace disk_params
            namespace sphere_params {
                static constexpr SText radius = "radius"sv; // float 1
                static constexpr SText zmin   = "zmin"sv;   // float -radius
                static constexpr SText zmax   = "zmax"sv;   // float radius
                static constexpr SText phimax = "phimax"sv; // float 360
            } // namespace sphere_params
            namespace trianglemesh_params {
                // P N S uv must be same size. only P is required
                static constexpr SText indices = "indices"sv; // required unless there are only 3 vertices, integer[]
                static constexpr SText P       = "P"sv;       // point3[]
                static constexpr SText N  = "N"sv;  // normal[], if present, shading normals are computed using these
                static constexpr SText S  = "S"sv;  // vector3[], per-vertex tangents
                static constexpr SText uv = "uv"sv; // point2[]
            } // namespace trianglemesh_params
            namespace plymesh_params {
                static constexpr SText filename = "filename"sv; // relative path of .ply or .ply.gz (gzip compressed)
                static constexpr SText displacement = "displacement"sv; // displacement texture
                static constexpr SText edgelength = "edgelength"sv; // edges of a triangle are split until this is met, def. 1.f
            } // namespace plymesh_params
            namespace loopsubdiv_params {                     // mesh which is subdivided
                static constexpr SText levels  = "levels"sv;  // integer 3
                static constexpr SText indices = "indices"sv; // integer[]
                static constexpr SText P       = "P"sv;       // point[]
            } // namespace loopsubdiv_params
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
            namespace distant_params {
                static constexpr SText L = "L"sv; // spectrum, spectral radiance, default = current color space illuminant
                static constexpr SText from = "from"sv; // point, 0 0 0
                static constexpr SText to   = "to"sv;   // point, 0 0 1
            } // namespace distant_params
            namespace goniometric_params {
                static constexpr SText filename = "filename"sv; // string, no default, required
                static constexpr SText I        = "I"sv;        // current color space's illuminant
            } // namespace goniometric_params
            namespace infinite_params {
                // either filename or L
                static constexpr SText filename = "filename"sv;
                static constexpr SText portal   = "portal"sv; // point3[4], window through which the light is visible
                static constexpr SText L        = "L"sv;      // radiance intensity = L * scale * power
            } // namespace infinite_params
            namespace point_params {
                static constexpr SText I = "I"sv; // spectrum, default = current color space illuminant. spectrad dist of light emitted radiant intensity
                static constexpr SText from = "from"sv; // point, 0 0 0, light location
            } // namespace point_params
            namespace projection_params {
                static constexpr SText I        = "I"sv;        // spectrum
                static constexpr SText fov      = "fov"sv;      // float, 90
                static constexpr SText filename = "filename"sv; // string, required
            } // namespace projection_params
            namespace spotlight_params {
                static constexpr SText I              = "I"sv;              // spectrum, spectral intensity
                static constexpr SText from           = "from"sv;           // point, 0 0 0
                static constexpr SText to             = "to"sv;             // point, 0 0 1
                static constexpr SText coneangle      = "coneangle"sv;      // float, 30
                static constexpr SText conedeltaangle = "conedeltaangle"sv; // float, 5
            } // namespace spotlight_params
        } // namespace light

        // describing the scene: area light
        namespace arealight {
            // arealight types
            static constexpr SText diffuse = "diffuse"sv;
            // diffuse
            namespace diffuse_params {
                static constexpr SText filename = "filename"sv; // string required no default
                static constexpr SText L        = "L"sv;        // spectrum, emitted spectral radiance distribution
                static constexpr SText twosided = "twosided"sv; // bool, false = emit light only in halfspace pointed by normal of shape
            } // namespace diffuse_params
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
            namespace coated_params { // coateddiffuse and coatedconductor
                static constexpr SText albedo = "albedo"sv; // spectrum texture, scattering albedo between interface and diffuse layers. in [0, 1]
                static constexpr SText g         = "g"sv;         // float texture, in [-1, 1]
                static constexpr SText maxdepth  = "maxdepth"sv;  // integer 10
                static constexpr SText nsamples  = "nsamples"sv;  // integer 1
                static constexpr SText thickness = "thickness"sv; // float 0.01f
                namespace diffuse_params {
                    static constexpr SText reflectance = "reflectance"sv; // spectrum texture, default 0.5
                }
                namespace conductor_params {
                    static constexpr SText conductor_eta = "conductor.eta"sv; // spectrum
                    static constexpr SText conductor_k   = "conductor.k"sv;   // spectrum
                    static constexpr SText reflectance   = "reflectance"sv;   // spectrum (NOT texture)
                } // namespace conductor_params
            } // namespace coated_params
            namespace conductor_params {
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
            } // namespace conductor_params
            namespace dielectric_params {
                static constexpr SText eta = "eta"sv; // float texture or spectrum texture. default float texture 1.5 constant
            }
            namespace diffuse_params {
                static constexpr SText reflectance = "reflectance"sv; // spectrum texture, def 0.5
            }
            namespace diffusetransmission_params {
                static constexpr SText reflectance   = "reflectance"sv;   // spectrum texture, def 0.25
                static constexpr SText transmittance = "transmittance"sv; // spectrum texture, def 0.25
                static constexpr SText scale         = "scale"sv;         // float texture, def 1
            } // namespace diffusetransmission_params
            namespace hair_params {
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
                static constexpr SText alpha  = "alpha"sv;  // float texture, def 2 degrees
            } // namespace hair_params
            namespace measured_params {
                static constexpr SText filename = "filename"sv; // string filename
            }
            namespace mix_params {
                static constexpr SText materials = "materials"sv; // string[2], material names
                static constexpr SText amount    = "amount"sv;    // texture float, def 0.5,
            } // namespace mix_params
            namespace subsurface_params {
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
            } // namespace subsurface_params
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

            namespace bilerp_params {
                static constexpr SText v00 = "v00"sv; // spectrum texture or float texture. def = float, 0
                static constexpr SText v01 = "v01"sv; // spectrum texture or float texture. def = float, 1
                static constexpr SText v10 = "v10"sv; // spectrum texture or float texture. def = float, 0
                static constexpr SText v11 = "v11"sv; // spectrum texture or float texture. def = float, 1
            } // namespace bilerp_params
            namespace checkerboard_params {
                static constexpr SText dimension = "dimension"sv; // integer, def = 2 (can be either 2 or 3)
                static constexpr SText tex1      = "tex1"sv;      // spectrum texture or float texture, def = float 1
                static constexpr SText tex2      = "tex2"sv;      // spectrum texture or float texture, def = float 0
            } // namespace checkerboard_params
            namespace constant_params {
                static constexpr SText value = "value"sv; // nbuckets values or rgb if spectrum, 1 value if float
            }
            namespace directionmix_params {
                static constexpr SText tex1 = "tex1"sv; // spectrum texture or float texture, def = float 0
                static constexpr SText tex2 = "tex2"sv; // spectrum texture or float texture, def = float 1
                static constexpr SText dir  = "dir"sv;  // vector, def = 0 1 0
            } // namespace directionmix_params
            namespace dots_params {
                static constexpr SText inside  = "inside"sv;  // spectrum texture or float texture, def = float 1
                static constexpr SText outside = "outside"sv; // spectrum texture or float texture, def = float 0
            } // namespace dots_params
            namespace perlin_params {                             // fbm, wrinkled, windy
                static constexpr SText octaves   = "octaves"sv;   // integer, def = 8
                static constexpr SText roughness = "roughness"sv; // float, def = 0.5
            } // namespace perlin_params
            namespace imagemap_params {
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
            } // namespace imagemap_params
            namespace marble_params {                             // still perlin
                static constexpr SText octaves   = "octaves"sv;   // integer, def = 8
                static constexpr SText roughness = "roughness"sv; // float, def = 0.5
                static constexpr SText scale     = "scale"sv;     // float, def = 1, scaling factor for inpouts
                static constexpr SText variation = "variation"sv; // float, def = 0.2, scaling factor for output
            } // namespace marble_params
            namespace mix_params {
                static constexpr SText tex1   = "tex1"sv;   // spectrum texture or float texture, def = float, 0
                static constexpr SText tex2   = "tex2"sv;   // spectrum texture or float texture, def = float, 1
                static constexpr SText amount = "amount"sv; // float texture, def = float 0.5
            } // namespace mix_params
            namespace ptex_params {
                static constexpr SText encoding = "encoding"sv; // enum, def = gamma 2.2
                static constexpr SText filename = "filename"sv; // stringfilename, end with ptex, required no def
                namespace filename_extensions {
                    static constexpr SText _ptex = ".ptex"sv;
                }
                static constexpr SText scale = "scale"sv; // float, def = 1
            } // namespace ptex_params
            namespace scale_params {
                static constexpr SText tex   = "tex"sv;   // spectrum texture or float texture to be scaled, def float 1
                static constexpr SText scale = "scale"sv; // float texture, def float 1
            } // namespace scale_params
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
            namespace homogeneous_params {
                static constexpr SText g       = "g"sv;       // Henyey Greenstein asymmetry, float, def 0, [-1,1]
                static constexpr SText Le      = "Le"sv;      // spectrum, def 0, distribution of emitted radiance
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText preset = "preset"sv; // TODO see https://github.com/mmp/pbrt-v4/blob/cdccb71cb1e153b63e538f624efcc13ab0f9bda2/src/pbrt/media.cpp#L79
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
            } // namespace homogeneous_params
            namespace uniformgrid_params { // generalization of homogeneous, so it takes all its parameters plus the following
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
            } // namespace uniformgrid_params
            namespace rgbgrid_params { // alternative to uniformgrid, so takes all parameters of homogeneous EXCEPT preset
                static constexpr SText g       = "g"sv;       // Henyey Greenstein asymmetry, float, def 0, [-1,1]
                static constexpr SText Le      = "Le"sv;      // spectrum, def 0, distribution of emitted radiance
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
                static constexpr SText p0 = "p0"sv; // point3, def 0 0 0, min bound of the density grid in medium space
                static constexpr SText p1 = "p1"sv; // point3, def 1 1 1, max bound of the density grid in medium space
            } // namespace rgbgrid_params
            namespace cloud_params {                // perlin
                static constexpr SText p0 = "p0"sv; // point3, def 0 0 0, min bound of the density grid in medium space
                static constexpr SText p1 = "p1"sv; // point3, def 1 1 1, max bound of the density grid in medium space
                static constexpr SText density   = "density"sv;   // float, def = 1
                static constexpr SText frequency = "frequency"sv; // float, def = 5
                static constexpr SText g         = "g"sv; // Henyey Greenstein asymmetry parameter, float, def 0, [-1,1]
                static constexpr SText sigma_a   = "sigma_a"sv;  // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s   = "sigma_s"sv;  // spectrum, scattering cross section, def 1
                static constexpr SText wispness  = "wispness"sv; // float, def 1
            } // namespace cloud_params
            namespace nanovdb_params {
                static constexpr SText g       = "g"sv; // Henyey Greenstein asymmetry parameter, float, def 0, [-1,1]
                static constexpr SText sigma_a = "sigma_a"sv; // spectrum, absorption cross section, def 1
                static constexpr SText sigma_s = "sigma_s"sv; // spectrum, scattering cross section, def 1
                static constexpr SText scale   = "scale"sv;   // float, scale factor of sigma_a and sigma_s, def = 1
                static constexpr SText Lescale = "Lescale"sv; // float, def = 1
                static constexpr SText temperatureoffset = "temperatureoffset"sv; // float, def = 0
                static constexpr SText temperaturescale  = "temperaturescale"sv;  // float, def = 1
                static constexpr SText filename          = "filename"sv;          // string
            } // namespace nanovdb_params
        } // namespace media
    } // namespace dict

    static constexpr std::array<sid_t, 2> pointTypes{dict::types::tPoint.sid, dict::types::tPoint3.sid};

    static constexpr bool activeTransformFromSid(sid_t type, EActiveTransform& out)
    {
        switch (type)
        {
            case dict::activetransform_literals::StartTime.sid: out = EActiveTransform::eStartTime; break;
            case dict::activetransform_literals::EndTime.sid: out = EActiveTransform::eEndTime; break;
            case dict::activetransform_literals::All.sid: out = EActiveTransform::eAll; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool materialTypeFromSid(sid_t type, EMaterialType& out)
    {
        switch (type)
        {
            case dict::material::coateddiffuse.sid: out = EMaterialType::eCoateddiffuse; break;
            case dict::material::coatedconductor.sid: out = EMaterialType::eCoatedconductor; break;
            case dict::material::conductor.sid: out = EMaterialType::eConductor; break;
            case dict::material::dielectric.sid: out = EMaterialType::eDielectric; break;
            case dict::material::diffuse.sid: out = EMaterialType::eDiffuse; break;
            case dict::material::diffusetransmission.sid: out = EMaterialType::eDiffusetransmission; break;
            case dict::material::hair.sid: out = EMaterialType::eHair; break;
            case dict::material::interface.sid: out = EMaterialType::eInterface; break;
            case dict::material::measured.sid: out = EMaterialType::eMeasured; break;
            case dict::material::mix.sid: out = EMaterialType::eMix; break;
            case dict::material::subsurface.sid: out = EMaterialType::eSubsurface; break;
            case dict::material::thindielectric.sid: out = EMaterialType::eThindielectric; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool textureTypeFromSid(sid_t type, ETextureType& out)
    {
        switch (type)
        {
            case dict::types::tSpectrum.sid: out = ETextureType::eSpectrum; break;
            case dict::types::tFloat.sid: out = ETextureType::eFloat; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool lightTypeFromSid(sid_t type, ELightType& out)
    {
        switch (type)
        {
            case dict::light::distant.sid: out = ELightType::eDistant; break;
            case dict::light::goniometric.sid: out = ELightType::eGoniometric; break;
            case dict::light::infinite.sid: out = ELightType::eInfinite; break;
            case dict::light::point.sid: out = ELightType::ePoint; break;
            case dict::light::projection.sid: out = ELightType::eProjection; break;
            case dict::light::spot.sid: out = ELightType::eSpot; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool areaLightTypeFromSid(sid_t type, EAreaLightType& out)
    {
        switch (type)
        {
            case dict::arealight::diffuse.sid: out = EAreaLightType::eDiffuse; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool shapeTypeFromSid(sid_t type, EShapeType& out)
    {
        switch (type)
        {
            case dict::shape::bilinearmesh.sid: out = EShapeType::eBilinearmesh; break;
            case dict::shape::curve.sid: out = EShapeType::eCurve; break;
            case dict::shape::cylinder.sid: out = EShapeType::eCylinder; break;
            case dict::shape::disk.sid: out = EShapeType::eDisk; break;
            case dict::shape::sphere.sid: out = EShapeType::eSphere; break;
            case dict::shape::trianglemesh.sid: out = EShapeType::eTrianglemesh; break;
            case dict::shape::loopsubdiv.sid: out = EShapeType::eLoopsubdiv; break;
            case dict::shape::plymesh.sid: out = EShapeType::ePlymesh; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool textureClassFromSid(sid_t type, ETextureClass& out)
    {
        switch (type)
        {
            case dict::texture::bilerp.sid: out = ETextureClass::eBilerp; break;
            case dict::texture::checkerboard.sid: out = ETextureClass::eCheckerboard; break;
            case dict::texture::constant.sid: out = ETextureClass::eConstant; break;
            case dict::texture::directionmix.sid: out = ETextureClass::eDirectionmix; break;
            case dict::texture::dots.sid: out = ETextureClass::eDots; break;
            case dict::texture::fbm.sid: out = ETextureClass::eFbm; break;
            case dict::texture::imagemap.sid: out = ETextureClass::eImagemap; break;
            case dict::texture::marble.sid: out = ETextureClass::eMarble; break;
            case dict::texture::mix.sid: out = ETextureClass::eMix; break;
            case dict::texture::ptex.sid: out = ETextureClass::ePtex; break;
            case dict::texture::scale.sid: out = ETextureClass::eScale; break;
            case dict::texture::windy.sid: out = ETextureClass::eWindy; break;
            case dict::texture::wrinkled.sid: out = ETextureClass::eWrinkled; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool renderCoordSysFromSid(sid_t type, ERenderCoordSys& out)
    {
        switch (type)
        {
            case dict::opts::rendercoordsys_literals::cameraworld.sid: out = ERenderCoordSys::eCameraWorld; break;
            case dict::opts::rendercoordsys_literals::camera.sid: out = ERenderCoordSys::eCamera; break;
            case dict::opts::rendercoordsys_literals::world.sid: out = ERenderCoordSys::eWorld; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool cameraTypeFromSid(sid_t type, ECameraType& out)
    {
        switch (type)
        {
            case dict::camera::orthographic.sid: out = ECameraType::eOrthographic; break;
            case dict::camera::perspective.sid: out = ECameraType::ePerspective; break;
            case dict::camera::realistic.sid: out = ECameraType::eRealistic; break;
            case dict::camera::spherical.sid: out = ECameraType::eSpherical; break;
            default: return false;
        }
        return true;
    }

    static constexpr sid_t cameraSidFromType(ECameraType in)
    {
        switch (in)
        {
            case ECameraType::eOrthographic: return dict::camera::orthographic.sid;
            case ECameraType::ePerspective: return dict::camera::perspective.sid;
            case ECameraType::eRealistic: return dict::camera::realistic.sid;
            case ECameraType::eSpherical: return dict::camera::spherical.sid;
        }
        [[unreachable]];
        return 0;
    }

    static constexpr bool sphericalMappingFromSid(sid_t type, ESphericalMapping& out)
    {
        switch (type)
        {
            case dict::camera::mapping_literals::equalarea.sid: out = ESphericalMapping::eEqualArea; break;
            case dict::camera::mapping_literals::equirectangular.sid: out = ESphericalMapping::eEquirectangular; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool samplerTypeFromSid(sid_t type, ESamplerType& out)
    {
        switch (type)
        {
            case dict::sampler::zsobol.sid: out = ESamplerType::eZSobol; break;
            case dict::sampler::halton.sid: out = ESamplerType::eHalton; break;
            case dict::sampler::independent.sid: out = ESamplerType::eIndependent; break;
            case dict::sampler::paddedsobol.sid: out = ESamplerType::ePaddedSobol; break;
            case dict::sampler::sobol.sid: out = ESamplerType::eSobol; break;
            case dict::sampler::stratified.sid: out = ESamplerType::eStratified; break;
            default: return false;
        }

        return true;
    }

    static constexpr bool randomizationFromSid(sid_t type, ERandomization& out)
    {
        switch (type)
        {
            case dict::sampler::randomization_literals::fastowen.sid: out = ERandomization::eFastOwen; break;
            case dict::sampler::randomization_literals::none.sid: out = ERandomization::eNone; break;
            case dict::sampler::randomization_literals::permutedigits.sid: out = ERandomization::ePermuteDigits; break;
            case dict::sampler::randomization_literals::owen.sid: out = ERandomization::eOwen; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool colorSpaceTypeFromSid(sid_t type, EColorSpaceType& out)
    {
        switch (type)
        {
            case dict::colorspace::srgb.sid: out = EColorSpaceType::eSRGB; break;
            case dict::colorspace::rec2020.sid: out = EColorSpaceType::eRec2020; break;
            case dict::colorspace::aces2065_1.sid: out = EColorSpaceType::eAces2065_1; break;
            case dict::colorspace::dci_p3.sid: out = EColorSpaceType::eDci_p3; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool filmTypeFromSid(sid_t type, EFilmType& out)
    {
        switch (type)
        {
            case dict::film::rgb.sid: out = EFilmType::eRGB; break;
            case dict::film::gbuffer.sid: out = EFilmType::eGBuffer; break;
            case dict::film::spectral.sid: out = EFilmType::eSpectral; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool sensorFromSid(sid_t type, ESensor& out)
    {
        switch (type)
        {
            case dict::film::sensor_literals::cie1931.sid: out = ESensor::eCIE1931; break;
            case dict::film::sensor_literals::canon_eos_100d.sid: out = ESensor::eCanon_eos_100d; break;
            case dict::film::sensor_literals::canon_eos_1dx_mkii.sid: out = ESensor::eCanon_eos_1dx_mkii; break;
            case dict::film::sensor_literals::canon_eos_200d.sid: out = ESensor::eCanon_eos_200d; break;
            case dict::film::sensor_literals::canon_eos_200d_mkii.sid: out = ESensor::eCanon_eos_200d_mkii; break;
            case dict::film::sensor_literals::canon_eos_5d.sid: out = ESensor::eCanon_eos_5d; break;
            case dict::film::sensor_literals::canon_eos_5d_mkii.sid: out = ESensor::eCanon_eos_5d_mkii; break;
            case dict::film::sensor_literals::canon_eos_5d_mkiii.sid: out = ESensor::eCanon_eos_5d_mkiii; break;
            case dict::film::sensor_literals::canon_eos_5d_mkiv.sid: out = ESensor::eCanon_eos_5d_mkiv; break;
            case dict::film::sensor_literals::canon_eos_5ds.sid: out = ESensor::eCanon_eos_5ds; break;
            case dict::film::sensor_literals::canon_eos_m.sid: out = ESensor::eCanon_eos_m; break;
            case dict::film::sensor_literals::hasselblad_l1d_20c.sid: out = ESensor::eHasselblad_l1d_20c; break;
            case dict::film::sensor_literals::nikon_d810.sid: out = ESensor::eNikon_d810; break;
            case dict::film::sensor_literals::nikon_d850.sid: out = ESensor::eNikon_d850; break;
            case dict::film::sensor_literals::sony_ilce_6400.sid: out = ESensor::eSony_ilce_6400; break;
            case dict::film::sensor_literals::sony_ilce_7m3.sid: out = ESensor::eSony_ilce_7m3; break;
            case dict::film::sensor_literals::sony_ilce_7rm3.sid: out = ESensor::eSony_ilce_7rm3; break;
            case dict::film::sensor_literals::sony_ilce_9.sid: out = ESensor::eSony_ilce_9; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool targetFromSid(sid_t type, ETarget& out)
    {
        switch (type)
        {
            case dict::target::shape.sid: out = ETarget::eShape; break;
            case dict::target::light.sid: out = ETarget::eLight; break;
            case dict::target::material.sid: out = ETarget::eMaterial; break;
            case dict::target::medium.sid: out = ETarget::eMedium; break;
            case dict::target::texture.sid: out = ETarget::eTexture; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool gBufferCoordSysFromSid(sid_t type, EGVufferCoordSys& out)
    {
        switch (type)
        {
            case dict::film::coordinatesystem_literals::camera.sid: out = EGVufferCoordSys::eCamera; break;
            case dict::film::coordinatesystem_literals::world.sid: out = EGVufferCoordSys::eWorld; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool filterTypeFromSid(sid_t type, EFilterType& out)
    {
        switch (type)
        {
            case dict::filter::box.sid: out = EFilterType::eBox; break;
            case dict::filter::gaussian.sid: out = EFilterType::eGaussian; break;
            case dict::filter::mitchell.sid: out = EFilterType::eMitchell; break;
            case dict::filter::sinc.sid: out = EFilterType::eSinc; break;
            case dict::filter::triangle.sid: out = EFilterType::eTriangle; break;
            default: return false;
        }
        return true;
    }

    float defaultRadiusFromFilterType(EFilterType e)
    {
        switch (e)
        {
            using enum EFilterType;
            case eBox: return 0.5f;
            case eMitchell: return 2.f;
            case eSinc: return 4.f;
            case eTriangle: return 2.f;
            case eGaussian: [[fallthrough]];
            default: return 1.5f;
        }
    }

    static constexpr bool integratorTypeFromSid(sid_t type, EIntegratorType& out)
    {
        switch (type)
        {
            case dict::integrator::volpath.sid: out = EIntegratorType::eVolPath; break;
            case dict::integrator::ambientocclusion.sid: out = EIntegratorType::eAmbientOcclusion; break;
            case dict::integrator::bdpt.sid: out = EIntegratorType::eBdpt; break;
            case dict::integrator::lightpath.sid: out = EIntegratorType::eLightPath; break;
            case dict::integrator::mlt.sid: out = EIntegratorType::eMLT; break;
            case dict::integrator::path.sid: out = EIntegratorType::ePath; break;
            case dict::integrator::randomwalk.sid: out = EIntegratorType::eRandomWalk; break;
            case dict::integrator::simplepath.sid: out = EIntegratorType::eSimplePath; break;
            case dict::integrator::simplevolpath.sid: out = EIntegratorType::eSimpleVolPath; break;
            case dict::integrator::sppm.sid: out = EIntegratorType::eSPPM; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool lightSamplerFromSid(sid_t type, ELightSampler& out)
    {
        switch (type)
        {
            case dict::integrator::lightsampler_literals::bvh.sid: out = ELightSampler::eBVH; break;
            case dict::integrator::lightsampler_literals::uniform.sid: out = ELightSampler::eUniform; break;
            case dict::integrator::lightsampler_literals::power.sid: out = ELightSampler::ePower; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool acceleratorTypeFromSid(sid_t type, EAcceletatorType& out)
    {
        switch (type)
        {
            case dict::accelerator::bvh.sid: out = EAcceletatorType::eBVH; break;
            case dict::accelerator::kdtree.sid: out = EAcceletatorType::eKdTree; break;
            default: return false;
        }
        return true;
    }

    static constexpr bool bvhSplitMethodFromSid(sid_t type, EBVHSplitMethod& out)
    {
        switch (type)
        {
            case dict::accelerator::splitmethod_literals::sah.sid: out = EBVHSplitMethod::eSAH; break;
            case dict::accelerator::splitmethod_literals::middle.sid: out = EBVHSplitMethod::eMiddle; break;
            case dict::accelerator::splitmethod_literals::equal.sid: out = EBVHSplitMethod::eEqual; break;
            case dict::accelerator::splitmethod_literals::hlbvh.sid: out = EBVHSplitMethod::eHLBVH; break;
            default: return false;
        }
        return true;
    }

    static constexpr sid_t lightSidFromType(ELightType type)
    {
        using enum ELightType;
        switch (type)
        {
            case eDistant: return dict::light::distant.sid;
            case eGoniometric: return dict::light::goniometric.sid;
            case eInfinite: return dict::light::infinite.sid;
            case ePoint: return dict::light::point.sid;
            case eProjection: return dict::light::projection.sid;
            case eSpot: return dict::light::spot.sid;
        }
        return 0;
    }

    static constexpr sid_t materialSidFromType(EMaterialType type)
    {
        using enum EMaterialType;
        switch (type)
        {
            case eCoateddiffuse: return dict::material::coateddiffuse.sid;
            case eCoatedconductor: return dict::material::coatedconductor.sid;
            case eConductor: return dict::material::conductor.sid;
            case eDielectric: return dict::material::dielectric.sid;
            case eDiffuse: return dict::material::diffuse.sid;
            case eDiffusetransmission: return dict::material::diffusetransmission.sid;
            case eHair: return dict::material::hair.sid;
            case eInterface: return dict::material::interface.sid;
            case eMeasured: return dict::material::measured.sid;
            case eMix: return dict::material::mix.sid;
            case eSubsurface: return dict::material::subsurface.sid;
            case eThindielectric: return dict::material::thindielectric.sid;
        }
        return 0;
    }

    // Parse and set a float value
    static bool parseAndSetFloat(ParamMap const& params, sid_t paramSid, float& target, float defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tFloat.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }
            float value = defaultValue;
            if (!parseFloat(values.valueAt(0), value))
            {
                return false; // Parsing failed
            }
            target = value;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    // Parse and set a std::string_view value
    static bool parseAndSetString(ParamMap const& params, sid_t paramSid, std::string& target, std::string_view defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tString.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }
            target = values.valueAt(0);
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    static bool parseAndSetString(ParamMap const& params, sid_t paramSid, char* target, uint32_t& outLength, std::string_view defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tString.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }
            std::memcpy(target, values.valueAt(0).data(), values.valueAt(0).size());
            outLength = static_cast<uint32_t>(values.valueAt(0).size());
        }
        else
        {
            std::memcpy(target, defaultValue.data(), defaultValue.size()); // Use default value if parameter is not found
            outLength = static_cast<uint32_t>(defaultValue.size());
        }
        return true;
    }

    template <typename CharT, typename Traits, typename Allocator>
    static bool parseAndSetString2(ParamMap const&                              params,
                                   sid_t                                        paramSid,
                                   std::basic_string<CharT, Traits, Allocator>& target,
                                   std::basic_string_view<CharT, Traits>        defaultValue,
                                   bool                                         required = false)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tString.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }

            // Extract the value and set it into the target string
            std::basic_string_view<CharT, Traits> const valueView{reinterpret_cast<CharT const*>(values.valueAt(0).data()),
                                                                  values.valueAt(0).size() / sizeof(CharT)};

            target.assign(valueView);
            return true;
        }
        else if (!required)
        {
            // Assign the default value to the target
            target.assign(defaultValue);
            return true;
        }
        else
            return false;
    }

    // Parse and set a boolean value
    static bool parseAndSetBool(ParamMap const& params, sid_t paramSid, bool& target, bool defaultValue)
    {
        using namespace std::string_view_literals;
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tBool.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }
            if (values.valueAt(0) == "true"sv)
                target = true;
            else if (values.valueAt(0) == "false"sv)
                target = false;
            else
                return false;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    bool parseAndSetFloat4(ParamMap const&             params,
                           sid_t                       paramSid,
                           std::array<float, 4>&       target,
                           std::array<float, 4> const& defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tFloat.sid || values.numParams() != 4)
            {
                return false; // Invalid type or incorrect number of parameters
            }

            std::array<float, 4> parsedValues = defaultValue;
            for (size_t i = 0; i < 4; ++i)
            {
                if (!parseFloat(values.valueAt(i), parsedValues[i]))
                {
                    return false; // Parsing failed for an element
                }
            }

            target = parsedValues;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    // Parse and set an int32_t value
    template <std::integral I>
    static bool parseAndSetInt(ParamMap const& params, sid_t paramSid, I& target, I defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tInteger.sid || values.numParams() != 1)
                return false; // Invalid type or number of parameters
            if (!parseInt(values.valueAt(0), target))
                return false;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    bool parseAndSetInt4(ParamMap const&               params,
                         sid_t                         paramSid,
                         std::array<int32_t, 4>&       target,
                         std::array<int32_t, 4> const& defaultValue)
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tInteger.sid || values.numParams() != 4)
                return false; // Invalid type or incorrect number of parameters

            std::array<int32_t, 4> parsedValues = defaultValue;
            for (size_t i = 0; i < 4; ++i)
            {
                if (!parseInt(values.valueAt(i), parsedValues[i]))
                    return false; // Parsing failed for an element
            }

            target = parsedValues;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    // Parse and set an enum value using a conversion function
    template <typename EnumType>
    static bool parseAndSetEnum(ParamMap const& params,
                                sid_t           paramSid,
                                EnumType&       target,
                                EnumType        defaultValue,
                                bool (*converter)(sid_t, EnumType&))
    {
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != dict::types::tString.sid || values.numParams() != 1)
            {
                return false; // Invalid type or number of parameters
            }
            if (!converter(hashCRC64(values.valueAt(0)), target))
                return false;
        }
        else
        {
            target = defaultValue; // Use default value if parameter is not found
        }
        return true;
    }

    static size_t parseAndSetFloatArray(
        ParamMap const&           params,
        sid_t                     paramSid,
        size_t                    targetSz,
        float* DMT_RESTRICT       target,
        float const* DMT_RESTRICT defaultVals,
        sid_t                     ttype = dict::types::tFloat.sid)
    {
        size_t ret = 0;
        if (auto it = params.find(paramSid); it != params.end())
        {
            ParamPair const& values = it->second;
            if (values.type != ttype || values.numParams() != targetSz)
                return 0;

            for (/**/; ret < values.numParams(); ++ret)
            {
                if (!parseFloat(values.valueAt(ret), target[ret]))
                    return ret;
            }
        }
        else
        {
            std::memcpy(target, defaultVals, targetSz * sizeof(float));
            ret = targetSz;
        }

        return ret;
    }

    template <std::size_t NTTPSize>
    static std::size_t parseFloatArray(ArgsDArray const& args, std::array<float, NTTPSize>& outArray)
    {
        // Check if the size of the input vector matches the expected size
        if (args.size() != NTTPSize)
        {
            return 0; // Mismatch in expected and actual size
        }

        std::size_t parsedCount = 0;

        for (std::size_t i = 0; i < NTTPSize; ++i)
        {
            // Parse each string into a float
            float value = 0.0f;
            if (!parseFloat(std::string_view(args[i]), value))
            {
                return parsedCount; // Stop and return the count parsed so far if there's an error
            }
            outArray[i] = value;
            ++parsedCount;
        }

        return parsedCount; // Should be NTTPSize upon success
    }

    template <Vector V>
    static bool parseAndSetVector(ParamMap const& _params, sid_t sid, V& target, V defaultV, std::span<sid_t const> allowedTypes)
    {
        if (auto it = _params.find(sid); it != _params.end())
        {
            ParamPair const& values = it->second;

            if (std::find(allowedTypes.begin(), allowedTypes.end(), values.type) == allowedTypes.end())
                return false;

            for (uint32_t i = 0; i < V::numComponents(); ++i)
            {
                float value = 0.f;
                if (!parseFloat(values.valueAt(i), value))
                    return false;
                target[i] = value;
            }
            return true;
        }
        else
        {
            target = defaultV;
            return true;
        }
    }

    // Entity Definitions ---------------------------------------------------------------------------------------------

    CameraSceneEntity::CameraSceneEntity(CameraSpec parameters, CameraTransform const& cameraTransform, sid_t medium) :
    SceneEntity(cameraSidFromType(parameters.type), {/*we have no unparsed parameters at the moment*/}),
    spec(parameters),
    cameraTransform(cameraTransform),
    medium(medium)
    {
    }

    TransformedSceneEntity::TransformedSceneEntity(sid_t name, AnimatedTransform const& t, ParamMap parameters) :
    SceneEntity(name, parameters),
    transform(t)
    {
    }

    LightEntity::LightEntity(ELightType               type,
                             AnimatedTransform const& t,
                             sid_t                    _medium,
                             EColorSpaceType          _colorSpace,
                             ParamMap const&          _params) :
    TransformedSceneEntity(lightSidFromType(type), t, _params),
    spec(type, _params.contains(dict::light::illuminance.sid), 0.f /*see insdie*/, 1.f /*see inside*/),
    medium(_medium),
    colorSpace(_colorSpace)
    { // TODO Error handling
        using namespace std::string_view_literals;
        if (spec.illum)
            parseAndSetFloat(parameters, dict::light::illuminance.sid, spec.po.illuminance, /*never*/ 1.f);
        else
            parseAndSetFloat(parameters, dict::light::power.sid, spec.po.power, 1.f);
        parseAndSetFloat(parameters, dict::light::scale.sid, spec.scale, 1.f);

        switch (type)
        {
            case ELightType::eDistant:
            {
                auto& param = spec.params.distant;
                //dict::light::distant_params::L.sid
                if (!parseAndSetVector(parameters, dict::light::distant_params::from.sid, param.from, {{0, 0, 0}}, pointTypes))
                    std::abort();
                if (!parseAndSetVector(parameters, dict::light::distant_params::to.sid, param.to, {{0, 0, 1}}, pointTypes))
                    std::abort();
                break;
            }
            case ELightType::eGoniometric:
            {
                auto& param = spec.params.goniometric;
                if (!parseAndSetString2(parameters,
                                        dict::light::goniometric_params::filename.sid,
                                        param.filename,
                                        u8""sv,
                                        true))
                    std::abort(); // TODO better

                //dict::light::goniometric_params::I.sid
                break;
            }
            case ELightType::eInfinite:
            {
                auto& param = spec.params.infinite;
                if (!parseAndSetString2(parameters, dict::light::infinite_params::filename.sid, param.filename, u8""sv))
                    std::abort(); // TODO better
                std::array<float, 12> def{};
                if (!parseAndSetFloatArray(parameters,
                                           dict::light::infinite_params::portal.sid,
                                           12,
                                           std::bit_cast<float*>(&param.portal[0][0]),
                                           &def[0],
                                           dict::types::tPoint3.sid))
                    std::abort(); // TODO better
                //dict::light::infinite_params::L.sid
                break;
            }
            case ELightType::ePoint:
            {
                auto& param = spec.params.point;
                //dict::light::point_params::I.sid
                if (!parseAndSetVector(parameters, dict::light::point_params::from.sid, param.from, {{0, 0, 0}}, pointTypes))
                    std::abort();
                break;
            }
            case ELightType::eProjection:
            {
                auto& param = spec.params.projection;
                //dict::light::projection_params::I.sid
                if (!parseAndSetFloat(parameters, dict::light::projection_params::fov.sid, param.fov, 90.f))
                    std::abort(); // TODO better
                if (!parseAndSetString2(parameters, dict::light::projection_params::filename.sid, param.filename, u8""sv, true))
                    std::abort(); // TODO better
                break;
            }
            case ELightType::eSpot:
            {
                auto& param = spec.params.spotlight;
                //dict::light::spotlight_params::I
                if (!parseAndSetVector(parameters, dict::light::spotlight_params::from.sid, param.from, {{0, 0, 0}}, pointTypes))
                    std::abort(); // TODO better
                if (!parseAndSetVector(parameters, dict::light::spotlight_params::to.sid, param.to, {{0, 0, 1}}, pointTypes))
                    std::abort(); // TODO better
                if (!parseAndSetFloat(parameters, dict::light::spotlight_params::coneangle.sid, param.coneangle, 30.f))
                    std::abort();
                if (!parseAndSetFloat(parameters, dict::light::spotlight_params::conedeltaangle.sid, param.conedeltaangle, 5.f))
                    std::abort();
                break;
            }
        }
    }
    // MaterialEntity ----------------------------------------------------------------------------------------------------
    /*MaterialEntity::MaterialEntity(EMaterialType type, EColorSpaceType _colorSpace, ParamMap const& _params) :
    SceneEntity(materiaSidFromType(type), _params),
    spec(type),
    colorSpace(_colorSpace)
    {
    }*/
    MaterialEntity::MaterialEntity(MaterialSpec const& spec, EColorSpaceType _colorSpace, ParamMap const& _params) :
    SceneEntity(materialSidFromType(type), _params),
    type(type),
    colorSpace(_colorSpace)
    {
        switch (spec.type)
        {
        }
    }

    // Parsing Helpers ------------------------------------------------------------------------------------------------
    static bool isDirective(sid_t token)
    {
        switch (token)
        {
            case dict::directive::Option.sid: [[fallthrough]];
            case dict::directive::Identity.sid: [[fallthrough]];
            case dict::directive::Camera.sid: [[fallthrough]];
            case dict::directive::Sampler.sid: [[fallthrough]];
            case dict::directive::ColorSpace.sid: [[fallthrough]];
            case dict::directive::Film.sid: [[fallthrough]];
            case dict::directive::PixelFilter.sid: [[fallthrough]];
            case dict::directive::Integrator.sid: [[fallthrough]];
            case dict::directive::Accelerator.sid: [[fallthrough]];
            case dict::directive::WorldBegin.sid: [[fallthrough]];
            case dict::directive::AttributeBegin.sid: [[fallthrough]];
            case dict::directive::AttributeEnd.sid: [[fallthrough]];
            case dict::directive::Include.sid: [[fallthrough]];
            case dict::directive::Import.sid: [[fallthrough]];
            case dict::directive::LookAt.sid: [[fallthrough]];
            case dict::directive::Translate.sid: [[fallthrough]];
            case dict::directive::Scale.sid: [[fallthrough]];
            case dict::directive::Rotate.sid: [[fallthrough]];
            case dict::directive::CoordinateSystem.sid: [[fallthrough]];
            case dict::directive::CoordSysTransform.sid: [[fallthrough]];
            case dict::directive::Transform.sid: [[fallthrough]];
            case dict::directive::ConcatTransform.sid: [[fallthrough]];
            case dict::directive::TransformTimes.sid: [[fallthrough]];
            case dict::directive::ActiveTransform.sid: [[fallthrough]];
            case dict::directive::ReverseOrientation.sid: [[fallthrough]];
            case dict::directive::Attribute.sid: [[fallthrough]];
            case dict::directive::Shape.sid: [[fallthrough]];
            case dict::directive::ObjectBegin.sid: [[fallthrough]];
            case dict::directive::ObjectEnd.sid: [[fallthrough]];
            case dict::directive::ObjectInstance.sid: [[fallthrough]];
            case dict::directive::LightSource.sid: [[fallthrough]];
            case dict::directive::AreaLightSource.sid: [[fallthrough]];
            case dict::directive::Material.sid: [[fallthrough]];
            case dict::directive::MakeNamedMaterial.sid: [[fallthrough]];
            case dict::directive::NamedMaterial.sid: [[fallthrough]];
            case dict::directive::Texture.sid: [[fallthrough]];
            case dict::directive::MakeNamedMedium.sid: [[fallthrough]];
            case dict::directive::MediumInterface.sid: return true;
            default: return false;
        }
    }

    static bool isZeroArgsDirective(sid_t token)
    {
        switch (token)
        {
            case dict::directive::Identity.sid: [[fallthrough]];
            case dict::directive::WorldBegin.sid: [[fallthrough]];
            case dict::directive::AttributeBegin.sid: [[fallthrough]];
            case dict::directive::AttributeEnd.sid: [[fallthrough]];
            case dict::directive::ReverseOrientation.sid: [[fallthrough]];
            case dict::directive::ObjectEnd.sid: return true;
            default: return false;
        }
    }

    struct ParamExractRet
    {
        sid_t            type;
        std::string_view name;
        sid_t            sid;
    };

    static ParamExractRet maybeExtractParam(AppContext& actx, std::string_view token)
    {
        ParamExractRet ret;
        ret.type = 0;
        ret.name = dequoteString(token);
        ret.sid  = hashCRC64(ret.name);
        if (token.starts_with('"'))
        {
            if (!token.ends_with('"'))
            {
                actx.error("syntax error file");
                std::abort();
            }
            size_t whiteSpacePos = findFirstWhitespace(ret.name);
            if (whiteSpacePos != std::string_view::npos)
            {
                std::string_view maybeType    = ret.name.substr(0, whiteSpacePos);
                sid_t            maybeTypeSid = hashCRC64(maybeType);
                switch (maybeTypeSid)
                {
                    case dict::types::tVector.sid: [[fallthrough]];
                    case dict::types::tPoint.sid: [[fallthrough]];
                    case dict::types::tNormal.sid: [[fallthrough]];
                    case dict::types::tBool.sid: [[fallthrough]];
                    case dict::types::tFloat.sid: [[fallthrough]];
                    case dict::types::tInteger.sid: [[fallthrough]];
                    case dict::types::tString.sid: [[fallthrough]];
                    case dict::types::tRGB.sid: [[fallthrough]];
                    case dict::types::tPoint2.sid: [[fallthrough]];
                    case dict::types::tPoint3.sid: [[fallthrough]];
                    case dict::types::tNormal3.sid: [[fallthrough]];
                    case dict::types::tBlackbody.sid: [[fallthrough]];
                    case dict::types::tSpectrum.sid: [[fallthrough]];
                    case dict::types::tVector2.sid: [[fallthrough]];
                    case dict::types::tVector3.sid: [[fallthrough]];
                    case dict::types::tTexture.sid:
                        ret.type = maybeTypeSid;
                        ret.name = trimStartWhitespace(ret.name.substr(whiteSpacePos));
                        ret.sid  = hashCRC64(ret.name);
                        break;
                    default: break;
                }
            }
        }

        return ret;
    }

    static bool isParameter(AppContext& actx, std::string_view token)
    {
        if (token.starts_with('"'))
        {
            if (!token.ends_with('"'))
            {
                actx.error("syntax error file");
                std::abort();
            }

            std::string_view name          = dequoteString(token);
            size_t           whiteSpacePos = findFirstWhitespace(name);
            if (whiteSpacePos != std::string_view::npos)
            {
                std::string_view maybeType    = name.substr(0, whiteSpacePos);
                sid_t            maybeTypeSid = hashCRC64(maybeType);
                switch (maybeTypeSid)
                {
                    case dict::types::tVector.sid: [[fallthrough]];
                    case dict::types::tPoint.sid: [[fallthrough]];
                    case dict::types::tNormal.sid: [[fallthrough]];
                    case dict::types::tBool.sid: [[fallthrough]];
                    case dict::types::tFloat.sid: [[fallthrough]];
                    case dict::types::tInteger.sid: [[fallthrough]];
                    case dict::types::tString.sid: [[fallthrough]];
                    case dict::types::tRGB.sid: [[fallthrough]];
                    case dict::types::tPoint2.sid: [[fallthrough]];
                    case dict::types::tPoint3.sid: [[fallthrough]];
                    case dict::types::tNormal3.sid: [[fallthrough]];
                    case dict::types::tBlackbody.sid: [[fallthrough]];
                    case dict::types::tSpectrum.sid: [[fallthrough]];
                    case dict::types::tVector2.sid: [[fallthrough]];
                    case dict::types::tVector3.sid: [[fallthrough]];
                    case dict::types::tTexture.sid: return true;
                    default: break;
                }
            }
        }

        return false;
    }

    static sid_t setCameraParams(CameraSpec& cameraSpec, ParamMap const& params, Options const& cmdOptions)
    { // TODO insert cmdOptions as default params
        // parse common params
        if (!parseAndSetFloat(params, dict::camera::shutteropen.sid, cameraSpec.shutteropen, 0.f) ||
            !parseAndSetFloat(params, dict::camera::shutterclose.sid, cameraSpec.shutterclose, 1.f))
        {
            return "camera"_side; // Early return on error
        }

        // parse class specific parameters
        switch (cameraSpec.type)
        {
            case ECameraType::ePerspective:
            {
                auto& projectingParams = cameraSpec.params.p;
                projectingParams       = std::remove_cvref_t<decltype(projectingParams)>{};
                if (!parseAndSetFloat(params, dict::camera::lensradius.sid, projectingParams.lensRadius, 0.f) ||
                    !parseAndSetFloat(params, dict::camera::focaldistance.sid, projectingParams.focalDistance, 1e30f))
                { // error
                    return "orthographic"_side;
                }
                if (!parseAndSetFloat(params, dict::camera::fov.sid, projectingParams.fov, 90.f))
                { // error
                    return dict::camera::fov.sid;
                }
                // NOTE: frameAspectRatio and defaultScreenWindow are set when the film is encountered
                break;
            }
            case ECameraType::eOrthographic:
            {
                auto& projectingParams = cameraSpec.params.p;
                projectingParams       = std::remove_cvref_t<decltype(projectingParams)>{};
                if (!parseAndSetFloat(params, dict::camera::lensradius.sid, projectingParams.lensRadius, 0.f) ||
                    !parseAndSetFloat(params, dict::camera::focaldistance.sid, projectingParams.focalDistance, 1e30f))
                { // error
                    return "orthographic"_side;
                }
                // NOTE: frameAspectRatio and defaultScreenWindow are set when the film is encountered
                break;
            }
            case ECameraType::eRealistic:
            {
                using namespace std::string_view_literals;
                auto& realisticParams = cameraSpec.params.r;
                realisticParams       = std::remove_cvref_t<decltype(realisticParams)>{};
                if (!parseAndSetFloat(params, dict::camera::aperturediameter.sid, realisticParams.apertureDiameter, 1.f) ||
                    !parseAndSetFloat(params, dict::camera::focusdistance.sid, realisticParams.focusDistance, 10.f) ||
                    !parseAndSetString(params, dict::camera::lensfile.sid, realisticParams.lensfile, ""sv) ||
                    !parseAndSetString(params,
                                       dict::camera::aperture.sid,
                                       realisticParams.aperture,
                                       dict::camera::aperture_builtin::circular.str))
                { // error
                    return "realistic"_side;
                }
                break;
            }
            case ECameraType::eSpherical:
            {
                auto& sphericalParams = cameraSpec.params.s;
                sphericalParams       = std::remove_cvref_t<decltype(sphericalParams)>{};
                if (!parseAndSetEnum(params,
                                     dict::camera::mapping.sid,
                                     sphericalParams.mapping,
                                     ESphericalMapping::eEqualArea,
                                     sphericalMappingFromSid))
                { // error
                    return dict::camera::mapping.sid;
                }
                break;
            }
        }

        return 0;
    }

    static sid_t setSamplerParams(SamplerSpec& samplerSpec, ParamMap const& params, Options const& cmdOptions)
    {
        // seed is used by almost all, so consume it anyways
        if (!parseAndSetInt(params, dict::sampler::seed.sid, samplerSpec.seed, cmdOptions.seed))
        {
            return dict::sampler::seed.sid;
        }

        // all but stratified sampler have num samples not subdivided by axes
        if (samplerSpec.type == ESamplerType::eStratified)
        {
            auto& stratifiedSamples = samplerSpec.samples.stratified;
            if (!parseAndSetBool(params, dict::sampler::jitter.sid, stratifiedSamples.jitter, true) ||
                !parseAndSetInt(params, dict::sampler::xsamples.sid, stratifiedSamples.x, 4) ||
                !parseAndSetInt(params, dict::sampler::ysamples.sid, stratifiedSamples.y, 4))
            {
                return "stratified"_side;
            }
        }
        else
        {
            if (!parseAndSetInt(params, dict::sampler::pixelsamples.sid, samplerSpec.samples.num, 16))
                return dict::sampler::pixelsamples.sid;

            if (samplerSpec.type == ESamplerType::eIndependent)
            {
                ERandomization def = samplerSpec.type == ESamplerType::eHalton ? ERandomization::ePermuteDigits
                                                                               : ERandomization::eFastOwen;
                if (!parseAndSetEnum(params, dict::sampler::randomization.sid, samplerSpec.randomization, def, randomizationFromSid))
                    return dict::sampler::randomization.sid;
            }
        }

        return 0;
    }

    static sid_t setFilmParams(FilmSpec& filmSpec, ParamMap const& params, Options const& unused)
    {
        using namespace std::string_view_literals;
        // common parameters
        if (!parseAndSetInt(params, dict::film::xresolution.sid, filmSpec.xResolution, 1280) ||
            !parseAndSetInt(params, dict::film::yresolution.sid, filmSpec.yResolution, 720) ||
            !parseAndSetFloat(params, dict::film::diagonal.sid, filmSpec.diagonal, 35.f) ||
            !parseAndSetFloat(params, dict::film::iso.sid, filmSpec.iso, 100.f) ||
            !parseAndSetFloat(params, dict::film::whitebalance.sid, filmSpec.whiteBalance, 0.f) ||
            !parseAndSetFloat(params,
                              dict::film::maxcomponentvalue.sid,
                              filmSpec.maxComponentValue,
                              std::numeric_limits<float>::infinity()) ||
            !parseAndSetEnum(params, dict::film::sensor.sid, filmSpec.sensor, ESensor::eCIE1931, sensorFromSid) ||
            !parseAndSetBool(params, dict::film::savefp16.sid, filmSpec.savefp16, true) ||
            !parseAndSetFloat4(params, dict::film::cropwindow.sid, filmSpec.cropWindow, {0.f, 1.f, 0.f, 1.f}) ||
            !parseAndSetInt4(params,
                             dict::film::pixelbounds.sid,
                             filmSpec.pixelBounds,
                             {0, filmSpec.xResolution, 0, filmSpec.yResolution}) ||
            !parseAndSetString(params, dict::film::filename.sid, filmSpec.fileName, "pbrt.exr"sv))
        {
            return "common"_side;
        }

        switch (filmSpec.type)
        {
            // if RGB,then filename extension can be one of .pfm, .exr, .qoi, .png
            case EFilmType::eRGB:
            {
                if (!endsWithAny(filmSpec.fileName, {".pfm"sv, ".exr"sv, ".qoi"sv, ".png"sv}))
                    return "rgb::filename"_side;
                break;
            }
            // gbuffer
            case EFilmType::eGBuffer:
            {
                if (!parseAndSetEnum(params,
                                     dict::film::coordinatesystem.sid,
                                     filmSpec.coordSys,
                                     EGVufferCoordSys::eCamera,
                                     gBufferCoordSysFromSid))
                    return "gbuffer::coordinatesystem"_side;
                break;
            }
            // spectral film
            case EFilmType::eSpectral:
            {
                if (!parseAndSetInt(params, dict::film::nbuckets.sid, filmSpec.nBuckets, static_cast<int16_t>(16)) ||
                    !parseAndSetFloat(params, dict::film::lambdamin.sid, filmSpec.lambdaMin, 360.f) ||
                    !parseAndSetFloat(params, dict::film::lambdamax.sid, filmSpec.lambdaMax, 830.f))
                {
                    return "spectral"_side;
                }
                break;
            }
            default: assert(false); break;
        }

        return 0;
    }

    static sid_t setFilterParams(FilterSpec& filterSpec, ParamMap const& params, Options const& unused)
    {
        static constexpr float oneThird      = 0x1.3333333p-2f;
        float                  defaultRadius = defaultRadiusFromFilterType(filterSpec.type);
        // set xRadius and yRadius
        if (!parseAndSetFloat(params, dict::filter::xradius.sid, filterSpec.xRadius, defaultRadius) ||
            !parseAndSetFloat(params, dict::filter::yradius.sid, filterSpec.yRadius, defaultRadius))
        {
            return "radius"_side;
        }
        switch (filterSpec.type)
        {
            case EFilterType::eGaussian:
            {
                auto& gauss = filterSpec.params.gaussian;
                if (!parseAndSetFloat(params, dict::filter::sigma.sid, gauss.sigma, 0.5f))
                    return "gaussian::sigma"_side;
                break;
            }
            case EFilterType::eMitchell:
            {
                auto& mitchell = filterSpec.params.mitchell;
                if (!parseAndSetFloat(params, dict::filter::B.sid, mitchell.b, oneThird) ||
                    !parseAndSetFloat(params, dict::filter::C.sid, mitchell.c, oneThird))
                {
                    return "mitchell"_side;
                }
                break;
            }
            case EFilterType::eSinc:
            {
                auto& sinc = filterSpec.params.sinc;
                if (!parseAndSetFloat(params, dict::filter::tau.sid, sinc.tau, 3.f))
                    return "sinc::tau"_side;
                break;
            }
            case EFilterType::eBox: break;
            case EFilterType::eTriangle: break;
            default: assert(false); break;
        }

        return 0;
    }

    static sid_t setIntegratorParams(IntegratorSpec& integratorSpec, ParamMap const& params, Options const& options)
    {
        assert(integratorSpec.type != EIntegratorType::eCount);
        // maxdepth: all but ambientocclusion
        if (integratorSpec.type != EIntegratorType::eAmbientOcclusion)
        {
            if (!parseAndSetInt(params, dict::integrator::maxdepth.sid, integratorSpec.maxDepth, 5))
                return dict::integrator::maxdepth.sid;
        }

        // lightsampler: path volpath wavefront/gpu
        if (isAnyEnum(integratorSpec.type, {EIntegratorType::eVolPath, EIntegratorType::ePath}) || wavefrontOrGPU(options))
        {
            if (!parseAndSetEnum(params,
                                 dict::integrator::lightsampler.sid,
                                 integratorSpec.lightSampler,
                                 ELightSampler::eBVH,
                                 lightSamplerFromSid))
                return dict::integrator::lightsampler.sid;
        }

        // regularize: bdpt mlt path volpath wavefront/gpu
        if (isAnyEnum(integratorSpec.type,
                      {EIntegratorType::eBdpt, EIntegratorType::eMLT, EIntegratorType::ePath, EIntegratorType::eVolPath}) ||
            wavefrontOrGPU(options))
        {
            if (!parseAndSetBool(params, dict::integrator::regularize.sid, integratorSpec.regularize, false))
                return dict::integrator::regularize.sid;
        }

        // integrator specific parameters
        switch (integratorSpec.type)
        {
            case EIntegratorType::eAmbientOcclusion:
            {
                auto& aoParams = integratorSpec.params.ao;
                if (!parseAndSetBool(params, dict::integrator::cossample.sid, aoParams.cosSample, true) ||
                    !parseAndSetFloat(params,
                                      dict::integrator::maxdistance.sid,
                                      aoParams.maxDistance,
                                      std::numeric_limits<float>::infinity()))
                {
                    return "ambientocclusion"_side;
                }
                break;
            }
            case EIntegratorType::eBdpt:
            {
                auto& bdptParams = integratorSpec.params.bdpt;
                if (!parseAndSetBool(params, dict::integrator::visualizestrategies.sid, bdptParams.visualizeStrategies, false) ||
                    !parseAndSetBool(params, dict::integrator::visualizeweights.sid, bdptParams.visualizeWeights, false))
                {
                    return "bdpt"_side;
                }
                break;
            }
            case EIntegratorType::eMLT:
            {
                auto& mltParams = integratorSpec.params.mlt;
                if (!parseAndSetInt(params, dict::integrator::bootstrapsamples.sid, mltParams.bootstraqpSamples, 100000) ||
                    !parseAndSetInt(params, dict::integrator::chains.sid, mltParams.chains, 1000) ||
                    !parseAndSetInt(params, dict::integrator::mutationsperpixel.sid, mltParams.mutationsPerPixel, 100) ||
                    !parseAndSetFloat(params, dict::integrator::largestepprobability.sid, mltParams.largestStepProbability, 0.3f) ||
                    !parseAndSetFloat(params, dict::integrator::sigma.sid, mltParams.sigma, 0.01f))
                {
                    return "mlt"_side;
                }
                break;
            }
            case EIntegratorType::eSimplePath:
            {
                auto& simplePathParams = integratorSpec.params.simplePath;
                if (!parseAndSetBool(params, dict::integrator::samplebsdf.sid, simplePathParams.sampleBSDF, true) ||
                    !parseAndSetBool(params, dict::integrator::samplelights.sid, simplePathParams.sampleLights, true))
                {
                    return "simplepath"_side;
                }
                break;
            }
            case EIntegratorType::eSPPM:
            {
                auto& sppmParams = integratorSpec.params.sppm;
                if (!parseAndSetInt(params, dict::integrator::photonsperiteration.sid, sppmParams.photonsPerIteration, -1) ||
                    !parseAndSetFloat(params, dict::integrator::radius.sid, sppmParams.radius, 1.f) ||
                    !parseAndSetInt(params, dict::integrator::seed.sid, sppmParams.seed, 0))
                {
                    return "sppm"_side;
                }
                break;
            }
            default: break;
        }

        return 0;
    }

    static sid_t setAcceleratorParams(AcceleratorSpec& acceleratorSpec, ParamMap const& params, Options const& unused)
    {
        switch (acceleratorSpec.type)
        {
            case EAcceletatorType::eBVH:
            {
                auto& bvhParams = acceleratorSpec.params.bvh;
                if (!parseAndSetInt(params, dict::accelerator::maxnodeprims.sid, bvhParams.maxNodePrims, 4) ||
                    !parseAndSetEnum(params,
                                     dict::accelerator::splitmethod.sid,
                                     bvhParams.splitMethod,
                                     EBVHSplitMethod::eSAH,
                                     bvhSplitMethodFromSid))
                {
                    return "bvh"_side;
                }
                break;
            }
            case EAcceletatorType::eKdTree:
            {
                auto& kdtreeParams = acceleratorSpec.params.kdtree;
                if (!parseAndSetInt(params, dict::accelerator::intersectcost.sid, kdtreeParams.intersectCost, 5) ||
                    !parseAndSetInt(params, dict::accelerator::traversalcost.sid, kdtreeParams.traversalCost, 1) ||
                    !parseAndSetFloat(params, dict::accelerator::emptybonus.sid, kdtreeParams.emptyBonus, 0.5f) ||
                    !parseAndSetInt(params, dict::accelerator::maxprims.sid, kdtreeParams.maxPrims, 1) ||
                    !parseAndSetInt(params, dict::accelerator::maxdepth.sid, kdtreeParams.maxDepth, -1))
                {
                    return "kdtree"_side;
                }
                break;
            }
            default: break;
        }
        return 0;
    }

    // Parsing --------------------------------------------------------------------------------------------------------
    // https://github.com/mmp/pbrt-v4/blob/88645ffd6a451bd030d062a55a70a701c58a55d0/src/pbrt/parser.cpp
    char WordParser::decodeEscaped(char c)
    {
        switch (c)
        {
            case 'b': return '\b';
            case 'f': return '\f';
            case 'n': return '\n';
            case 'r': return '\r';
            case 't': return '\t';
            case '\\': return '\\';
            case '\'': return '\'';
            case '\"': return '\"';
            default: assert(false && "invalid escaped character"); std::abort();
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

    bool WordParser::needsContinuation() const { return m_needsContinuation; }

    uint32_t WordParser::numCharReadLast() const { return m_numCharReadLastTime; }

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

    // TokenStream ----------------------------------------------------------------------------------------------------
    TokenStream::TokenStream(AppContext& actx, std::string_view filePath)
    { // TODO proper memory allocation with tag and all
        std::construct_at(&m_delayedCtor.reader, actx.mctx().pctx, filePath.data(), chunkSize);
        if (!m_delayedCtor.reader)
        {
            actx.error("Couldn't open file {}", {filePath});
            std::abort();
        }

        m_buffer = reinterpret_cast<char*>(actx.stackAllocate(chunkSize, 8, EMemoryTag::eUnknown, (sid_t)0));
        if (!m_buffer)
        {
            actx.error("Couldn't allocate 512 B for token buffer");
            std::abort();
        }
        using Tokenizer = std::remove_pointer_t<decltype(m_tokenizer)>;
        m_tokenizer     = reinterpret_cast<decltype(m_tokenizer)>(
            actx.stackAllocate(sizeof(Tokenizer), alignof(Tokenizer), EMemoryTag::eUnknown, (sid_t)0));
        if (!m_tokenizer)
        {
            actx.error("Couldn't allocate word tokenizer");
            std::abort();
        }
        std::construct_at(m_tokenizer);

        // request first chunk
        bool success = m_delayedCtor.reader.requestChunk(actx.mctx().pctx, m_buffer, 0);
        assert(success);
    }

    TokenStream::~TokenStream() noexcept
    {
        assert(m_chunkNum == m_delayedCtor.reader.numChunks());
        std::destroy_at(&m_delayedCtor.reader);
        std::destroy_at(m_tokenizer);
    }

    std::string TokenStream::next(AppContext& actx)
    { // TODO stack allocated
        advance(actx);
        return peek();
    }

    void TokenStream::advance(AppContext& actx)
    {
        while (true)
        {
            if (m_newChunk)
            {
                bool completed = false;
                while (!completed)
                {
                    completed = m_delayedCtor.reader.waitForPendingChunk(actx.mctx().pctx, 1000);
                }
                m_chunk    = {m_buffer, m_delayedCtor.reader.lastNumBytesRead()};
                m_newChunk = false;
            }

            m_needAdvance = true;
            while (m_needAdvance)
            { // TODO utf8 token normalization
                std::string_view token = m_tokenizer->nextWord(m_chunk);
                if (!token.empty() && !m_tokenizer->needsContinuation())
                {
                    m_chunk = m_chunk.substr(m_tokenizer->numCharReadLast());
                    m_token = token;
                    return;
                }
                else
                {
                    m_needAdvance = false;
                    m_newChunk    = true;
                    if (++m_chunkNum == m_delayedCtor.reader.numChunks())
                    {
                        m_token.clear();
                        return;
                    }
                    bool success = m_delayedCtor.reader.requestChunk(actx.mctx().pctx, m_buffer, m_chunkNum);
                    assert(success);
                }
            }
        }
    }

    std::string TokenStream::peek() { return m_token; }

    // SceneParser ----------------------------------------------------------------------------------------------------
    SceneParser::SceneParser(AppContext& actx, IParserTarget* pTarget, std::string_view filePath) : m_pTarget(pTarget)
    {
        if (!m_pTarget)
        {
            actx.error("Valid parser target needed");
            std::abort();
        }

        pushFile(actx, filePath, true);
        char             separators[2] = {pathSeparator(), '/'};
        std::string_view sepView       = {separators, 2};
        size_t           pos           = filePath.find_last_of(sepView);
        if (pos == std::string_view::npos)
        {
            actx.error("Invalid path {}", {filePath});
        }

        m_basePath = filePath.substr(0, pos);
        m_basePath += '/';
        assert(m_basePath.back() == separators[0] || m_basePath.back() == separators[1]);
    }

    void SceneParser::parse(AppContext& actx, Options& inOutOptions)
    {
        using namespace std::string_view_literals;

        //check/get Arg and check/get params
        auto typeAndParamListParsing = [this]<typename Enum>
            requires(std::is_enum_v<Enum>)
        (AppContext & actx,
         SText const& directive,
         TokenStream& currentStream,
         ArgsDArray&  outArgs,
         ParamMap&    outParams,
         bool (*fromSidFunc)(sid_t, Enum&),
         Enum& out) {
            currentStream.advance(actx);

            //check number of Args and get enum
            if (parseArgs(actx, currentStream, outArgs) != 1 || !fromSidFunc(hashCRC64(dequoteString(outArgs[0])), out))
            {
                actx.error("Unexpected argument {} for directive {}", {outArgs[0], directive.str});
                std::abort();
            }
            //check number of parameters and store them in outParams
            if (parseParams(actx, currentStream, outParams) == 0)
            {
                actx.error("Expected at least a parameter for directive {}", {directive.str});
                std::abort();
            }
        };


        auto typeHeaderDirectiveParsing =
            [this]<typename Spec, typename EnumType>
            requires requires(Spec spec) {
                spec.type;
                requires std::is_same_v<decltype(spec.type), EnumType> && std::is_enum_v<EnumType> &&
                             std::is_default_constructible_v<Spec>;
            }(AppContext & actx,
              SText const&   directive,
              Options const& options,
              TokenStream&   currentStream,
              ArgsDArray&    outArgs,
              ParamMap&      outParams,
              bool (*fromSidFunc)(sid_t, EnumType&),
              sid_t (*setParams)(Spec&, ParamMap const&, Options const&),
              void (IParserTarget::*apiFunc)(Spec const&),
              EEncounteredHeaderDirective eDirective,
              void (*callback)(SceneParser& self, Spec const& spec) = nullptr)
        {
            Spec spec;
            //check if you are out from world block
            if (!transitionToHeaderIfFirstHeaderDirective(actx, options, eDirective))
                std::abort();
            currentStream.advance(actx);
            //Check Arg
            if (uint32_t num = parseArgs(actx, currentStream, outArgs); num != 1)
            {
                actx.error("Unexpected number of arguments for {} directive. Should be 1, got {}", {directive.str, num});
                std::abort();
            }
            if (!fromSidFunc(hashCRC64(dequoteString(outArgs[0])), spec.type))
            {
                actx.error("Unexpected type argument for {} directive. got {}. Consult docs to see possible values.",
                           {directive.str, outArgs[0]});
                std::abort();
            }
            //check and get
            parseParams(actx, currentStream, outParams);

            if (setParams(spec, outParams, options) != 0)
            {
                actx.error("Encountered error while parsing {} parameters", {directive.str});
                std::abort();
            }
            (m_pTarget->*apiFunc)(spec);
            if (callback)
                callback(*this, spec);
        };

        //usefull for the transforms fix leng of params
        auto parseArgumentFloats = [this]<size_t size> // TODO rework without template
            requires(size == 3 || size == 16 || size == 9 || size == 4 || size == 2)
        (AppContext & actx,
         SText const&             directive,
         TokenStream&             currentStream,
         ArgsDArray&              outArgs,
         std::array<float, size>& outArray) {
            currentStream.advance(actx);
            if (parseArgs(actx, currentStream, outArgs) != size)
            {
                actx.error("Unexpected number of parameters for {} directive", {directive.str});
                std::abort();
            }
            if (parseFloatArray(outArgs, outArray) != size)
            {
                actx.error("Error while parsing float arguments for the {} directive", {directive.str});
                std::abort();
            }
        };

        //directives that has only string arg
        auto parseArgumentNames =
            [this](AppContext&  actx,
                   SText const& directive,
                   TokenStream& currentStream,
                   ArgsDArray&  outArgs,
                   sid_t*       pSids,
                   uint32_t     num) {
            currentStream.advance(actx);
            if (parseArgs(actx, currentStream, outArgs) != num ||
                std::reduce(outArgs.begin(), outArgs.begin() + num, false, [](bool curr, std::string const& elem) {
                return curr || !startsWithEndsWith(elem, '"', '"');
            }))
            {
                actx.error("Directive {} expects {} quoted string argument(s)", {directive.str, num});
                std::abort();
            }
            for (uint32_t i = 0; i < num; ++i)
                pSids[i] = hashCRC64(dequoteString(outArgs[i]));
        };

        //reads all file
        while (!m_fileStack.empty())
        {
            TokenStream& currentStream = topFile();
            currentStream.advance(actx);

            //read file
            for (std::string token = currentStream.peek(); !token.empty(); token = currentStream.peek())
            { // token advancement handled by either parseArgs, parseParams, or by a switch case
                ArgsDArray args;
                ParamMap   params;
                sid_t      tokenSid = hashCRC64(token);

                //comment statement
                if (token.starts_with('#'))
                {
                    currentStream.advance(actx);
                    continue;
                }


                if (!isDirective(tokenSid))
                {
                    //actx.error("Invalid directive {}", {token});
                    //std::abort();
                    currentStream.advance(actx);
                    continue;
                }

                //check directives
                switch (tokenSid)
                { // TODO args parsing type with error handling
                    case dict::directive::Option.sid:
                    { // 1 parameter, allowed only in step Options
                        if (m_parsingStep != EParsingStep::eOptions)
                        {
                            actx.error("Option directive allowed only before any other Directives in the Header");
                            std::abort();
                        }
                        parseParams(actx, currentStream, params);
                        if (!setOptionParam(actx, params, inOutOptions))
                            std::abort();
                        m_pTarget->Option(params.begin()->first, params.begin()->second);
                        break;
                    }
                    case dict::directive::Identity.sid:
                    {
                        //to verify
                        m_pTarget->Identity();
                        //currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::Camera.sid:
                    {
                        void (*copyCamera)(SceneParser& self, CameraSpec const& spec) =
                            [](SceneParser& self, CameraSpec const& spec) { self.m_parsingState.cameraSpec = spec; };
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::Camera,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   cameraTypeFromSid,
                                                   setCameraParams,
                                                   &IParserTarget::Camera,
                                                   EEncounteredHeaderDirective::eCamera,
                                                   copyCamera);
                        break;
                    }
                    case dict::directive::Sampler.sid:
                    {
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::Sampler,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   samplerTypeFromSid,
                                                   setSamplerParams,
                                                   &IParserTarget::Sampler,
                                                   EEncounteredHeaderDirective::eSampler);
                        break;
                    }
                    case dict::directive::ColorSpace.sid:
                    {
                        if (!transitionToHeaderIfFirstHeaderDirective(actx, inOutOptions, EEncounteredHeaderDirective::eColorSpace))
                            std::abort();
                        currentStream.advance(actx);
                        if (parseArgs(actx, currentStream, args) != 1)
                        {
                            actx.error("Unexpected number of arguments for ColorSpace directive");
                            std::abort();
                        }
                        EColorSpaceType type;
                        if (!colorSpaceTypeFromSid(hashCRC64(args[0]), type))
                        {
                            actx.error("Unknown color space inserted {} in directive {}",
                                       {args[0], dict::directive::ColorSpace.str});
                            std::abort();
                        }
                        m_pTarget->ColorSpace(type);
                        break;
                    }
                    case dict::directive::Film.sid:
                    {
                        void (*storeResolution)(SceneParser& self, FilmSpec const& spec) =
                            [](SceneParser& self, FilmSpec const& spec) {
                            self.m_parsingState.xResolution = spec.xResolution;
                            self.m_parsingState.yResolution = spec.yResolution;
                        };
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::Film,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   filmTypeFromSid,
                                                   setFilmParams,
                                                   &IParserTarget::Film,
                                                   EEncounteredHeaderDirective::eFilm,
                                                   storeResolution);
                        break;
                    }
                    case dict::directive::PixelFilter.sid:
                    {
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::PixelFilter,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   filterTypeFromSid,
                                                   setFilterParams,
                                                   &IParserTarget::PixelFilter,
                                                   EEncounteredHeaderDirective::ePixelFilter);
                        break;
                    }
                    case dict::directive::Integrator.sid:
                    {
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::Integrator,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   integratorTypeFromSid,
                                                   setIntegratorParams,
                                                   &IParserTarget::Integrator,
                                                   EEncounteredHeaderDirective::eIntegrator);
                        break;
                    }
                    case dict::directive::Accelerator.sid:
                    {
                        typeHeaderDirectiveParsing(actx,
                                                   dict::directive::Accelerator,
                                                   inOutOptions,
                                                   currentStream,
                                                   args,
                                                   params,
                                                   acceleratorTypeFromSid,
                                                   setAcceleratorParams,
                                                   &IParserTarget::Accelerator,
                                                   EEncounteredHeaderDirective::eAccelerator);
                        break;
                    }
                    case dict::directive::WorldBegin.sid:
                    {
                        if (m_parsingStep == EParsingStep::eWorld)
                        {
                            actx.error("Encountered WorldBegin more than once");
                            std::abort();
                        }
                        // Complete Camera Parsing and pass the complete camera
                        if (m_parsingState.cameraSpec.type == ECameraType::ePerspective ||
                            m_parsingState.cameraSpec.type == ECameraType::eOrthographic)
                        {
                            if (m_parsingState.xResolution == -1)
                            {
                                FilmSpec defaultFilm;
                                m_parsingState.xResolution = defaultFilm.xResolution;
                                m_parsingState.yResolution = defaultFilm.yResolution;
                            }
                            auto& projParams = m_parsingState.cameraSpec.params.p;
                            if (projParams.frameAspectRatio == CameraSpec::invalidAspectRatio)
                            {
                                projParams.frameAspectRatio = static_cast<float>(m_parsingState.xResolution) /
                                                              m_parsingState.yResolution;
                            }
                            if (projParams.screenWindow.p.maxX == CameraSpec::invalidScreen)
                            {
                                float    largeValue = projParams.frameAspectRatio > 1.f
                                                          ? projParams.frameAspectRatio
                                                          : (1.f / projParams.frameAspectRatio);
                                uint32_t idxMaxAxis = projParams.frameAspectRatio > 1.f ? 0 : 2;
                                uint32_t idxMinAxis = (idxMaxAxis + 2) & (projParams.screenWindow.arr.size() - 1);
                                projParams.screenWindow.arr[idxMinAxis]     = -1.f;
                                projParams.screenWindow.arr[idxMinAxis + 1] = 1.f;
                                projParams.screenWindow.arr[idxMaxAxis]     = -largeValue;
                                projParams.screenWindow.arr[idxMaxAxis + 1] = largeValue;
                            }
                        }

                        EndOfHeaderInfo info{};
                        info.cameraSpec = m_parsingState.cameraSpec;
                        m_pTarget->EndOfHeader(info);

                        m_parsingStep = EParsingStep::eWorld;
                        m_pTarget->WorldBegin();
                        break;
                    }
                    case dict::directive::AttributeBegin.sid:
                    {
                        pushScope(EScope::eAttribute);
                        m_pTarget->AttributeBegin();
                        break;
                    }
                    case dict::directive::AttributeEnd.sid:
                    {
                        if (!hasScope() || currentScope() != EScope::eAttribute)
                        {
                            actx.error("Unbalanced AttributeBegin and AttributeEnd");
                            std::abort();
                        }
                        popScope();
                        m_pTarget->AttributeEnd();
                        break;
                    }
                    case dict::directive::Include.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::Import.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::LookAt.sid:
                    {
                        std::array<float, 9> look{};
                        parseArgumentFloats(actx, dict::directive::LookAt, currentStream, args, look);
                        m_pTarget->LookAt(look[0], look[1], look[2], look[3], look[4], look[5], look[6], look[7], look[8]);
                        break;
                    }
                    case dict::directive::Translate.sid:
                    {
                        std::array<float, 3> translate{};
                        parseArgumentFloats(actx, dict::directive::Translate, currentStream, args, translate);
                        m_pTarget->Translate(translate[0], translate[1], translate[2]);
                        break;
                    }
                    case dict::directive::Scale.sid:
                    {
                        std::array<float, 3> scale{};
                        parseArgumentFloats(actx, dict::directive::Scale, currentStream, args, scale);
                        m_pTarget->Scale(scale[0], scale[1], scale[2]);
                        break;
                    }
                    case dict::directive::Rotate.sid:
                    {
                        std::array<float, 4> rotate{};
                        parseArgumentFloats(actx, dict::directive::Rotate, currentStream, args, rotate);
                        m_pTarget->Rotate(rotate[0], rotate[1], rotate[2], rotate[3]);
                        break;
                    }
                    case dict::directive::CoordinateSystem.sid:
                    {
                        sid_t coordSys = 0;
                        parseArgumentNames(actx, dict::directive::CoordinateSystem, currentStream, args, &coordSys, 1);
                        m_pTarget->CoordinateSystem(coordSys);
                        break;
                    }
                    case dict::directive::CoordSysTransform.sid:
                    {
                        sid_t coordSys = 0;
                        parseArgumentNames(actx, dict::directive::CoordSysTransform, currentStream, args, &coordSys, 1);
                        m_pTarget->CoordSysTransform(coordSys);
                        break;
                    }
                    case dict::directive::Transform.sid:
                    {
                        std::array<float, 16> transform{1, 0, 0, 0, /**/ 0, 1, 0, 0, /**/ 0, 0, 1, 0, /**/ 0, 0, 0, 1};
                        parseArgumentFloats(actx, dict::directive::Transform, currentStream, args, transform);
                        m_pTarget->Transform(transform);
                        break;
                    }
                    case dict::directive::ConcatTransform.sid:
                    {
                        std::array<float, 16> transform{1, 0, 0, 0, /**/ 0, 1, 0, 0, /**/ 0, 0, 1, 0, /**/ 0, 0, 0, 1};
                        parseArgumentFloats(actx, dict::directive::ConcatTransform, currentStream, args, transform);
                        m_pTarget->ConcatTransform(transform);
                        break;
                    }
                    case dict::directive::TransformTimes.sid:
                    {
                        std::array<float, 2> startEnd{0, 1};
                        parseArgumentFloats(actx, dict::directive::TransformTimes, currentStream, args, startEnd);
                        m_pTarget->TransformTimes(startEnd[0], startEnd[1]);
                        break;
                    }
                    case dict::directive::ActiveTransform.sid:
                    {
                        EActiveTransform transform = EActiveTransform::eStartTime;
                        currentStream.advance(actx);
                        if (parseArgs(actx, currentStream, args) != 1 ||
                            !activeTransformFromSid(hashCRC64(args[0]), transform))
                        {
                            actx.error("illegal or absent argument for directive {}",
                                       {dict::directive::ActiveTransform.str});
                            std::abort();
                        }
                        switch (transform)
                        {
                            case EActiveTransform::eStartTime: m_pTarget->ActiveTransformStartTime(); break;
                            case EActiveTransform::eEndTime: m_pTarget->ActiveTransformEndTime(); break;
                            case EActiveTransform::eAll: m_pTarget->ActiveTransformAll(); break;
                        }
                        break;
                    }
                    case dict::directive::ReverseOrientation.sid:
                    {
                        m_pTarget->ReverseOrientation();
                        break;
                    }
                    case dict::directive::Attribute.sid:
                    {
                        ETarget target = ETarget::eShape;
                        typeAndParamListParsing(actx, dict::directive::Attribute, currentStream, args, params, targetFromSid, target);
                        m_pTarget->Attribute(target, params);
                        break;
                    }
                    case dict::directive::Shape.sid:
                    { // TODO create file
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::ObjectBegin.sid:
                    {
                        sid_t name = 0;
                        parseArgumentNames(actx, dict::directive::ObjectBegin, currentStream, args, &name, 1);
                        pushScope(EScope::eObject);
                        m_pTarget->ObjectBegin(name);
                        break;
                    }
                    case dict::directive::ObjectEnd.sid:
                    {
                        if (!hasScope() || currentScope() != EScope::eObject)
                        {
                            actx.error("Unbalanced ObjectBegin and ObjectEnd");
                            std::abort();
                        }
                        popScope();
                        m_pTarget->ObjectEnd();
                        break;
                    }
                    case dict::directive::ObjectInstance.sid:
                    {
                        if (m_parsingStep != EParsingStep::eWorld)
                        {
                            actx.error("Encontered directive {} outside of the World Block",
                                       {dict::directive::ObjectInstance.str});
                            std::abort();
                        }
                        sid_t name = 0;
                        parseArgumentNames(actx, dict::directive::ObjectBegin, currentStream, args, &name, 1);
                        m_pTarget->ObjectInstance(name);
                        break;
                    }
                    case dict::directive::LightSource.sid:
                    {
                        ELightType out;
                        typeAndParamListParsing(actx, dict::directive::LightSource, currentStream, args, params, lightTypeFromSid, out);
                        m_pTarget->LightSource(out, params);
                        break;
                    }
                    case dict::directive::AreaLightSource.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::Material.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::MakeNamedMaterial.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::NamedMaterial.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::Texture.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::MakeNamedMedium.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    case dict::directive::MediumInterface.sid:
                    {
                        //to remove
                        if (!isZeroArgsDirective(tokenSid))
                            currentStream.advance(actx);
                        break;
                    }
                    default:
                        actx.error("Unrecognized directive {}.", {token});
                        std::abort();
                        break;
                }

                if (isZeroArgsDirective(tokenSid))
                    currentStream.advance(actx);
            }
        }
    }

    bool SceneParser::setOptionParam(AppContext& actx, ParamMap const& params, Options& outOptions)
    {
        using namespace std::string_view_literals;
        if (params.size() != 1)
        {
            actx.error("Option directive: expected a single parameter");
            return false;
        }
        sid_t            name  = params.begin()->first;
        ParamPair const& param = params.begin()->second;
        bool             b     = false;
        switch (name)
        {
            case dict::opts::disablepixeljitter.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing disablepixeljitter");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efDisablepixeljitter;
                break;
            case dict::opts::disabletexturefiltering.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing disabletexturefiltering");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efDisabletexturefiltering;
                break;
            case dict::opts::disablewavelengthjitter.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing disablewavelengthjitter");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efDisablewavelengthjitter;
                break;
            case dict::opts::displacementedgescale.sid:
                if (!parseAndSetFloat(params, name, outOptions.diespacementEdgeScale, 1.f))
                {
                    actx.error("Error while parsing displacementedgescale");
                    return false;
                }
                break;
            case dict::opts::msereferenceimage.sid:
                outOptions.mseReferenceOutput = reinterpret_cast<char*>(
                    actx.stackAllocate(256, 8, EMemoryTag::eUnknown, (sid_t)0));
                if (!outOptions.mseReferenceOutput ||
                    !parseAndSetString(params,
                                       name,
                                       outOptions.mseReferenceImage,
                                       outOptions.mseReferenceImageLength,
                                       ""sv))
                {
                    actx.error("Error while parsing msereferenceimage");
                    return false;
                }
                break;
            case dict::opts::msereferenceout.sid: // TODO
                outOptions.mseReferenceOutput = reinterpret_cast<char*>(
                    actx.stackAllocate(256, 8, EMemoryTag::eUnknown, (sid_t)0));
                if (!outOptions.mseReferenceOutput ||
                    !parseAndSetString(params,
                                       name,
                                       outOptions.mseReferenceOutput,
                                       outOptions.mseReferenceOutputLength,
                                       ""sv))
                {
                    actx.error("Error while parsing msereferenceout");
                    return false;
                }
                break;
            case dict::opts::rendercoordsys.sid:
                if (!parseAndSetEnum(params, name, outOptions.renderCoord, ERenderCoordSys::eCameraWorld, renderCoordSysFromSid))
                {
                    actx.error("Error while parsing rendercoordsys");
                    return false;
                }
                break;
            case dict::opts::seed.sid:
                if (!parseAndSetInt(params, name, outOptions.seed, 0))
                {
                    actx.error("Error while parsing seed");
                    return false;
                }
                break;
            case dict::opts::forcediffuse.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing forcediffuse");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efForcediffuse;
                break;
            case dict::opts::pixelstats.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing pixelstats");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efPixelstats;
                break;
            case dict::opts::wavefront.sid:
                if (!parseAndSetBool(params, name, b, false))
                {
                    actx.error("Error while parsing wavefront");
                    return false;
                }
                if (b)
                    outOptions.flags |= EBoolOptions::efWavefront;
                break;
            default: actx.error("Option directive: unexpected parameter"); return false;
        }

        return true;
    }

    uint32_t SceneParser::parseArgs(AppContext& actx, TokenStream& stream, ArgsDArray& outArr)
    {
        uint32_t i = 0;
        for (std::string token = stream.peek(); !token.empty(); stream.advance(actx), token = stream.peek())
        {
            if (token.starts_with('#'))
                continue;

            sid_t tokenSid = hashCRC64(dequoteString(token));
            if (isDirective(tokenSid) || isParameter(actx, token))
                break;

            outArr.emplace_back(std::move(token));
            ++i;
        }

        if (i == 0)
        {
            actx.error("Expected at least 1 argument for the current directive, got none");
            // abort done by caller, which checks the number of args and their type
        }

        return i;
    }

    uint32_t SceneParser::parseParams(AppContext& actx, TokenStream& stream, ParamMap& outParams)
    {
        uint32_t i = 0;
        for (std::string token = stream.peek(); !token.empty(); /**/)
        {
            if (token.starts_with('#')) // TODO isComment function
            {
                stream.advance(actx);
                token = stream.peek();
                continue;
            }

            sid_t tokenSid = hashCRC64(dequoteString(token));
            if (isDirective(tokenSid))
                break;

            ParamExractRet param   = maybeExtractParam(actx, token);
            auto [it, wasInserted] = outParams.try_emplace(param.sid, param.type);
            if (!wasInserted)
            {
                actx.error("Unexpected error, couldn't insert parameter into map, token: ", {token});
                std::abort();
            }

            stream.advance(actx);
            tokenSid = hashCRC64(dequoteString(token));
            for (token = stream.peek(); !token.empty() && !isParameter(actx, token) && !isDirective(tokenSid);
                 stream.advance(actx), token = stream.peek(), tokenSid = hashCRC64(dequoteString(token)))
            {
                if ((token.size() == 1 && (token[0] == '[' || token[0] == ']')) || token.starts_with('#'))
                    continue;
                it->second.addParamValue(dequoteString(token));
            }
            if (it->second.values.size() < 1)
            {
                actx.error("Found a parameter without values, aborting...");
                std::abort();
            }

            ++i;
            if (token.empty() || isDirective(tokenSid))
                break;
        }

        if (i == 0)
        {
            actx.error("Expected at least 1 parameter for the current directive, got none");
            // abort done by caller, which checks the number of args and their type
        }

        return i;
    }

    bool SceneParser::transitionToHeaderIfFirstHeaderDirective(AppContext&                 actx,
                                                               Options const&              outOptions,
                                                               EEncounteredHeaderDirective val)
    {
        if (m_parsingStep == EParsingStep::eWorld)
        {
            actx.error("Unexpected directive found in World block");
            return false;
        }
        if (m_parsingStep == EParsingStep::eOptions)
        {
            m_parsingStep = EParsingStep::eHeader;
            m_pTarget->EndOfOptions(outOptions);
        }
        if (hasFlag(m_encounteredHeaders, val))
        {
            actx.error("Already encountered directive {}", {toStr(val)});
            return false;
        }
        putFlag(m_encounteredHeaders, val);
        return true;
    }

    void SceneParser::pushFile(AppContext& actx, std::string_view filePath, bool isImport)
    {
        m_fileStack.emplace_front(actx, filePath);
        if (isImport)
            m_scopeStacks.emplace_back();
    }

    void SceneParser::popFile(bool isImport)
    {
        m_fileStack.pop_front();
        if (isImport)
            m_scopeStacks.pop_back();
    }

    TokenStream& SceneParser::topFile() { return m_fileStack.front(); }

    bool SceneParser::hasScope() const { return !m_scopeStacks.empty(); }

    EScope SceneParser::currentScope() const
    {
        assert(hasScope() && !m_scopeStacks.back().empty());
        return m_scopeStacks.back().back();
    }

    void SceneParser::popScope()
    {
        assert(hasScope() && !m_scopeStacks.back().empty());
        m_scopeStacks.back().pop_back();
    }

    void SceneParser::pushScope(EScope scope)
    {
        assert(hasScope());
        m_scopeStacks.back().push_back(scope);
    }

    // SceneDescription -----------------------------------------------------------------------------------------------
    void SceneDescription::Scale(float sx, float sy, float sz)
    {
        m_graphicsState.ForActiveTransforms(
            [=](dmt::Transform t) { // takes a parameter by copy and returns it, NRVO should kick in (TODO check assembly)
            t.scale_(Vector3f{{sx, sy, sz}});
            return t;
        });
    }

    void SceneDescription::Shape(EShapeType type, ParamMap const& params) {}

    void SceneDescription::Option(sid_t name, ParamPair const&) {}

    void SceneDescription::Identity()
    {
        m_graphicsState.ForActiveTransforms([](dmt::Transform t) { return dmt::Transform{}; });
    }

    void SceneDescription::Translate(float dx, float dy, float dz)
    {
        m_graphicsState.ForActiveTransforms([=](dmt::Transform t) {
            t.translate_(Vector3f{{dx, dy, dz}});
            return t;
        });
    }

    void SceneDescription::Rotate(float angle, float ax, float ay, float az)
    {
        m_graphicsState.ForActiveTransforms([=](dmt::Transform t) {
            t.rotate_(angle, Vector3f{{ax, ay, az}});
            return t;
        });
    }

    void SceneDescription::LookAt(float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz)
    {
        m_graphicsState.ForActiveTransforms([=](dmt::Transform t) {
            t.lookAt_(Vector3f{{ex, ey, ez}}, Vector3f{{lx, ly, lz}}, Vector3f{{ux, uy, uz}});
            return t;
        });
    }

    void SceneDescription::ConcatTransform(std::array<float, 16> const& transform) {}

    void SceneDescription::Transform(std::array<float, 16> transform) {}

    void SceneDescription::CoordinateSystem(sid_t name) {}

    void SceneDescription::CoordSysTransform(sid_t name) {}

    void SceneDescription::ActiveTransformAll()
    {
        m_graphicsState.activeTransformBits = std::numeric_limits<uint32_t>::max();
    }

    void SceneDescription::ActiveTransformEndTime() { m_graphicsState.activeTransformBits = 1 << 1; }

    void SceneDescription::ActiveTransformStartTime() { m_graphicsState.activeTransformBits = 1 << 0; }

    void SceneDescription::TransformTimes(float start, float end)
    {
        //verify option
        m_graphicsState.transformStartTime = start;
        m_graphicsState.transformEndTime   = end;
    }

    void SceneDescription::ColorSpace(EColorSpaceType colorSpace) { m_graphicsState.colorSpace = colorSpace; }

    void SceneDescription::PixelFilter(FilterSpec const& spec) {}

    void SceneDescription::Film(FilmSpec const& spec) {}

    void SceneDescription::Accelerator(AcceleratorSpec const& spec) {}

    void SceneDescription::Integrator(IntegratorSpec const& spec) {}

    void SceneDescription::Camera(CameraSpec const& params)
    {
        //association params and colorSpace
        TransformSet cameraFormWorld                          = m_graphicsState.ctm;
        TransformSet worldFromCamera                          = Inverse(m_graphicsState.ctm);
        m_namedCoordinateSystems[dict::directive::Camera.sid] = worldFromCamera;
        CameraTransform cameraTransform(AnimatedTransform(worldFromCamera[0],
                                                          m_graphicsState.transformStartTime,
                                                          worldFromCamera[1],
                                                          m_graphicsState.transformEndTime),
                                        m_options.renderCoord);
        m_renderFromWorld = cameraTransform.renderFromWorld();
        m_camera          = CameraSceneEntity(params, cameraTransform, m_graphicsState.currentOutsideMedium);
    }

    void SceneDescription::MakeNamedMedium(sid_t name, ParamMap const& params) {}

    void SceneDescription::MediumInterface(sid_t insideName, sid_t outsideName) {}

    void SceneDescription::Sampler(SamplerSpec const& spec) {}

    void SceneDescription::WorldBegin() {}

    void SceneDescription::AttributeBegin() {}

    void SceneDescription::AttributeEnd() {}

    void SceneDescription::Attribute(ETarget target, ParamMap const& params) {}

    void SceneDescription::Texture(sid_t name, ETextureType type, ETextureClass texname, ParamMap const& params) {}

    void SceneDescription::Material(EMaterialType type, ParamMap const& params)
    {
        MaterialEntity material{type, m_graphicsState.colorSpace, params};
        m_materials.push_back(material);
        m_graphicsState.currentMaterialName = 0;
    }

    void SceneDescription::MakeNamedMaterial(sid_t name, ParamMap const& params)
    {
        SceneEntity material{name, params};
        m_materials.push_back(material);
    }

    void SceneDescription::NamedMaterial(sid_t name) {}

    void SceneDescription::LightSource(ELightType type, ParamMap const& params)
    {
        AnimatedTransform const t(m_graphicsState.ctm[0],
                                  m_graphicsState.transformStartTime,
                                  m_graphicsState.ctm[1],
                                  m_graphicsState.transformEndTime);
        LightEntity entity{type, t, m_graphicsState.currentOutsideMedium, m_graphicsState.colorSpace, params};
        m_lights.push_back(entity);
        //m_lights.emplace_back(type, t, m_graphicsState.currentOutsideMedium, m_graphicsState.colorSpace, params);
    }

    void SceneDescription::AreaLightSource(EAreaLightType type, ParamMap const& params) {}

    void SceneDescription::ReverseOrientation()
    {
        //verify_world
        m_graphicsState.reverseOrientation = !m_graphicsState.reverseOrientation;
    }

    void SceneDescription::ObjectBegin(sid_t name) {}

    void SceneDescription::ObjectEnd() {}

    void SceneDescription::ObjectInstance(sid_t name) {}

    void SceneDescription::EndOfOptions(Options const& options) { m_options = options; }

    void SceneDescription::EndOfFiles() {}

    void SceneDescription::EndOfHeader(EndOfHeaderInfo const& info)
    {
        if (m_camera.spec.type == ECameraType::eOrthographic || m_camera.spec.type == ECameraType::ePerspective)
        {
            m_camera.spec.params.p.screenWindow.arr = info.cameraSpec.params.p.screenWindow.arr;
        }
    }

    // Spec Functions -------------------------------------------------------------------------------------------------
    LightSourceSpec::LightSourceSpec(ELightType type, bool illum, float powerOrIlluminance, float scale) :
    scale(scale),
    type(type),
    illum(illum)
    {
        switch (type)
        {
            case ELightType::eDistant:
            {
                std::construct_at(&params.distant);
                break;
            }
            case ELightType::eGoniometric:
            {
                std::construct_at(&params.goniometric);
                break;
            }
            case ELightType::eInfinite:
            {
                std::construct_at(&params.infinite);
                break;
            }
            case ELightType::ePoint:
            {
                std::construct_at(&params.point);
                break;
            }
            case ELightType::eProjection:
            {
                std::construct_at(&params.projection);
                break;
            }
            case ELightType::eSpot:
            {
                std::construct_at(&params.spotlight);
                break;
            }
        }
        if (illum)
        {
            po.illuminance = powerOrIlluminance;
        }
        else
            po.power = powerOrIlluminance;
    }
    // Since realistic camrea stores two filenames as strings, we cannot use memcpy for copy semantics
    LightSourceSpec::LightSourceSpec(LightSourceSpec const& other) :
    scale(other.scale),
    type(other.type),
    illum(other.illum)
    {
        if (other.illum)
            po.illuminance = other.po.illuminance;
        else
            po.power = other.po.power;

        switch (other.type)
        {
            case ELightType::eDistant:
            {
                std::construct_at(&params.distant);
                params.distant = other.params.distant;
                break;
            }
            case ELightType::eGoniometric:
            {
                std::construct_at(&params.goniometric);
                params.goniometric = other.params.goniometric;
                break;
            }
            case ELightType::eInfinite:
            {
                std::construct_at(&params.infinite);
                params.infinite = other.params.infinite;
                break;
            }
            case ELightType::ePoint:
            {
                std::construct_at(&params.point);
                params.point = other.params.point;
                break;
            }
            case ELightType::eProjection:
            {
                params.projection = other.params.projection;
                break;
            }
            case ELightType::eSpot:
            {
                std::construct_at(&params.spotlight);
                params.spotlight = other.params.spotlight;
                break;
            }
        }
    }

    LightSourceSpec::LightSourceSpec(LightSourceSpec&& _that) noexcept :
    scale(std::exchange(_that.scale, 1.f)),
    type(std::exchange(_that.type, ELightType::eCount)),
    illum(std::exchange(_that.illum, false))
    {
        if (illum)
            po.illuminance = _that.po.illuminance;
        else
            po.power = _that.po.power;

        switch (type)
        {
            case ELightType::eDistant:
            {
                std::construct_at(&params.distant);
                params.distant = std::move(_that.params.distant);
                break;
            }
            case ELightType::eGoniometric:
            {
                std::construct_at(&params.goniometric);
                params.goniometric = std::move(_that.params.goniometric);
                break;
            }
            case ELightType::eInfinite:
            {
                std::construct_at(&params.infinite);
                params.infinite = std::move(_that.params.infinite);
                break;
            }
            case ELightType::ePoint:
            {
                std::construct_at(&params.point);
                params.point = std::move(_that.params.point);
                break;
            }
            case ELightType::eProjection:
            {
                std::construct_at(&params.projection);
                params.projection = std::move(_that.params.projection);
                break;
            }
            case ELightType::eSpot:
            {
                std::construct_at(&params.spotlight);
                params.spotlight = std::move(_that.params.spotlight);
                break;
            }
        }
    }

    LightSourceSpec& LightSourceSpec::operator=(LightSourceSpec const& _that)
    {
        if (this != &_that)
        {
            // destroy preexisting object first
            cleanup();

            // then assign the new object
            scale = _that.scale;
            type  = _that.type;
            illum = _that.illum;

            switch (type)
            {
                case ELightType::eDistant:
                {
                    params.distant = _that.params.distant;
                    break;
                }
                case ELightType::eGoniometric:
                {
                    params.goniometric = _that.params.goniometric;
                    break;
                }
                case ELightType::eInfinite:
                {
                    params.infinite = _that.params.infinite;
                    break;
                }
                case ELightType::ePoint:
                {
                    params.point = _that.params.point;
                    break;
                }
                case ELightType::eProjection:
                {
                    params.projection = _that.params.projection;
                    break;
                }
                case ELightType::eSpot:
                {
                    params.spotlight = _that.params.spotlight;
                    break;
                }
            }
        }
        return *this;
    }

    LightSourceSpec& LightSourceSpec::operator=(LightSourceSpec&& _that) noexcept
    {
        if (this != &_that)
        {
            // destroy preexisting object first
            cleanup();

            // then assign the new object
            scale = std::exchange(_that.scale, 1.f);
            type  = std::exchange(_that.type, ELightType::eCount);
            illum = std::exchange(_that.illum, false);

            switch (type)
            {
                case ELightType::eDistant:
                {
                    params.distant = std::move(_that.params.distant);
                    break;
                }
                case ELightType::eGoniometric:
                {
                    params.goniometric = std::move(_that.params.goniometric);
                    break;
                }
                case ELightType::eInfinite:
                {
                    params.infinite = std::move(_that.params.infinite);
                    break;
                }
                case ELightType::ePoint:
                {
                    params.point = std::move(_that.params.point);
                    break;
                }
                case ELightType::eProjection:
                {
                    params.projection = std::move(_that.params.projection);
                    break;
                }
                case ELightType::eSpot:
                {
                    params.spotlight = std::move(_that.params.spotlight);
                    break;
                }
            }

            // then destroy the moved from object
            _that.cleanup();
        }
        return *this;
    }

    LightSourceSpec::~LightSourceSpec() noexcept { cleanup(); }

    void LightSourceSpec::cleanup() noexcept
    {
        switch (type)
        {
            case ELightType::eDistant:
            {
                std::destroy_at(&params.distant);
                break;
            }
            case ELightType::eGoniometric:
            {
                std::destroy_at(&params.goniometric);
                break;
            }
            case ELightType::eInfinite:
            {
                std::destroy_at(&params.infinite);
                break;
            }
            case ELightType::ePoint:
            {

                std::destroy_at(&params.point);
                break;
            }
            case ELightType::eProjection:
            {
                std::destroy_at(&params.projection);
                break;
            }
            case ELightType::eSpot:
            {
                std::destroy_at(&params.spotlight);
                break;
            }
        }
    }
    //MaterialSpec
    MaterialSpec::MaterialSpec(EMaterialType type) : type(type)
    {
        switch (type)
        {
            case EMaterialType::eCoateddiffuse:
            {
                std::construct_at(&params.coateddiffuse);
                break;
            }
            case EMaterialType::eCoatedconductor:
            {
                std::construct_at(&params.coatedconductor);
                break;
            }
            case EMaterialType::eConductor:
            {
                std::construct_at(&params.conductor);
                break;
            }
            case EMaterialType::eDielectric:
            {
                std::construct_at(&params.dielectric);
                break;
            }
            case EMaterialType::eDiffusetransmission:
            {
                std::construct_at(&params.diffuseTransmission);
                break;
            }
            case EMaterialType::eHair:
            {
                std::construct_at(&params.hair);
                break;
            }
            case EMaterialType::eInterface:
            {
                std::construct_at(&params.interface);
                break;
            }
            case EMaterialType::eMeasured:
            {
                std::construct_at(&params.mesured);
                break;
            }
            case EMaterialType::eMix:
            {
                std::construct_at(&params.mix);
                break;
            }
            case EMaterialType::eSubsurface:
            {
                std::construct_at(&params.subsurface);
                break;
            }
            case EMaterialType::eThindielectric:
            {
                std::construct_at(&params.thindielectric);
                break;
            }
        }
    }
    // Since realistic camrea stores two filenames as strings, we cannot use memcpy for copy semantics
    MaterialSpec::MaterialSpec(MaterialSpec const& other) : type(other.type)
    {
        switch (other.type)
        {
            case EMaterialType::eCoateddiffuse:
            {
                std::construct_at(&params.coateddiffuse);
                params.coateddiffuse = other.params.coateddiffuse;
                break;
            }
            case EMaterialType::eCoatedconductor:
            {
                std::construct_at(&params.coatedconductor);
                params.coatedconductor = other.params.coatedconductor;
                break;
            }
            case EMaterialType::eConductor:
            {
                std::construct_at(&params.conductor);
                params.conductor = other.params.conductor;
                break;
            }
            case EMaterialType::eDielectric:
            {
                std::construct_at(&params.dielectric);
                params.dielectric = other.params.dielectric;
                break;
            }
            case EMaterialType::eDiffusetransmission:
            {
                std::construct_at(&params.diffuseTransmission);
                params.diffuseTransmission = other.params.diffuseTransmission;
                break;
            }
            case EMaterialType::eHair:
            {
                std::construct_at(&params.hair);
                params.hair = other.params.hair;
                break;
            }
            case EMaterialType::eInterface:
            {
                std::construct_at(&params.interface);
                params.interface = other.params.interface;
                break;
            }
            case EMaterialType::eMeasured:
            {
                std::construct_at(&params.mesured);
                params.mesured = other.params.mesured;
                break;
            }
            case EMaterialType::eMix:
            {
                std::construct_at(&params.mix);
                params.mix = other.params.mix;
                break;
            }
            case EMaterialType::eSubsurface:
            {
                std::construct_at(&params.subsurface);
                params.subsurface = other.params.subsurface;
                break;
            }
            case EMaterialType::eThindielectric:
            {
                std::construct_at(&params.thindielectric);
                params.thindielectric = other.params.thindielectric;
                break;
            }
        }
    }

    MaterialSpec::MaterialSpec(MaterialSpec&& _that) noexcept : type(std::exchange(_that.type, EMaterialType::eCount))
    {
        switch (type)
        {
            case EMaterialType::eCoateddiffuse:
            {
                std::construct_at(&params.coateddiffuse);
                params.coateddiffuse = std::move(_that.params.coateddiffuse);
                break;
            }
            case EMaterialType::eCoatedconductor:
            {
                std::construct_at(&params.coatedconductor);
                params.coatedconductor = std::move(_that.params.coatedconductor);
                break;
            }
            case EMaterialType::eConductor:
            {
                std::construct_at(&params.conductor);
                params.conductor = std::move(_that.params.conductor);
                break;
            }
            case EMaterialType::eDielectric:
            {
                std::construct_at(&params.dielectric);
                params.dielectric = std::move(_that.params.dielectric);
                break;
            }
            case EMaterialType::eDiffusetransmission:
            {
                std::construct_at(&params.diffuseTransmission);
                params.diffuseTransmission = std::move(_that.params.diffuseTransmission);
                break;
            }
            case EMaterialType::eHair:
            {
                std::construct_at(&params.hair);
                params.hair = std::move(_that.params.hair);
                break;
            }
            case EMaterialType::eInterface:
            {
                std::construct_at(&params.interface);
                params.interface = std::move(_that.params.interface);
                break;
            }
            case EMaterialType::eMeasured:
            {
                std::construct_at(&params.mesured);
                params.mesured = std::move(_that.params.mesured);
                break;
            }
            case EMaterialType::eMix:
            {
                std::construct_at(&params.mix);
                params.mix = std::move(_that.params.mix);
                break;
            }
            case EMaterialType::eSubsurface:
            {
                std::construct_at(&params.subsurface);
                params.subsurface = std::move(_that.params.subsurface);
                break;
            }
            case EMaterialType::eThindielectric:
            {
                std::construct_at(&params.thindielectric);
                params.thindielectric = std::move(_that.params.thindielectric);
                break;
            }
        }
    }

    MaterialSpec& MaterialSpec::operator=(MaterialSpec const& _that)
    {
        if (this != &_that)
        {
            // destroy preexisting object first
            cleanup();

            // then assign the new object
            type = _that.type;

            switch (type)
            {
                case EMaterialType::eCoateddiffuse:
                {
                    params.coateddiffuse = _that.params.coateddiffuse;
                    break;
                }
                case EMaterialType::eCoatedconductor:
                {
                    params.coatedconductor = _that.params.coatedconductor;
                    break;
                }
                case EMaterialType::eConductor:
                {
                    params.conductor = _that.params.conductor;
                    break;
                }
                case EMaterialType::eDielectric:
                {
                    params.dielectric = _that.params.dielectric;
                    break;
                }
                case EMaterialType::eDiffusetransmission:
                {
                    params.diffuseTransmission = _that.params.diffuseTransmission;
                    break;
                }
                case EMaterialType::eHair:
                {
                    params.hair = _that.params.hair;
                    break;
                }
                case EMaterialType::eInterface:
                {
                    params.interface = _that.params.interface;
                    break;
                }
                case EMaterialType::eMeasured:
                {
                    params.mesured = _that.params.mesured;
                    break;
                }
                case EMaterialType::eMix:
                {
                    params.mix = _that.params.mix;
                    break;
                }
                case EMaterialType::eSubsurface:
                {
                    params.subsurface = _that.params.subsurface;
                    break;
                }
                case EMaterialType::eThindielectric:
                {
                    params.thindielectric = _that.params.thindielectric;
                    break;
                }
            }
        }
        return *this;
    }

    MaterialSpec& MaterialSpec::operator=(MaterialSpec&& _that) noexcept
    {
        if (this != &_that)
        {
            // destroy preexisting object first
            cleanup();

            // then assign the new object
            type = std::exchange(_that.type, EMaterialType::eCount);

            switch (type)
            {
                case EMaterialType::eCoateddiffuse:
                {
                    params.coateddiffuse = std::move(_that.params.coateddiffuse);
                    break;
                }
                case EMaterialType::eCoatedconductor:
                {
                    params.coatedconductor = std::move(_that.params.coatedconductor);
                    break;
                }
                case EMaterialType::eConductor:
                {
                    params.conductor = std::move(_that.params.conductor);
                    break;
                }
                case EMaterialType::eDielectric:
                {
                    params.dielectric = std::move(_that.params.dielectric);
                    break;
                }
                case EMaterialType::eDiffusetransmission:
                {
                    params.diffuseTransmission = std::move(_that.params.diffuseTransmission);
                    break;
                }
                case EMaterialType::eHair:
                {
                    params.hair = std::move(_that.params.hair);
                    break;
                }
                case EMaterialType::eInterface:
                {
                    params.interface = std::move(_that.params.interface);
                    break;
                }
                case EMaterialType::eMeasured:
                {
                    params.mesured = std::move(_that.params.mesured);
                    break;
                }
                case EMaterialType::eMix:
                {
                    params.mix = std::move(_that.params.mix);
                    break;
                }
                case EMaterialType::eSubsurface:
                {
                    params.subsurface = std::move(_that.params.subsurface);
                    break;
                }
                case EMaterialType::eThindielectric:
                {
                    params.thindielectric = std::move(_that.params.thindielectric);
                    break;
                }
            }

            // then destroy the moved from object
            _that.cleanup();
        }
        return *this;
    }

    MaterialSpec::~MaterialSpec() noexcept { cleanup(); }

    void MaterialSpec::cleanup() noexcept
    {
        switch (type)
        {
            case EMaterialType::eCoateddiffuse:
            {
                std::destroy_at(&params.coateddiffuse);
                break;
            }
            case EMaterialType::eCoatedconductor:
            {
                std::destroy_at(&params.coatedconductor);
                break;
            }
            case EMaterialType::eConductor:
            {
                std::destroy_at(&params.conductor);
                break;
            }
            case EMaterialType::eDielectric:
            {
                std::destroy_at(&params.dielectric);
                break;
            }
            case EMaterialType::eDiffusetransmission:
            {
                std::destroy_at(&params.diffuseTransmission);
                break;
            }
            case EMaterialType::eHair:
            {
                std::destroy_at(&params.hair);
                break;
            }
            case EMaterialType::eInterface:
            {
                std::destroy_at(&params.interface);
                break;
            }
            case EMaterialType::eMeasured:
            {
                std::destroy_at(&params.mesured);
                break;
            }
            case EMaterialType::eMix:
            {
                std::destroy_at(&params.mix);
                break;
            }
            case EMaterialType::eSubsurface:
            {
                std::destroy_at(&params.subsurface);
                break;
            }
            case EMaterialType::eThindielectric:
            {
                std::destroy_at(&params.thindielectric);
                break;
            }
        }
    }

    //MaterialS----------------------------------------------------------
    MaterialS::MaterialS(EMaterialType type) : type(type) {}
    // CameraSpec -------------------------------------------------------
    CameraSpec::CameraSpec(CameraSpec const& other) :
    shutteropen(other.shutteropen),
    shutterclose(other.shutterclose),
    type(other.type)
    {
        assignParams(other.params, false);
    }

    CameraSpec::CameraSpec(CameraSpec&& that) noexcept :
    shutteropen(that.shutteropen),
    shutterclose(that.shutterclose),
    type(that.type)
    {
        assignParams(that.params, true);
    }

    void CameraSpec::assignParams(Params const& that, bool move)
    {
        static_assert(std::is_trivial_v<Spherical>);
        static_assert(std::is_trivially_copyable_v<Projecting>);
        switch (type)
        {
            case ECameraType::eOrthographic: [[fallthrough]];
            case ECameraType::ePerspective: params.p = that.p; break;
            case ECameraType::eRealistic: // contains two std::string
                if (move)
                    params.r = std::move(that.r);
                else
                    params.r = that.r;
                break;
            case ECameraType::eSpherical: params.s = that.s; break;
            default: assert(false); break;
        }
    }

    CameraSpec& CameraSpec::operator=(CameraSpec const& that)
    {
        if (this != &that)
        {
            shutteropen  = that.shutteropen;
            shutterclose = that.shutterclose;
            type         = that.type;
            assignParams(that.params, false);
        }
        return *this;
    }

    CameraSpec& CameraSpec::operator=(CameraSpec&& that) noexcept
    {
        if (this != &that)
        {
            shutteropen  = that.shutteropen;
            shutterclose = that.shutterclose;
            type         = that.type;
            assignParams(that.params, true);
        }
        return *this;
    }

    CameraSpec::~CameraSpec() noexcept
    {
        if (type == ECameraType::eRealistic)
        {
            std::destroy_at(&params.r.lensfile);
            std::destroy_at(&params.r.aperture);
        }
    }
} // namespace dmt