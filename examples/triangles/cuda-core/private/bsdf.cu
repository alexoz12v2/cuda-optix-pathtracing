#include "bsdf.cuh"

#include "extra_math.cuh"
#include "sampling.cuh"

#include <cooperative_groups.h>

#include <cassert>
#include <numbers>

namespace cg = cooperative_groups;

static int constexpr GGX_E_TABLE_COLS = 32;
static int constexpr GGX_E_TABLE_ROWS = 1024 / GGX_E_TABLE_COLS;
static constexpr float table_ggx_E[GGX_E_TABLE_ROWS * GGX_E_TABLE_COLS]{
    1.000000f, 0.980405f, 0.994967f, 0.997749f, 0.998725f, 0.999173f, 0.999411f,
    0.999550f, 0.999634f, 0.999686f, 0.999716f, 0.999732f, 0.999738f, 0.999735f,
    0.999726f, 0.999712f, 0.999693f, 0.999671f, 0.999645f, 0.999615f, 0.999583f,
    0.999548f, 0.999511f, 0.999471f, 0.999429f, 0.999385f, 0.999338f, 0.999290f,
    0.999240f, 0.999188f, 0.999134f, 0.999079f, 1.000000f, 0.999451f, 0.990086f,
    0.954714f, 0.911203f, 0.891678f, 0.893893f, 0.905010f, 0.917411f, 0.928221f,
    0.936755f, 0.943104f, 0.947567f, 0.950455f, 0.952036f, 0.952526f, 0.952096f,
    0.950879f, 0.948983f, 0.946495f, 0.943484f, 0.940010f, 0.936122f, 0.931864f,
    0.927272f, 0.922380f, 0.917217f, 0.911810f, 0.906184f, 0.900361f, 0.894361f,
    0.888202f, 1.000000f, 0.999866f, 0.997676f, 0.987331f, 0.962386f, 0.929174f,
    0.902886f, 0.890270f, 0.888687f, 0.893114f, 0.899716f, 0.906297f, 0.911810f,
    0.915853f, 0.918345f, 0.919348f, 0.918982f, 0.917380f, 0.914671f, 0.910973f,
    0.906393f, 0.901026f, 0.894957f, 0.888261f, 0.881006f, 0.873254f, 0.865061f,
    0.856478f, 0.847553f, 0.838329f, 0.828845f, 0.819138f, 1.000000f, 0.999941f,
    0.998997f, 0.994519f, 0.982075f, 0.959460f, 0.931758f, 0.907714f, 0.892271f,
    0.885248f, 0.884058f, 0.885962f, 0.888923f, 0.891666f, 0.893488f, 0.894054f,
    0.893246f, 0.891062f, 0.887565f, 0.882845f, 0.877003f, 0.870143f, 0.862365f,
    0.853766f, 0.844436f, 0.834459f, 0.823915f, 0.812876f, 0.801410f, 0.789581f,
    0.777445f, 0.765057f, 1.000000f, 0.999967f, 0.999442f, 0.996987f, 0.989925f,
    0.975437f, 0.953654f, 0.929078f, 0.907414f, 0.891877f, 0.882635f, 0.878166f,
    0.876572f, 0.876236f, 0.876004f, 0.875144f, 0.873238f, 0.870080f, 0.865601f,
    0.859812f, 0.852774f, 0.844571f, 0.835302f, 0.825069f, 0.813976f, 0.802123f,
    0.789609f, 0.776523f, 0.762954f, 0.748982f, 0.734682f, 0.720122f, 1.000000f,
    0.999979f, 0.999644f, 0.998096f, 0.993618f, 0.983950f, 0.967765f, 0.946488f,
    0.923989f, 0.904160f, 0.889039f, 0.878704f, 0.872099f, 0.867839f, 0.864665f,
    0.861611f, 0.858015f, 0.853468f, 0.847745f, 0.840752f, 0.832480f, 0.822973f,
    0.812312f, 0.800592f, 0.787922f, 0.774411f, 0.760171f, 0.745308f, 0.729926f,
    0.714121f, 0.697985f, 0.681600f, 1.000000f, 0.999985f, 0.999752f, 0.998684f,
    0.995603f, 0.988803f, 0.976727f, 0.959272f, 0.938451f, 0.917446f, 0.898932f,
    0.884154f, 0.873043f, 0.864773f, 0.858278f, 0.852565f, 0.846845f, 0.840555f,
    0.833331f, 0.824969f, 0.815378f, 0.804550f, 0.792533f, 0.779405f, 0.765271f,
    0.750246f, 0.734447f, 0.717996f, 0.701008f, 0.683595f, 0.665862f, 0.647905f,
    1.000000f, 0.999989f, 0.999816f, 0.999032f, 0.996781f, 0.991766f, 0.982561f,
    0.968421f, 0.950089f, 0.929705f, 0.909775f, 0.892112f, 0.877428f, 0.865535f,
    0.855737f, 0.847185f, 0.839084f, 0.830796f, 0.821858f, 0.811969f, 0.800961f,
    0.788766f, 0.775391f, 0.760894f, 0.745367f, 0.728923f, 0.711685f, 0.693783f,
    0.675344f, 0.656491f, 0.637343f, 0.618009f, 1.000000f, 0.999991f, 0.999858f,
    0.999255f, 0.997533f, 0.993686f, 0.986490f, 0.974988f, 0.959173f, 0.940264f,
    0.920251f, 0.901030f, 0.883794f, 0.868900f, 0.856078f, 0.844728f, 0.834154f,
    0.823720f, 0.812914f, 0.801368f, 0.788846f, 0.775227f, 0.760476f, 0.744623f,
    0.727744f, 0.709946f, 0.691355f, 0.672105f, 0.652332f, 0.632172f, 0.611753f,
    0.591195f, 1.000000f, 0.999993f, 0.999886f, 0.999406f, 0.998040f, 0.994992f,
    0.989227f, 0.979769f, 0.966203f, 0.949069f, 0.929766f, 0.909990f, 0.891126f,
    0.873921f, 0.858498f, 0.844545f, 0.831539f, 0.818909f, 0.806146f, 0.792847f,
    0.778733f, 0.763633f, 0.747476f, 0.730264f, 0.712056f, 0.692949f, 0.673066f,
    0.652547f, 0.631533f, 0.610170f, 0.588596f, 0.566939f, 1.000000f, 0.999995f,
    0.999906f, 0.999513f, 0.998399f, 0.995917f, 0.991195f, 0.983312f, 0.971656f,
    0.956303f, 0.938125f, 0.918488f, 0.898757f, 0.879901f, 0.862353f, 0.846089f,
    0.830787f, 0.815992f, 0.801242f, 0.786135f, 0.770368f, 0.753741f, 0.736147f,
    0.717566f, 0.698037f, 0.677649f, 0.656520f, 0.634791f, 0.612611f, 0.590130f,
    0.567495f, 0.544843f, 1.000000f, 0.999996f, 0.999921f, 0.999591f, 0.998661f,
    0.996594f, 0.992650f, 0.985988f, 0.975917f, 0.962217f, 0.945336f, 0.926280f,
    0.906266f, 0.886338f, 0.867144f, 0.848905f, 0.831508f, 0.814643f, 0.797929f,
    0.780995f, 0.763540f, 0.745347f, 0.726289f, 0.706324f, 0.685477f, 0.663824f,
    0.641483f, 0.618591f, 0.595303f, 0.571774f, 0.548156f, 0.524596f, 1.000000f,
    0.999996f, 0.999933f, 0.999651f, 0.998859f, 0.997104f, 0.993752f, 0.988046f,
    0.979280f, 0.967055f, 0.951498f, 0.933282f, 0.913407f, 0.892888f, 0.872490f,
    0.852624f, 0.833371f, 0.814578f, 0.795964f, 0.777219f, 0.758062f, 0.738279f,
    0.717734f, 0.696371f, 0.674202f, 0.651296f, 0.627765f, 0.603747f, 0.579398f,
    0.554878f, 0.530346f, 0.505951f, 1.000000f, 0.999997f, 0.999941f, 0.999696f,
    0.999012f, 0.997497f, 0.994604f, 0.989656f, 0.981964f, 0.971026f, 0.956742f,
    0.939495f, 0.920052f, 0.899322f, 0.878109f, 0.856955f, 0.836102f, 0.815549f,
    0.795133f, 0.774620f, 0.753771f, 0.732389f, 0.710341f, 0.687567f, 0.664071f,
    0.639918f, 0.615213f, 0.590097f, 0.564725f, 0.539263f, 0.513871f, 0.488706f,
    1.000000f, 0.999997f, 0.999949f, 0.999733f, 0.999132f, 0.997806f, 0.995276f,
    0.990935f, 0.984129f, 0.974304f, 0.961200f, 0.944967f, 0.926144f, 0.905496f,
    0.883799f, 0.861667f, 0.839472f, 0.817348f, 0.795251f, 0.773035f, 0.750522f,
    0.727545f, 0.703988f, 0.679793f, 0.654965f, 0.629565f, 0.603699f, 0.577506f,
    0.551143f, 0.524778f, 0.498576f, 0.472695f, 1.000000f, 0.999997f, 0.999954f,
    0.999762f, 0.999228f, 0.998053f, 0.995814f, 0.991966f, 0.985893f, 0.977026f,
    0.964995f, 0.949768f, 0.931677f, 0.911324f, 0.889415f, 0.866586f, 0.843295f,
    0.819794f, 0.796153f, 0.772321f, 0.748187f, 0.723632f, 0.698566f, 0.672945f,
    0.646780f, 0.620134f, 0.593114f, 0.565860f, 0.538532f, 0.511298f, 0.484328f,
    0.457779f, 1.000000f, 0.999998f, 0.999959f, 0.999786f, 0.999307f, 0.998254f,
    0.996252f, 0.992808f, 0.987348f, 0.979302f, 0.968235f, 0.953973f, 0.936669f,
    0.916763f, 0.894859f, 0.871576f, 0.847421f, 0.822737f, 0.797698f, 0.772347f,
    0.746651f, 0.720547f, 0.693982f, 0.666934f, 0.639428f, 0.611535f, 0.583366f,
    0.555065f, 0.526792f, 0.498719f, 0.471015f, 0.443841f, 1.000000f, 0.999998f,
    0.999962f, 0.999805f, 0.999371f, 0.998420f, 0.996612f, 0.993502f, 0.988557f,
    0.981218f, 0.971011f, 0.957656f, 0.941156f, 0.921798f, 0.900070f, 0.876539f,
    0.851730f, 0.826051f, 0.799763f, 0.773002f, 0.745814f, 0.718199f, 0.690150f,
    0.661679f, 0.632832f, 0.603691f, 0.574377f, 0.545037f, 0.515837f, 0.486949f,
    0.458544f, 0.430781f, 1.000000f, 0.999998f, 0.999966f, 0.999822f, 0.999425f,
    0.998558f, 0.996912f, 0.994082f, 0.989572f, 0.982843f, 0.973398f, 0.960884f,
    0.945180f, 0.926433f, 0.905008f, 0.881402f, 0.856128f, 0.829631f, 0.802244f,
    0.774186f, 0.745583f, 0.716504f, 0.686997f, 0.657112f, 0.626924f, 0.596535f,
    0.566077f, 0.535706f, 0.505592f, 0.475910f, 0.446831f, 0.418514f, 1.000000f,
    0.999998f, 0.999968f, 0.999836f, 0.999471f, 0.998674f, 0.997165f, 0.994570f,
    0.990430f, 0.984229f, 0.975460f, 0.963718f, 0.948785f, 0.930682f, 0.909656f,
    0.886116f, 0.860541f, 0.833390f, 0.805050f, 0.775811f, 0.745878f, 0.715390f,
    0.684454f, 0.653169f, 0.621644f, 0.590007f, 0.558407f, 0.527011f, 0.495995f,
    0.465536f, 0.435808f, 0.406965f, 1.000000f, 0.999999f, 0.999970f, 0.999848f,
    0.999509f, 0.998773f, 0.997379f, 0.994985f, 0.991163f, 0.985419f, 0.977249f,
    0.966213f, 0.952014f, 0.934568f, 0.914006f, 0.890646f, 0.864912f, 0.837258f,
    0.808105f, 0.777803f, 0.746627f, 0.714789f, 0.682460f, 0.649794f, 0.616939f,
    0.584055f, 0.551314f, 0.518896f, 0.486987f, 0.455768f, 0.425410f, 0.396069f,
    1.000000f, 0.999999f, 0.999973f, 0.999858f, 0.999542f, 0.998857f, 0.997562f,
    0.995341f, 0.991792f, 0.986447f, 0.978809f, 0.968414f, 0.954908f, 0.938115f,
    0.918063f, 0.894970f, 0.869199f, 0.841178f, 0.811344f, 0.780093f, 0.747765f,
    0.714642f, 0.680962f, 0.646935f, 0.612760f, 0.578633f, 0.544751f, 0.511315f,
    0.478520f, 0.446553f, 0.415586f, 0.385770f, 1.000000f, 0.999999f, 0.999974f,
    0.999867f, 0.999571f, 0.998930f, 0.997720f, 0.995648f, 0.992336f, 0.987340f,
    0.980174f, 0.970361f, 0.957504f, 0.941351f, 0.921833f, 0.899078f, 0.873371f,
    0.845104f, 0.814712f, 0.782625f, 0.749237f, 0.714896f, 0.679909f, 0.644547f,
    0.609064f, 0.573697f, 0.538677f, 0.504225f, 0.470550f, 0.437846f, 0.406286f,
    0.376017f, 1.000000f, 0.999999f, 0.999976f, 0.999874f, 0.999596f, 0.998993f,
    0.997858f, 0.995914f, 0.992810f, 0.988121f, 0.981374f, 0.972090f, 0.959837f,
    0.944301f, 0.925331f, 0.902963f, 0.877406f, 0.848999f, 0.818163f, 0.785347f,
    0.750991f, 0.715502f, 0.679256f, 0.642589f, 0.605812f, 0.569211f, 0.533054f,
    0.497587f, 0.463038f, 0.429607f, 0.397469f, 0.366766f, 1.000000f, 0.999999f,
    0.999977f, 0.999881f, 0.999618f, 0.999049f, 0.997978f, 0.996147f, 0.993224f,
    0.988806f, 0.982434f, 0.973628f, 0.961935f, 0.946991f, 0.928572f, 0.906628f,
    0.881288f, 0.852834f, 0.821659f, 0.788217f, 0.752981f, 0.716417f, 0.678962f,
    0.641022f, 0.602968f, 0.565141f, 0.527849f, 0.491370f, 0.455949f, 0.421800f,
    0.389096f, 0.357977f, 1.000000f, 0.999999f, 0.999978f, 0.999887f, 0.999637f,
    0.999097f, 0.998083f, 0.996352f, 0.993588f, 0.989410f, 0.983373f, 0.975002f,
    0.963827f, 0.949446f, 0.931571f, 0.910076f, 0.885009f, 0.856589f, 0.825169f,
    0.791196f, 0.755170f, 0.717601f, 0.678991f, 0.639812f, 0.600501f, 0.561455f,
    0.523031f, 0.485541f, 0.449253f, 0.414392f, 0.381135f, 0.349616f, 1.000000f,
    0.999999f, 0.999979f, 0.999892f, 0.999655f, 0.999141f, 0.998177f, 0.996533f,
    0.993910f, 0.989945f, 0.984209f, 0.976232f, 0.965536f, 0.951688f, 0.934346f,
    0.913314f, 0.888564f, 0.860245f, 0.828665f, 0.794253f, 0.757521f, 0.719020f,
    0.679309f, 0.638927f, 0.598380f, 0.558126f, 0.518574f, 0.480074f, 0.442922f,
    0.407354f, 0.373554f, 0.341651f, 1.000000f, 0.999999f, 0.999980f, 0.999897f,
    0.999669f, 0.999179f, 0.998260f, 0.996693f, 0.994197f, 0.990422f, 0.984956f,
    0.977337f, 0.967083f, 0.953737f, 0.936913f, 0.916351f, 0.891951f, 0.863792f,
    0.832128f, 0.797360f, 0.760004f, 0.720642f, 0.679885f, 0.638338f, 0.596578f,
    0.555128f, 0.514452f, 0.474944f, 0.436929f, 0.400661f, 0.366328f, 0.334053f,
    1.000000f, 0.999999f, 0.999981f, 0.999901f, 0.999683f, 0.999213f, 0.998334f,
    0.996836f, 0.994452f, 0.990848f, 0.985625f, 0.978332f, 0.968486f, 0.955612f,
    0.939288f, 0.919197f, 0.895172f, 0.867221f, 0.835539f, 0.800494f, 0.762592f,
    0.722438f, 0.680691f, 0.638020f, 0.595071f, 0.552438f, 0.510643f, 0.470129f,
    0.431253f, 0.394289f, 0.359430f, 0.326796f, 1.000000f, 0.999999f, 0.999982f,
    0.999905f, 0.999695f, 0.999244f, 0.998400f, 0.996965f, 0.994681f, 0.991230f,
    0.986227f, 0.979231f, 0.969761f, 0.957330f, 0.941486f, 0.921863f, 0.898229f,
    0.870526f, 0.838887f, 0.803634f, 0.765260f, 0.724383f, 0.681702f, 0.637948f,
    0.593837f, 0.550034f, 0.507126f, 0.465607f, 0.425873f, 0.388216f, 0.352839f,
    0.319858f, 1.000000f, 0.999999f, 0.999982f, 0.999908f, 0.999706f, 0.999271f,
    0.998459f, 0.997080f, 0.994886f, 0.991574f, 0.986770f, 0.980045f, 0.970922f,
    0.958906f, 0.943521f, 0.924358f, 0.901128f, 0.873705f, 0.842159f, 0.806765f,
    0.767988f, 0.726453f, 0.682895f, 0.638100f, 0.592854f, 0.547896f, 0.503881f,
    0.461361f, 0.420769f, 0.382424f, 0.346535f, 0.313216f, 1.000000f, 0.999999f,
    0.999983f, 0.999911f, 0.999716f, 0.999296f, 0.998513f, 0.997184f, 0.995072f,
    0.991884f, 0.987261f, 0.980785f, 0.971982f, 0.960355f, 0.945407f, 0.926694f,
    0.903873f, 0.876756f, 0.845349f, 0.809871f, 0.770757f, 0.728630f, 0.684249f,
    0.638454f, 0.592102f, 0.546006f, 0.500893f, 0.457373f, 0.415925f, 0.376893f,
    0.340499f, 0.306853f};

static int constexpr GGX_EAVG_TABLE_COUNT = 32;
static constexpr float table_ggx_Eavg[GGX_EAVG_TABLE_COUNT] = {
    1.000000f, 0.999992f, 0.999897f, 0.999548f, 0.998729f, 0.997199f, 0.994703f,
    0.990986f, 0.985805f, 0.978930f, 0.970160f, 0.959321f, 0.946279f, 0.930937f,
    0.913247f, 0.893209f, 0.870874f, 0.846345f, 0.819774f, 0.791360f, 0.761345f,
    0.730001f, 0.697631f, 0.664547f, 0.631068f, 0.597509f, 0.564165f, 0.531311f,
    0.499191f, 0.468013f, 0.437950f, 0.409137f};

// table_ggx_E[1024] → 4 KB
// table_ggx_Eavg[32] → 128 B
// | Access pattern         | Best choice?   |
// | ---------------------- | -------------- |
// | Coherent, same index   | `__constant__` |
// | Semi-random            | `__ldg`        |
// | Random + interpolation | **Texture**    |
// | 2D LUT                 | **Texture**    |

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory
// https://stackoverflow.com/questions/14450029/cudaarray-vs-device-pointer
// static __constant__ float d_table_ggx_E[GGX_E_TABLE_COUNT];
// static __constant__ float d_table_ggx_Eavg[GGX_EAVG_TABLE_COUNT];

static cudaArray_t array_ggx_E;
static cudaArray_t array_ggx_Eavg;

static cudaTextureObject_t tex_ggx_E = 0;
static cudaTextureObject_t tex_ggx_Eavg = 0;

static __constant__ cudaTextureObject_t d_tex_ggx_E = 0;
static __constant__ cudaTextureObject_t d_tex_ggx_Eavg = 0;

__host__ void allocateDeviceGGXEnergyPreservingTables() {
  // 1. allocate array backings for our textures. Arrays are 2D matrices.
  cudaChannelFormatDesc const desc = cudaCreateChannelDesc<float>();
  CUDA_CHECK(cudaMallocArray(&array_ggx_Eavg, &desc, GGX_EAVG_TABLE_COUNT, 1));
  CUDA_CHECK(
      cudaMallocArray(&array_ggx_E, &desc, GGX_E_TABLE_COLS, GGX_E_TABLE_ROWS));

  CUDA_CHECK(cudaMemcpy2DToArray(array_ggx_Eavg, 0, 0, table_ggx_Eavg,
                                 GGX_EAVG_TABLE_COUNT * sizeof(float),
                                 GGX_EAVG_TABLE_COUNT * sizeof(float), 1,
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy2DToArray(array_ggx_E, 0, 0, table_ggx_E,
                                 GGX_E_TABLE_COLS * sizeof(float),
                                 GGX_E_TABLE_COLS * sizeof(float),
                                 GGX_E_TABLE_ROWS, cudaMemcpyHostToDevice));
  // 2. create texture object with linear filtering
  {
    cudaResourceDesc ggxEavgDesc{};
    ggxEavgDesc.resType = cudaResourceTypeArray;  // backing is array
    ggxEavgDesc.res.array.array = array_ggx_Eavg;

    cudaTextureDesc ggxEavgTexDesc{};                   // clamp is default
    ggxEavgTexDesc.readMode = cudaReadModeElementType;  // should be floats
    ggxEavgTexDesc.filterMode = cudaFilterModeLinear;
    ggxEavgTexDesc.normalizedCoords = true;
    CUDA_CHECK(cudaCreateTextureObject(&tex_ggx_Eavg, &ggxEavgDesc,
                                       &ggxEavgTexDesc, nullptr));
  }
  {
    cudaResourceDesc ggxEDesc{};
    ggxEDesc.resType = cudaResourceTypeArray;
    ggxEDesc.res.array.array = array_ggx_E;

    cudaTextureDesc ggxEDescTexDesc{};
    ggxEDescTexDesc.readMode = cudaReadModeElementType;
    ggxEDescTexDesc.filterMode = cudaFilterModeLinear;
    ggxEDescTexDesc.normalizedCoords = true;
    CUDA_CHECK(cudaCreateTextureObject(&tex_ggx_E, &ggxEDesc, &ggxEDescTexDesc,
                                       nullptr));
  }
  // 3. Upload texture handles to device constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_tex_ggx_Eavg, &tex_ggx_Eavg,
                                sizeof(cudaTextureObject_t)));

  CUDA_CHECK(
      cudaMemcpyToSymbol(d_tex_ggx_E, &tex_ggx_E, sizeof(cudaTextureObject_t)));
}

__host__ void freeDeviceGGXEnergyPreservingTables() {
  // 0. Clear device-side constant texture handles
  cudaTextureObject_t zero = 0;
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_tex_ggx_Eavg, &zero, sizeof(cudaTextureObject_t)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(d_tex_ggx_E, &zero, sizeof(cudaTextureObject_t)));

  // 1. Destroy texture objects (must be done before freeing arrays)
  if (tex_ggx_Eavg) {
    CUDA_CHECK(cudaDestroyTextureObject(tex_ggx_Eavg));
    tex_ggx_Eavg = 0;
  }

  if (tex_ggx_E) {
    CUDA_CHECK(cudaDestroyTextureObject(tex_ggx_E));
    tex_ggx_E = 0;
  }

  // 2. Free backing CUDA arrays
  if (array_ggx_Eavg) {
    CUDA_CHECK(cudaFreeArray(array_ggx_Eavg));
    array_ggx_Eavg = nullptr;
  }

  if (array_ggx_E) {
    CUDA_CHECK(cudaFreeArray(array_ggx_E));
    array_ggx_E = nullptr;
  }
}

// ---------------------------------------------------------------------------
// BSDF-Specific sampling/evaluation functions
// ---------------------------------------------------------------------------

namespace {
__host__ __device__ __forceinline__ float3 tangentFromPhi(const float3 ns,
                                                          float const phi0) {
  // Choose a reference vector not parallel to ns
  float3 const ref = fabsf(ns.x) < 0.999f ? make_float3(1.0f, 0.0f, 0.0f)
                                          : make_float3(0.0f, 1.0f, 0.0f);

  // Build an orthonormal basis around ns
  float3 const t = normalize(cross(ref, ns));
  float3 const b = cross(ns, t);

  // Rotate in tangent plane
  float const s = sinf(phi0);
  float const c = cosf(phi0);

  return c * t + s * b;
}

// simple branches as this should be compiled in PTX/SASS as conditional
// instructions. TODO Check
__host__ __device__ __forceinline__ float3 faceForward(const float3 n,
                                                       const float3 v) {
  return dot(n, v) < 0.0f ? -n : n;
}

__host__ __device__ __forceinline__ float3 sampleGGX_VNDF(float3 wo, float2 u,
                                                          float ax, float ay) {
  // stretch view (anisotropic roughness)
  float3 const V = normalize(make_float3(ax * wo.x, ay * wo.y, wo.z));

  // orthonormal basis (Frame) (azimuthal only)
  float3 T1, T2;
  if (float const lensq = V.x * V.x + V.y * V.y; lensq > 1e-7f) {
    float const invLen = rsqrtf(lensq);
    T1 = make_float3(-V.y * invLen, V.x * invLen, 0.f);
    T2 = cross(V, T1);
  } else {
    T1 = make_float3(1, 0, 0);
    T2 = make_float3(0, 1, 0);
  }

  // sample disk
  float2 t = sampleUniformDisk(u);
  t.y = lerp(safeSqrt(1.f - t.x * t.x), t.y, 0.5f * (1.f + V.z));

  // recombine and unstretch (disk_to_hemisphere + to_global)
  float3 Nh = t.x * T1 + t.y * T2 + safeSqrt(1.f - dot(t, t)) * V;

  // transform normal back to ellipsoid configuration
  Nh = normalize(make_float3(ax * Nh.x, ay * Nh.y, fmaxf(0.f, Nh.z)));
  return Nh;
}

__host__ __device__ __forceinline__ void microfacetFresnel(
    BSDF const& bsdf, float const cos_HO, float* cos_HI, float3* reflectance,
    float3* transmittance) {
  EBSDFType const type = bsdf.type();
#ifndef __CUDA_ARCH__
  assert(type == EBSDFType::eGGXConductor || type == EBSDFType::eGGXDielectric);
#endif
  // TODO how can we improve this branching?
  if (type == EBSDFType::eGGXDielectric) {  // dielectric tint
    auto const& dielectric = bsdf.data.ggx.mat.dielectric;
    float const F = reflectanceFresnelDielectric(
        cos_HO, half_bits_to_float(dielectric.eta), cos_HI);
    *reflectance = F * half_vec_to_float3(dielectric.reflectanceTint);
    *transmittance =
        (1.f - F) * half_vec_to_float3(dielectric.transmittanceTint);
  } else {  // conductor
    auto const& conductor = bsdf.data.ggx.mat.conductor;
    // doesn't populate cosine as it's cosTheta_T, hence used only on trans.
    *reflectance =
        reflectanceFresnelConductor(cos_HO, half_vec_to_float3(conductor.eta),
                                    half_vec_to_float3(conductor.kappa));
    *transmittance = make_float3(0, 0, 0);
  }
}

// assumes normal is properly oriented, hence dot(n,i) > 0
// starts with cosThetaT already computed
__host__ __device__ __forceinline__ float3 refractAngle(float3 const incident,
                                                        float3 const normal,
                                                        float const cosThetaT,
                                                        float const invEta) {
  return (invEta * dot(normal, incident) + cosThetaT) * normal -
         invEta * incident;
}

// ---------------------------------------------------------------------------
// BSDF: Microfacet with Generalized Torrance Sparrow and GGX NDF
// ---------------------------------------------------------------------------

// common computation between isotropic and anisotropic auxiliary function
// https://jcgt.org/published/0003/02/03/
__host__ __device__ __forceinline__ float ggx_lambda_from_sqr_alpha_tan_n(
    float const sqr_alpha_tan_n) {
  return 0.5f * (sqrtf(1.f + sqr_alpha_tan_n) - 1.f);
}

__host__ __device__ __forceinline__ float ggx_D(float const alpha2,
                                                float const cos_NH) {
  float const cos_NH2 = fminf(cos_NH * cos_NH, 1.f);
  float const one_minus_cos_NH2 = 1.f - cos_NH2;
  return alpha2 / (std::numbers::pi_v<float> *
                   sqrf(one_minus_cos_NH2 + alpha2 * cos_NH2));
}
__host__ __device__ __forceinline__ float ggx_lambda(float const alpha2,
                                                     float const cos_N) {
  float const sqr_alpha_tan_n = alpha2 * fmaxf(0.f, 1.f / sqrf(cos_N));
  return ggx_lambda_from_sqr_alpha_tan_n(sqr_alpha_tan_n);
}
__host__ __device__ __forceinline__ float ggx_aniso_D(float const alphax,
                                                      float const alphay,
                                                      float3 local_H) {
  local_H /= make_float3(alphax, alphay, 1.f);
  // float const cos_NH2 = sqrf(local_H.z); // beckmann only
  float const alpha2 = alphax * alphay;
  return std::numbers::inv_pi_v<float> / (alpha2 * sqrf(dot(local_H, local_H)));
}
__host__ __device__ __forceinline__ float ggx_aniso_lambda(float const alphax,
                                                           float const alphay,
                                                           float3 const V) {
  float const sqr_alpha_tan_n =
      (sqrf(alphax * V.x) + sqrf(alphay * V.y)) / sqrf(V.z);
  return ggx_lambda_from_sqr_alpha_tan_n(sqr_alpha_tan_n);
}

// TODO unit test
// assumes albedo already estimated
__host__ __device__ __forceinline__ void energyPreservingGGXScale(
    BSDF& bsdf, float const alpha2, float const cos_NO, float3 const Fss) {
#ifndef __CUDA_ARCH__
  float const E = lookupTableRead2D(table_ggx_E, alpha2, cos_NO,
                                    GGX_E_TABLE_COLS, GGX_E_TABLE_ROWS);
  float const Eavg =
      lookupTableRead(table_ggx_Eavg, alpha2, GGX_EAVG_TABLE_COUNT);
#else
  float const E = tex2D<float>(d_tex_ggx_E, alpha2, cos_NO);
  float const Eavg = tex1D<float>(d_tex_ggx_Eavg, alpha2);
#endif

  // assumes that Single Scattering doesn't cover the full energy
  float const missingFactor = (1.f - E) / E;
  bsdf.data.ggx.energyScale = 1.f + missingFactor;
  float3 const Fms = Fss * Eavg / (make_float3(1, 1, 1) - Fss * (1.f - Eavg));

  bsdf.setWeight(bsdf.weight() * (1.f + Fms * missingFactor) /
                 bsdf.data.ggx.energyScale);
}

__host__ __device__ __forceinline__ void bsdfInit(BSDF& bsdf, EBSDFType type,
                                                  float3 albedoEstimate) {
  bsdf.weightStorage[0] = float_to_half_bits(albedoEstimate.x);
  bsdf.weightStorage[1] = float_to_half_bits(albedoEstimate.y);
  bsdf.weightStorage[2] = float_to_half_bits(albedoEstimate.z);
  bsdf.weightStorage[3] = static_cast<uint16_t>(type);
}

__host__ __device__ __forceinline__ void bsdfGGXCommon(BSDF& bsdf, float alphax,
                                                       float alphay,
                                                       float phi0) {
  // roughness
  bsdf.data.ggx.alphax =
      static_cast<uint16_t>(fminf(fmaxf(alphax * UINT16_MAX, 0.f), UINT16_MAX));
  bsdf.data.ggx.alphay =
      static_cast<uint16_t>(fminf(fmaxf(alphay * UINT16_MAX, 0.f), UINT16_MAX));
  // energy scale estimated during prepare function, where wo is known
  bsdf.data.ggx.energyScale = 1.f;
  // tangent encoding through angle to azimuth of local tangent axis
  bsdf.data.ggx.phi0 = static_cast<uint16_t>(
      fminf(fmaxf(phi0 / (2.f * std::numbers::pi_v<float>)*UINT16_MAX, 0.f),
            UINT16_MAX));
}

}  // namespace

// cycles swaps the notion of wi and wo
static constexpr float throughputEps = 1e-6f;

__host__ __device__ BSDFSample sampleGGX(BSDF const& bsdf, float3 wo, float3 ns,
                                         float3 ng, float2 u, float uc) {
  BSDFSample sample{};
  float const cos_NO = dot(ns, wo);
#if 0  // already checked in dispatcher function
  if (cos_NI <= 0) {
    // incident angles from lower hemisphere are invalid. If you are within
    // a material after a transmission, it's the caller's responsibility
    // to flip the normals so that cosines are positive
    return sample;
  }
#endif
  bool const isotropic = bsdf.data.ggx.alphax == bsdf.data.ggx.alphay;
  float const alphax = static_cast<float>(bsdf.data.ggx.alphax) / UINT16_MAX;
  float const alphay = static_cast<float>(bsdf.data.ggx.alphay) / UINT16_MAX;
  float3 const tangent = tangentFromPhi(ns, bsdf.data.ggx.getPhi0());
  // eta always refers to outside/inside. hence, if last was transmission,
  // the caller should flip it. Here we initialize it to 1 as we don't know
  // whether the GGX is conductor or dielectric yet.
  sample.eta = 1.f;
  float invEta = 1.f;
  // below a certain roughness, GGX becomes effectively specular
  sample.delta = fmaxf(alphax, alphay) < 1e-3f;  // TODO tweak?

  // half vector (global space) and anisotropic params
  float3 H{}, local_H{}, local_O{};
  if (sample.delta) {
    H = ns;
  } else {
    // make tangent-space frame. Consider BSDF tangent if anisotropic
    float3 X{}, Y{};
    // TODO remove branch
    if (isotropic) {
      gramSchmidt(ns, &X, &Y);
    } else {
      orthonormalTangent(ns, tangent, &X, &Y);
    }
    // importance sampling of Distribution of Visible Normals
    // https://jcgt.org/published/0007/04/01/
    local_O = make_float3(dot(X, wo), dot(Y, wo), cos_NO);
    local_H = sampleGGX_VNDF(local_O, u, alphax, alphay);
    // to global
    H = local_H.x * X + local_H.y * Y + local_H.z * ns;
  }
  float const cos_HO = dot(H, wo);
  // angle betwen half vector and refracted ray. not used in reflection
  float cos_HI{};
  float3 reflectance{};
  float3 transmittance{};
  microfacetFresnel(bsdf, cos_HO, &cos_HI, &reflectance, &transmittance);
  // TODO should assert positive or zero values for all components
  if (nearZeroPos(reflectance, throughputEps) &&
      nearZeroPos(transmittance, throughputEps)) {
    return sample;  // TODO better branching?
  }

  const float pdfReflect =  // TODO assert [0,1]
      luminance(reflectance) / luminance(reflectance + transmittance);
  sample.refract = uc >= pdfReflect;
  // reflected/refracted direction
  sample.wi =
      sample.refract
          ? refractAngle(
                wo, H, cos_HI,
                (invEta = 1.f / (sample.eta = half_bits_to_float(
                                     bsdf.data.ggx.mat.dielectric.eta))))
          : 2.f * cos_HO * H - wo;
  // either normal and direction are same hemisphere or refraction
  if (dot(ng, sample.wi) < 0 && !sample.refract) {  // TODO branching
    return sample;  // pdf still 0 here, so falsy sample
  }

  // BSDF/PDF computation
  if (sample.refract) {
    sample.f = transmittance;
    sample.pdf = 1.f - pdfReflect;
    // if IOR is near to 1.0, then specular
    sample.delta |= fabsf(sample.eta - 1.f) < 1e-4f;
  } else {
    sample.pdf = pdfReflect;
  }
  // adjust for singular, otherwise apply Generalized Cook Torrance
  if (sample.delta) {
    // TODO: Check if this is necessary, since we don't do MIS for delta
    sample.pdf *= 1e6f;
    sample.f *= 1e6f;
  } else {
    float D{}, lambdaI{}, lambdaO{};

    // as cycles, we don't support anisotropic for transmission
    if (isotropic || sample.refract) {  // isotropic D and auxiliary
      float const alpha2 = alphax * alphay;
      float const cos_NH = local_H.z;
      float const cos_NI = dot(ns, sample.wi);

      D = ggx_D(alpha2, cos_NH);
      lambdaI = ggx_lambda(alpha2, cos_NI);
      lambdaO = ggx_lambda(alpha2, cos_NO);
    } else {  // anisotropic D and auxiliary
      float3 const local_I = 2.f * cos_HO * local_H - local_O;

      D = ggx_aniso_D(alphax, alphay, local_H);
      lambdaI = ggx_aniso_lambda(alphax, alphay, local_I);
      lambdaO = ggx_aniso_lambda(alphax, alphay, local_O);
    }

    // fused cook torrance formula for transmission and reflection
    float const common = D / cos_NO *
                         (sample.refract ? fabsf(cos_HO * cos_HI) /
                                               sqrf(cos_HI + cos_HO * invEta)
                                         : 0.25f);
    sample.pdf *= common / (1.f + lambdaO);
    sample.f *= common / (1.f + lambdaO + lambdaI);
  }

  return sample;
}

__host__ __device__ float3 evalGGX(BSDF const& bsdf, float3 wo, float3 wi,
                                   float3 ns, float3 ng, float* pdf) {
  float const energyScale = bsdf.data.ggx.energyScale;
  bool const conductor = bsdf.type() == EBSDFType::eGGXConductor;
  const bool hasReflection =
      conductor
          ? true
          : luminance(half_vec_to_float3(
                bsdf.data.ggx.mat.dielectric.reflectanceTint)) > throughputEps;
  const bool hasTransmission =
      conductor ? false
                : luminance(half_vec_to_float3(
                      bsdf.data.ggx.mat.dielectric.transmittanceTint)) >
                      throughputEps;
  float const alphax = static_cast<float>(bsdf.data.ggx.alphax) / UINT16_MAX;
  float const alphay = static_cast<float>(bsdf.data.ggx.alphay) / UINT16_MAX;
  bool const isotropic = alphax == alphay;
  float const cos_NO = dot(ns, wo);
  float const cos_NI = dot(ns, wi);
  float const cos_NgI = dot(ng, wi);
  bool const isTransmission = cos_NI < 0.f;  // assumes dielectric
  float const ior = isTransmission
                        ? half_bits_to_float(bsdf.data.ggx.mat.dielectric.eta)
                        : 1.f;
  bool const effectivelySpecular =
      fmaxf(alphax, alphay) < 1e-3f;  // TODO tweak?
  // - outgoing direction and normals (both) must be in the same hemisphere
  // - specular are not evaluated (dirac)
  // - incoming direction must be in the same hemisphere of normal (both) for
  // reflection
  // - purely reflective -> no transmission. purely transmissive -> no
  // reflection
  if (cos_NO <= 0.f || (cos_NgI < 0) != isTransmission ||
      (effectivelySpecular) || (!hasReflection && cos_NgI > 0.f) ||
      (!hasTransmission && cos_NgI < 0.f)) {
    *pdf = 0.f;
    return make_float3(0, 0, 0);
  }
  // half vector
  float3 H = isTransmission ? -(ior * wi + wo) : (wi + wo);
  float const invLen_H = rsqrtf(dot(H, H));  // TODO safe division for zero
  H *= invLen_H;

  // fresnel
  float const cos_HO = dot(H, wo);
  float unused{};
  float3 reflectance{};
  float3 transmittance{};
  microfacetFresnel(bsdf, cos_HO, &unused, &reflectance, &transmittance);
  if (nearZeroPos(reflectance, throughputEps) &&
      nearZeroPos(transmittance, throughputEps)) {
    *pdf = 0.f;
    return make_float3(0, 0, 0);
  }
  // D and lambda
  float const cos_NH = dot(ns, H);
  float D{}, lambdaI{}, lambdaO{};
  if (isotropic || isTransmission) {
    float const alpha2 = alphax * alphay;
    D = ggx_D(alpha2, cos_NH);
    lambdaI = ggx_lambda(alpha2, cos_NI);
    lambdaO = ggx_lambda(alpha2, cos_NO);
  } else {
    // anisotropic only for reflection
    // make tangent-space frame. Consider BSDF tangent if anisotropic
    float3 const tangent = tangentFromPhi(ns, bsdf.data.ggx.getPhi0());
    float3 X{}, Y{};
    orthonormalTangent(ns, tangent, &X, &Y);
    // to local
    float3 const local_H = make_float3(dot(X, H), dot(Y, H), dot(ns, H));
    float3 const local_O = make_float3(dot(X, wo), dot(Y, wo), cos_NO);
    float3 const local_I = make_float3(dot(X, wi), dot(Y, wi), cos_NI);

    D = ggx_aniso_D(alphax, alphay, local_H);
    lambdaI = ggx_aniso_lambda(alphax, alphay, local_I);
    lambdaO = ggx_aniso_lambda(alphax, alphay, local_O);
  }
  // generalized cook torrance
  float const common =
      D / cos_NO *
      (isTransmission ? sqrf(ior * invLen_H) * fabsf(cos_HO * dot(H, wi))
                      : 0.25f);
  float const pdfReflect =
      luminance(reflectance) / luminance(reflectance + transmittance);
  float const lobePdf = isTransmission ? 1.f - pdfReflect : pdfReflect;
  *pdf = lobePdf * common / (1.f + lambdaO);
#if 0
  printf(
      "\t*pdf = lobePdf * common / (1.f + lambdaO);\n\t\t%f = %f * %f / (1 + "
      "%f)\n",
      *pdf, lobePdf, common, lambdaO);
#endif
#if defined(__CUDA_ARCH__) && 0
  if (cg::coalesced_threads().thread_rank() == 0 && isTransmission) {
    printf(
        "energyScale * (isTransmission ? transmittance : reflectance) * common/"
        "(1.f + lambdaO + lambdaI):\n\t%f * (%d ? %f %f %f : %f %f %f) * %f / "
        "(1 + %f + %f)\n\n",
        energyScale, isTransmission, transmittance.x, transmittance.y,
        transmittance.z, reflectance.x, reflectance.y, reflectance.z, common,
        lambdaO, lambdaI);
  }
#endif
  return energyScale * (isTransmission ? transmittance : reflectance) * common /
         (1.f + lambdaO + lambdaI);
}

__host__ __device__ BSDF makeGGXDielectric(float3 reflectanceTint,
                                           float3 transmittanceTint, float phi0,
                                           float eta, float alphax,
                                           float alphay) {
  BSDF bsdf{};
  // albedo is estimated in the prepare function, where wo is known
  bsdfInit(bsdf, EBSDFType::eGGXDielectric, make_float3(1, 1, 1));
  bsdfGGXCommon(bsdf, alphax, alphay, phi0);
  auto& dielectric = bsdf.data.ggx.mat.dielectric;

  dielectric.eta = float_to_half_bits(eta);
  dielectric.reflectanceTint[0] = float_to_half_bits(reflectanceTint.x);
  dielectric.reflectanceTint[1] = float_to_half_bits(reflectanceTint.y);
  dielectric.reflectanceTint[2] = float_to_half_bits(reflectanceTint.z);
  dielectric.transmittanceTint[0] = float_to_half_bits(transmittanceTint.x);
  dielectric.transmittanceTint[1] = float_to_half_bits(transmittanceTint.y);
  dielectric.transmittanceTint[2] = float_to_half_bits(transmittanceTint.z);

  return bsdf;
}

__host__ __device__ BSDF makeGGXConductor(float3 eta, float3 kappa, float phi0,
                                          float alphax, float alphay) {
  BSDF bsdf{};
  // albedo is estimated in the prepare function, where wo is known
  bsdfInit(bsdf, EBSDFType::eGGXConductor, make_float3(1, 1, 1));
  bsdfGGXCommon(bsdf, alphax, alphay, phi0);
  auto& conductor = bsdf.data.ggx.mat.conductor;

  conductor.eta[0] = float_to_half_bits(eta.x);
  conductor.eta[1] = float_to_half_bits(eta.y);
  conductor.eta[2] = float_to_half_bits(eta.z);
  conductor.kappa[0] = float_to_half_bits(kappa.x);
  conductor.kappa[1] = float_to_half_bits(kappa.y);
  conductor.kappa[2] = float_to_half_bits(kappa.z);

  return bsdf;
}

// ---------------------------------------------------------------------------
// BSDF: Oren Nayar
// ---------------------------------------------------------------------------

// From cycles: improved Oren-Nayar model by Yasuhiro Fujii
// https://mimosa-pudica.net/improved-oren-nayar.html
// energy-preserving multi-scattering term based on the OpenPBR specification
// https://academysoftwarefoundation.github.io/OpenPBR
__host__ __device__ __forceinline__ float orenNayar_G(float const cosTheta) {
  static float constexpr piOver2 = std::numbers::pi_v<float> / 2;
  static float constexpr _2Over3 = 2.f / 3.f;
  static float constexpr piOver2m2Over3 = piOver2 - _2Over3;
  if (cosTheta < 1e-6f) {
    return piOver2m2Over3 - cosTheta;
  }
  // TODO check that SASS/PTX compiles to condition ops and not branching
  float const sinTheta = sin_from_cos(cosTheta);
  float const theta = safeacos(cosTheta);
  return sinTheta * (theta - _2Over3 - sinTheta * cosTheta) +
         _2Over3 * (sinTheta / cosTheta) * (1.f - sqrf(sinTheta) * sinTheta);
}

__host__ __device__ __forceinline__ float3
orenNayar_intensity(BSDF::BSDFUnion::OrenNayar const& params, float3 const n,
                    float3 const v, float3 const l) {
  float const bsdf_a = half_bits_to_float(params.a);
  float const bsdf_b = half_bits_to_float(params.b);
  float3 const bsdf_multiScatter = half_vec_to_float3(params.multiScatter);

  float const nl = fmaxf(dot(n, l), 0.f);
  if (bsdf_b <= 0) {  // should never happen
    float const r = nl * std::numbers::inv_pi_v<float>;
    return make_float3(r, r, r);
  }
  float const nv = fmaxf(dot(n, v), 0.f);
  float t = dot(l, v) - nl * nv;
  // TODO check that SASS/PTX compiles to condition ops and not branching
  if (t > 0.f) {
    t /= fmaxf(nl, nv) + FLT_MIN;
  }
  float const singleScatter = bsdf_a + bsdf_b * t;
  float const El =
      bsdf_a * std::numbers::pi_v<float> + bsdf_b * orenNayar_G(nl);
  float3 const multiScatter = bsdf_multiScatter * (1.f - El);
  return nl * (singleScatter + multiScatter);
}

__host__ __device__ BSDFSample sampleOrenNayar(BSDF const& bsdf, float3 wo,
                                               float3 ns, float3 ng, float2 u,
                                               float uc) {
  BSDFSample sample{};
  sample.eta = 1.f;
  sample.wi = sampleCosHemisphere(ns, u, &sample.pdf);
  if (dot(ng, sample.wi) > 0.f) {
    sample.f = orenNayar_intensity(bsdf.data.orenNayar, ns, wo, sample.wi);
  } else {  // TODO better branching
    sample.pdf = 0;
  }
  return sample;
}

__host__ __device__ float3 evalOrenNayar(BSDF const& bsdf, float3 wo, float3 wi,
                                         float3 ns, float3 ng, float* pdf) {
  float const cos_NI = dot(ns, wi);
  if (cos_NI > 0.f) {
    *pdf = cos_NI * std::numbers::inv_pi_v<float>;
    return orenNayar_intensity(bsdf.data.orenNayar, ns, wo, wi);
  }
  return make_float3(0, 0, 0);
}

// ---------------------------------------------------------------------------
// BSDF makers
// ---------------------------------------------------------------------------
__host__ __device__ BSDF makeOrenNayar(float3 color, float roughness) {
  static constexpr float piOver2Minus2Over3 =
      (std::numbers::pi_v<float> / 2.f) - 2.f / 3.f;
  float3 const albedo =
      make_float3(fmaxf(0, fminf(color.x, 1)), fmaxf(0, fminf(color.y, 1)),
                  fmaxf(0, fminf(color.z, 1)));

  BSDF bsdf{};
  bsdfInit(bsdf, EBSDFType::eOrenNayar, albedo);
  auto& orenNayar = bsdf.data.orenNayar;

  orenNayar.roughness = float_to_half_bits(
      fmaxf(0, fminf(roughness, std::numbers::pi_v<float> / 2.f)));
  float const sigma = half_bits_to_float(orenNayar.roughness);

  orenNayar.a = float_to_half_bits(
      1.f / (std::numbers::pi_v<float> + piOver2Minus2Over3 * sigma));
  float const a = half_bits_to_float(orenNayar.a);

  orenNayar.b = float_to_half_bits(a * sigma);

  // multi-scatter initialized in prepare function, where wo is known
  orenNayar.multiScatter[0] = float_to_half_bits(1.f);
  orenNayar.multiScatter[1] = float_to_half_bits(1.f);
  orenNayar.multiScatter[2] = float_to_half_bits(1.f);

  return bsdf;
}

// ---------------------------------------------------------------------------
// BSDF sampling/evaluation functions
// ---------------------------------------------------------------------------

// TODO is it ok to synchronize the coalesced warp?
__host__ __device__ BSDFSample sampleBsdf(BSDF const& bsdf, float3 wo,
                                          float3 ns, float3 ng, float2 u,
                                          float uc) {
  BSDFSample sample{};
#ifdef __CUDA_ARCH__
  cg::coalesced_group const theWarp = cg::coalesced_threads();
#endif
  // Ensure outgoing direction is same hemisphere of geo normal
  if (dot(wo, ng) > 0.0f) {
    // Shading normal must not change hemisphere
    ns = faceForward(ns, ng);

    switch (bsdf.type()) {
      case EBSDFType::eOrenNayar:
        sample = sampleOrenNayar(bsdf, wo, ns, ng, u, uc);
        break;
      case EBSDFType::eGGXDielectric:
      case EBSDFType::eGGXConductor:
        sample = sampleGGX(bsdf, wo, ns, ng, u, uc);
        break;
    }
  }
#ifdef __CUDA_ARCH__
  theWarp.sync();
#endif
  return sample;
}

__host__ __device__ float3 evalBsdf(BSDF const& bsdf, float3 wo, float3 wi,
                                    float3 ns, float3 ng, float* pdf) {
  float3 f = make_float3(0, 0, 0);
  *pdf = 0;
#ifdef __CUDA_ARCH__
  cg::coalesced_group const theWarp = cg::coalesced_threads();
#endif
  EBSDFType const type = bsdf.type();
  switch (type) {
    case EBSDFType::eOrenNayar:
      f = evalOrenNayar(bsdf, wo, wi, ns, ng, pdf);
      break;
    case EBSDFType::eGGXConductor:
    case EBSDFType::eGGXDielectric:
      f = evalGGX(bsdf, wo, wi, ns, ng, pdf);
      break;
  }
#ifdef __CUDA_ARCH__
  theWarp.sync();
#endif
  return f;
}

__host__ __device__ void prepareBSDF(BSDF* bsdf, float3 ns, float3 wo) {
#ifdef __CUDA_ARCH__
  cg::coalesced_group const theWarp = cg::coalesced_threads();
#endif
  EBSDFType const type = bsdf->type();
  switch (type) {
    case EBSDFType::eOrenNayar: {
      static constexpr float _2piM5_6Over3 =
          (2.f * std::numbers::pi_v<float> - 5.6f) / 3.f;
      static constexpr float _1OverPi = std::numbers::inv_pi_v<float>;

      // energy preserving multi-scattering term
      float const bsdf_a = half_bits_to_float(bsdf->data.orenNayar.a);
      float const bsdf_b = half_bits_to_float(bsdf->data.orenNayar.b);
      float3 const bsdf_albedo = bsdf->weight();

#if 0
      float const Eavg = bsdf_a * bsdf_a + _2piM5_6Over3 * bsdf_b;
      float3 const Ems = _1OverPi - bsdf_albedo * bsdf_albedo *
                                        (Eavg / (1.f - Eavg)) /
                                        (1.f - bsdf_albedo * (1.f - Eavg));
      float const nv = fmaxf(0.f, dot(ns, wo));  // check done at sample/eval
      float const Ev =
          bsdf_a * std::numbers::pi_v<float> + bsdf_b * orenNayar_G(nv);

      float3 const multiScatter = Ems * (1.f - Ev);
#else
      float const nl = fmaxf(0.f, dot(ns, wo));
      // Compute multi-scatter energy term using Nayar 1994
      float const Ev =
          bsdf_a * std::numbers::pi_v<float> + bsdf_b * orenNayar_G(nl);
      float3 multiScatter =
          bsdf_albedo * (1.f - Ev);  // energy preserving scaling
      // Clamp to avoid negatives
      multiScatter.x = fmaxf(multiScatter.x, 0.f);
      multiScatter.y = fmaxf(multiScatter.y, 0.f);
      multiScatter.z = fmaxf(multiScatter.z, 0.f);
#endif
      bsdf->data.orenNayar.multiScatter[0] = float_to_half_bits(multiScatter.x);
      bsdf->data.orenNayar.multiScatter[1] = float_to_half_bits(multiScatter.y);
      bsdf->data.orenNayar.multiScatter[2] = float_to_half_bits(multiScatter.z);
      break;
    }
    case EBSDFType::eGGXDielectric:
    case EBSDFType::eGGXConductor: {
      // estimate albedo (redundant call to fresnel?)
      {
        // checking same hemisphere is done in sample and eval
        float const cos_HO = fabsf(dot(wo, ns));
        float unused{};
        float3 reflectance;
        float3 transmittance;
        microfacetFresnel(*bsdf, cos_HO, &unused, &reflectance, &transmittance);
        bsdf->setWeight(reflectance + transmittance);
      }

      float3 Fss{};
      if (type == EBSDFType::eGGXDielectric) {
        // assume dielectric tint = transmission tint
        Fss =
            half_vec_to_float3(bsdf->data.ggx.mat.dielectric.transmittanceTint);
      } else {  // EBSDFType::eGGXConductor
        float3 const eta = half_vec_to_float3(bsdf->data.ggx.mat.conductor.eta);
        float3 const kappa =
            half_vec_to_float3(bsdf->data.ggx.mat.conductor.kappa);
        // F82-tint
        float3 const F0 = reflectanceFresnelConductor(1.f, eta, kappa);
        float3 const F82 = reflectanceFresnelConductor(1.f / 7.f, eta, kappa);

        float3 const B =
            (lerp(F0, make_float3(1, 1, 1), 0.46266436f) - F82) * 17.651384f;
        Fss = lerp(F0, make_float3(1, 1, 1), 1.f / 21.f) - B * (1.f / 126.f);
      }
      // compute energy scale from table lookup
      float const alphax =
          static_cast<float>(bsdf->data.ggx.alphax) / UINT16_MAX;
      float const alphay =
          static_cast<float>(bsdf->data.ggx.alphay) / UINT16_MAX;
      float const alpha2 = alphax * alphay;
      float const cos_NO =
          fmaxf(0.f, dot(ns, wo));  // check done at sample/eval

      energyPreservingGGXScale(*bsdf, alpha2, cos_NO, Fss);
      break;
    }
  }
#ifdef __CUDA_ARCH__
  theWarp.sync();
#endif
}
