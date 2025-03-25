#include "stuff.h"

bool loadNvcudaFunctions(dmt::os::LibraryLoader const& loader, NvcudaLibraryFunctions* funcList)
{
    using namespace std::string_view_literals;

#if defined(DMT_OS_WINDOWS)
    dmt::os::Path path  = dmt::os::Path::root("C:") / "Windows" / "System32";
    funcList->m_library = loader.loadLibrary("nvcuda.dll"sv, true, &path);
#elif defined(DMT_OS_LINUX)
    dmt::os::Path path  = dmt::os::Path::root("");
    funcList->m_library = loader.loadLibrary(""sv, true, &path);
#else
#error "unsupported platform"
#endif
    if (!funcList->m_library)
    {
        return false;
    }
    funcList->cuArray3DCreate = reinterpret_cast<NvcudaLibraryFunctions::cuArray3DCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArray3DCreate_v2"));
    if (!funcList->cuArray3DCreate)
    {
        return false;
    }
    funcList->cuArray3DGetDescriptor = reinterpret_cast<NvcudaLibraryFunctions::cuArray3DGetDescriptor_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArray3DGetDescriptor_v2"));
    if (!funcList->cuArray3DGetDescriptor)
    {
        return false;
    }
    funcList->cuArrayCreate = reinterpret_cast<NvcudaLibraryFunctions::cuArrayCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayCreate_v2"));
    if (!funcList->cuArrayCreate)
    {
        return false;
    }
    funcList->cuArrayDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuArrayDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayDestroy"));
    if (!funcList->cuArrayDestroy)
    {
        return false;
    }
    funcList->cuArrayGetDescriptor = reinterpret_cast<NvcudaLibraryFunctions::cuArrayGetDescriptor_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayGetDescriptor_v2"));
    if (!funcList->cuArrayGetDescriptor)
    {
        return false;
    }
    funcList->cuArrayGetMemoryRequirements = reinterpret_cast<NvcudaLibraryFunctions::cuArrayGetMemoryRequirements_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayGetMemoryRequirements"));
    if (!funcList->cuArrayGetMemoryRequirements)
    {
        return false;
    }
    funcList->cuArrayGetPlane = reinterpret_cast<NvcudaLibraryFunctions::cuArrayGetPlane_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayGetPlane"));
    if (!funcList->cuArrayGetPlane)
    {
        return false;
    }
    funcList->cuArrayGetSparseProperties = reinterpret_cast<NvcudaLibraryFunctions::cuArrayGetSparseProperties_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuArrayGetSparseProperties"));
    if (!funcList->cuArrayGetSparseProperties)
    {
        return false;
    }
    funcList->cuCheckpointProcessCheckpoint = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessCheckpoint_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessCheckpoint"));
    if (!funcList->cuCheckpointProcessCheckpoint)
    {
        return false;
    }
    funcList->cuCheckpointProcessGetRestoreThreadId = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessGetRestoreThreadId_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessGetRestoreThreadId"));
    if (!funcList->cuCheckpointProcessGetRestoreThreadId)
    {
        return false;
    }
    funcList->cuCheckpointProcessGetState = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessGetState_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessGetState"));
    if (!funcList->cuCheckpointProcessGetState)
    {
        return false;
    }
    funcList->cuCheckpointProcessLock = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessLock_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessLock"));
    if (!funcList->cuCheckpointProcessLock)
    {
        return false;
    }
    funcList->cuCheckpointProcessRestore = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessRestore_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessRestore"));
    if (!funcList->cuCheckpointProcessRestore)
    {
        return false;
    }
    funcList->cuCheckpointProcessUnlock = reinterpret_cast<NvcudaLibraryFunctions::cuCheckpointProcessUnlock_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCheckpointProcessUnlock"));
    if (!funcList->cuCheckpointProcessUnlock)
    {
        return false;
    }
    funcList->cuCoredumpGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuCoredumpGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCoredumpGetAttribute"));
    if (!funcList->cuCoredumpGetAttribute)
    {
        return false;
    }
    funcList->cuCoredumpGetAttributeGlobal = reinterpret_cast<NvcudaLibraryFunctions::cuCoredumpGetAttributeGlobal_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCoredumpGetAttributeGlobal"));
    if (!funcList->cuCoredumpGetAttributeGlobal)
    {
        return false;
    }
    funcList->cuCoredumpSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuCoredumpSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCoredumpSetAttribute"));
    if (!funcList->cuCoredumpSetAttribute)
    {
        return false;
    }
    funcList->cuCoredumpSetAttributeGlobal = reinterpret_cast<NvcudaLibraryFunctions::cuCoredumpSetAttributeGlobal_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCoredumpSetAttributeGlobal"));
    if (!funcList->cuCoredumpSetAttributeGlobal)
    {
        return false;
    }
    funcList->cuCtxAttach = reinterpret_cast<NvcudaLibraryFunctions::cuCtxAttach_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxAttach"));
    if (!funcList->cuCtxAttach)
    {
        return false;
    }
    funcList->cuCtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuCtxCreate_v4_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxCreate_v4"));
    if (!funcList->cuCtxCreate)
    {
        return false;
    }
    funcList->cuCtxDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuCtxDestroy_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxDestroy_v2"));
    if (!funcList->cuCtxDestroy)
    {
        return false;
    }
    funcList->cuCtxDetach = reinterpret_cast<NvcudaLibraryFunctions::cuCtxDetach_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxDetach"));
    if (!funcList->cuCtxDetach)
    {
        return false;
    }
    funcList->cuCtxDisablePeerAccess = reinterpret_cast<NvcudaLibraryFunctions::cuCtxDisablePeerAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxDisablePeerAccess"));
    if (!funcList->cuCtxDisablePeerAccess)
    {
        return false;
    }
    funcList->cuCtxEnablePeerAccess = reinterpret_cast<NvcudaLibraryFunctions::cuCtxEnablePeerAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxEnablePeerAccess"));
    if (!funcList->cuCtxEnablePeerAccess)
    {
        return false;
    }
    funcList->cuCtxFromGreenCtx = reinterpret_cast<NvcudaLibraryFunctions::cuCtxFromGreenCtx_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxFromGreenCtx"));
    if (!funcList->cuCtxFromGreenCtx)
    {
        return false;
    }
    funcList->cuCtxGetApiVersion = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetApiVersion_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetApiVersion"));
    if (!funcList->cuCtxGetApiVersion)
    {
        return false;
    }
    funcList->cuCtxGetCacheConfig = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetCacheConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetCacheConfig"));
    if (!funcList->cuCtxGetCacheConfig)
    {
        return false;
    }
    funcList->cuCtxGetCurrent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetCurrent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetCurrent"));
    if (!funcList->cuCtxGetCurrent)
    {
        return false;
    }
    funcList->cuCtxGetDevResource = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetDevResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetDevResource"));
    if (!funcList->cuCtxGetDevResource)
    {
        return false;
    }
    funcList->cuCtxGetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetDevice"));
    if (!funcList->cuCtxGetDevice)
    {
        return false;
    }
    funcList->cuCtxGetExecAffinity = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetExecAffinity_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetExecAffinity"));
    if (!funcList->cuCtxGetExecAffinity)
    {
        return false;
    }
    funcList->cuCtxGetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetFlags"));
    if (!funcList->cuCtxGetFlags)
    {
        return false;
    }
    funcList->cuCtxGetId = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetId_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetId"));
    if (!funcList->cuCtxGetId)
    {
        return false;
    }
    funcList->cuCtxGetLimit = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetLimit_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetLimit"));
    if (!funcList->cuCtxGetLimit)
    {
        return false;
    }
    funcList->cuCtxGetSharedMemConfig = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetSharedMemConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetSharedMemConfig"));
    if (!funcList->cuCtxGetSharedMemConfig)
    {
        return false;
    }
    funcList->cuCtxGetStreamPriorityRange = reinterpret_cast<NvcudaLibraryFunctions::cuCtxGetStreamPriorityRange_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxGetStreamPriorityRange"));
    if (!funcList->cuCtxGetStreamPriorityRange)
    {
        return false;
    }
    funcList->cuCtxPopCurrent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxPopCurrent_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxPopCurrent_v2"));
    if (!funcList->cuCtxPopCurrent)
    {
        return false;
    }
    funcList->cuCtxPushCurrent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxPushCurrent_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxPushCurrent_v2"));
    if (!funcList->cuCtxPushCurrent)
    {
        return false;
    }
    funcList->cuCtxRecordEvent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxRecordEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxRecordEvent"));
    if (!funcList->cuCtxRecordEvent)
    {
        return false;
    }
    funcList->cuCtxResetPersistingL2Cache = reinterpret_cast<NvcudaLibraryFunctions::cuCtxResetPersistingL2Cache_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxResetPersistingL2Cache"));
    if (!funcList->cuCtxResetPersistingL2Cache)
    {
        return false;
    }
    funcList->cuCtxSetCacheConfig = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSetCacheConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSetCacheConfig"));
    if (!funcList->cuCtxSetCacheConfig)
    {
        return false;
    }
    funcList->cuCtxSetCurrent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSetCurrent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSetCurrent"));
    if (!funcList->cuCtxSetCurrent)
    {
        return false;
    }
    funcList->cuCtxSetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSetFlags"));
    if (!funcList->cuCtxSetFlags)
    {
        return false;
    }
    funcList->cuCtxSetLimit = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSetLimit_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSetLimit"));
    if (!funcList->cuCtxSetLimit)
    {
        return false;
    }
    funcList->cuCtxSetSharedMemConfig = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSetSharedMemConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSetSharedMemConfig"));
    if (!funcList->cuCtxSetSharedMemConfig)
    {
        return false;
    }
    funcList->cuCtxSynchronize = reinterpret_cast<NvcudaLibraryFunctions::cuCtxSynchronize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxSynchronize"));
    if (!funcList->cuCtxSynchronize)
    {
        return false;
    }
    funcList->cuCtxWaitEvent = reinterpret_cast<NvcudaLibraryFunctions::cuCtxWaitEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuCtxWaitEvent"));
    if (!funcList->cuCtxWaitEvent)
    {
        return false;
    }
    funcList->cuD3D10CtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10CtxCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10CtxCreate_v2"));
    if (!funcList->cuD3D10CtxCreate)
    {
        return false;
    }
    funcList->cuD3D10CtxCreateOnDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10CtxCreateOnDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10CtxCreateOnDevice"));
    if (!funcList->cuD3D10CtxCreateOnDevice)
    {
        return false;
    }
    funcList->cuD3D10GetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10GetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10GetDevice"));
    if (!funcList->cuD3D10GetDevice)
    {
        return false;
    }
    funcList->cuD3D10GetDevices = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10GetDevices_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10GetDevices"));
    if (!funcList->cuD3D10GetDevices)
    {
        return false;
    }
    funcList->cuD3D10GetDirect3DDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10GetDirect3DDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10GetDirect3DDevice"));
    if (!funcList->cuD3D10GetDirect3DDevice)
    {
        return false;
    }
    funcList->cuD3D10MapResources = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10MapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10MapResources"));
    if (!funcList->cuD3D10MapResources)
    {
        return false;
    }
    funcList->cuD3D10RegisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10RegisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10RegisterResource"));
    if (!funcList->cuD3D10RegisterResource)
    {
        return false;
    }
    funcList->cuD3D10ResourceGetMappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceGetMappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceGetMappedArray"));
    if (!funcList->cuD3D10ResourceGetMappedArray)
    {
        return false;
    }
    funcList->cuD3D10ResourceGetMappedPitch = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceGetMappedPitch_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceGetMappedPitch_v2"));
    if (!funcList->cuD3D10ResourceGetMappedPitch)
    {
        return false;
    }
    funcList->cuD3D10ResourceGetMappedPointer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceGetMappedPointer_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceGetMappedPointer_v2"));
    if (!funcList->cuD3D10ResourceGetMappedPointer)
    {
        return false;
    }
    funcList->cuD3D10ResourceGetMappedSize = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceGetMappedSize_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceGetMappedSize_v2"));
    if (!funcList->cuD3D10ResourceGetMappedSize)
    {
        return false;
    }
    funcList->cuD3D10ResourceGetSurfaceDimensions = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceGetSurfaceDimensions_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceGetSurfaceDimensions_v2"));
    if (!funcList->cuD3D10ResourceGetSurfaceDimensions)
    {
        return false;
    }
    funcList->cuD3D10ResourceSetMapFlags = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10ResourceSetMapFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10ResourceSetMapFlags"));
    if (!funcList->cuD3D10ResourceSetMapFlags)
    {
        return false;
    }
    funcList->cuD3D10UnmapResources = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10UnmapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10UnmapResources"));
    if (!funcList->cuD3D10UnmapResources)
    {
        return false;
    }
    funcList->cuD3D10UnregisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuD3D10UnregisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D10UnregisterResource"));
    if (!funcList->cuD3D10UnregisterResource)
    {
        return false;
    }
    funcList->cuD3D11CtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuD3D11CtxCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D11CtxCreate_v2"));
    if (!funcList->cuD3D11CtxCreate)
    {
        return false;
    }
    funcList->cuD3D11CtxCreateOnDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D11CtxCreateOnDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D11CtxCreateOnDevice"));
    if (!funcList->cuD3D11CtxCreateOnDevice)
    {
        return false;
    }
    funcList->cuD3D11GetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D11GetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D11GetDevice"));
    if (!funcList->cuD3D11GetDevice)
    {
        return false;
    }
    funcList->cuD3D11GetDevices = reinterpret_cast<NvcudaLibraryFunctions::cuD3D11GetDevices_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D11GetDevices"));
    if (!funcList->cuD3D11GetDevices)
    {
        return false;
    }
    funcList->cuD3D11GetDirect3DDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D11GetDirect3DDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D11GetDirect3DDevice"));
    if (!funcList->cuD3D11GetDirect3DDevice)
    {
        return false;
    }
    funcList->cuD3D9Begin = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9Begin_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9Begin"));
    if (!funcList->cuD3D9Begin)
    {
        return false;
    }
    funcList->cuD3D9CtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9CtxCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9CtxCreate_v2"));
    if (!funcList->cuD3D9CtxCreate)
    {
        return false;
    }
    funcList->cuD3D9CtxCreateOnDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9CtxCreateOnDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9CtxCreateOnDevice"));
    if (!funcList->cuD3D9CtxCreateOnDevice)
    {
        return false;
    }
    funcList->cuD3D9End = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9End_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9End"));
    if (!funcList->cuD3D9End)
    {
        return false;
    }
    funcList->cuD3D9GetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9GetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9GetDevice"));
    if (!funcList->cuD3D9GetDevice)
    {
        return false;
    }
    funcList->cuD3D9GetDevices = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9GetDevices_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9GetDevices"));
    if (!funcList->cuD3D9GetDevices)
    {
        return false;
    }
    funcList->cuD3D9GetDirect3DDevice = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9GetDirect3DDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9GetDirect3DDevice"));
    if (!funcList->cuD3D9GetDirect3DDevice)
    {
        return false;
    }
    funcList->cuD3D9MapResources = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9MapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9MapResources"));
    if (!funcList->cuD3D9MapResources)
    {
        return false;
    }
    funcList->cuD3D9MapVertexBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9MapVertexBuffer_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9MapVertexBuffer_v2"));
    if (!funcList->cuD3D9MapVertexBuffer)
    {
        return false;
    }
    funcList->cuD3D9RegisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9RegisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9RegisterResource"));
    if (!funcList->cuD3D9RegisterResource)
    {
        return false;
    }
    funcList->cuD3D9RegisterVertexBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9RegisterVertexBuffer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9RegisterVertexBuffer"));
    if (!funcList->cuD3D9RegisterVertexBuffer)
    {
        return false;
    }
    funcList->cuD3D9ResourceGetMappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceGetMappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceGetMappedArray"));
    if (!funcList->cuD3D9ResourceGetMappedArray)
    {
        return false;
    }
    funcList->cuD3D9ResourceGetMappedPitch = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceGetMappedPitch_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceGetMappedPitch_v2"));
    if (!funcList->cuD3D9ResourceGetMappedPitch)
    {
        return false;
    }
    funcList->cuD3D9ResourceGetMappedPointer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceGetMappedPointer_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceGetMappedPointer_v2"));
    if (!funcList->cuD3D9ResourceGetMappedPointer)
    {
        return false;
    }
    funcList->cuD3D9ResourceGetMappedSize = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceGetMappedSize_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceGetMappedSize_v2"));
    if (!funcList->cuD3D9ResourceGetMappedSize)
    {
        return false;
    }
    funcList->cuD3D9ResourceGetSurfaceDimensions = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceGetSurfaceDimensions_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceGetSurfaceDimensions_v2"));
    if (!funcList->cuD3D9ResourceGetSurfaceDimensions)
    {
        return false;
    }
    funcList->cuD3D9ResourceSetMapFlags = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9ResourceSetMapFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9ResourceSetMapFlags"));
    if (!funcList->cuD3D9ResourceSetMapFlags)
    {
        return false;
    }
    funcList->cuD3D9UnmapResources = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9UnmapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9UnmapResources"));
    if (!funcList->cuD3D9UnmapResources)
    {
        return false;
    }
    funcList->cuD3D9UnmapVertexBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9UnmapVertexBuffer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9UnmapVertexBuffer"));
    if (!funcList->cuD3D9UnmapVertexBuffer)
    {
        return false;
    }
    funcList->cuD3D9UnregisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9UnregisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9UnregisterResource"));
    if (!funcList->cuD3D9UnregisterResource)
    {
        return false;
    }
    funcList->cuD3D9UnregisterVertexBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuD3D9UnregisterVertexBuffer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuD3D9UnregisterVertexBuffer"));
    if (!funcList->cuD3D9UnregisterVertexBuffer)
    {
        return false;
    }
    funcList->cuDestroyExternalMemory = reinterpret_cast<NvcudaLibraryFunctions::cuDestroyExternalMemory_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDestroyExternalMemory"));
    if (!funcList->cuDestroyExternalMemory)
    {
        return false;
    }
    funcList->cuDestroyExternalSemaphore = reinterpret_cast<NvcudaLibraryFunctions::cuDestroyExternalSemaphore_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDestroyExternalSemaphore"));
    if (!funcList->cuDestroyExternalSemaphore)
    {
        return false;
    }
    funcList->cuDevResourceGenerateDesc = reinterpret_cast<NvcudaLibraryFunctions::cuDevResourceGenerateDesc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevResourceGenerateDesc"));
    if (!funcList->cuDevResourceGenerateDesc)
    {
        return false;
    }
    funcList->cuDevSmResourceSplitByCount = reinterpret_cast<NvcudaLibraryFunctions::cuDevSmResourceSplitByCount_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevSmResourceSplitByCount"));
    if (!funcList->cuDevSmResourceSplitByCount)
    {
        return false;
    }
    funcList->cuDeviceCanAccessPeer = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceCanAccessPeer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceCanAccessPeer"));
    if (!funcList->cuDeviceCanAccessPeer)
    {
        return false;
    }
    funcList->cuDeviceComputeCapability = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceComputeCapability_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceComputeCapability"));
    if (!funcList->cuDeviceComputeCapability)
    {
        return false;
    }
    funcList->cuDeviceGet = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGet_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGet"));
    if (!funcList->cuDeviceGet)
    {
        return false;
    }
    funcList->cuDeviceGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetAttribute"));
    if (!funcList->cuDeviceGetAttribute)
    {
        return false;
    }
    funcList->cuDeviceGetByPCIBusId = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetByPCIBusId_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetByPCIBusId"));
    if (!funcList->cuDeviceGetByPCIBusId)
    {
        return false;
    }
    funcList->cuDeviceGetCount = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetCount_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetCount"));
    if (!funcList->cuDeviceGetCount)
    {
        return false;
    }
    funcList->cuDeviceGetDefaultMemPool = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetDefaultMemPool_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetDefaultMemPool"));
    if (!funcList->cuDeviceGetDefaultMemPool)
    {
        return false;
    }
    funcList->cuDeviceGetDevResource = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetDevResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetDevResource"));
    if (!funcList->cuDeviceGetDevResource)
    {
        return false;
    }
    funcList->cuDeviceGetExecAffinitySupport = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetExecAffinitySupport_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetExecAffinitySupport"));
    if (!funcList->cuDeviceGetExecAffinitySupport)
    {
        return false;
    }
    funcList->cuDeviceGetGraphMemAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetGraphMemAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetGraphMemAttribute"));
    if (!funcList->cuDeviceGetGraphMemAttribute)
    {
        return false;
    }
    funcList->cuDeviceGetLuid = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetLuid_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetLuid"));
    if (!funcList->cuDeviceGetLuid)
    {
        return false;
    }
    funcList->cuDeviceGetMemPool = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetMemPool_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetMemPool"));
    if (!funcList->cuDeviceGetMemPool)
    {
        return false;
    }
    funcList->cuDeviceGetName = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetName_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetName"));
    if (!funcList->cuDeviceGetName)
    {
        return false;
    }
    funcList->cuDeviceGetP2PAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetP2PAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetP2PAttribute"));
    if (!funcList->cuDeviceGetP2PAttribute)
    {
        return false;
    }
    funcList->cuDeviceGetPCIBusId = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetPCIBusId_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetPCIBusId"));
    if (!funcList->cuDeviceGetPCIBusId)
    {
        return false;
    }
    funcList->cuDeviceGetProperties = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetProperties_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetProperties"));
    if (!funcList->cuDeviceGetProperties)
    {
        return false;
    }
    funcList->cuDeviceGetTexture1DLinearMaxWidth = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetTexture1DLinearMaxWidth_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetTexture1DLinearMaxWidth"));
    if (!funcList->cuDeviceGetTexture1DLinearMaxWidth)
    {
        return false;
    }
    funcList->cuDeviceGetUuid = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGetUuid_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGetUuid_v2"));
    if (!funcList->cuDeviceGetUuid)
    {
        return false;
    }
    funcList->cuDeviceGraphMemTrim = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceGraphMemTrim_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceGraphMemTrim"));
    if (!funcList->cuDeviceGraphMemTrim)
    {
        return false;
    }
    funcList->cuDevicePrimaryCtxGetState = reinterpret_cast<NvcudaLibraryFunctions::cuDevicePrimaryCtxGetState_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevicePrimaryCtxGetState"));
    if (!funcList->cuDevicePrimaryCtxGetState)
    {
        return false;
    }
    funcList->cuDevicePrimaryCtxRelease = reinterpret_cast<NvcudaLibraryFunctions::cuDevicePrimaryCtxRelease_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevicePrimaryCtxRelease_v2"));
    if (!funcList->cuDevicePrimaryCtxRelease)
    {
        return false;
    }
    funcList->cuDevicePrimaryCtxReset = reinterpret_cast<NvcudaLibraryFunctions::cuDevicePrimaryCtxReset_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevicePrimaryCtxReset_v2"));
    if (!funcList->cuDevicePrimaryCtxReset)
    {
        return false;
    }
    funcList->cuDevicePrimaryCtxRetain = reinterpret_cast<NvcudaLibraryFunctions::cuDevicePrimaryCtxRetain_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevicePrimaryCtxRetain"));
    if (!funcList->cuDevicePrimaryCtxRetain)
    {
        return false;
    }
    funcList->cuDevicePrimaryCtxSetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuDevicePrimaryCtxSetFlags_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDevicePrimaryCtxSetFlags_v2"));
    if (!funcList->cuDevicePrimaryCtxSetFlags)
    {
        return false;
    }
    funcList->cuDeviceRegisterAsyncNotification = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceRegisterAsyncNotification_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceRegisterAsyncNotification"));
    if (!funcList->cuDeviceRegisterAsyncNotification)
    {
        return false;
    }
    funcList->cuDeviceSetGraphMemAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceSetGraphMemAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceSetGraphMemAttribute"));
    if (!funcList->cuDeviceSetGraphMemAttribute)
    {
        return false;
    }
    funcList->cuDeviceSetMemPool = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceSetMemPool_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceSetMemPool"));
    if (!funcList->cuDeviceSetMemPool)
    {
        return false;
    }
    funcList->cuDeviceTotalMem = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceTotalMem_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceTotalMem_v2"));
    if (!funcList->cuDeviceTotalMem)
    {
        return false;
    }
    funcList->cuDeviceUnregisterAsyncNotification = reinterpret_cast<NvcudaLibraryFunctions::cuDeviceUnregisterAsyncNotification_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDeviceUnregisterAsyncNotification"));
    if (!funcList->cuDeviceUnregisterAsyncNotification)
    {
        return false;
    }
    funcList->cuDriverGetVersion = reinterpret_cast<NvcudaLibraryFunctions::cuDriverGetVersion_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuDriverGetVersion"));
    if (!funcList->cuDriverGetVersion)
    {
        return false;
    }
    funcList->cuEventCreate = reinterpret_cast<NvcudaLibraryFunctions::cuEventCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventCreate"));
    if (!funcList->cuEventCreate)
    {
        return false;
    }
    funcList->cuEventDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuEventDestroy_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventDestroy_v2"));
    if (!funcList->cuEventDestroy)
    {
        return false;
    }
    funcList->cuEventElapsedTime = reinterpret_cast<NvcudaLibraryFunctions::cuEventElapsedTime_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventElapsedTime_v2"));
    if (!funcList->cuEventElapsedTime)
    {
        return false;
    }
    funcList->cuEventQuery = reinterpret_cast<NvcudaLibraryFunctions::cuEventQuery_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventQuery"));
    if (!funcList->cuEventQuery)
    {
        return false;
    }
    funcList->cuEventRecord = reinterpret_cast<NvcudaLibraryFunctions::cuEventRecord_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventRecord"));
    if (!funcList->cuEventRecord)
    {
        return false;
    }
    funcList->cuEventRecordWithFlags = reinterpret_cast<NvcudaLibraryFunctions::cuEventRecordWithFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventRecordWithFlags"));
    if (!funcList->cuEventRecordWithFlags)
    {
        return false;
    }
    funcList->cuEventRecordWithFlags_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuEventRecordWithFlags_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventRecordWithFlags_ptsz"));
    if (!funcList->cuEventRecordWithFlags_ptsz)
    {
        return false;
    }
    funcList->cuEventRecord_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuEventRecord_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventRecord_ptsz"));
    if (!funcList->cuEventRecord_ptsz)
    {
        return false;
    }
    funcList->cuEventSynchronize = reinterpret_cast<NvcudaLibraryFunctions::cuEventSynchronize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuEventSynchronize"));
    if (!funcList->cuEventSynchronize)
    {
        return false;
    }
    funcList->cuExternalMemoryGetMappedBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuExternalMemoryGetMappedBuffer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuExternalMemoryGetMappedBuffer"));
    if (!funcList->cuExternalMemoryGetMappedBuffer)
    {
        return false;
    }
    funcList->cuExternalMemoryGetMappedMipmappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuExternalMemoryGetMappedMipmappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuExternalMemoryGetMappedMipmappedArray"));
    if (!funcList->cuExternalMemoryGetMappedMipmappedArray)
    {
        return false;
    }
    funcList->cuFlushGPUDirectRDMAWrites = reinterpret_cast<NvcudaLibraryFunctions::cuFlushGPUDirectRDMAWrites_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFlushGPUDirectRDMAWrites"));
    if (!funcList->cuFlushGPUDirectRDMAWrites)
    {
        return false;
    }
    funcList->cuFuncGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuFuncGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncGetAttribute"));
    if (!funcList->cuFuncGetAttribute)
    {
        return false;
    }
    funcList->cuFuncGetModule = reinterpret_cast<NvcudaLibraryFunctions::cuFuncGetModule_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncGetModule"));
    if (!funcList->cuFuncGetModule)
    {
        return false;
    }
    funcList->cuFuncGetName = reinterpret_cast<NvcudaLibraryFunctions::cuFuncGetName_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncGetName"));
    if (!funcList->cuFuncGetName)
    {
        return false;
    }
    funcList->cuFuncGetParamInfo = reinterpret_cast<NvcudaLibraryFunctions::cuFuncGetParamInfo_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncGetParamInfo"));
    if (!funcList->cuFuncGetParamInfo)
    {
        return false;
    }
    funcList->cuFuncIsLoaded = reinterpret_cast<NvcudaLibraryFunctions::cuFuncIsLoaded_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncIsLoaded"));
    if (!funcList->cuFuncIsLoaded)
    {
        return false;
    }
    funcList->cuFuncLoad = reinterpret_cast<NvcudaLibraryFunctions::cuFuncLoad_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncLoad"));
    if (!funcList->cuFuncLoad)
    {
        return false;
    }
    funcList->cuFuncSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuFuncSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncSetAttribute"));
    if (!funcList->cuFuncSetAttribute)
    {
        return false;
    }
    funcList->cuFuncSetBlockShape = reinterpret_cast<NvcudaLibraryFunctions::cuFuncSetBlockShape_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncSetBlockShape"));
    if (!funcList->cuFuncSetBlockShape)
    {
        return false;
    }
    funcList->cuFuncSetCacheConfig = reinterpret_cast<NvcudaLibraryFunctions::cuFuncSetCacheConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncSetCacheConfig"));
    if (!funcList->cuFuncSetCacheConfig)
    {
        return false;
    }
    funcList->cuFuncSetSharedMemConfig = reinterpret_cast<NvcudaLibraryFunctions::cuFuncSetSharedMemConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncSetSharedMemConfig"));
    if (!funcList->cuFuncSetSharedMemConfig)
    {
        return false;
    }
    funcList->cuFuncSetSharedSize = reinterpret_cast<NvcudaLibraryFunctions::cuFuncSetSharedSize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuFuncSetSharedSize"));
    if (!funcList->cuFuncSetSharedSize)
    {
        return false;
    }
    funcList->cuGLCtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuGLCtxCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLCtxCreate_v2"));
    if (!funcList->cuGLCtxCreate)
    {
        return false;
    }
    funcList->cuGLGetDevices = reinterpret_cast<NvcudaLibraryFunctions::cuGLGetDevices_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLGetDevices_v2"));
    if (!funcList->cuGLGetDevices)
    {
        return false;
    }
    funcList->cuGLInit = reinterpret_cast<NvcudaLibraryFunctions::cuGLInit_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLInit"));
    if (!funcList->cuGLInit)
    {
        return false;
    }
    funcList->cuGLMapBufferObject = reinterpret_cast<NvcudaLibraryFunctions::cuGLMapBufferObject_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLMapBufferObject_v2"));
    if (!funcList->cuGLMapBufferObject)
    {
        return false;
    }
    funcList->cuGLMapBufferObjectAsync = reinterpret_cast<NvcudaLibraryFunctions::cuGLMapBufferObjectAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLMapBufferObjectAsync_v2"));
    if (!funcList->cuGLMapBufferObjectAsync)
    {
        return false;
    }
    funcList->cuGLRegisterBufferObject = reinterpret_cast<NvcudaLibraryFunctions::cuGLRegisterBufferObject_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLRegisterBufferObject"));
    if (!funcList->cuGLRegisterBufferObject)
    {
        return false;
    }
    funcList->cuGLSetBufferObjectMapFlags = reinterpret_cast<NvcudaLibraryFunctions::cuGLSetBufferObjectMapFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLSetBufferObjectMapFlags"));
    if (!funcList->cuGLSetBufferObjectMapFlags)
    {
        return false;
    }
    funcList->cuGLUnmapBufferObject = reinterpret_cast<NvcudaLibraryFunctions::cuGLUnmapBufferObject_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLUnmapBufferObject"));
    if (!funcList->cuGLUnmapBufferObject)
    {
        return false;
    }
    funcList->cuGLUnmapBufferObjectAsync = reinterpret_cast<NvcudaLibraryFunctions::cuGLUnmapBufferObjectAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLUnmapBufferObjectAsync"));
    if (!funcList->cuGLUnmapBufferObjectAsync)
    {
        return false;
    }
    funcList->cuGLUnregisterBufferObject = reinterpret_cast<NvcudaLibraryFunctions::cuGLUnregisterBufferObject_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGLUnregisterBufferObject"));
    if (!funcList->cuGLUnregisterBufferObject)
    {
        return false;
    }
    funcList->cuGetErrorName = reinterpret_cast<NvcudaLibraryFunctions::cuGetErrorName_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGetErrorName"));
    if (!funcList->cuGetErrorName)
    {
        return false;
    }
    funcList->cuGetErrorString = reinterpret_cast<NvcudaLibraryFunctions::cuGetErrorString_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGetErrorString"));
    if (!funcList->cuGetErrorString)
    {
        return false;
    }
    funcList->cuGetExportTable = reinterpret_cast<NvcudaLibraryFunctions::cuGetExportTable_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGetExportTable"));
    if (!funcList->cuGetExportTable)
    {
        return false;
    }
    funcList->cuGetProcAddress = reinterpret_cast<NvcudaLibraryFunctions::cuGetProcAddress_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGetProcAddress_v2"));
    if (!funcList->cuGetProcAddress)
    {
        return false;
    }
    funcList->cuGraphAddBatchMemOpNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddBatchMemOpNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddBatchMemOpNode"));
    if (!funcList->cuGraphAddBatchMemOpNode)
    {
        return false;
    }
    funcList->cuGraphAddChildGraphNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddChildGraphNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddChildGraphNode"));
    if (!funcList->cuGraphAddChildGraphNode)
    {
        return false;
    }
    funcList->cuGraphAddDependencies = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddDependencies_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddDependencies_v2"));
    if (!funcList->cuGraphAddDependencies)
    {
        return false;
    }
    funcList->cuGraphAddEmptyNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddEmptyNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddEmptyNode"));
    if (!funcList->cuGraphAddEmptyNode)
    {
        return false;
    }
    funcList->cuGraphAddEventRecordNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddEventRecordNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddEventRecordNode"));
    if (!funcList->cuGraphAddEventRecordNode)
    {
        return false;
    }
    funcList->cuGraphAddEventWaitNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddEventWaitNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddEventWaitNode"));
    if (!funcList->cuGraphAddEventWaitNode)
    {
        return false;
    }
    funcList->cuGraphAddExternalSemaphoresSignalNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddExternalSemaphoresSignalNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddExternalSemaphoresSignalNode"));
    if (!funcList->cuGraphAddExternalSemaphoresSignalNode)
    {
        return false;
    }
    funcList->cuGraphAddExternalSemaphoresWaitNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddExternalSemaphoresWaitNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddExternalSemaphoresWaitNode"));
    if (!funcList->cuGraphAddExternalSemaphoresWaitNode)
    {
        return false;
    }
    funcList->cuGraphAddHostNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddHostNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddHostNode"));
    if (!funcList->cuGraphAddHostNode)
    {
        return false;
    }
    funcList->cuGraphAddKernelNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddKernelNode_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddKernelNode_v2"));
    if (!funcList->cuGraphAddKernelNode)
    {
        return false;
    }
    funcList->cuGraphAddMemAllocNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddMemAllocNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddMemAllocNode"));
    if (!funcList->cuGraphAddMemAllocNode)
    {
        return false;
    }
    funcList->cuGraphAddMemFreeNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddMemFreeNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddMemFreeNode"));
    if (!funcList->cuGraphAddMemFreeNode)
    {
        return false;
    }
    funcList->cuGraphAddMemcpyNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddMemcpyNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddMemcpyNode"));
    if (!funcList->cuGraphAddMemcpyNode)
    {
        return false;
    }
    funcList->cuGraphAddMemsetNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddMemsetNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddMemsetNode"));
    if (!funcList->cuGraphAddMemsetNode)
    {
        return false;
    }
    funcList->cuGraphAddNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphAddNode_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphAddNode_v2"));
    if (!funcList->cuGraphAddNode)
    {
        return false;
    }
    funcList->cuGraphBatchMemOpNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphBatchMemOpNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphBatchMemOpNodeGetParams"));
    if (!funcList->cuGraphBatchMemOpNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphBatchMemOpNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphBatchMemOpNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphBatchMemOpNodeSetParams"));
    if (!funcList->cuGraphBatchMemOpNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphChildGraphNodeGetGraph = reinterpret_cast<NvcudaLibraryFunctions::cuGraphChildGraphNodeGetGraph_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphChildGraphNodeGetGraph"));
    if (!funcList->cuGraphChildGraphNodeGetGraph)
    {
        return false;
    }
    funcList->cuGraphClone = reinterpret_cast<NvcudaLibraryFunctions::cuGraphClone_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphClone"));
    if (!funcList->cuGraphClone)
    {
        return false;
    }
    funcList->cuGraphConditionalHandleCreate = reinterpret_cast<NvcudaLibraryFunctions::cuGraphConditionalHandleCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphConditionalHandleCreate"));
    if (!funcList->cuGraphConditionalHandleCreate)
    {
        return false;
    }
    funcList->cuGraphCreate = reinterpret_cast<NvcudaLibraryFunctions::cuGraphCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphCreate"));
    if (!funcList->cuGraphCreate)
    {
        return false;
    }
    funcList->cuGraphDebugDotPrint = reinterpret_cast<NvcudaLibraryFunctions::cuGraphDebugDotPrint_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphDebugDotPrint"));
    if (!funcList->cuGraphDebugDotPrint)
    {
        return false;
    }
    funcList->cuGraphDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuGraphDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphDestroy"));
    if (!funcList->cuGraphDestroy)
    {
        return false;
    }
    funcList->cuGraphDestroyNode = reinterpret_cast<NvcudaLibraryFunctions::cuGraphDestroyNode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphDestroyNode"));
    if (!funcList->cuGraphDestroyNode)
    {
        return false;
    }
    funcList->cuGraphEventRecordNodeGetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphEventRecordNodeGetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphEventRecordNodeGetEvent"));
    if (!funcList->cuGraphEventRecordNodeGetEvent)
    {
        return false;
    }
    funcList->cuGraphEventRecordNodeSetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphEventRecordNodeSetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphEventRecordNodeSetEvent"));
    if (!funcList->cuGraphEventRecordNodeSetEvent)
    {
        return false;
    }
    funcList->cuGraphEventWaitNodeGetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphEventWaitNodeGetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphEventWaitNodeGetEvent"));
    if (!funcList->cuGraphEventWaitNodeGetEvent)
    {
        return false;
    }
    funcList->cuGraphEventWaitNodeSetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphEventWaitNodeSetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphEventWaitNodeSetEvent"));
    if (!funcList->cuGraphEventWaitNodeSetEvent)
    {
        return false;
    }
    funcList->cuGraphExecBatchMemOpNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecBatchMemOpNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecBatchMemOpNodeSetParams"));
    if (!funcList->cuGraphExecBatchMemOpNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecChildGraphNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecChildGraphNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecChildGraphNodeSetParams"));
    if (!funcList->cuGraphExecChildGraphNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecDestroy"));
    if (!funcList->cuGraphExecDestroy)
    {
        return false;
    }
    funcList->cuGraphExecEventRecordNodeSetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecEventRecordNodeSetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecEventRecordNodeSetEvent"));
    if (!funcList->cuGraphExecEventRecordNodeSetEvent)
    {
        return false;
    }
    funcList->cuGraphExecEventWaitNodeSetEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecEventWaitNodeSetEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecEventWaitNodeSetEvent"));
    if (!funcList->cuGraphExecEventWaitNodeSetEvent)
    {
        return false;
    }
    funcList->cuGraphExecExternalSemaphoresSignalNodeSetParams = reinterpret_cast<
        NvcudaLibraryFunctions::cuGraphExecExternalSemaphoresSignalNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecExternalSemaphoresSignalNodeSetParams"));
    if (!funcList->cuGraphExecExternalSemaphoresSignalNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecExternalSemaphoresWaitNodeSetParams = reinterpret_cast<
        NvcudaLibraryFunctions::cuGraphExecExternalSemaphoresWaitNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecExternalSemaphoresWaitNodeSetParams"));
    if (!funcList->cuGraphExecExternalSemaphoresWaitNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecGetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecGetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecGetFlags"));
    if (!funcList->cuGraphExecGetFlags)
    {
        return false;
    }
    funcList->cuGraphExecHostNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecHostNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecHostNodeSetParams"));
    if (!funcList->cuGraphExecHostNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecKernelNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecKernelNodeSetParams_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecKernelNodeSetParams_v2"));
    if (!funcList->cuGraphExecKernelNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecMemcpyNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecMemcpyNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecMemcpyNodeSetParams"));
    if (!funcList->cuGraphExecMemcpyNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecMemsetNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecMemsetNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecMemsetNodeSetParams"));
    if (!funcList->cuGraphExecMemsetNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecNodeSetParams"));
    if (!funcList->cuGraphExecNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExecUpdate = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExecUpdate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExecUpdate_v2"));
    if (!funcList->cuGraphExecUpdate)
    {
        return false;
    }
    funcList->cuGraphExternalSemaphoresSignalNodeGetParams = reinterpret_cast<
        NvcudaLibraryFunctions::cuGraphExternalSemaphoresSignalNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExternalSemaphoresSignalNodeGetParams"));
    if (!funcList->cuGraphExternalSemaphoresSignalNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphExternalSemaphoresSignalNodeSetParams = reinterpret_cast<
        NvcudaLibraryFunctions::cuGraphExternalSemaphoresSignalNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExternalSemaphoresSignalNodeSetParams"));
    if (!funcList->cuGraphExternalSemaphoresSignalNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphExternalSemaphoresWaitNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExternalSemaphoresWaitNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExternalSemaphoresWaitNodeGetParams"));
    if (!funcList->cuGraphExternalSemaphoresWaitNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphExternalSemaphoresWaitNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphExternalSemaphoresWaitNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphExternalSemaphoresWaitNodeSetParams"));
    if (!funcList->cuGraphExternalSemaphoresWaitNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphGetEdges = reinterpret_cast<NvcudaLibraryFunctions::cuGraphGetEdges_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphGetEdges_v2"));
    if (!funcList->cuGraphGetEdges)
    {
        return false;
    }
    funcList->cuGraphGetNodes = reinterpret_cast<NvcudaLibraryFunctions::cuGraphGetNodes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphGetNodes"));
    if (!funcList->cuGraphGetNodes)
    {
        return false;
    }
    funcList->cuGraphGetRootNodes = reinterpret_cast<NvcudaLibraryFunctions::cuGraphGetRootNodes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphGetRootNodes"));
    if (!funcList->cuGraphGetRootNodes)
    {
        return false;
    }
    funcList->cuGraphHostNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphHostNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphHostNodeGetParams"));
    if (!funcList->cuGraphHostNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphHostNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphHostNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphHostNodeSetParams"));
    if (!funcList->cuGraphHostNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphInstantiate = reinterpret_cast<NvcudaLibraryFunctions::cuGraphInstantiate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphInstantiate_v2"));
    if (!funcList->cuGraphInstantiate)
    {
        return false;
    }
    funcList->cuGraphInstantiateWithFlags = reinterpret_cast<NvcudaLibraryFunctions::cuGraphInstantiateWithFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphInstantiateWithFlags"));
    if (!funcList->cuGraphInstantiateWithFlags)
    {
        return false;
    }
    funcList->cuGraphInstantiateWithParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphInstantiateWithParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphInstantiateWithParams"));
    if (!funcList->cuGraphInstantiateWithParams)
    {
        return false;
    }
    funcList->cuGraphInstantiateWithParams_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuGraphInstantiateWithParams_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphInstantiateWithParams_ptsz"));
    if (!funcList->cuGraphInstantiateWithParams_ptsz)
    {
        return false;
    }
    funcList->cuGraphKernelNodeCopyAttributes = reinterpret_cast<NvcudaLibraryFunctions::cuGraphKernelNodeCopyAttributes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphKernelNodeCopyAttributes"));
    if (!funcList->cuGraphKernelNodeCopyAttributes)
    {
        return false;
    }
    funcList->cuGraphKernelNodeGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuGraphKernelNodeGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphKernelNodeGetAttribute"));
    if (!funcList->cuGraphKernelNodeGetAttribute)
    {
        return false;
    }
    funcList->cuGraphKernelNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphKernelNodeGetParams_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphKernelNodeGetParams_v2"));
    if (!funcList->cuGraphKernelNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphKernelNodeSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuGraphKernelNodeSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphKernelNodeSetAttribute"));
    if (!funcList->cuGraphKernelNodeSetAttribute)
    {
        return false;
    }
    funcList->cuGraphKernelNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphKernelNodeSetParams_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphKernelNodeSetParams_v2"));
    if (!funcList->cuGraphKernelNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphLaunch = reinterpret_cast<NvcudaLibraryFunctions::cuGraphLaunch_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphLaunch"));
    if (!funcList->cuGraphLaunch)
    {
        return false;
    }
    funcList->cuGraphLaunch_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuGraphLaunch_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphLaunch_ptsz"));
    if (!funcList->cuGraphLaunch_ptsz)
    {
        return false;
    }
    funcList->cuGraphMemAllocNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemAllocNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemAllocNodeGetParams"));
    if (!funcList->cuGraphMemAllocNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphMemFreeNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemFreeNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemFreeNodeGetParams"));
    if (!funcList->cuGraphMemFreeNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphMemcpyNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemcpyNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemcpyNodeGetParams"));
    if (!funcList->cuGraphMemcpyNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphMemcpyNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemcpyNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemcpyNodeSetParams"));
    if (!funcList->cuGraphMemcpyNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphMemsetNodeGetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemsetNodeGetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemsetNodeGetParams"));
    if (!funcList->cuGraphMemsetNodeGetParams)
    {
        return false;
    }
    funcList->cuGraphMemsetNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphMemsetNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphMemsetNodeSetParams"));
    if (!funcList->cuGraphMemsetNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphNodeFindInClone = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeFindInClone_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeFindInClone"));
    if (!funcList->cuGraphNodeFindInClone)
    {
        return false;
    }
    funcList->cuGraphNodeGetDependencies = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeGetDependencies_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeGetDependencies_v2"));
    if (!funcList->cuGraphNodeGetDependencies)
    {
        return false;
    }
    funcList->cuGraphNodeGetDependentNodes = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeGetDependentNodes_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeGetDependentNodes_v2"));
    if (!funcList->cuGraphNodeGetDependentNodes)
    {
        return false;
    }
    funcList->cuGraphNodeGetEnabled = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeGetEnabled_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeGetEnabled"));
    if (!funcList->cuGraphNodeGetEnabled)
    {
        return false;
    }
    funcList->cuGraphNodeGetType = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeGetType_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeGetType"));
    if (!funcList->cuGraphNodeGetType)
    {
        return false;
    }
    funcList->cuGraphNodeSetEnabled = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeSetEnabled_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeSetEnabled"));
    if (!funcList->cuGraphNodeSetEnabled)
    {
        return false;
    }
    funcList->cuGraphNodeSetParams = reinterpret_cast<NvcudaLibraryFunctions::cuGraphNodeSetParams_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphNodeSetParams"));
    if (!funcList->cuGraphNodeSetParams)
    {
        return false;
    }
    funcList->cuGraphReleaseUserObject = reinterpret_cast<NvcudaLibraryFunctions::cuGraphReleaseUserObject_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphReleaseUserObject"));
    if (!funcList->cuGraphReleaseUserObject)
    {
        return false;
    }
    funcList->cuGraphRemoveDependencies = reinterpret_cast<NvcudaLibraryFunctions::cuGraphRemoveDependencies_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphRemoveDependencies_v2"));
    if (!funcList->cuGraphRemoveDependencies)
    {
        return false;
    }
    funcList->cuGraphRetainUserObject = reinterpret_cast<NvcudaLibraryFunctions::cuGraphRetainUserObject_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphRetainUserObject"));
    if (!funcList->cuGraphRetainUserObject)
    {
        return false;
    }
    funcList->cuGraphUpload = reinterpret_cast<NvcudaLibraryFunctions::cuGraphUpload_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphUpload"));
    if (!funcList->cuGraphUpload)
    {
        return false;
    }
    funcList->cuGraphUpload_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuGraphUpload_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphUpload_ptsz"));
    if (!funcList->cuGraphUpload_ptsz)
    {
        return false;
    }
    funcList->cuGraphicsD3D10RegisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsD3D10RegisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsD3D10RegisterResource"));
    if (!funcList->cuGraphicsD3D10RegisterResource)
    {
        return false;
    }
    funcList->cuGraphicsD3D11RegisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsD3D11RegisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsD3D11RegisterResource"));
    if (!funcList->cuGraphicsD3D11RegisterResource)
    {
        return false;
    }
    funcList->cuGraphicsD3D9RegisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsD3D9RegisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsD3D9RegisterResource"));
    if (!funcList->cuGraphicsD3D9RegisterResource)
    {
        return false;
    }
    funcList->cuGraphicsGLRegisterBuffer = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsGLRegisterBuffer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsGLRegisterBuffer"));
    if (!funcList->cuGraphicsGLRegisterBuffer)
    {
        return false;
    }
    funcList->cuGraphicsGLRegisterImage = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsGLRegisterImage_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsGLRegisterImage"));
    if (!funcList->cuGraphicsGLRegisterImage)
    {
        return false;
    }
    funcList->cuGraphicsMapResources = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsMapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsMapResources"));
    if (!funcList->cuGraphicsMapResources)
    {
        return false;
    }
    funcList->cuGraphicsMapResources_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsMapResources_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsMapResources_ptsz"));
    if (!funcList->cuGraphicsMapResources_ptsz)
    {
        return false;
    }
    funcList->cuGraphicsResourceGetMappedMipmappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsResourceGetMappedMipmappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsResourceGetMappedMipmappedArray"));
    if (!funcList->cuGraphicsResourceGetMappedMipmappedArray)
    {
        return false;
    }
    funcList->cuGraphicsResourceGetMappedPointer = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsResourceGetMappedPointer_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsResourceGetMappedPointer_v2"));
    if (!funcList->cuGraphicsResourceGetMappedPointer)
    {
        return false;
    }
    funcList->cuGraphicsResourceSetMapFlags = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsResourceSetMapFlags_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsResourceSetMapFlags_v2"));
    if (!funcList->cuGraphicsResourceSetMapFlags)
    {
        return false;
    }
    funcList->cuGraphicsSubResourceGetMappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsSubResourceGetMappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsSubResourceGetMappedArray"));
    if (!funcList->cuGraphicsSubResourceGetMappedArray)
    {
        return false;
    }
    funcList->cuGraphicsUnmapResources = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsUnmapResources_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsUnmapResources"));
    if (!funcList->cuGraphicsUnmapResources)
    {
        return false;
    }
    funcList->cuGraphicsUnmapResources_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsUnmapResources_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsUnmapResources_ptsz"));
    if (!funcList->cuGraphicsUnmapResources_ptsz)
    {
        return false;
    }
    funcList->cuGraphicsUnregisterResource = reinterpret_cast<NvcudaLibraryFunctions::cuGraphicsUnregisterResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGraphicsUnregisterResource"));
    if (!funcList->cuGraphicsUnregisterResource)
    {
        return false;
    }
    funcList->cuGreenCtxCreate = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxCreate"));
    if (!funcList->cuGreenCtxCreate)
    {
        return false;
    }
    funcList->cuGreenCtxDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxDestroy"));
    if (!funcList->cuGreenCtxDestroy)
    {
        return false;
    }
    funcList->cuGreenCtxGetDevResource = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxGetDevResource_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxGetDevResource"));
    if (!funcList->cuGreenCtxGetDevResource)
    {
        return false;
    }
    funcList->cuGreenCtxRecordEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxRecordEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxRecordEvent"));
    if (!funcList->cuGreenCtxRecordEvent)
    {
        return false;
    }
    funcList->cuGreenCtxStreamCreate = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxStreamCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxStreamCreate"));
    if (!funcList->cuGreenCtxStreamCreate)
    {
        return false;
    }
    funcList->cuGreenCtxWaitEvent = reinterpret_cast<NvcudaLibraryFunctions::cuGreenCtxWaitEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuGreenCtxWaitEvent"));
    if (!funcList->cuGreenCtxWaitEvent)
    {
        return false;
    }
    funcList->cuImportExternalMemory = reinterpret_cast<NvcudaLibraryFunctions::cuImportExternalMemory_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuImportExternalMemory"));
    if (!funcList->cuImportExternalMemory)
    {
        return false;
    }
    funcList->cuImportExternalSemaphore = reinterpret_cast<NvcudaLibraryFunctions::cuImportExternalSemaphore_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuImportExternalSemaphore"));
    if (!funcList->cuImportExternalSemaphore)
    {
        return false;
    }
    funcList->cuInit = reinterpret_cast<NvcudaLibraryFunctions::cuInit_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuInit"));
    if (!funcList->cuInit)
    {
        return false;
    }
    funcList->cuIpcCloseMemHandle = reinterpret_cast<NvcudaLibraryFunctions::cuIpcCloseMemHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuIpcCloseMemHandle"));
    if (!funcList->cuIpcCloseMemHandle)
    {
        return false;
    }
    funcList->cuIpcGetEventHandle = reinterpret_cast<NvcudaLibraryFunctions::cuIpcGetEventHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuIpcGetEventHandle"));
    if (!funcList->cuIpcGetEventHandle)
    {
        return false;
    }
    funcList->cuIpcGetMemHandle = reinterpret_cast<NvcudaLibraryFunctions::cuIpcGetMemHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuIpcGetMemHandle"));
    if (!funcList->cuIpcGetMemHandle)
    {
        return false;
    }
    funcList->cuIpcOpenEventHandle = reinterpret_cast<NvcudaLibraryFunctions::cuIpcOpenEventHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuIpcOpenEventHandle"));
    if (!funcList->cuIpcOpenEventHandle)
    {
        return false;
    }
    funcList->cuIpcOpenMemHandle = reinterpret_cast<NvcudaLibraryFunctions::cuIpcOpenMemHandle_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuIpcOpenMemHandle_v2"));
    if (!funcList->cuIpcOpenMemHandle)
    {
        return false;
    }
    funcList->cuKernelGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuKernelGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelGetAttribute"));
    if (!funcList->cuKernelGetAttribute)
    {
        return false;
    }
    funcList->cuKernelGetFunction = reinterpret_cast<NvcudaLibraryFunctions::cuKernelGetFunction_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelGetFunction"));
    if (!funcList->cuKernelGetFunction)
    {
        return false;
    }
    funcList->cuKernelGetLibrary = reinterpret_cast<NvcudaLibraryFunctions::cuKernelGetLibrary_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelGetLibrary"));
    if (!funcList->cuKernelGetLibrary)
    {
        return false;
    }
    funcList->cuKernelGetName = reinterpret_cast<NvcudaLibraryFunctions::cuKernelGetName_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelGetName"));
    if (!funcList->cuKernelGetName)
    {
        return false;
    }
    funcList->cuKernelGetParamInfo = reinterpret_cast<NvcudaLibraryFunctions::cuKernelGetParamInfo_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelGetParamInfo"));
    if (!funcList->cuKernelGetParamInfo)
    {
        return false;
    }
    funcList->cuKernelSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuKernelSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelSetAttribute"));
    if (!funcList->cuKernelSetAttribute)
    {
        return false;
    }
    funcList->cuKernelSetCacheConfig = reinterpret_cast<NvcudaLibraryFunctions::cuKernelSetCacheConfig_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuKernelSetCacheConfig"));
    if (!funcList->cuKernelSetCacheConfig)
    {
        return false;
    }
    funcList->cuLaunch = reinterpret_cast<NvcudaLibraryFunctions::cuLaunch_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunch"));
    if (!funcList->cuLaunch)
    {
        return false;
    }
    funcList->cuLaunchCooperativeKernel = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchCooperativeKernel_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchCooperativeKernel"));
    if (!funcList->cuLaunchCooperativeKernel)
    {
        return false;
    }
    funcList->cuLaunchCooperativeKernelMultiDevice = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchCooperativeKernelMultiDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchCooperativeKernelMultiDevice"));
    if (!funcList->cuLaunchCooperativeKernelMultiDevice)
    {
        return false;
    }
    funcList->cuLaunchCooperativeKernel_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchCooperativeKernel_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchCooperativeKernel_ptsz"));
    if (!funcList->cuLaunchCooperativeKernel_ptsz)
    {
        return false;
    }
    funcList->cuLaunchGrid = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchGrid_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchGrid"));
    if (!funcList->cuLaunchGrid)
    {
        return false;
    }
    funcList->cuLaunchGridAsync = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchGridAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchGridAsync"));
    if (!funcList->cuLaunchGridAsync)
    {
        return false;
    }
    funcList->cuLaunchHostFunc = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchHostFunc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchHostFunc"));
    if (!funcList->cuLaunchHostFunc)
    {
        return false;
    }
    funcList->cuLaunchHostFunc_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchHostFunc_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchHostFunc_ptsz"));
    if (!funcList->cuLaunchHostFunc_ptsz)
    {
        return false;
    }
    funcList->cuLaunchKernel = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchKernel_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchKernel"));
    if (!funcList->cuLaunchKernel)
    {
        return false;
    }
    funcList->cuLaunchKernelEx = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchKernelEx_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchKernelEx"));
    if (!funcList->cuLaunchKernelEx)
    {
        return false;
    }
    funcList->cuLaunchKernelEx_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchKernelEx_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchKernelEx_ptsz"));
    if (!funcList->cuLaunchKernelEx_ptsz)
    {
        return false;
    }
    funcList->cuLaunchKernel_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuLaunchKernel_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLaunchKernel_ptsz"));
    if (!funcList->cuLaunchKernel_ptsz)
    {
        return false;
    }
    funcList->cuLibraryEnumerateKernels = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryEnumerateKernels_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryEnumerateKernels"));
    if (!funcList->cuLibraryEnumerateKernels)
    {
        return false;
    }
    funcList->cuLibraryGetGlobal = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetGlobal_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetGlobal"));
    if (!funcList->cuLibraryGetGlobal)
    {
        return false;
    }
    funcList->cuLibraryGetKernel = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetKernel_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetKernel"));
    if (!funcList->cuLibraryGetKernel)
    {
        return false;
    }
    funcList->cuLibraryGetKernelCount = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetKernelCount_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetKernelCount"));
    if (!funcList->cuLibraryGetKernelCount)
    {
        return false;
    }
    funcList->cuLibraryGetManaged = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetManaged_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetManaged"));
    if (!funcList->cuLibraryGetManaged)
    {
        return false;
    }
    funcList->cuLibraryGetModule = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetModule_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetModule"));
    if (!funcList->cuLibraryGetModule)
    {
        return false;
    }
    funcList->cuLibraryGetUnifiedFunction = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryGetUnifiedFunction_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryGetUnifiedFunction"));
    if (!funcList->cuLibraryGetUnifiedFunction)
    {
        return false;
    }
    funcList->cuLibraryLoadData = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryLoadData_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryLoadData"));
    if (!funcList->cuLibraryLoadData)
    {
        return false;
    }
    funcList->cuLibraryLoadFromFile = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryLoadFromFile_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryLoadFromFile"));
    if (!funcList->cuLibraryLoadFromFile)
    {
        return false;
    }
    funcList->cuLibraryUnload = reinterpret_cast<NvcudaLibraryFunctions::cuLibraryUnload_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLibraryUnload"));
    if (!funcList->cuLibraryUnload)
    {
        return false;
    }
    funcList->cuLinkAddData = reinterpret_cast<NvcudaLibraryFunctions::cuLinkAddData_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLinkAddData_v2"));
    if (!funcList->cuLinkAddData)
    {
        return false;
    }
    funcList->cuLinkAddFile = reinterpret_cast<NvcudaLibraryFunctions::cuLinkAddFile_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLinkAddFile_v2"));
    if (!funcList->cuLinkAddFile)
    {
        return false;
    }
    funcList->cuLinkComplete = reinterpret_cast<NvcudaLibraryFunctions::cuLinkComplete_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLinkComplete"));
    if (!funcList->cuLinkComplete)
    {
        return false;
    }
    funcList->cuLinkCreate = reinterpret_cast<NvcudaLibraryFunctions::cuLinkCreate_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLinkCreate_v2"));
    if (!funcList->cuLinkCreate)
    {
        return false;
    }
    funcList->cuLinkDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuLinkDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuLinkDestroy"));
    if (!funcList->cuLinkDestroy)
    {
        return false;
    }
    funcList->cuMemAddressFree = reinterpret_cast<NvcudaLibraryFunctions::cuMemAddressFree_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAddressFree"));
    if (!funcList->cuMemAddressFree)
    {
        return false;
    }
    funcList->cuMemAddressReserve = reinterpret_cast<NvcudaLibraryFunctions::cuMemAddressReserve_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAddressReserve"));
    if (!funcList->cuMemAddressReserve)
    {
        return false;
    }
    funcList->cuMemAdvise = reinterpret_cast<NvcudaLibraryFunctions::cuMemAdvise_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAdvise_v2"));
    if (!funcList->cuMemAdvise)
    {
        return false;
    }
    funcList->cuMemAlloc = reinterpret_cast<NvcudaLibraryFunctions::cuMemAlloc_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAlloc_v2"));
    if (!funcList->cuMemAlloc)
    {
        return false;
    }
    funcList->cuMemAllocAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocAsync"));
    if (!funcList->cuMemAllocAsync)
    {
        return false;
    }
    funcList->cuMemAllocAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocAsync_ptsz"));
    if (!funcList->cuMemAllocAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemAllocFromPoolAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocFromPoolAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocFromPoolAsync"));
    if (!funcList->cuMemAllocFromPoolAsync)
    {
        return false;
    }
    funcList->cuMemAllocFromPoolAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocFromPoolAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocFromPoolAsync_ptsz"));
    if (!funcList->cuMemAllocFromPoolAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemAllocHost = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocHost_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocHost_v2"));
    if (!funcList->cuMemAllocHost)
    {
        return false;
    }
    funcList->cuMemAllocManaged = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocManaged_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocManaged"));
    if (!funcList->cuMemAllocManaged)
    {
        return false;
    }
    funcList->cuMemAllocPitch = reinterpret_cast<NvcudaLibraryFunctions::cuMemAllocPitch_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemAllocPitch_v2"));
    if (!funcList->cuMemAllocPitch)
    {
        return false;
    }
    funcList->cuMemBatchDecompressAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemBatchDecompressAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemBatchDecompressAsync"));
    if (!funcList->cuMemBatchDecompressAsync)
    {
        return false;
    }
    funcList->cuMemBatchDecompressAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemBatchDecompressAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemBatchDecompressAsync_ptsz"));
    if (!funcList->cuMemBatchDecompressAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemCreate = reinterpret_cast<NvcudaLibraryFunctions::cuMemCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemCreate"));
    if (!funcList->cuMemCreate)
    {
        return false;
    }
    funcList->cuMemExportToShareableHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemExportToShareableHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemExportToShareableHandle"));
    if (!funcList->cuMemExportToShareableHandle)
    {
        return false;
    }
    funcList->cuMemFree = reinterpret_cast<NvcudaLibraryFunctions::cuMemFree_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemFree_v2"));
    if (!funcList->cuMemFree)
    {
        return false;
    }
    funcList->cuMemFreeAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemFreeAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemFreeAsync"));
    if (!funcList->cuMemFreeAsync)
    {
        return false;
    }
    funcList->cuMemFreeAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemFreeAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemFreeAsync_ptsz"));
    if (!funcList->cuMemFreeAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemFreeHost = reinterpret_cast<NvcudaLibraryFunctions::cuMemFreeHost_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemFreeHost"));
    if (!funcList->cuMemFreeHost)
    {
        return false;
    }
    funcList->cuMemGetAccess = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetAccess"));
    if (!funcList->cuMemGetAccess)
    {
        return false;
    }
    funcList->cuMemGetAddressRange = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetAddressRange_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetAddressRange_v2"));
    if (!funcList->cuMemGetAddressRange)
    {
        return false;
    }
    funcList->cuMemGetAllocationGranularity = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetAllocationGranularity_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetAllocationGranularity"));
    if (!funcList->cuMemGetAllocationGranularity)
    {
        return false;
    }
    funcList->cuMemGetAllocationPropertiesFromHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetAllocationPropertiesFromHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetAllocationPropertiesFromHandle"));
    if (!funcList->cuMemGetAllocationPropertiesFromHandle)
    {
        return false;
    }
    funcList->cuMemGetHandleForAddressRange = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetHandleForAddressRange_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetHandleForAddressRange"));
    if (!funcList->cuMemGetHandleForAddressRange)
    {
        return false;
    }
    funcList->cuMemGetInfo = reinterpret_cast<NvcudaLibraryFunctions::cuMemGetInfo_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemGetInfo_v2"));
    if (!funcList->cuMemGetInfo)
    {
        return false;
    }
    funcList->cuMemHostAlloc = reinterpret_cast<NvcudaLibraryFunctions::cuMemHostAlloc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemHostAlloc"));
    if (!funcList->cuMemHostAlloc)
    {
        return false;
    }
    funcList->cuMemHostGetDevicePointer = reinterpret_cast<NvcudaLibraryFunctions::cuMemHostGetDevicePointer_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemHostGetDevicePointer_v2"));
    if (!funcList->cuMemHostGetDevicePointer)
    {
        return false;
    }
    funcList->cuMemHostGetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuMemHostGetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemHostGetFlags"));
    if (!funcList->cuMemHostGetFlags)
    {
        return false;
    }
    funcList->cuMemHostRegister = reinterpret_cast<NvcudaLibraryFunctions::cuMemHostRegister_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemHostRegister_v2"));
    if (!funcList->cuMemHostRegister)
    {
        return false;
    }
    funcList->cuMemHostUnregister = reinterpret_cast<NvcudaLibraryFunctions::cuMemHostUnregister_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemHostUnregister"));
    if (!funcList->cuMemHostUnregister)
    {
        return false;
    }
    funcList->cuMemImportFromShareableHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemImportFromShareableHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemImportFromShareableHandle"));
    if (!funcList->cuMemImportFromShareableHandle)
    {
        return false;
    }
    funcList->cuMemMap = reinterpret_cast<NvcudaLibraryFunctions::cuMemMap_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemMap"));
    if (!funcList->cuMemMap)
    {
        return false;
    }
    funcList->cuMemMapArrayAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemMapArrayAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemMapArrayAsync"));
    if (!funcList->cuMemMapArrayAsync)
    {
        return false;
    }
    funcList->cuMemMapArrayAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemMapArrayAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemMapArrayAsync_ptsz"));
    if (!funcList->cuMemMapArrayAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemPoolCreate = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolCreate"));
    if (!funcList->cuMemPoolCreate)
    {
        return false;
    }
    funcList->cuMemPoolDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolDestroy"));
    if (!funcList->cuMemPoolDestroy)
    {
        return false;
    }
    funcList->cuMemPoolExportPointer = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolExportPointer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolExportPointer"));
    if (!funcList->cuMemPoolExportPointer)
    {
        return false;
    }
    funcList->cuMemPoolExportToShareableHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolExportToShareableHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolExportToShareableHandle"));
    if (!funcList->cuMemPoolExportToShareableHandle)
    {
        return false;
    }
    funcList->cuMemPoolGetAccess = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolGetAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolGetAccess"));
    if (!funcList->cuMemPoolGetAccess)
    {
        return false;
    }
    funcList->cuMemPoolGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolGetAttribute"));
    if (!funcList->cuMemPoolGetAttribute)
    {
        return false;
    }
    funcList->cuMemPoolImportFromShareableHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolImportFromShareableHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolImportFromShareableHandle"));
    if (!funcList->cuMemPoolImportFromShareableHandle)
    {
        return false;
    }
    funcList->cuMemPoolImportPointer = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolImportPointer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolImportPointer"));
    if (!funcList->cuMemPoolImportPointer)
    {
        return false;
    }
    funcList->cuMemPoolSetAccess = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolSetAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolSetAccess"));
    if (!funcList->cuMemPoolSetAccess)
    {
        return false;
    }
    funcList->cuMemPoolSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolSetAttribute"));
    if (!funcList->cuMemPoolSetAttribute)
    {
        return false;
    }
    funcList->cuMemPoolTrimTo = reinterpret_cast<NvcudaLibraryFunctions::cuMemPoolTrimTo_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPoolTrimTo"));
    if (!funcList->cuMemPoolTrimTo)
    {
        return false;
    }
    funcList->cuMemPrefetchAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemPrefetchAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPrefetchAsync_v2"));
    if (!funcList->cuMemPrefetchAsync)
    {
        return false;
    }
    funcList->cuMemPrefetchAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemPrefetchAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemPrefetchAsync_ptsz"));
    if (!funcList->cuMemPrefetchAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemRangeGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuMemRangeGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemRangeGetAttribute"));
    if (!funcList->cuMemRangeGetAttribute)
    {
        return false;
    }
    funcList->cuMemRangeGetAttributes = reinterpret_cast<NvcudaLibraryFunctions::cuMemRangeGetAttributes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemRangeGetAttributes"));
    if (!funcList->cuMemRangeGetAttributes)
    {
        return false;
    }
    funcList->cuMemRelease = reinterpret_cast<NvcudaLibraryFunctions::cuMemRelease_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemRelease"));
    if (!funcList->cuMemRelease)
    {
        return false;
    }
    funcList->cuMemRetainAllocationHandle = reinterpret_cast<NvcudaLibraryFunctions::cuMemRetainAllocationHandle_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemRetainAllocationHandle"));
    if (!funcList->cuMemRetainAllocationHandle)
    {
        return false;
    }
    funcList->cuMemSetAccess = reinterpret_cast<NvcudaLibraryFunctions::cuMemSetAccess_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemSetAccess"));
    if (!funcList->cuMemSetAccess)
    {
        return false;
    }
    funcList->cuMemUnmap = reinterpret_cast<NvcudaLibraryFunctions::cuMemUnmap_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemUnmap"));
    if (!funcList->cuMemUnmap)
    {
        return false;
    }
    funcList->cuMemcpy = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy"));
    if (!funcList->cuMemcpy)
    {
        return false;
    }
    funcList->cuMemcpy2D = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy2D_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy2D_v2"));
    if (!funcList->cuMemcpy2D)
    {
        return false;
    }
    funcList->cuMemcpy2DAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy2DAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy2DAsync_v2"));
    if (!funcList->cuMemcpy2DAsync)
    {
        return false;
    }
    funcList->cuMemcpy2DUnaligned = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy2DUnaligned_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy2DUnaligned_v2"));
    if (!funcList->cuMemcpy2DUnaligned)
    {
        return false;
    }
    funcList->cuMemcpy3D = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3D_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3D_v2"));
    if (!funcList->cuMemcpy3D)
    {
        return false;
    }
    funcList->cuMemcpy3DAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DAsync_v2"));
    if (!funcList->cuMemcpy3DAsync)
    {
        return false;
    }
    funcList->cuMemcpy3DBatchAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DBatchAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DBatchAsync"));
    if (!funcList->cuMemcpy3DBatchAsync)
    {
        return false;
    }
    funcList->cuMemcpy3DBatchAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DBatchAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DBatchAsync_ptsz"));
    if (!funcList->cuMemcpy3DBatchAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemcpy3DPeer = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DPeer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DPeer"));
    if (!funcList->cuMemcpy3DPeer)
    {
        return false;
    }
    funcList->cuMemcpy3DPeerAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DPeerAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DPeerAsync"));
    if (!funcList->cuMemcpy3DPeerAsync)
    {
        return false;
    }
    funcList->cuMemcpy3DPeerAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DPeerAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DPeerAsync_ptsz"));
    if (!funcList->cuMemcpy3DPeerAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemcpy3DPeer_ptds = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy3DPeer_ptds_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy3DPeer_ptds"));
    if (!funcList->cuMemcpy3DPeer_ptds)
    {
        return false;
    }
    funcList->cuMemcpyAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAsync"));
    if (!funcList->cuMemcpyAsync)
    {
        return false;
    }
    funcList->cuMemcpyAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAsync_ptsz"));
    if (!funcList->cuMemcpyAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemcpyAtoA = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAtoA_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAtoA_v2"));
    if (!funcList->cuMemcpyAtoA)
    {
        return false;
    }
    funcList->cuMemcpyAtoD = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAtoD_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAtoD_v2"));
    if (!funcList->cuMemcpyAtoD)
    {
        return false;
    }
    funcList->cuMemcpyAtoH = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAtoH_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAtoH_v2"));
    if (!funcList->cuMemcpyAtoH)
    {
        return false;
    }
    funcList->cuMemcpyAtoHAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyAtoHAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyAtoHAsync_v2"));
    if (!funcList->cuMemcpyAtoHAsync)
    {
        return false;
    }
    funcList->cuMemcpyBatchAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyBatchAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyBatchAsync"));
    if (!funcList->cuMemcpyBatchAsync)
    {
        return false;
    }
    funcList->cuMemcpyBatchAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyBatchAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyBatchAsync_ptsz"));
    if (!funcList->cuMemcpyBatchAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemcpyDtoA = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyDtoA_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyDtoA_v2"));
    if (!funcList->cuMemcpyDtoA)
    {
        return false;
    }
    funcList->cuMemcpyDtoD = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyDtoD_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyDtoD_v2"));
    if (!funcList->cuMemcpyDtoD)
    {
        return false;
    }
    funcList->cuMemcpyDtoDAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyDtoDAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyDtoDAsync_v2"));
    if (!funcList->cuMemcpyDtoDAsync)
    {
        return false;
    }
    funcList->cuMemcpyDtoH = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyDtoH_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyDtoH_v2"));
    if (!funcList->cuMemcpyDtoH)
    {
        return false;
    }
    funcList->cuMemcpyDtoHAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyDtoHAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyDtoHAsync_v2"));
    if (!funcList->cuMemcpyDtoHAsync)
    {
        return false;
    }
    funcList->cuMemcpyHtoA = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyHtoA_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyHtoA_v2"));
    if (!funcList->cuMemcpyHtoA)
    {
        return false;
    }
    funcList->cuMemcpyHtoAAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyHtoAAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyHtoAAsync_v2"));
    if (!funcList->cuMemcpyHtoAAsync)
    {
        return false;
    }
    funcList->cuMemcpyHtoD = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyHtoD_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyHtoD_v2"));
    if (!funcList->cuMemcpyHtoD)
    {
        return false;
    }
    funcList->cuMemcpyHtoDAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyHtoDAsync_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyHtoDAsync_v2"));
    if (!funcList->cuMemcpyHtoDAsync)
    {
        return false;
    }
    funcList->cuMemcpyPeer = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyPeer_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyPeer"));
    if (!funcList->cuMemcpyPeer)
    {
        return false;
    }
    funcList->cuMemcpyPeerAsync = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyPeerAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyPeerAsync"));
    if (!funcList->cuMemcpyPeerAsync)
    {
        return false;
    }
    funcList->cuMemcpyPeerAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyPeerAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyPeerAsync_ptsz"));
    if (!funcList->cuMemcpyPeerAsync_ptsz)
    {
        return false;
    }
    funcList->cuMemcpyPeer_ptds = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpyPeer_ptds_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpyPeer_ptds"));
    if (!funcList->cuMemcpyPeer_ptds)
    {
        return false;
    }
    funcList->cuMemcpy_ptds = reinterpret_cast<NvcudaLibraryFunctions::cuMemcpy_ptds_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemcpy_ptds"));
    if (!funcList->cuMemcpy_ptds)
    {
        return false;
    }
    funcList->cuMemsetD16 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD16_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD16_v2"));
    if (!funcList->cuMemsetD16)
    {
        return false;
    }
    funcList->cuMemsetD16Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD16Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD16Async"));
    if (!funcList->cuMemsetD16Async)
    {
        return false;
    }
    funcList->cuMemsetD16Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD16Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD16Async_ptsz"));
    if (!funcList->cuMemsetD16Async_ptsz)
    {
        return false;
    }
    funcList->cuMemsetD2D16 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D16_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D16_v2"));
    if (!funcList->cuMemsetD2D16)
    {
        return false;
    }
    funcList->cuMemsetD2D16Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D16Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D16Async"));
    if (!funcList->cuMemsetD2D16Async)
    {
        return false;
    }
    funcList->cuMemsetD2D16Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D16Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D16Async_ptsz"));
    if (!funcList->cuMemsetD2D16Async_ptsz)
    {
        return false;
    }
    funcList->cuMemsetD2D32 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D32_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D32_v2"));
    if (!funcList->cuMemsetD2D32)
    {
        return false;
    }
    funcList->cuMemsetD2D32Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D32Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D32Async"));
    if (!funcList->cuMemsetD2D32Async)
    {
        return false;
    }
    funcList->cuMemsetD2D32Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D32Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D32Async_ptsz"));
    if (!funcList->cuMemsetD2D32Async_ptsz)
    {
        return false;
    }
    funcList->cuMemsetD2D8 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D8_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D8_v2"));
    if (!funcList->cuMemsetD2D8)
    {
        return false;
    }
    funcList->cuMemsetD2D8Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D8Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D8Async"));
    if (!funcList->cuMemsetD2D8Async)
    {
        return false;
    }
    funcList->cuMemsetD2D8Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD2D8Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD2D8Async_ptsz"));
    if (!funcList->cuMemsetD2D8Async_ptsz)
    {
        return false;
    }
    funcList->cuMemsetD32 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD32_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD32_v2"));
    if (!funcList->cuMemsetD32)
    {
        return false;
    }
    funcList->cuMemsetD32Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD32Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD32Async"));
    if (!funcList->cuMemsetD32Async)
    {
        return false;
    }
    funcList->cuMemsetD32Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD32Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD32Async_ptsz"));
    if (!funcList->cuMemsetD32Async_ptsz)
    {
        return false;
    }
    funcList->cuMemsetD8 = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD8_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD8_v2"));
    if (!funcList->cuMemsetD8)
    {
        return false;
    }
    funcList->cuMemsetD8Async = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD8Async_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD8Async"));
    if (!funcList->cuMemsetD8Async)
    {
        return false;
    }
    funcList->cuMemsetD8Async_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuMemsetD8Async_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMemsetD8Async_ptsz"));
    if (!funcList->cuMemsetD8Async_ptsz)
    {
        return false;
    }
    funcList->cuMipmappedArrayCreate = reinterpret_cast<NvcudaLibraryFunctions::cuMipmappedArrayCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMipmappedArrayCreate"));
    if (!funcList->cuMipmappedArrayCreate)
    {
        return false;
    }
    funcList->cuMipmappedArrayDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuMipmappedArrayDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMipmappedArrayDestroy"));
    if (!funcList->cuMipmappedArrayDestroy)
    {
        return false;
    }
    funcList->cuMipmappedArrayGetLevel = reinterpret_cast<NvcudaLibraryFunctions::cuMipmappedArrayGetLevel_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMipmappedArrayGetLevel"));
    if (!funcList->cuMipmappedArrayGetLevel)
    {
        return false;
    }
    funcList->cuMipmappedArrayGetMemoryRequirements = reinterpret_cast<NvcudaLibraryFunctions::cuMipmappedArrayGetMemoryRequirements_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMipmappedArrayGetMemoryRequirements"));
    if (!funcList->cuMipmappedArrayGetMemoryRequirements)
    {
        return false;
    }
    funcList->cuMipmappedArrayGetSparseProperties = reinterpret_cast<NvcudaLibraryFunctions::cuMipmappedArrayGetSparseProperties_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMipmappedArrayGetSparseProperties"));
    if (!funcList->cuMipmappedArrayGetSparseProperties)
    {
        return false;
    }
    funcList->cuModuleEnumerateFunctions = reinterpret_cast<NvcudaLibraryFunctions::cuModuleEnumerateFunctions_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleEnumerateFunctions"));
    if (!funcList->cuModuleEnumerateFunctions)
    {
        return false;
    }
    funcList->cuModuleGetFunction = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetFunction_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetFunction"));
    if (!funcList->cuModuleGetFunction)
    {
        return false;
    }
    funcList->cuModuleGetFunctionCount = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetFunctionCount_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetFunctionCount"));
    if (!funcList->cuModuleGetFunctionCount)
    {
        return false;
    }
    funcList->cuModuleGetGlobal = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetGlobal_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetGlobal_v2"));
    if (!funcList->cuModuleGetGlobal)
    {
        return false;
    }
    funcList->cuModuleGetLoadingMode = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetLoadingMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetLoadingMode"));
    if (!funcList->cuModuleGetLoadingMode)
    {
        return false;
    }
    funcList->cuModuleGetSurfRef = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetSurfRef_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetSurfRef"));
    if (!funcList->cuModuleGetSurfRef)
    {
        return false;
    }
    funcList->cuModuleGetTexRef = reinterpret_cast<NvcudaLibraryFunctions::cuModuleGetTexRef_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleGetTexRef"));
    if (!funcList->cuModuleGetTexRef)
    {
        return false;
    }
    funcList->cuModuleLoad = reinterpret_cast<NvcudaLibraryFunctions::cuModuleLoad_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleLoad"));
    if (!funcList->cuModuleLoad)
    {
        return false;
    }
    funcList->cuModuleLoadData = reinterpret_cast<NvcudaLibraryFunctions::cuModuleLoadData_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleLoadData"));
    if (!funcList->cuModuleLoadData)
    {
        return false;
    }
    funcList->cuModuleLoadDataEx = reinterpret_cast<NvcudaLibraryFunctions::cuModuleLoadDataEx_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleLoadDataEx"));
    if (!funcList->cuModuleLoadDataEx)
    {
        return false;
    }
    funcList->cuModuleLoadFatBinary = reinterpret_cast<NvcudaLibraryFunctions::cuModuleLoadFatBinary_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleLoadFatBinary"));
    if (!funcList->cuModuleLoadFatBinary)
    {
        return false;
    }
    funcList->cuModuleUnload = reinterpret_cast<NvcudaLibraryFunctions::cuModuleUnload_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuModuleUnload"));
    if (!funcList->cuModuleUnload)
    {
        return false;
    }
    funcList->cuMulticastAddDevice = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastAddDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastAddDevice"));
    if (!funcList->cuMulticastAddDevice)
    {
        return false;
    }
    funcList->cuMulticastBindAddr = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastBindAddr_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastBindAddr"));
    if (!funcList->cuMulticastBindAddr)
    {
        return false;
    }
    funcList->cuMulticastBindMem = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastBindMem_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastBindMem"));
    if (!funcList->cuMulticastBindMem)
    {
        return false;
    }
    funcList->cuMulticastCreate = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastCreate"));
    if (!funcList->cuMulticastCreate)
    {
        return false;
    }
    funcList->cuMulticastGetGranularity = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastGetGranularity_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastGetGranularity"));
    if (!funcList->cuMulticastGetGranularity)
    {
        return false;
    }
    funcList->cuMulticastUnbind = reinterpret_cast<NvcudaLibraryFunctions::cuMulticastUnbind_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuMulticastUnbind"));
    if (!funcList->cuMulticastUnbind)
    {
        return false;
    }
    funcList->cuOccupancyAvailableDynamicSMemPerBlock = reinterpret_cast<NvcudaLibraryFunctions::cuOccupancyAvailableDynamicSMemPerBlock_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyAvailableDynamicSMemPerBlock"));
    if (!funcList->cuOccupancyAvailableDynamicSMemPerBlock)
    {
        return false;
    }
    funcList->cuOccupancyMaxActiveBlocksPerMultiprocessor = reinterpret_cast<
        NvcudaLibraryFunctions::cuOccupancyMaxActiveBlocksPerMultiprocessor_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxActiveBlocksPerMultiprocessor"));
    if (!funcList->cuOccupancyMaxActiveBlocksPerMultiprocessor)
    {
        return false;
    }
    funcList->cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = reinterpret_cast<
        NvcudaLibraryFunctions::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
    if (!funcList->cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
    {
        return false;
    }
    funcList->cuOccupancyMaxActiveClusters = reinterpret_cast<NvcudaLibraryFunctions::cuOccupancyMaxActiveClusters_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxActiveClusters"));
    if (!funcList->cuOccupancyMaxActiveClusters)
    {
        return false;
    }
    funcList->cuOccupancyMaxPotentialBlockSize = reinterpret_cast<NvcudaLibraryFunctions::cuOccupancyMaxPotentialBlockSize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxPotentialBlockSize"));
    if (!funcList->cuOccupancyMaxPotentialBlockSize)
    {
        return false;
    }
    funcList->cuOccupancyMaxPotentialBlockSizeWithFlags = reinterpret_cast<NvcudaLibraryFunctions::cuOccupancyMaxPotentialBlockSizeWithFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxPotentialBlockSizeWithFlags"));
    if (!funcList->cuOccupancyMaxPotentialBlockSizeWithFlags)
    {
        return false;
    }
    funcList->cuOccupancyMaxPotentialClusterSize = reinterpret_cast<NvcudaLibraryFunctions::cuOccupancyMaxPotentialClusterSize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuOccupancyMaxPotentialClusterSize"));
    if (!funcList->cuOccupancyMaxPotentialClusterSize)
    {
        return false;
    }
    funcList->cuParamSetSize = reinterpret_cast<NvcudaLibraryFunctions::cuParamSetSize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuParamSetSize"));
    if (!funcList->cuParamSetSize)
    {
        return false;
    }
    funcList->cuParamSetTexRef = reinterpret_cast<NvcudaLibraryFunctions::cuParamSetTexRef_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuParamSetTexRef"));
    if (!funcList->cuParamSetTexRef)
    {
        return false;
    }
    funcList->cuParamSetf = reinterpret_cast<NvcudaLibraryFunctions::cuParamSetf_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuParamSetf"));
    if (!funcList->cuParamSetf)
    {
        return false;
    }
    funcList->cuParamSeti = reinterpret_cast<NvcudaLibraryFunctions::cuParamSeti_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuParamSeti"));
    if (!funcList->cuParamSeti)
    {
        return false;
    }
    funcList->cuParamSetv = reinterpret_cast<NvcudaLibraryFunctions::cuParamSetv_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuParamSetv"));
    if (!funcList->cuParamSetv)
    {
        return false;
    }
    funcList->cuPointerGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuPointerGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuPointerGetAttribute"));
    if (!funcList->cuPointerGetAttribute)
    {
        return false;
    }
    funcList->cuPointerGetAttributes = reinterpret_cast<NvcudaLibraryFunctions::cuPointerGetAttributes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuPointerGetAttributes"));
    if (!funcList->cuPointerGetAttributes)
    {
        return false;
    }
    funcList->cuPointerSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuPointerSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuPointerSetAttribute"));
    if (!funcList->cuPointerSetAttribute)
    {
        return false;
    }
    funcList->cuProfilerInitialize = reinterpret_cast<NvcudaLibraryFunctions::cuProfilerInitialize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuProfilerInitialize"));
    if (!funcList->cuProfilerInitialize)
    {
        return false;
    }
    funcList->cuProfilerStart = reinterpret_cast<NvcudaLibraryFunctions::cuProfilerStart_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuProfilerStart"));
    if (!funcList->cuProfilerStart)
    {
        return false;
    }
    funcList->cuProfilerStop = reinterpret_cast<NvcudaLibraryFunctions::cuProfilerStop_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuProfilerStop"));
    if (!funcList->cuProfilerStop)
    {
        return false;
    }
    funcList->cuSignalExternalSemaphoresAsync = reinterpret_cast<NvcudaLibraryFunctions::cuSignalExternalSemaphoresAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSignalExternalSemaphoresAsync"));
    if (!funcList->cuSignalExternalSemaphoresAsync)
    {
        return false;
    }
    funcList->cuSignalExternalSemaphoresAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuSignalExternalSemaphoresAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSignalExternalSemaphoresAsync_ptsz"));
    if (!funcList->cuSignalExternalSemaphoresAsync_ptsz)
    {
        return false;
    }
    funcList->cuStreamAddCallback = reinterpret_cast<NvcudaLibraryFunctions::cuStreamAddCallback_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamAddCallback"));
    if (!funcList->cuStreamAddCallback)
    {
        return false;
    }
    funcList->cuStreamAddCallback_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamAddCallback_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamAddCallback_ptsz"));
    if (!funcList->cuStreamAddCallback_ptsz)
    {
        return false;
    }
    funcList->cuStreamAttachMemAsync = reinterpret_cast<NvcudaLibraryFunctions::cuStreamAttachMemAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamAttachMemAsync"));
    if (!funcList->cuStreamAttachMemAsync)
    {
        return false;
    }
    funcList->cuStreamAttachMemAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamAttachMemAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamAttachMemAsync_ptsz"));
    if (!funcList->cuStreamAttachMemAsync_ptsz)
    {
        return false;
    }
    funcList->cuStreamBatchMemOp = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBatchMemOp_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBatchMemOp_v2"));
    if (!funcList->cuStreamBatchMemOp)
    {
        return false;
    }
    funcList->cuStreamBatchMemOp_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBatchMemOp_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBatchMemOp_ptsz"));
    if (!funcList->cuStreamBatchMemOp_ptsz)
    {
        return false;
    }
    funcList->cuStreamBeginCapture = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBeginCapture_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBeginCapture_v2"));
    if (!funcList->cuStreamBeginCapture)
    {
        return false;
    }
    funcList->cuStreamBeginCaptureToGraph = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBeginCaptureToGraph_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBeginCaptureToGraph"));
    if (!funcList->cuStreamBeginCaptureToGraph)
    {
        return false;
    }
    funcList->cuStreamBeginCaptureToGraph_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBeginCaptureToGraph_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBeginCaptureToGraph_ptsz"));
    if (!funcList->cuStreamBeginCaptureToGraph_ptsz)
    {
        return false;
    }
    funcList->cuStreamBeginCapture_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamBeginCapture_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamBeginCapture_ptsz"));
    if (!funcList->cuStreamBeginCapture_ptsz)
    {
        return false;
    }
    funcList->cuStreamCopyAttributes = reinterpret_cast<NvcudaLibraryFunctions::cuStreamCopyAttributes_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamCopyAttributes"));
    if (!funcList->cuStreamCopyAttributes)
    {
        return false;
    }
    funcList->cuStreamCopyAttributes_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamCopyAttributes_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamCopyAttributes_ptsz"));
    if (!funcList->cuStreamCopyAttributes_ptsz)
    {
        return false;
    }
    funcList->cuStreamCreate = reinterpret_cast<NvcudaLibraryFunctions::cuStreamCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamCreate"));
    if (!funcList->cuStreamCreate)
    {
        return false;
    }
    funcList->cuStreamCreateWithPriority = reinterpret_cast<NvcudaLibraryFunctions::cuStreamCreateWithPriority_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamCreateWithPriority"));
    if (!funcList->cuStreamCreateWithPriority)
    {
        return false;
    }
    funcList->cuStreamDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuStreamDestroy_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamDestroy_v2"));
    if (!funcList->cuStreamDestroy)
    {
        return false;
    }
    funcList->cuStreamEndCapture = reinterpret_cast<NvcudaLibraryFunctions::cuStreamEndCapture_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamEndCapture"));
    if (!funcList->cuStreamEndCapture)
    {
        return false;
    }
    funcList->cuStreamEndCapture_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamEndCapture_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamEndCapture_ptsz"));
    if (!funcList->cuStreamEndCapture_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetAttribute"));
    if (!funcList->cuStreamGetAttribute)
    {
        return false;
    }
    funcList->cuStreamGetAttribute_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetAttribute_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetAttribute_ptsz"));
    if (!funcList->cuStreamGetAttribute_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetCaptureInfo = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetCaptureInfo_v3_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetCaptureInfo_v3"));
    if (!funcList->cuStreamGetCaptureInfo)
    {
        return false;
    }
    funcList->cuStreamGetCaptureInfo_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetCaptureInfo_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetCaptureInfo_ptsz"));
    if (!funcList->cuStreamGetCaptureInfo_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetCtx = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetCtx_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetCtx_v2"));
    if (!funcList->cuStreamGetCtx)
    {
        return false;
    }
    funcList->cuStreamGetCtx_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetCtx_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetCtx_ptsz"));
    if (!funcList->cuStreamGetCtx_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetDevice"));
    if (!funcList->cuStreamGetDevice)
    {
        return false;
    }
    funcList->cuStreamGetDevice_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetDevice_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetDevice_ptsz"));
    if (!funcList->cuStreamGetDevice_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetFlags"));
    if (!funcList->cuStreamGetFlags)
    {
        return false;
    }
    funcList->cuStreamGetFlags_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetFlags_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetFlags_ptsz"));
    if (!funcList->cuStreamGetFlags_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetGreenCtx = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetGreenCtx_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetGreenCtx"));
    if (!funcList->cuStreamGetGreenCtx)
    {
        return false;
    }
    funcList->cuStreamGetId = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetId_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetId"));
    if (!funcList->cuStreamGetId)
    {
        return false;
    }
    funcList->cuStreamGetId_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetId_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetId_ptsz"));
    if (!funcList->cuStreamGetId_ptsz)
    {
        return false;
    }
    funcList->cuStreamGetPriority = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetPriority_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetPriority"));
    if (!funcList->cuStreamGetPriority)
    {
        return false;
    }
    funcList->cuStreamGetPriority_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamGetPriority_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamGetPriority_ptsz"));
    if (!funcList->cuStreamGetPriority_ptsz)
    {
        return false;
    }
    funcList->cuStreamIsCapturing = reinterpret_cast<NvcudaLibraryFunctions::cuStreamIsCapturing_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamIsCapturing"));
    if (!funcList->cuStreamIsCapturing)
    {
        return false;
    }
    funcList->cuStreamIsCapturing_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamIsCapturing_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamIsCapturing_ptsz"));
    if (!funcList->cuStreamIsCapturing_ptsz)
    {
        return false;
    }
    funcList->cuStreamQuery = reinterpret_cast<NvcudaLibraryFunctions::cuStreamQuery_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamQuery"));
    if (!funcList->cuStreamQuery)
    {
        return false;
    }
    funcList->cuStreamQuery_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamQuery_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamQuery_ptsz"));
    if (!funcList->cuStreamQuery_ptsz)
    {
        return false;
    }
    funcList->cuStreamSetAttribute = reinterpret_cast<NvcudaLibraryFunctions::cuStreamSetAttribute_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamSetAttribute"));
    if (!funcList->cuStreamSetAttribute)
    {
        return false;
    }
    funcList->cuStreamSetAttribute_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamSetAttribute_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamSetAttribute_ptsz"));
    if (!funcList->cuStreamSetAttribute_ptsz)
    {
        return false;
    }
    funcList->cuStreamSynchronize = reinterpret_cast<NvcudaLibraryFunctions::cuStreamSynchronize_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamSynchronize"));
    if (!funcList->cuStreamSynchronize)
    {
        return false;
    }
    funcList->cuStreamSynchronize_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamSynchronize_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamSynchronize_ptsz"));
    if (!funcList->cuStreamSynchronize_ptsz)
    {
        return false;
    }
    funcList->cuStreamUpdateCaptureDependencies = reinterpret_cast<NvcudaLibraryFunctions::cuStreamUpdateCaptureDependencies_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamUpdateCaptureDependencies_v2"));
    if (!funcList->cuStreamUpdateCaptureDependencies)
    {
        return false;
    }
    funcList->cuStreamUpdateCaptureDependencies_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamUpdateCaptureDependencies_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamUpdateCaptureDependencies_ptsz"));
    if (!funcList->cuStreamUpdateCaptureDependencies_ptsz)
    {
        return false;
    }
    funcList->cuStreamWaitEvent = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitEvent_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitEvent"));
    if (!funcList->cuStreamWaitEvent)
    {
        return false;
    }
    funcList->cuStreamWaitEvent_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitEvent_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitEvent_ptsz"));
    if (!funcList->cuStreamWaitEvent_ptsz)
    {
        return false;
    }
    funcList->cuStreamWaitValue32 = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitValue32_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitValue32_v2"));
    if (!funcList->cuStreamWaitValue32)
    {
        return false;
    }
    funcList->cuStreamWaitValue32_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitValue32_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitValue32_ptsz"));
    if (!funcList->cuStreamWaitValue32_ptsz)
    {
        return false;
    }
    funcList->cuStreamWaitValue64 = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitValue64_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitValue64_v2"));
    if (!funcList->cuStreamWaitValue64)
    {
        return false;
    }
    funcList->cuStreamWaitValue64_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWaitValue64_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWaitValue64_ptsz"));
    if (!funcList->cuStreamWaitValue64_ptsz)
    {
        return false;
    }
    funcList->cuStreamWriteValue32 = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWriteValue32_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWriteValue32_v2"));
    if (!funcList->cuStreamWriteValue32)
    {
        return false;
    }
    funcList->cuStreamWriteValue32_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWriteValue32_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWriteValue32_ptsz"));
    if (!funcList->cuStreamWriteValue32_ptsz)
    {
        return false;
    }
    funcList->cuStreamWriteValue64 = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWriteValue64_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWriteValue64_v2"));
    if (!funcList->cuStreamWriteValue64)
    {
        return false;
    }
    funcList->cuStreamWriteValue64_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuStreamWriteValue64_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuStreamWriteValue64_ptsz"));
    if (!funcList->cuStreamWriteValue64_ptsz)
    {
        return false;
    }
    funcList->cuSurfObjectCreate = reinterpret_cast<NvcudaLibraryFunctions::cuSurfObjectCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSurfObjectCreate"));
    if (!funcList->cuSurfObjectCreate)
    {
        return false;
    }
    funcList->cuSurfObjectDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuSurfObjectDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSurfObjectDestroy"));
    if (!funcList->cuSurfObjectDestroy)
    {
        return false;
    }
    funcList->cuSurfObjectGetResourceDesc = reinterpret_cast<NvcudaLibraryFunctions::cuSurfObjectGetResourceDesc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSurfObjectGetResourceDesc"));
    if (!funcList->cuSurfObjectGetResourceDesc)
    {
        return false;
    }
    funcList->cuSurfRefGetArray = reinterpret_cast<NvcudaLibraryFunctions::cuSurfRefGetArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSurfRefGetArray"));
    if (!funcList->cuSurfRefGetArray)
    {
        return false;
    }
    funcList->cuSurfRefSetArray = reinterpret_cast<NvcudaLibraryFunctions::cuSurfRefSetArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuSurfRefSetArray"));
    if (!funcList->cuSurfRefSetArray)
    {
        return false;
    }
    funcList->cuTensorMapEncodeIm2col = reinterpret_cast<NvcudaLibraryFunctions::cuTensorMapEncodeIm2col_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTensorMapEncodeIm2col"));
    if (!funcList->cuTensorMapEncodeIm2col)
    {
        return false;
    }
    funcList->cuTensorMapEncodeIm2colWide = reinterpret_cast<NvcudaLibraryFunctions::cuTensorMapEncodeIm2colWide_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTensorMapEncodeIm2colWide"));
    if (!funcList->cuTensorMapEncodeIm2colWide)
    {
        return false;
    }
    funcList->cuTensorMapEncodeTiled = reinterpret_cast<NvcudaLibraryFunctions::cuTensorMapEncodeTiled_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTensorMapEncodeTiled"));
    if (!funcList->cuTensorMapEncodeTiled)
    {
        return false;
    }
    funcList->cuTensorMapReplaceAddress = reinterpret_cast<NvcudaLibraryFunctions::cuTensorMapReplaceAddress_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTensorMapReplaceAddress"));
    if (!funcList->cuTensorMapReplaceAddress)
    {
        return false;
    }
    funcList->cuTexObjectCreate = reinterpret_cast<NvcudaLibraryFunctions::cuTexObjectCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexObjectCreate"));
    if (!funcList->cuTexObjectCreate)
    {
        return false;
    }
    funcList->cuTexObjectDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuTexObjectDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexObjectDestroy"));
    if (!funcList->cuTexObjectDestroy)
    {
        return false;
    }
    funcList->cuTexObjectGetResourceDesc = reinterpret_cast<NvcudaLibraryFunctions::cuTexObjectGetResourceDesc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexObjectGetResourceDesc"));
    if (!funcList->cuTexObjectGetResourceDesc)
    {
        return false;
    }
    funcList->cuTexObjectGetResourceViewDesc = reinterpret_cast<NvcudaLibraryFunctions::cuTexObjectGetResourceViewDesc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexObjectGetResourceViewDesc"));
    if (!funcList->cuTexObjectGetResourceViewDesc)
    {
        return false;
    }
    funcList->cuTexObjectGetTextureDesc = reinterpret_cast<NvcudaLibraryFunctions::cuTexObjectGetTextureDesc_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexObjectGetTextureDesc"));
    if (!funcList->cuTexObjectGetTextureDesc)
    {
        return false;
    }
    funcList->cuTexRefCreate = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefCreate"));
    if (!funcList->cuTexRefCreate)
    {
        return false;
    }
    funcList->cuTexRefDestroy = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefDestroy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefDestroy"));
    if (!funcList->cuTexRefDestroy)
    {
        return false;
    }
    funcList->cuTexRefGetAddress = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetAddress_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetAddress_v2"));
    if (!funcList->cuTexRefGetAddress)
    {
        return false;
    }
    funcList->cuTexRefGetAddressMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetAddressMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetAddressMode"));
    if (!funcList->cuTexRefGetAddressMode)
    {
        return false;
    }
    funcList->cuTexRefGetArray = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetArray"));
    if (!funcList->cuTexRefGetArray)
    {
        return false;
    }
    funcList->cuTexRefGetBorderColor = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetBorderColor_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetBorderColor"));
    if (!funcList->cuTexRefGetBorderColor)
    {
        return false;
    }
    funcList->cuTexRefGetFilterMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetFilterMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetFilterMode"));
    if (!funcList->cuTexRefGetFilterMode)
    {
        return false;
    }
    funcList->cuTexRefGetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetFlags"));
    if (!funcList->cuTexRefGetFlags)
    {
        return false;
    }
    funcList->cuTexRefGetFormat = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetFormat_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetFormat"));
    if (!funcList->cuTexRefGetFormat)
    {
        return false;
    }
    funcList->cuTexRefGetMaxAnisotropy = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetMaxAnisotropy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetMaxAnisotropy"));
    if (!funcList->cuTexRefGetMaxAnisotropy)
    {
        return false;
    }
    funcList->cuTexRefGetMipmapFilterMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetMipmapFilterMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetMipmapFilterMode"));
    if (!funcList->cuTexRefGetMipmapFilterMode)
    {
        return false;
    }
    funcList->cuTexRefGetMipmapLevelBias = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetMipmapLevelBias_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetMipmapLevelBias"));
    if (!funcList->cuTexRefGetMipmapLevelBias)
    {
        return false;
    }
    funcList->cuTexRefGetMipmapLevelClamp = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetMipmapLevelClamp_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetMipmapLevelClamp"));
    if (!funcList->cuTexRefGetMipmapLevelClamp)
    {
        return false;
    }
    funcList->cuTexRefGetMipmappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefGetMipmappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefGetMipmappedArray"));
    if (!funcList->cuTexRefGetMipmappedArray)
    {
        return false;
    }
    funcList->cuTexRefSetAddress = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetAddress_v2_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetAddress_v2"));
    if (!funcList->cuTexRefSetAddress)
    {
        return false;
    }
    funcList->cuTexRefSetAddress2D = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetAddress2D_v3_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetAddress2D_v3"));
    if (!funcList->cuTexRefSetAddress2D)
    {
        return false;
    }
    funcList->cuTexRefSetAddressMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetAddressMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetAddressMode"));
    if (!funcList->cuTexRefSetAddressMode)
    {
        return false;
    }
    funcList->cuTexRefSetArray = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetArray"));
    if (!funcList->cuTexRefSetArray)
    {
        return false;
    }
    funcList->cuTexRefSetBorderColor = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetBorderColor_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetBorderColor"));
    if (!funcList->cuTexRefSetBorderColor)
    {
        return false;
    }
    funcList->cuTexRefSetFilterMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetFilterMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetFilterMode"));
    if (!funcList->cuTexRefSetFilterMode)
    {
        return false;
    }
    funcList->cuTexRefSetFlags = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetFlags_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetFlags"));
    if (!funcList->cuTexRefSetFlags)
    {
        return false;
    }
    funcList->cuTexRefSetFormat = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetFormat_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetFormat"));
    if (!funcList->cuTexRefSetFormat)
    {
        return false;
    }
    funcList->cuTexRefSetMaxAnisotropy = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetMaxAnisotropy_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetMaxAnisotropy"));
    if (!funcList->cuTexRefSetMaxAnisotropy)
    {
        return false;
    }
    funcList->cuTexRefSetMipmapFilterMode = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetMipmapFilterMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetMipmapFilterMode"));
    if (!funcList->cuTexRefSetMipmapFilterMode)
    {
        return false;
    }
    funcList->cuTexRefSetMipmapLevelBias = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetMipmapLevelBias_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetMipmapLevelBias"));
    if (!funcList->cuTexRefSetMipmapLevelBias)
    {
        return false;
    }
    funcList->cuTexRefSetMipmapLevelClamp = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetMipmapLevelClamp_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetMipmapLevelClamp"));
    if (!funcList->cuTexRefSetMipmapLevelClamp)
    {
        return false;
    }
    funcList->cuTexRefSetMipmappedArray = reinterpret_cast<NvcudaLibraryFunctions::cuTexRefSetMipmappedArray_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuTexRefSetMipmappedArray"));
    if (!funcList->cuTexRefSetMipmappedArray)
    {
        return false;
    }
    funcList->cuThreadExchangeStreamCaptureMode = reinterpret_cast<NvcudaLibraryFunctions::cuThreadExchangeStreamCaptureMode_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuThreadExchangeStreamCaptureMode"));
    if (!funcList->cuThreadExchangeStreamCaptureMode)
    {
        return false;
    }
    funcList->cuUserObjectCreate = reinterpret_cast<NvcudaLibraryFunctions::cuUserObjectCreate_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuUserObjectCreate"));
    if (!funcList->cuUserObjectCreate)
    {
        return false;
    }
    funcList->cuUserObjectRelease = reinterpret_cast<NvcudaLibraryFunctions::cuUserObjectRelease_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuUserObjectRelease"));
    if (!funcList->cuUserObjectRelease)
    {
        return false;
    }
    funcList->cuUserObjectRetain = reinterpret_cast<NvcudaLibraryFunctions::cuUserObjectRetain_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuUserObjectRetain"));
    if (!funcList->cuUserObjectRetain)
    {
        return false;
    }
    funcList->cuWGLGetDevice = reinterpret_cast<NvcudaLibraryFunctions::cuWGLGetDevice_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuWGLGetDevice"));
    if (!funcList->cuWGLGetDevice)
    {
        return false;
    }
    funcList->cuWaitExternalSemaphoresAsync = reinterpret_cast<NvcudaLibraryFunctions::cuWaitExternalSemaphoresAsync_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuWaitExternalSemaphoresAsync"));
    if (!funcList->cuWaitExternalSemaphoresAsync)
    {
        return false;
    }
    funcList->cuWaitExternalSemaphoresAsync_ptsz = reinterpret_cast<NvcudaLibraryFunctions::cuWaitExternalSemaphoresAsync_ptsz_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cuWaitExternalSemaphoresAsync_ptsz"));
    if (!funcList->cuWaitExternalSemaphoresAsync_ptsz)
    {
        return false;
    }
    funcList->cudbgApiAttach = reinterpret_cast<NvcudaLibraryFunctions::cudbgApiAttach_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgApiAttach"));
    if (!funcList->cudbgApiAttach)
    {
        return false;
    }
    funcList->cudbgApiDetach = reinterpret_cast<NvcudaLibraryFunctions::cudbgApiDetach_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgApiDetach"));
    if (!funcList->cudbgApiDetach)
    {
        return false;
    }
    funcList->cudbgApiInit = reinterpret_cast<NvcudaLibraryFunctions::cudbgApiInit_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgApiInit"));
    if (!funcList->cudbgApiInit)
    {
        return false;
    }
    funcList->cudbgGetAPI = reinterpret_cast<NvcudaLibraryFunctions::cudbgGetAPI_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgGetAPI"));
    if (!funcList->cudbgGetAPI)
    {
        return false;
    }
    funcList->cudbgGetAPIVersion = reinterpret_cast<NvcudaLibraryFunctions::cudbgGetAPIVersion_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgGetAPIVersion"));
    if (!funcList->cudbgGetAPIVersion)
    {
        return false;
    }
    funcList->cudbgMain = reinterpret_cast<NvcudaLibraryFunctions::cudbgMain_t>(
        dmt::os::lib::getFunc(funcList->m_library, "cudbgMain"));
    if (!funcList->cudbgMain)
    {
        return false;
    }
    return true;
}
