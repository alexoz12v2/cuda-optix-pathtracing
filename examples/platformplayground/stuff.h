#pragma once
#include "platform/platform-utils.h"


#include "cuda.h"

#ifdef cuArray3DCreate
#undef cuArray3DCreate
#endif
#ifdef cuArray3DGetDescriptor
#undef cuArray3DGetDescriptor
#endif
#ifdef cuArrayCreate
#undef cuArrayCreate
#endif
#ifdef cuArrayDestroy
#undef cuArrayDestroy
#endif
#ifdef cuArrayGetDescriptor
#undef cuArrayGetDescriptor
#endif
#ifdef cuArrayGetMemoryRequirements
#undef cuArrayGetMemoryRequirements
#endif
#ifdef cuArrayGetPlane
#undef cuArrayGetPlane
#endif
#ifdef cuArrayGetSparseProperties
#undef cuArrayGetSparseProperties
#endif
#ifdef cuCheckpointProcessCheckpoint
#undef cuCheckpointProcessCheckpoint
#endif
#ifdef cuCheckpointProcessGetRestoreThreadId
#undef cuCheckpointProcessGetRestoreThreadId
#endif
#ifdef cuCheckpointProcessGetState
#undef cuCheckpointProcessGetState
#endif
#ifdef cuCheckpointProcessLock
#undef cuCheckpointProcessLock
#endif
#ifdef cuCheckpointProcessRestore
#undef cuCheckpointProcessRestore
#endif
#ifdef cuCheckpointProcessUnlock
#undef cuCheckpointProcessUnlock
#endif
#ifdef cuCoredumpGetAttribute
#undef cuCoredumpGetAttribute
#endif
#ifdef cuCoredumpGetAttributeGlobal
#undef cuCoredumpGetAttributeGlobal
#endif
#ifdef cuCoredumpSetAttribute
#undef cuCoredumpSetAttribute
#endif
#ifdef cuCoredumpSetAttributeGlobal
#undef cuCoredumpSetAttributeGlobal
#endif
#ifdef cuCtxAttach
#undef cuCtxAttach
#endif
#ifdef cuCtxCreate
#undef cuCtxCreate
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
#ifdef cuCtxDetach
#undef cuCtxDetach
#endif
#ifdef cuCtxDisablePeerAccess
#undef cuCtxDisablePeerAccess
#endif
#ifdef cuCtxEnablePeerAccess
#undef cuCtxEnablePeerAccess
#endif
#ifdef cuCtxFromGreenCtx
#undef cuCtxFromGreenCtx
#endif
#ifdef cuCtxGetApiVersion
#undef cuCtxGetApiVersion
#endif
#ifdef cuCtxGetCacheConfig
#undef cuCtxGetCacheConfig
#endif
#ifdef cuCtxGetCurrent
#undef cuCtxGetCurrent
#endif
#ifdef cuCtxGetDevResource
#undef cuCtxGetDevResource
#endif
#ifdef cuCtxGetDevice
#undef cuCtxGetDevice
#endif
#ifdef cuCtxGetExecAffinity
#undef cuCtxGetExecAffinity
#endif
#ifdef cuCtxGetFlags
#undef cuCtxGetFlags
#endif
#ifdef cuCtxGetId
#undef cuCtxGetId
#endif
#ifdef cuCtxGetLimit
#undef cuCtxGetLimit
#endif
#ifdef cuCtxGetSharedMemConfig
#undef cuCtxGetSharedMemConfig
#endif
#ifdef cuCtxGetStreamPriorityRange
#undef cuCtxGetStreamPriorityRange
#endif
#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif
#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif
#ifdef cuCtxRecordEvent
#undef cuCtxRecordEvent
#endif
#ifdef cuCtxResetPersistingL2Cache
#undef cuCtxResetPersistingL2Cache
#endif
#ifdef cuCtxSetCacheConfig
#undef cuCtxSetCacheConfig
#endif
#ifdef cuCtxSetCurrent
#undef cuCtxSetCurrent
#endif
#ifdef cuCtxSetFlags
#undef cuCtxSetFlags
#endif
#ifdef cuCtxSetLimit
#undef cuCtxSetLimit
#endif
#ifdef cuCtxSetSharedMemConfig
#undef cuCtxSetSharedMemConfig
#endif
#ifdef cuCtxSynchronize
#undef cuCtxSynchronize
#endif
#ifdef cuCtxWaitEvent
#undef cuCtxWaitEvent
#endif
#ifdef cuD3D10CtxCreate
#undef cuD3D10CtxCreate
#endif
#ifdef cuD3D10CtxCreateOnDevice
#undef cuD3D10CtxCreateOnDevice
#endif
#ifdef cuD3D10GetDevice
#undef cuD3D10GetDevice
#endif
#ifdef cuD3D10GetDevices
#undef cuD3D10GetDevices
#endif
#ifdef cuD3D10GetDirect3DDevice
#undef cuD3D10GetDirect3DDevice
#endif
#ifdef cuD3D10MapResources
#undef cuD3D10MapResources
#endif
#ifdef cuD3D10RegisterResource
#undef cuD3D10RegisterResource
#endif
#ifdef cuD3D10ResourceGetMappedArray
#undef cuD3D10ResourceGetMappedArray
#endif
#ifdef cuD3D10ResourceGetMappedPitch
#undef cuD3D10ResourceGetMappedPitch
#endif
#ifdef cuD3D10ResourceGetMappedPointer
#undef cuD3D10ResourceGetMappedPointer
#endif
#ifdef cuD3D10ResourceGetMappedSize
#undef cuD3D10ResourceGetMappedSize
#endif
#ifdef cuD3D10ResourceGetSurfaceDimensions
#undef cuD3D10ResourceGetSurfaceDimensions
#endif
#ifdef cuD3D10ResourceSetMapFlags
#undef cuD3D10ResourceSetMapFlags
#endif
#ifdef cuD3D10UnmapResources
#undef cuD3D10UnmapResources
#endif
#ifdef cuD3D10UnregisterResource
#undef cuD3D10UnregisterResource
#endif
#ifdef cuD3D11CtxCreate
#undef cuD3D11CtxCreate
#endif
#ifdef cuD3D11CtxCreateOnDevice
#undef cuD3D11CtxCreateOnDevice
#endif
#ifdef cuD3D11GetDevice
#undef cuD3D11GetDevice
#endif
#ifdef cuD3D11GetDevices
#undef cuD3D11GetDevices
#endif
#ifdef cuD3D11GetDirect3DDevice
#undef cuD3D11GetDirect3DDevice
#endif
#ifdef cuD3D9Begin
#undef cuD3D9Begin
#endif
#ifdef cuD3D9CtxCreate
#undef cuD3D9CtxCreate
#endif
#ifdef cuD3D9CtxCreateOnDevice
#undef cuD3D9CtxCreateOnDevice
#endif
#ifdef cuD3D9End
#undef cuD3D9End
#endif
#ifdef cuD3D9GetDevice
#undef cuD3D9GetDevice
#endif
#ifdef cuD3D9GetDevices
#undef cuD3D9GetDevices
#endif
#ifdef cuD3D9GetDirect3DDevice
#undef cuD3D9GetDirect3DDevice
#endif
#ifdef cuD3D9MapResources
#undef cuD3D9MapResources
#endif
#ifdef cuD3D9MapVertexBuffer
#undef cuD3D9MapVertexBuffer
#endif
#ifdef cuD3D9RegisterResource
#undef cuD3D9RegisterResource
#endif
#ifdef cuD3D9RegisterVertexBuffer
#undef cuD3D9RegisterVertexBuffer
#endif
#ifdef cuD3D9ResourceGetMappedArray
#undef cuD3D9ResourceGetMappedArray
#endif
#ifdef cuD3D9ResourceGetMappedPitch
#undef cuD3D9ResourceGetMappedPitch
#endif
#ifdef cuD3D9ResourceGetMappedPointer
#undef cuD3D9ResourceGetMappedPointer
#endif
#ifdef cuD3D9ResourceGetMappedSize
#undef cuD3D9ResourceGetMappedSize
#endif
#ifdef cuD3D9ResourceGetSurfaceDimensions
#undef cuD3D9ResourceGetSurfaceDimensions
#endif
#ifdef cuD3D9ResourceSetMapFlags
#undef cuD3D9ResourceSetMapFlags
#endif
#ifdef cuD3D9UnmapResources
#undef cuD3D9UnmapResources
#endif
#ifdef cuD3D9UnmapVertexBuffer
#undef cuD3D9UnmapVertexBuffer
#endif
#ifdef cuD3D9UnregisterResource
#undef cuD3D9UnregisterResource
#endif
#ifdef cuD3D9UnregisterVertexBuffer
#undef cuD3D9UnregisterVertexBuffer
#endif
#ifdef cuDestroyExternalMemory
#undef cuDestroyExternalMemory
#endif
#ifdef cuDestroyExternalSemaphore
#undef cuDestroyExternalSemaphore
#endif
#ifdef cuDevResourceGenerateDesc
#undef cuDevResourceGenerateDesc
#endif
#ifdef cuDevSmResourceSplitByCount
#undef cuDevSmResourceSplitByCount
#endif
#ifdef cuDeviceCanAccessPeer
#undef cuDeviceCanAccessPeer
#endif
#ifdef cuDeviceComputeCapability
#undef cuDeviceComputeCapability
#endif
#ifdef cuDeviceGet
#undef cuDeviceGet
#endif
#ifdef cuDeviceGetAttribute
#undef cuDeviceGetAttribute
#endif
#ifdef cuDeviceGetByPCIBusId
#undef cuDeviceGetByPCIBusId
#endif
#ifdef cuDeviceGetCount
#undef cuDeviceGetCount
#endif
#ifdef cuDeviceGetDefaultMemPool
#undef cuDeviceGetDefaultMemPool
#endif
#ifdef cuDeviceGetDevResource
#undef cuDeviceGetDevResource
#endif
#ifdef cuDeviceGetExecAffinitySupport
#undef cuDeviceGetExecAffinitySupport
#endif
#ifdef cuDeviceGetGraphMemAttribute
#undef cuDeviceGetGraphMemAttribute
#endif
#ifdef cuDeviceGetLuid
#undef cuDeviceGetLuid
#endif
#ifdef cuDeviceGetMemPool
#undef cuDeviceGetMemPool
#endif
#ifdef cuDeviceGetName
#undef cuDeviceGetName
#endif
#ifdef cuDeviceGetP2PAttribute
#undef cuDeviceGetP2PAttribute
#endif
#ifdef cuDeviceGetPCIBusId
#undef cuDeviceGetPCIBusId
#endif
#ifdef cuDeviceGetProperties
#undef cuDeviceGetProperties
#endif
#ifdef cuDeviceGetTexture1DLinearMaxWidth
#undef cuDeviceGetTexture1DLinearMaxWidth
#endif
#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
#ifdef cuDeviceGraphMemTrim
#undef cuDeviceGraphMemTrim
#endif
#ifdef cuDevicePrimaryCtxGetState
#undef cuDevicePrimaryCtxGetState
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuDevicePrimaryCtxRetain
#undef cuDevicePrimaryCtxRetain
#endif
#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
#ifdef cuDeviceRegisterAsyncNotification
#undef cuDeviceRegisterAsyncNotification
#endif
#ifdef cuDeviceSetGraphMemAttribute
#undef cuDeviceSetGraphMemAttribute
#endif
#ifdef cuDeviceSetMemPool
#undef cuDeviceSetMemPool
#endif
#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
#ifdef cuDeviceUnregisterAsyncNotification
#undef cuDeviceUnregisterAsyncNotification
#endif
#ifdef cuDriverGetVersion
#undef cuDriverGetVersion
#endif
#ifdef cuEventCreate
#undef cuEventCreate
#endif
#ifdef cuEventDestroy
#undef cuEventDestroy
#endif
#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
#ifdef cuEventQuery
#undef cuEventQuery
#endif
#ifdef cuEventRecord
#undef cuEventRecord
#endif
#ifdef cuEventRecordWithFlags
#undef cuEventRecordWithFlags
#endif
#ifdef cuEventRecordWithFlags_ptsz
#undef cuEventRecordWithFlags_ptsz
#endif
#ifdef cuEventRecord_ptsz
#undef cuEventRecord_ptsz
#endif
#ifdef cuEventSynchronize
#undef cuEventSynchronize
#endif
#ifdef cuExternalMemoryGetMappedBuffer
#undef cuExternalMemoryGetMappedBuffer
#endif
#ifdef cuExternalMemoryGetMappedMipmappedArray
#undef cuExternalMemoryGetMappedMipmappedArray
#endif
#ifdef cuFlushGPUDirectRDMAWrites
#undef cuFlushGPUDirectRDMAWrites
#endif
#ifdef cuFuncGetAttribute
#undef cuFuncGetAttribute
#endif
#ifdef cuFuncGetModule
#undef cuFuncGetModule
#endif
#ifdef cuFuncGetName
#undef cuFuncGetName
#endif
#ifdef cuFuncGetParamInfo
#undef cuFuncGetParamInfo
#endif
#ifdef cuFuncIsLoaded
#undef cuFuncIsLoaded
#endif
#ifdef cuFuncLoad
#undef cuFuncLoad
#endif
#ifdef cuFuncSetAttribute
#undef cuFuncSetAttribute
#endif
#ifdef cuFuncSetBlockShape
#undef cuFuncSetBlockShape
#endif
#ifdef cuFuncSetCacheConfig
#undef cuFuncSetCacheConfig
#endif
#ifdef cuFuncSetSharedMemConfig
#undef cuFuncSetSharedMemConfig
#endif
#ifdef cuFuncSetSharedSize
#undef cuFuncSetSharedSize
#endif
#ifdef cuGLCtxCreate
#undef cuGLCtxCreate
#endif
#ifdef cuGLGetDevices
#undef cuGLGetDevices
#endif
#ifdef cuGLInit
#undef cuGLInit
#endif
#ifdef cuGLMapBufferObject
#undef cuGLMapBufferObject
#endif
#ifdef cuGLMapBufferObjectAsync
#undef cuGLMapBufferObjectAsync
#endif
#ifdef cuGLRegisterBufferObject
#undef cuGLRegisterBufferObject
#endif
#ifdef cuGLSetBufferObjectMapFlags
#undef cuGLSetBufferObjectMapFlags
#endif
#ifdef cuGLUnmapBufferObject
#undef cuGLUnmapBufferObject
#endif
#ifdef cuGLUnmapBufferObjectAsync
#undef cuGLUnmapBufferObjectAsync
#endif
#ifdef cuGLUnregisterBufferObject
#undef cuGLUnregisterBufferObject
#endif
#ifdef cuGetErrorName
#undef cuGetErrorName
#endif
#ifdef cuGetErrorString
#undef cuGetErrorString
#endif
#ifdef cuGetExportTable
#undef cuGetExportTable
#endif
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuGraphAddBatchMemOpNode
#undef cuGraphAddBatchMemOpNode
#endif
#ifdef cuGraphAddChildGraphNode
#undef cuGraphAddChildGraphNode
#endif
#ifdef cuGraphAddDependencies
#undef cuGraphAddDependencies
#endif
#ifdef cuGraphAddEmptyNode
#undef cuGraphAddEmptyNode
#endif
#ifdef cuGraphAddEventRecordNode
#undef cuGraphAddEventRecordNode
#endif
#ifdef cuGraphAddEventWaitNode
#undef cuGraphAddEventWaitNode
#endif
#ifdef cuGraphAddExternalSemaphoresSignalNode
#undef cuGraphAddExternalSemaphoresSignalNode
#endif
#ifdef cuGraphAddExternalSemaphoresWaitNode
#undef cuGraphAddExternalSemaphoresWaitNode
#endif
#ifdef cuGraphAddHostNode
#undef cuGraphAddHostNode
#endif
#ifdef cuGraphAddKernelNode
#undef cuGraphAddKernelNode
#endif
#ifdef cuGraphAddMemAllocNode
#undef cuGraphAddMemAllocNode
#endif
#ifdef cuGraphAddMemFreeNode
#undef cuGraphAddMemFreeNode
#endif
#ifdef cuGraphAddMemcpyNode
#undef cuGraphAddMemcpyNode
#endif
#ifdef cuGraphAddMemsetNode
#undef cuGraphAddMemsetNode
#endif
#ifdef cuGraphAddNode
#undef cuGraphAddNode
#endif
#ifdef cuGraphBatchMemOpNodeGetParams
#undef cuGraphBatchMemOpNodeGetParams
#endif
#ifdef cuGraphBatchMemOpNodeSetParams
#undef cuGraphBatchMemOpNodeSetParams
#endif
#ifdef cuGraphChildGraphNodeGetGraph
#undef cuGraphChildGraphNodeGetGraph
#endif
#ifdef cuGraphClone
#undef cuGraphClone
#endif
#ifdef cuGraphConditionalHandleCreate
#undef cuGraphConditionalHandleCreate
#endif
#ifdef cuGraphCreate
#undef cuGraphCreate
#endif
#ifdef cuGraphDebugDotPrint
#undef cuGraphDebugDotPrint
#endif
#ifdef cuGraphDestroy
#undef cuGraphDestroy
#endif
#ifdef cuGraphDestroyNode
#undef cuGraphDestroyNode
#endif
#ifdef cuGraphEventRecordNodeGetEvent
#undef cuGraphEventRecordNodeGetEvent
#endif
#ifdef cuGraphEventRecordNodeSetEvent
#undef cuGraphEventRecordNodeSetEvent
#endif
#ifdef cuGraphEventWaitNodeGetEvent
#undef cuGraphEventWaitNodeGetEvent
#endif
#ifdef cuGraphEventWaitNodeSetEvent
#undef cuGraphEventWaitNodeSetEvent
#endif
#ifdef cuGraphExecBatchMemOpNodeSetParams
#undef cuGraphExecBatchMemOpNodeSetParams
#endif
#ifdef cuGraphExecChildGraphNodeSetParams
#undef cuGraphExecChildGraphNodeSetParams
#endif
#ifdef cuGraphExecDestroy
#undef cuGraphExecDestroy
#endif
#ifdef cuGraphExecEventRecordNodeSetEvent
#undef cuGraphExecEventRecordNodeSetEvent
#endif
#ifdef cuGraphExecEventWaitNodeSetEvent
#undef cuGraphExecEventWaitNodeSetEvent
#endif
#ifdef cuGraphExecExternalSemaphoresSignalNodeSetParams
#undef cuGraphExecExternalSemaphoresSignalNodeSetParams
#endif
#ifdef cuGraphExecExternalSemaphoresWaitNodeSetParams
#undef cuGraphExecExternalSemaphoresWaitNodeSetParams
#endif
#ifdef cuGraphExecGetFlags
#undef cuGraphExecGetFlags
#endif
#ifdef cuGraphExecHostNodeSetParams
#undef cuGraphExecHostNodeSetParams
#endif
#ifdef cuGraphExecKernelNodeSetParams
#undef cuGraphExecKernelNodeSetParams
#endif
#ifdef cuGraphExecMemcpyNodeSetParams
#undef cuGraphExecMemcpyNodeSetParams
#endif
#ifdef cuGraphExecMemsetNodeSetParams
#undef cuGraphExecMemsetNodeSetParams
#endif
#ifdef cuGraphExecNodeSetParams
#undef cuGraphExecNodeSetParams
#endif
#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif
#ifdef cuGraphExternalSemaphoresSignalNodeGetParams
#undef cuGraphExternalSemaphoresSignalNodeGetParams
#endif
#ifdef cuGraphExternalSemaphoresSignalNodeSetParams
#undef cuGraphExternalSemaphoresSignalNodeSetParams
#endif
#ifdef cuGraphExternalSemaphoresWaitNodeGetParams
#undef cuGraphExternalSemaphoresWaitNodeGetParams
#endif
#ifdef cuGraphExternalSemaphoresWaitNodeSetParams
#undef cuGraphExternalSemaphoresWaitNodeSetParams
#endif
#ifdef cuGraphGetEdges
#undef cuGraphGetEdges
#endif
#ifdef cuGraphGetNodes
#undef cuGraphGetNodes
#endif
#ifdef cuGraphGetRootNodes
#undef cuGraphGetRootNodes
#endif
#ifdef cuGraphHostNodeGetParams
#undef cuGraphHostNodeGetParams
#endif
#ifdef cuGraphHostNodeSetParams
#undef cuGraphHostNodeSetParams
#endif
#ifdef cuGraphInstantiate
#undef cuGraphInstantiate
#endif
#ifdef cuGraphInstantiateWithFlags
#undef cuGraphInstantiateWithFlags
#endif
#ifdef cuGraphInstantiateWithParams
#undef cuGraphInstantiateWithParams
#endif
#ifdef cuGraphInstantiateWithParams_ptsz
#undef cuGraphInstantiateWithParams_ptsz
#endif
#ifdef cuGraphKernelNodeCopyAttributes
#undef cuGraphKernelNodeCopyAttributes
#endif
#ifdef cuGraphKernelNodeGetAttribute
#undef cuGraphKernelNodeGetAttribute
#endif
#ifdef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeGetParams
#endif
#ifdef cuGraphKernelNodeSetAttribute
#undef cuGraphKernelNodeSetAttribute
#endif
#ifdef cuGraphKernelNodeSetParams
#undef cuGraphKernelNodeSetParams
#endif
#ifdef cuGraphLaunch
#undef cuGraphLaunch
#endif
#ifdef cuGraphLaunch_ptsz
#undef cuGraphLaunch_ptsz
#endif
#ifdef cuGraphMemAllocNodeGetParams
#undef cuGraphMemAllocNodeGetParams
#endif
#ifdef cuGraphMemFreeNodeGetParams
#undef cuGraphMemFreeNodeGetParams
#endif
#ifdef cuGraphMemcpyNodeGetParams
#undef cuGraphMemcpyNodeGetParams
#endif
#ifdef cuGraphMemcpyNodeSetParams
#undef cuGraphMemcpyNodeSetParams
#endif
#ifdef cuGraphMemsetNodeGetParams
#undef cuGraphMemsetNodeGetParams
#endif
#ifdef cuGraphMemsetNodeSetParams
#undef cuGraphMemsetNodeSetParams
#endif
#ifdef cuGraphNodeFindInClone
#undef cuGraphNodeFindInClone
#endif
#ifdef cuGraphNodeGetDependencies
#undef cuGraphNodeGetDependencies
#endif
#ifdef cuGraphNodeGetDependentNodes
#undef cuGraphNodeGetDependentNodes
#endif
#ifdef cuGraphNodeGetEnabled
#undef cuGraphNodeGetEnabled
#endif
#ifdef cuGraphNodeGetType
#undef cuGraphNodeGetType
#endif
#ifdef cuGraphNodeSetEnabled
#undef cuGraphNodeSetEnabled
#endif
#ifdef cuGraphNodeSetParams
#undef cuGraphNodeSetParams
#endif
#ifdef cuGraphReleaseUserObject
#undef cuGraphReleaseUserObject
#endif
#ifdef cuGraphRemoveDependencies
#undef cuGraphRemoveDependencies
#endif
#ifdef cuGraphRetainUserObject
#undef cuGraphRetainUserObject
#endif
#ifdef cuGraphUpload
#undef cuGraphUpload
#endif
#ifdef cuGraphUpload_ptsz
#undef cuGraphUpload_ptsz
#endif
#ifdef cuGraphicsD3D10RegisterResource
#undef cuGraphicsD3D10RegisterResource
#endif
#ifdef cuGraphicsD3D11RegisterResource
#undef cuGraphicsD3D11RegisterResource
#endif
#ifdef cuGraphicsD3D9RegisterResource
#undef cuGraphicsD3D9RegisterResource
#endif
#ifdef cuGraphicsGLRegisterBuffer
#undef cuGraphicsGLRegisterBuffer
#endif
#ifdef cuGraphicsGLRegisterImage
#undef cuGraphicsGLRegisterImage
#endif
#ifdef cuGraphicsMapResources
#undef cuGraphicsMapResources
#endif
#ifdef cuGraphicsMapResources_ptsz
#undef cuGraphicsMapResources_ptsz
#endif
#ifdef cuGraphicsResourceGetMappedMipmappedArray
#undef cuGraphicsResourceGetMappedMipmappedArray
#endif
#ifdef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceGetMappedPointer
#endif
#ifdef cuGraphicsResourceSetMapFlags
#undef cuGraphicsResourceSetMapFlags
#endif
#ifdef cuGraphicsSubResourceGetMappedArray
#undef cuGraphicsSubResourceGetMappedArray
#endif
#ifdef cuGraphicsUnmapResources
#undef cuGraphicsUnmapResources
#endif
#ifdef cuGraphicsUnmapResources_ptsz
#undef cuGraphicsUnmapResources_ptsz
#endif
#ifdef cuGraphicsUnregisterResource
#undef cuGraphicsUnregisterResource
#endif
#ifdef cuGreenCtxCreate
#undef cuGreenCtxCreate
#endif
#ifdef cuGreenCtxDestroy
#undef cuGreenCtxDestroy
#endif
#ifdef cuGreenCtxGetDevResource
#undef cuGreenCtxGetDevResource
#endif
#ifdef cuGreenCtxRecordEvent
#undef cuGreenCtxRecordEvent
#endif
#ifdef cuGreenCtxStreamCreate
#undef cuGreenCtxStreamCreate
#endif
#ifdef cuGreenCtxWaitEvent
#undef cuGreenCtxWaitEvent
#endif
#ifdef cuImportExternalMemory
#undef cuImportExternalMemory
#endif
#ifdef cuImportExternalSemaphore
#undef cuImportExternalSemaphore
#endif
#ifdef cuInit
#undef cuInit
#endif
#ifdef cuIpcCloseMemHandle
#undef cuIpcCloseMemHandle
#endif
#ifdef cuIpcGetEventHandle
#undef cuIpcGetEventHandle
#endif
#ifdef cuIpcGetMemHandle
#undef cuIpcGetMemHandle
#endif
#ifdef cuIpcOpenEventHandle
#undef cuIpcOpenEventHandle
#endif
#ifdef cuIpcOpenMemHandle
#undef cuIpcOpenMemHandle
#endif
#ifdef cuKernelGetAttribute
#undef cuKernelGetAttribute
#endif
#ifdef cuKernelGetFunction
#undef cuKernelGetFunction
#endif
#ifdef cuKernelGetLibrary
#undef cuKernelGetLibrary
#endif
#ifdef cuKernelGetName
#undef cuKernelGetName
#endif
#ifdef cuKernelGetParamInfo
#undef cuKernelGetParamInfo
#endif
#ifdef cuKernelSetAttribute
#undef cuKernelSetAttribute
#endif
#ifdef cuKernelSetCacheConfig
#undef cuKernelSetCacheConfig
#endif
#ifdef cuLaunch
#undef cuLaunch
#endif
#ifdef cuLaunchCooperativeKernel
#undef cuLaunchCooperativeKernel
#endif
#ifdef cuLaunchCooperativeKernelMultiDevice
#undef cuLaunchCooperativeKernelMultiDevice
#endif
#ifdef cuLaunchCooperativeKernel_ptsz
#undef cuLaunchCooperativeKernel_ptsz
#endif
#ifdef cuLaunchGrid
#undef cuLaunchGrid
#endif
#ifdef cuLaunchGridAsync
#undef cuLaunchGridAsync
#endif
#ifdef cuLaunchHostFunc
#undef cuLaunchHostFunc
#endif
#ifdef cuLaunchHostFunc_ptsz
#undef cuLaunchHostFunc_ptsz
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
#ifdef cuLaunchKernelEx
#undef cuLaunchKernelEx
#endif
#ifdef cuLaunchKernelEx_ptsz
#undef cuLaunchKernelEx_ptsz
#endif
#ifdef cuLaunchKernel_ptsz
#undef cuLaunchKernel_ptsz
#endif
#ifdef cuLibraryEnumerateKernels
#undef cuLibraryEnumerateKernels
#endif
#ifdef cuLibraryGetGlobal
#undef cuLibraryGetGlobal
#endif
#ifdef cuLibraryGetKernel
#undef cuLibraryGetKernel
#endif
#ifdef cuLibraryGetKernelCount
#undef cuLibraryGetKernelCount
#endif
#ifdef cuLibraryGetManaged
#undef cuLibraryGetManaged
#endif
#ifdef cuLibraryGetModule
#undef cuLibraryGetModule
#endif
#ifdef cuLibraryGetUnifiedFunction
#undef cuLibraryGetUnifiedFunction
#endif
#ifdef cuLibraryLoadData
#undef cuLibraryLoadData
#endif
#ifdef cuLibraryLoadFromFile
#undef cuLibraryLoadFromFile
#endif
#ifdef cuLibraryUnload
#undef cuLibraryUnload
#endif
#ifdef cuLinkAddData
#undef cuLinkAddData
#endif
#ifdef cuLinkAddFile
#undef cuLinkAddFile
#endif
#ifdef cuLinkComplete
#undef cuLinkComplete
#endif
#ifdef cuLinkCreate
#undef cuLinkCreate
#endif
#ifdef cuLinkDestroy
#undef cuLinkDestroy
#endif
#ifdef cuMemAddressFree
#undef cuMemAddressFree
#endif
#ifdef cuMemAddressReserve
#undef cuMemAddressReserve
#endif
#ifdef cuMemAdvise
#undef cuMemAdvise
#endif
#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
#ifdef cuMemAllocAsync
#undef cuMemAllocAsync
#endif
#ifdef cuMemAllocAsync_ptsz
#undef cuMemAllocAsync_ptsz
#endif
#ifdef cuMemAllocFromPoolAsync
#undef cuMemAllocFromPoolAsync
#endif
#ifdef cuMemAllocFromPoolAsync_ptsz
#undef cuMemAllocFromPoolAsync_ptsz
#endif
#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
#ifdef cuMemAllocManaged
#undef cuMemAllocManaged
#endif
#ifdef cuMemAllocPitch
#undef cuMemAllocPitch
#endif
#ifdef cuMemBatchDecompressAsync
#undef cuMemBatchDecompressAsync
#endif
#ifdef cuMemBatchDecompressAsync_ptsz
#undef cuMemBatchDecompressAsync_ptsz
#endif
#ifdef cuMemCreate
#undef cuMemCreate
#endif
#ifdef cuMemExportToShareableHandle
#undef cuMemExportToShareableHandle
#endif
#ifdef cuMemFree
#undef cuMemFree
#endif
#ifdef cuMemFreeAsync
#undef cuMemFreeAsync
#endif
#ifdef cuMemFreeAsync_ptsz
#undef cuMemFreeAsync_ptsz
#endif
#ifdef cuMemFreeHost
#undef cuMemFreeHost
#endif
#ifdef cuMemGetAccess
#undef cuMemGetAccess
#endif
#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
#ifdef cuMemGetAllocationGranularity
#undef cuMemGetAllocationGranularity
#endif
#ifdef cuMemGetAllocationPropertiesFromHandle
#undef cuMemGetAllocationPropertiesFromHandle
#endif
#ifdef cuMemGetHandleForAddressRange
#undef cuMemGetHandleForAddressRange
#endif
#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
#ifdef cuMemHostAlloc
#undef cuMemHostAlloc
#endif
#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
#ifdef cuMemHostGetFlags
#undef cuMemHostGetFlags
#endif
#ifdef cuMemHostRegister
#undef cuMemHostRegister
#endif
#ifdef cuMemHostUnregister
#undef cuMemHostUnregister
#endif
#ifdef cuMemImportFromShareableHandle
#undef cuMemImportFromShareableHandle
#endif
#ifdef cuMemMap
#undef cuMemMap
#endif
#ifdef cuMemMapArrayAsync
#undef cuMemMapArrayAsync
#endif
#ifdef cuMemMapArrayAsync_ptsz
#undef cuMemMapArrayAsync_ptsz
#endif
#ifdef cuMemPoolCreate
#undef cuMemPoolCreate
#endif
#ifdef cuMemPoolDestroy
#undef cuMemPoolDestroy
#endif
#ifdef cuMemPoolExportPointer
#undef cuMemPoolExportPointer
#endif
#ifdef cuMemPoolExportToShareableHandle
#undef cuMemPoolExportToShareableHandle
#endif
#ifdef cuMemPoolGetAccess
#undef cuMemPoolGetAccess
#endif
#ifdef cuMemPoolGetAttribute
#undef cuMemPoolGetAttribute
#endif
#ifdef cuMemPoolImportFromShareableHandle
#undef cuMemPoolImportFromShareableHandle
#endif
#ifdef cuMemPoolImportPointer
#undef cuMemPoolImportPointer
#endif
#ifdef cuMemPoolSetAccess
#undef cuMemPoolSetAccess
#endif
#ifdef cuMemPoolSetAttribute
#undef cuMemPoolSetAttribute
#endif
#ifdef cuMemPoolTrimTo
#undef cuMemPoolTrimTo
#endif
#ifdef cuMemPrefetchAsync
#undef cuMemPrefetchAsync
#endif
#ifdef cuMemPrefetchAsync_ptsz
#undef cuMemPrefetchAsync_ptsz
#endif
#ifdef cuMemRangeGetAttribute
#undef cuMemRangeGetAttribute
#endif
#ifdef cuMemRangeGetAttributes
#undef cuMemRangeGetAttributes
#endif
#ifdef cuMemRelease
#undef cuMemRelease
#endif
#ifdef cuMemRetainAllocationHandle
#undef cuMemRetainAllocationHandle
#endif
#ifdef cuMemSetAccess
#undef cuMemSetAccess
#endif
#ifdef cuMemUnmap
#undef cuMemUnmap
#endif
#ifdef cuMemcpy
#undef cuMemcpy
#endif
#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
#ifdef cuMemcpy2DAsync
#undef cuMemcpy2DAsync
#endif
#ifdef cuMemcpy2DUnaligned
#undef cuMemcpy2DUnaligned
#endif
#ifdef cuMemcpy3D
#undef cuMemcpy3D
#endif
#ifdef cuMemcpy3DAsync
#undef cuMemcpy3DAsync
#endif
#ifdef cuMemcpy3DBatchAsync
#undef cuMemcpy3DBatchAsync
#endif
#ifdef cuMemcpy3DBatchAsync_ptsz
#undef cuMemcpy3DBatchAsync_ptsz
#endif
#ifdef cuMemcpy3DPeer
#undef cuMemcpy3DPeer
#endif
#ifdef cuMemcpy3DPeerAsync
#undef cuMemcpy3DPeerAsync
#endif
#ifdef cuMemcpy3DPeerAsync_ptsz
#undef cuMemcpy3DPeerAsync_ptsz
#endif
#ifdef cuMemcpy3DPeer_ptds
#undef cuMemcpy3DPeer_ptds
#endif
#ifdef cuMemcpyAsync
#undef cuMemcpyAsync
#endif
#ifdef cuMemcpyAsync_ptsz
#undef cuMemcpyAsync_ptsz
#endif
#ifdef cuMemcpyAtoA
#undef cuMemcpyAtoA
#endif
#ifdef cuMemcpyAtoD
#undef cuMemcpyAtoD
#endif
#ifdef cuMemcpyAtoH
#undef cuMemcpyAtoH
#endif
#ifdef cuMemcpyAtoHAsync
#undef cuMemcpyAtoHAsync
#endif
#ifdef cuMemcpyBatchAsync
#undef cuMemcpyBatchAsync
#endif
#ifdef cuMemcpyBatchAsync_ptsz
#undef cuMemcpyBatchAsync_ptsz
#endif
#ifdef cuMemcpyDtoA
#undef cuMemcpyDtoA
#endif
#ifdef cuMemcpyDtoD
#undef cuMemcpyDtoD
#endif
#ifdef cuMemcpyDtoDAsync
#undef cuMemcpyDtoDAsync
#endif
#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
#ifdef cuMemcpyHtoA
#undef cuMemcpyHtoA
#endif
#ifdef cuMemcpyHtoAAsync
#undef cuMemcpyHtoAAsync
#endif
#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
#ifdef cuMemcpyPeer
#undef cuMemcpyPeer
#endif
#ifdef cuMemcpyPeerAsync
#undef cuMemcpyPeerAsync
#endif
#ifdef cuMemcpyPeerAsync_ptsz
#undef cuMemcpyPeerAsync_ptsz
#endif
#ifdef cuMemcpyPeer_ptds
#undef cuMemcpyPeer_ptds
#endif
#ifdef cuMemcpy_ptds
#undef cuMemcpy_ptds
#endif
#ifdef cuMemsetD16
#undef cuMemsetD16
#endif
#ifdef cuMemsetD16Async
#undef cuMemsetD16Async
#endif
#ifdef cuMemsetD16Async_ptsz
#undef cuMemsetD16Async_ptsz
#endif
#ifdef cuMemsetD2D16
#undef cuMemsetD2D16
#endif
#ifdef cuMemsetD2D16Async
#undef cuMemsetD2D16Async
#endif
#ifdef cuMemsetD2D16Async_ptsz
#undef cuMemsetD2D16Async_ptsz
#endif
#ifdef cuMemsetD2D32
#undef cuMemsetD2D32
#endif
#ifdef cuMemsetD2D32Async
#undef cuMemsetD2D32Async
#endif
#ifdef cuMemsetD2D32Async_ptsz
#undef cuMemsetD2D32Async_ptsz
#endif
#ifdef cuMemsetD2D8
#undef cuMemsetD2D8
#endif
#ifdef cuMemsetD2D8Async
#undef cuMemsetD2D8Async
#endif
#ifdef cuMemsetD2D8Async_ptsz
#undef cuMemsetD2D8Async_ptsz
#endif
#ifdef cuMemsetD32
#undef cuMemsetD32
#endif
#ifdef cuMemsetD32Async
#undef cuMemsetD32Async
#endif
#ifdef cuMemsetD32Async_ptsz
#undef cuMemsetD32Async_ptsz
#endif
#ifdef cuMemsetD8
#undef cuMemsetD8
#endif
#ifdef cuMemsetD8Async
#undef cuMemsetD8Async
#endif
#ifdef cuMemsetD8Async_ptsz
#undef cuMemsetD8Async_ptsz
#endif
#ifdef cuMipmappedArrayCreate
#undef cuMipmappedArrayCreate
#endif
#ifdef cuMipmappedArrayDestroy
#undef cuMipmappedArrayDestroy
#endif
#ifdef cuMipmappedArrayGetLevel
#undef cuMipmappedArrayGetLevel
#endif
#ifdef cuMipmappedArrayGetMemoryRequirements
#undef cuMipmappedArrayGetMemoryRequirements
#endif
#ifdef cuMipmappedArrayGetSparseProperties
#undef cuMipmappedArrayGetSparseProperties
#endif
#ifdef cuModuleEnumerateFunctions
#undef cuModuleEnumerateFunctions
#endif
#ifdef cuModuleGetFunction
#undef cuModuleGetFunction
#endif
#ifdef cuModuleGetFunctionCount
#undef cuModuleGetFunctionCount
#endif
#ifdef cuModuleGetGlobal
#undef cuModuleGetGlobal
#endif
#ifdef cuModuleGetLoadingMode
#undef cuModuleGetLoadingMode
#endif
#ifdef cuModuleGetSurfRef
#undef cuModuleGetSurfRef
#endif
#ifdef cuModuleGetTexRef
#undef cuModuleGetTexRef
#endif
#ifdef cuModuleLoad
#undef cuModuleLoad
#endif
#ifdef cuModuleLoadData
#undef cuModuleLoadData
#endif
#ifdef cuModuleLoadDataEx
#undef cuModuleLoadDataEx
#endif
#ifdef cuModuleLoadFatBinary
#undef cuModuleLoadFatBinary
#endif
#ifdef cuModuleUnload
#undef cuModuleUnload
#endif
#ifdef cuMulticastAddDevice
#undef cuMulticastAddDevice
#endif
#ifdef cuMulticastBindAddr
#undef cuMulticastBindAddr
#endif
#ifdef cuMulticastBindMem
#undef cuMulticastBindMem
#endif
#ifdef cuMulticastCreate
#undef cuMulticastCreate
#endif
#ifdef cuMulticastGetGranularity
#undef cuMulticastGetGranularity
#endif
#ifdef cuMulticastUnbind
#undef cuMulticastUnbind
#endif
#ifdef cuOccupancyAvailableDynamicSMemPerBlock
#undef cuOccupancyAvailableDynamicSMemPerBlock
#endif
#ifdef cuOccupancyMaxActiveBlocksPerMultiprocessor
#undef cuOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifdef cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#undef cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
#endif
#ifdef cuOccupancyMaxActiveClusters
#undef cuOccupancyMaxActiveClusters
#endif
#ifdef cuOccupancyMaxPotentialBlockSize
#undef cuOccupancyMaxPotentialBlockSize
#endif
#ifdef cuOccupancyMaxPotentialBlockSizeWithFlags
#undef cuOccupancyMaxPotentialBlockSizeWithFlags
#endif
#ifdef cuOccupancyMaxPotentialClusterSize
#undef cuOccupancyMaxPotentialClusterSize
#endif
#ifdef cuParamSetSize
#undef cuParamSetSize
#endif
#ifdef cuParamSetTexRef
#undef cuParamSetTexRef
#endif
#ifdef cuParamSetf
#undef cuParamSetf
#endif
#ifdef cuParamSeti
#undef cuParamSeti
#endif
#ifdef cuParamSetv
#undef cuParamSetv
#endif
#ifdef cuPointerGetAttribute
#undef cuPointerGetAttribute
#endif
#ifdef cuPointerGetAttributes
#undef cuPointerGetAttributes
#endif
#ifdef cuPointerSetAttribute
#undef cuPointerSetAttribute
#endif
#ifdef cuProfilerInitialize
#undef cuProfilerInitialize
#endif
#ifdef cuProfilerStart
#undef cuProfilerStart
#endif
#ifdef cuProfilerStop
#undef cuProfilerStop
#endif
#ifdef cuSignalExternalSemaphoresAsync
#undef cuSignalExternalSemaphoresAsync
#endif
#ifdef cuSignalExternalSemaphoresAsync_ptsz
#undef cuSignalExternalSemaphoresAsync_ptsz
#endif
#ifdef cuStreamAddCallback
#undef cuStreamAddCallback
#endif
#ifdef cuStreamAddCallback_ptsz
#undef cuStreamAddCallback_ptsz
#endif
#ifdef cuStreamAttachMemAsync
#undef cuStreamAttachMemAsync
#endif
#ifdef cuStreamAttachMemAsync_ptsz
#undef cuStreamAttachMemAsync_ptsz
#endif
#ifdef cuStreamBatchMemOp
#undef cuStreamBatchMemOp
#endif
#ifdef cuStreamBatchMemOp_ptsz
#undef cuStreamBatchMemOp_ptsz
#endif
#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
#ifdef cuStreamBeginCaptureToGraph
#undef cuStreamBeginCaptureToGraph
#endif
#ifdef cuStreamBeginCaptureToGraph_ptsz
#undef cuStreamBeginCaptureToGraph_ptsz
#endif
#ifdef cuStreamBeginCapture_ptsz
#undef cuStreamBeginCapture_ptsz
#endif
#ifdef cuStreamCopyAttributes
#undef cuStreamCopyAttributes
#endif
#ifdef cuStreamCopyAttributes_ptsz
#undef cuStreamCopyAttributes_ptsz
#endif
#ifdef cuStreamCreate
#undef cuStreamCreate
#endif
#ifdef cuStreamCreateWithPriority
#undef cuStreamCreateWithPriority
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#ifdef cuStreamEndCapture
#undef cuStreamEndCapture
#endif
#ifdef cuStreamEndCapture_ptsz
#undef cuStreamEndCapture_ptsz
#endif
#ifdef cuStreamGetAttribute
#undef cuStreamGetAttribute
#endif
#ifdef cuStreamGetAttribute_ptsz
#undef cuStreamGetAttribute_ptsz
#endif
#ifdef cuStreamGetCaptureInfo
#undef cuStreamGetCaptureInfo
#endif
#ifdef cuStreamGetCaptureInfo_ptsz
#undef cuStreamGetCaptureInfo_ptsz
#endif
#ifdef cuStreamGetCtx
#undef cuStreamGetCtx
#endif
#ifdef cuStreamGetCtx_ptsz
#undef cuStreamGetCtx_ptsz
#endif
#ifdef cuStreamGetDevice
#undef cuStreamGetDevice
#endif
#ifdef cuStreamGetDevice_ptsz
#undef cuStreamGetDevice_ptsz
#endif
#ifdef cuStreamGetFlags
#undef cuStreamGetFlags
#endif
#ifdef cuStreamGetFlags_ptsz
#undef cuStreamGetFlags_ptsz
#endif
#ifdef cuStreamGetGreenCtx
#undef cuStreamGetGreenCtx
#endif
#ifdef cuStreamGetId
#undef cuStreamGetId
#endif
#ifdef cuStreamGetId_ptsz
#undef cuStreamGetId_ptsz
#endif
#ifdef cuStreamGetPriority
#undef cuStreamGetPriority
#endif
#ifdef cuStreamGetPriority_ptsz
#undef cuStreamGetPriority_ptsz
#endif
#ifdef cuStreamIsCapturing
#undef cuStreamIsCapturing
#endif
#ifdef cuStreamIsCapturing_ptsz
#undef cuStreamIsCapturing_ptsz
#endif
#ifdef cuStreamQuery
#undef cuStreamQuery
#endif
#ifdef cuStreamQuery_ptsz
#undef cuStreamQuery_ptsz
#endif
#ifdef cuStreamSetAttribute
#undef cuStreamSetAttribute
#endif
#ifdef cuStreamSetAttribute_ptsz
#undef cuStreamSetAttribute_ptsz
#endif
#ifdef cuStreamSynchronize
#undef cuStreamSynchronize
#endif
#ifdef cuStreamSynchronize_ptsz
#undef cuStreamSynchronize_ptsz
#endif
#ifdef cuStreamUpdateCaptureDependencies
#undef cuStreamUpdateCaptureDependencies
#endif
#ifdef cuStreamUpdateCaptureDependencies_ptsz
#undef cuStreamUpdateCaptureDependencies_ptsz
#endif
#ifdef cuStreamWaitEvent
#undef cuStreamWaitEvent
#endif
#ifdef cuStreamWaitEvent_ptsz
#undef cuStreamWaitEvent_ptsz
#endif
#ifdef cuStreamWaitValue32
#undef cuStreamWaitValue32
#endif
#ifdef cuStreamWaitValue32_ptsz
#undef cuStreamWaitValue32_ptsz
#endif
#ifdef cuStreamWaitValue64
#undef cuStreamWaitValue64
#endif
#ifdef cuStreamWaitValue64_ptsz
#undef cuStreamWaitValue64_ptsz
#endif
#ifdef cuStreamWriteValue32
#undef cuStreamWriteValue32
#endif
#ifdef cuStreamWriteValue32_ptsz
#undef cuStreamWriteValue32_ptsz
#endif
#ifdef cuStreamWriteValue64
#undef cuStreamWriteValue64
#endif
#ifdef cuStreamWriteValue64_ptsz
#undef cuStreamWriteValue64_ptsz
#endif
#ifdef cuSurfObjectCreate
#undef cuSurfObjectCreate
#endif
#ifdef cuSurfObjectDestroy
#undef cuSurfObjectDestroy
#endif
#ifdef cuSurfObjectGetResourceDesc
#undef cuSurfObjectGetResourceDesc
#endif
#ifdef cuSurfRefGetArray
#undef cuSurfRefGetArray
#endif
#ifdef cuSurfRefSetArray
#undef cuSurfRefSetArray
#endif
#ifdef cuTensorMapEncodeIm2col
#undef cuTensorMapEncodeIm2col
#endif
#ifdef cuTensorMapEncodeIm2colWide
#undef cuTensorMapEncodeIm2colWide
#endif
#ifdef cuTensorMapEncodeTiled
#undef cuTensorMapEncodeTiled
#endif
#ifdef cuTensorMapReplaceAddress
#undef cuTensorMapReplaceAddress
#endif
#ifdef cuTexObjectCreate
#undef cuTexObjectCreate
#endif
#ifdef cuTexObjectDestroy
#undef cuTexObjectDestroy
#endif
#ifdef cuTexObjectGetResourceDesc
#undef cuTexObjectGetResourceDesc
#endif
#ifdef cuTexObjectGetResourceViewDesc
#undef cuTexObjectGetResourceViewDesc
#endif
#ifdef cuTexObjectGetTextureDesc
#undef cuTexObjectGetTextureDesc
#endif
#ifdef cuTexRefCreate
#undef cuTexRefCreate
#endif
#ifdef cuTexRefDestroy
#undef cuTexRefDestroy
#endif
#ifdef cuTexRefGetAddress
#undef cuTexRefGetAddress
#endif
#ifdef cuTexRefGetAddressMode
#undef cuTexRefGetAddressMode
#endif
#ifdef cuTexRefGetArray
#undef cuTexRefGetArray
#endif
#ifdef cuTexRefGetBorderColor
#undef cuTexRefGetBorderColor
#endif
#ifdef cuTexRefGetFilterMode
#undef cuTexRefGetFilterMode
#endif
#ifdef cuTexRefGetFlags
#undef cuTexRefGetFlags
#endif
#ifdef cuTexRefGetFormat
#undef cuTexRefGetFormat
#endif
#ifdef cuTexRefGetMaxAnisotropy
#undef cuTexRefGetMaxAnisotropy
#endif
#ifdef cuTexRefGetMipmapFilterMode
#undef cuTexRefGetMipmapFilterMode
#endif
#ifdef cuTexRefGetMipmapLevelBias
#undef cuTexRefGetMipmapLevelBias
#endif
#ifdef cuTexRefGetMipmapLevelClamp
#undef cuTexRefGetMipmapLevelClamp
#endif
#ifdef cuTexRefGetMipmappedArray
#undef cuTexRefGetMipmappedArray
#endif
#ifdef cuTexRefSetAddress
#undef cuTexRefSetAddress
#endif
#ifdef cuTexRefSetAddress2D
#undef cuTexRefSetAddress2D
#endif
#ifdef cuTexRefSetAddressMode
#undef cuTexRefSetAddressMode
#endif
#ifdef cuTexRefSetArray
#undef cuTexRefSetArray
#endif
#ifdef cuTexRefSetBorderColor
#undef cuTexRefSetBorderColor
#endif
#ifdef cuTexRefSetFilterMode
#undef cuTexRefSetFilterMode
#endif
#ifdef cuTexRefSetFlags
#undef cuTexRefSetFlags
#endif
#ifdef cuTexRefSetFormat
#undef cuTexRefSetFormat
#endif
#ifdef cuTexRefSetMaxAnisotropy
#undef cuTexRefSetMaxAnisotropy
#endif
#ifdef cuTexRefSetMipmapFilterMode
#undef cuTexRefSetMipmapFilterMode
#endif
#ifdef cuTexRefSetMipmapLevelBias
#undef cuTexRefSetMipmapLevelBias
#endif
#ifdef cuTexRefSetMipmapLevelClamp
#undef cuTexRefSetMipmapLevelClamp
#endif
#ifdef cuTexRefSetMipmappedArray
#undef cuTexRefSetMipmappedArray
#endif
#ifdef cuThreadExchangeStreamCaptureMode
#undef cuThreadExchangeStreamCaptureMode
#endif
#ifdef cuUserObjectCreate
#undef cuUserObjectCreate
#endif
#ifdef cuUserObjectRelease
#undef cuUserObjectRelease
#endif
#ifdef cuUserObjectRetain
#undef cuUserObjectRetain
#endif
#ifdef cuWGLGetDevice
#undef cuWGLGetDevice
#endif
#ifdef cuWaitExternalSemaphoresAsync
#undef cuWaitExternalSemaphoresAsync
#endif
#ifdef cuWaitExternalSemaphoresAsync_ptsz
#undef cuWaitExternalSemaphoresAsync_ptsz
#endif
#ifdef cudbgApiAttach
#undef cudbgApiAttach
#endif
#ifdef cudbgApiDetach
#undef cudbgApiDetach
#endif
#ifdef cudbgApiInit
#undef cudbgApiInit
#endif
#ifdef cudbgGetAPI
#undef cudbgGetAPI
#endif
#ifdef cudbgGetAPIVersion
#undef cudbgGetAPIVersion
#endif
#ifdef cudbgMain
#undef cudbgMain
#endif

class NvcudaLibraryFunctions
{
public:
    using cuArray3DCreate_v2_t = CUresult(CUDAAPI*)(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
    using cuArray3DGetDescriptor_v2_t = CUresult(CUDAAPI*)(CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
    using cuArrayCreate_v2_t        = CUresult(CUDAAPI*)(CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray);
    using cuArrayDestroy_t          = CUresult(CUDAAPI*)(CUarray hArray);
    using cuArrayGetDescriptor_v2_t = CUresult(CUDAAPI*)(CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
    using cuArrayGetMemoryRequirements_t = void (*)();
    using cuArrayGetPlane_t = CUresult(CUDAAPI*)(CUarray* pPlaneArray, CUarray hArray, unsigned int planeIdx);
    using cuArrayGetSparseProperties_t = CUresult(CUDAAPI*)(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties, CUarray array);
    using cuCheckpointProcessCheckpoint_t         = void (*)();
    using cuCheckpointProcessGetRestoreThreadId_t = void (*)();
    using cuCheckpointProcessGetState_t           = void (*)();
    using cuCheckpointProcessLock_t               = void (*)();
    using cuCheckpointProcessRestore_t            = void (*)();
    using cuCheckpointProcessUnlock_t             = void (*)();
    using cuCoredumpGetAttribute_t                = void (*)();
    using cuCoredumpGetAttributeGlobal_t          = void (*)();
    using cuCoredumpSetAttribute_t                = void (*)();
    using cuCoredumpSetAttributeGlobal_t          = void (*)();
    using cuCtxAttach_t                           = CUresult(CUDAAPI*)(CUcontext* pctx, unsigned int flags);
    using cuCtxCreate_v4_t                     = CUresult(CUDAAPI*)(CUcontext* pctx, unsigned int flags, CUdevice dev);
    using cuCtxDestroy_v2_t                    = CUresult(CUDAAPI*)(CUcontext ctx);
    using cuCtxDetach_t                        = CUresult(CUDAAPI*)(CUcontext ctx);
    using cuCtxDisablePeerAccess_t             = CUresult(CUDAAPI*)(CUcontext peerContext);
    using cuCtxEnablePeerAccess_t              = CUresult(CUDAAPI*)(CUcontext peerContext, unsigned int Flags);
    using cuCtxFromGreenCtx_t                  = void (*)();
    using cuCtxGetApiVersion_t                 = CUresult(CUDAAPI*)(CUcontext ctx, unsigned int* version);
    using cuCtxGetCacheConfig_t                = CUresult(CUDAAPI*)(CUfunc_cache* pconfig);
    using cuCtxGetCurrent_t                    = CUresult(CUDAAPI*)(CUcontext* pctx);
    using cuCtxGetDevResource_t                = void (*)();
    using cuCtxGetDevice_t                     = CUresult(CUDAAPI*)(CUdevice* device);
    using cuCtxGetExecAffinity_t               = void (*)();
    using cuCtxGetFlags_t                      = CUresult(CUDAAPI*)(unsigned int* flags);
    using cuCtxGetId_t                         = void (*)();
    using cuCtxGetLimit_t                      = CUresult(CUDAAPI*)(size_t* pvalue, CUlimit limit);
    using cuCtxGetSharedMemConfig_t            = CUresult(CUDAAPI*)(CUsharedconfig* pConfig);
    using cuCtxGetStreamPriorityRange_t        = CUresult(CUDAAPI*)(int* leastPriority, int* greatestPriority);
    using cuCtxPopCurrent_v2_t                 = CUresult(CUDAAPI*)(CUcontext* pctx);
    using cuCtxPushCurrent_v2_t                = CUresult(CUDAAPI*)(CUcontext ctx);
    using cuCtxRecordEvent_t                   = void (*)();
    using cuCtxResetPersistingL2Cache_t        = CUresult(CUDAAPI*)(void);
    using cuCtxSetCacheConfig_t                = CUresult(CUDAAPI*)(CUfunc_cache config);
    using cuCtxSetCurrent_t                    = CUresult(CUDAAPI*)(CUcontext ctx);
    using cuCtxSetFlags_t                      = void (*)();
    using cuCtxSetLimit_t                      = CUresult(CUDAAPI*)(CUlimit limit, size_t value);
    using cuCtxSetSharedMemConfig_t            = CUresult(CUDAAPI*)(CUsharedconfig config);
    using cuCtxSynchronize_t                   = CUresult(CUDAAPI*)(void);
    using cuCtxWaitEvent_t                     = void (*)();
    using cuD3D10CtxCreate_v2_t                = void (*)();
    using cuD3D10CtxCreateOnDevice_t           = void (*)();
    using cuD3D10GetDevice_t                   = void (*)();
    using cuD3D10GetDevices_t                  = void (*)();
    using cuD3D10GetDirect3DDevice_t           = void (*)();
    using cuD3D10MapResources_t                = void (*)();
    using cuD3D10RegisterResource_t            = void (*)();
    using cuD3D10ResourceGetMappedArray_t      = void (*)();
    using cuD3D10ResourceGetMappedPitch_v2_t   = void (*)();
    using cuD3D10ResourceGetMappedPointer_v2_t = void (*)();
    using cuD3D10ResourceGetMappedSize_v2_t    = void (*)();
    using cuD3D10ResourceGetSurfaceDimensions_v2_t = void (*)();
    using cuD3D10ResourceSetMapFlags_t             = void (*)();
    using cuD3D10UnmapResources_t                  = void (*)();
    using cuD3D10UnregisterResource_t              = void (*)();
    using cuD3D11CtxCreate_v2_t                    = void (*)();
    using cuD3D11CtxCreateOnDevice_t               = void (*)();
    using cuD3D11GetDevice_t                       = void (*)();
    using cuD3D11GetDevices_t                      = void (*)();
    using cuD3D11GetDirect3DDevice_t               = void (*)();
    using cuD3D9Begin_t                            = void (*)();
    using cuD3D9CtxCreate_v2_t                     = void (*)();
    using cuD3D9CtxCreateOnDevice_t                = void (*)();
    using cuD3D9End_t                              = void (*)();
    using cuD3D9GetDevice_t                        = void (*)();
    using cuD3D9GetDevices_t                       = void (*)();
    using cuD3D9GetDirect3DDevice_t                = void (*)();
    using cuD3D9MapResources_t                     = void (*)();
    using cuD3D9MapVertexBuffer_v2_t               = void (*)();
    using cuD3D9RegisterResource_t                 = void (*)();
    using cuD3D9RegisterVertexBuffer_t             = void (*)();
    using cuD3D9ResourceGetMappedArray_t           = void (*)();
    using cuD3D9ResourceGetMappedPitch_v2_t        = void (*)();
    using cuD3D9ResourceGetMappedPointer_v2_t      = void (*)();
    using cuD3D9ResourceGetMappedSize_v2_t         = void (*)();
    using cuD3D9ResourceGetSurfaceDimensions_v2_t  = void (*)();
    using cuD3D9ResourceSetMapFlags_t              = void (*)();
    using cuD3D9UnmapResources_t                   = void (*)();
    using cuD3D9UnmapVertexBuffer_t                = void (*)();
    using cuD3D9UnregisterResource_t               = void (*)();
    using cuD3D9UnregisterVertexBuffer_t           = void (*)();
    using cuDestroyExternalMemory_t                = CUresult(CUDAAPI*)(CUexternalMemory extMem);
    using cuDestroyExternalSemaphore_t             = CUresult(CUDAAPI*)(CUexternalSemaphore extSem);
    using cuDevResourceGenerateDesc_t              = void (*)();
    using cuDevSmResourceSplitByCount_t            = void (*)();
    using cuDeviceCanAccessPeer_t          = CUresult(CUDAAPI*)(int* canAccessPeer, CUdevice dev, CUdevice peerDev);
    using cuDeviceComputeCapability_t      = CUresult(CUDAAPI*)(int* major, int* minor, CUdevice dev);
    using cuDeviceGet_t                    = CUresult(CUDAAPI*)(CUdevice* device, int ordinal);
    using cuDeviceGetAttribute_t           = CUresult(CUDAAPI*)(int* pi, CUdevice_attribute attrib, CUdevice dev);
    using cuDeviceGetByPCIBusId_t          = CUresult(CUDAAPI*)(CUdevice* dev, char const* pciBusId);
    using cuDeviceGetCount_t               = CUresult(CUDAAPI*)(int* count);
    using cuDeviceGetDefaultMemPool_t      = CUresult(CUDAAPI*)(CUmemoryPool* pool_out, CUdevice dev);
    using cuDeviceGetDevResource_t         = void (*)();
    using cuDeviceGetExecAffinitySupport_t = void (*)();
    using cuDeviceGetGraphMemAttribute_t   = void (*)();
    using cuDeviceGetLuid_t                = CUresult(CUDAAPI*)(char* luid, unsigned int* deviceNodeMask, CUdevice dev);
    using cuDeviceGetMemPool_t             = CUresult(CUDAAPI*)(CUmemoryPool* pool, CUdevice dev);
    using cuDeviceGetName_t                = CUresult(CUDAAPI*)(char* name, int len, CUdevice dev);
    using cuDeviceGetP2PAttribute_t =
        CUresult(CUDAAPI*)(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);
    using cuDeviceGetPCIBusId_t   = CUresult(CUDAAPI*)(char* pciBusId, int len, CUdevice dev);
    using cuDeviceGetProperties_t = CUresult(CUDAAPI*)(CUdevprop* prop, CUdevice dev);
    using cuDeviceGetTexture1DLinearMaxWidth_t =
        CUresult(CUDAAPI*)(size_t* maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev);
    using cuDeviceGetUuid_v2_t                  = CUresult(CUDAAPI*)(CUuuid* uuid, CUdevice dev);
    using cuDeviceGraphMemTrim_t                = void (*)();
    using cuDevicePrimaryCtxGetState_t          = CUresult(CUDAAPI*)(CUdevice dev, unsigned int* flags, int* active);
    using cuDevicePrimaryCtxRelease_v2_t        = CUresult(CUDAAPI*)(CUdevice dev);
    using cuDevicePrimaryCtxReset_v2_t          = CUresult(CUDAAPI*)(CUdevice dev);
    using cuDevicePrimaryCtxRetain_t            = CUresult(CUDAAPI*)(CUcontext* pctx, CUdevice dev);
    using cuDevicePrimaryCtxSetFlags_v2_t       = CUresult(CUDAAPI*)(CUdevice dev, unsigned int flags);
    using cuDeviceRegisterAsyncNotification_t   = void (*)();
    using cuDeviceSetGraphMemAttribute_t        = void (*)();
    using cuDeviceSetMemPool_t                  = CUresult(CUDAAPI*)(CUdevice dev, CUmemoryPool pool);
    using cuDeviceTotalMem_v2_t                 = CUresult(CUDAAPI*)(size_t* bytes, CUdevice dev);
    using cuDeviceUnregisterAsyncNotification_t = void (*)();
    using cuDriverGetVersion_t                  = CUresult(CUDAAPI*)(int* driverVersion);
    using cuEventCreate_t                       = CUresult(CUDAAPI*)(CUevent* phEvent, unsigned int Flags);
    using cuEventDestroy_v2_t                   = CUresult(CUDAAPI*)(CUevent hEvent);
    using cuEventElapsedTime_v2_t           = CUresult(CUDAAPI*)(float* pMilliseconds, CUevent hStart, CUevent hEnd);
    using cuEventQuery_t                    = CUresult(CUDAAPI*)(CUevent hEvent);
    using cuEventRecord_t                   = CUresult(CUDAAPI*)(CUevent hEvent, CUstream hStream);
    using cuEventRecordWithFlags_t          = CUresult(CUDAAPI*)(CUevent hEvent, CUstream hStream, unsigned int flags);
    using cuEventRecordWithFlags_ptsz_t     = void (*)();
    using cuEventRecord_ptsz_t              = void (*)();
    using cuEventSynchronize_t              = CUresult(CUDAAPI*)(CUevent hEvent);
    using cuExternalMemoryGetMappedBuffer_t = CUresult(
        CUDAAPI*)(CUdeviceptr* devPtr, CUexternalMemory extMem, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc);
    using cuExternalMemoryGetMappedMipmappedArray_t = CUresult(CUDAAPI*)(
        CUmipmappedArray*                                mipmap,
        CUexternalMemory                                 extMem,
        const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmapDesc);
    using cuFlushGPUDirectRDMAWrites_t  = void (*)();
    using cuFuncGetAttribute_t          = CUresult(CUDAAPI*)(int* pi, CUfunction_attribute attrib, CUfunction hfunc);
    using cuFuncGetModule_t             = CUresult(CUDAAPI*)(CUmodule* hmod, CUfunction hfunc);
    using cuFuncGetName_t               = void (*)();
    using cuFuncGetParamInfo_t          = void (*)();
    using cuFuncIsLoaded_t              = void (*)();
    using cuFuncLoad_t                  = void (*)();
    using cuFuncSetAttribute_t          = CUresult(CUDAAPI*)(CUfunction hfunc, CUfunction_attribute attrib, int value);
    using cuFuncSetBlockShape_t         = CUresult(CUDAAPI*)(CUfunction hfunc, int x, int y, int z);
    using cuFuncSetCacheConfig_t        = CUresult(CUDAAPI*)(CUfunction hfunc, CUfunc_cache config);
    using cuFuncSetSharedMemConfig_t    = CUresult(CUDAAPI*)(CUfunction hfunc, CUsharedconfig config);
    using cuFuncSetSharedSize_t         = CUresult(CUDAAPI*)(CUfunction hfunc, unsigned int bytes);
    using cuGLCtxCreate_v2_t            = void (*)();
    using cuGLGetDevices_v2_t           = void (*)();
    using cuGLInit_t                    = void (*)();
    using cuGLMapBufferObject_v2_t      = void (*)();
    using cuGLMapBufferObjectAsync_v2_t = void (*)();
    using cuGLRegisterBufferObject_t    = void (*)();
    using cuGLSetBufferObjectMapFlags_t = void (*)();
    using cuGLUnmapBufferObject_t       = void (*)();
    using cuGLUnmapBufferObjectAsync_t  = void (*)();
    using cuGLUnregisterBufferObject_t  = void (*)();
    using cuGetErrorName_t              = CUresult(CUDAAPI*)(CUresult error, char const** pStr);
    using cuGetErrorString_t            = CUresult(CUDAAPI*)(CUresult error, char const** pStr);
    using cuGetExportTable_t            = CUresult(CUDAAPI*)(void const** ppExportTable, CUuuid const* pExportTableId);
    using cuGetProcAddress_v2_t         = void (*)();
    using cuGraphAddBatchMemOpNode_t    = void (*)();
    using cuGraphAddChildGraphNode_t    = CUresult(CUDAAPI*)(
        CUgraphNode*       phGraphNode,
        CUgraph            hGraph,
        CUgraphNode const* dependencies,
        size_t             numDependencies,
        CUgraph            childGraph);
    using cuGraphAddDependencies_v2_t =
        CUresult(CUDAAPI*)(CUgraph hGraph, CUgraphNode const* from, CUgraphNode const* to, size_t numDependencies);
    using cuGraphAddEmptyNode_t = CUresult(
        CUDAAPI*)(CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode const* dependencies, size_t numDependencies);
    using cuGraphAddEventRecordNode_t = CUresult(
        CUDAAPI*)(CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode const* dependencies, size_t numDependencies, CUevent event);
    using cuGraphAddEventWaitNode_t = CUresult(
        CUDAAPI*)(CUgraphNode* phGraphNode, CUgraph hGraph, CUgraphNode const* dependencies, size_t numDependencies, CUevent event);
    using cuGraphAddExternalSemaphoresSignalNode_t = CUresult(CUDAAPI*)(
        CUgraphNode*                           phGraphNode,
        CUgraph                                hGraph,
        CUgraphNode const*                     dependencies,
        size_t                                 numDependencies,
        const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
    using cuGraphAddExternalSemaphoresWaitNode_t = CUresult(CUDAAPI*)(
        CUgraphNode*                         phGraphNode,
        CUgraph                              hGraph,
        CUgraphNode const*                   dependencies,
        size_t                               numDependencies,
        const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
    using cuGraphAddHostNode_t = CUresult(CUDAAPI*)(
        CUgraphNode*                 phGraphNode,
        CUgraph                      hGraph,
        CUgraphNode const*           dependencies,
        size_t                       numDependencies,
        const CUDA_HOST_NODE_PARAMS* nodeParams);
    using cuGraphAddKernelNode_v2_t = CUresult(CUDAAPI*)(
        CUgraphNode*                   phGraphNode,
        CUgraph                        hGraph,
        CUgraphNode const*             dependencies,
        size_t                         numDependencies,
        const CUDA_KERNEL_NODE_PARAMS* nodeParams);
    using cuGraphAddMemAllocNode_t = void (*)();
    using cuGraphAddMemFreeNode_t  = void (*)();
    using cuGraphAddMemcpyNode_t   = CUresult(CUDAAPI*)(
        CUgraphNode*         phGraphNode,
        CUgraph              hGraph,
        CUgraphNode const*   dependencies,
        size_t               numDependencies,
        const CUDA_MEMCPY3D* copyParams,
        CUcontext            ctx);
    using cuGraphAddMemsetNode_t = CUresult(CUDAAPI*)(
        CUgraphNode*                   phGraphNode,
        CUgraph                        hGraph,
        CUgraphNode const*             dependencies,
        size_t                         numDependencies,
        const CUDA_MEMSET_NODE_PARAMS* memsetParams,
        CUcontext                      ctx);
    using cuGraphAddNode_v2_t                  = void (*)();
    using cuGraphBatchMemOpNodeGetParams_t     = void (*)();
    using cuGraphBatchMemOpNodeSetParams_t     = void (*)();
    using cuGraphChildGraphNodeGetGraph_t      = CUresult(CUDAAPI*)(CUgraphNode hNode, CUgraph* phGraph);
    using cuGraphClone_t                       = CUresult(CUDAAPI*)(CUgraph* phGraphClone, CUgraph originalGraph);
    using cuGraphConditionalHandleCreate_t     = void (*)();
    using cuGraphCreate_t                      = CUresult(CUDAAPI*)(CUgraph* phGraph, unsigned int flags);
    using cuGraphDebugDotPrint_t               = void (*)();
    using cuGraphDestroy_t                     = CUresult(CUDAAPI*)(CUgraph hGraph);
    using cuGraphDestroyNode_t                 = CUresult(CUDAAPI*)(CUgraphNode hNode);
    using cuGraphEventRecordNodeGetEvent_t     = CUresult(CUDAAPI*)(CUgraphNode hNode, CUevent* event_out);
    using cuGraphEventRecordNodeSetEvent_t     = CUresult(CUDAAPI*)(CUgraphNode hNode, CUevent event);
    using cuGraphEventWaitNodeGetEvent_t       = CUresult(CUDAAPI*)(CUgraphNode hNode, CUevent* event_out);
    using cuGraphEventWaitNodeSetEvent_t       = CUresult(CUDAAPI*)(CUgraphNode hNode, CUevent event);
    using cuGraphExecBatchMemOpNodeSetParams_t = void (*)();
    using cuGraphExecChildGraphNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, CUgraph childGraph);
    using cuGraphExecDestroy_t = CUresult(CUDAAPI*)(CUgraphExec hGraphExec);
    using cuGraphExecEventRecordNodeSetEvent_t = CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
    using cuGraphExecEventWaitNodeSetEvent_t = CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, CUevent event);
    using cuGraphExecExternalSemaphoresSignalNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
    using cuGraphExecExternalSemaphoresWaitNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
    using cuGraphExecGetFlags_t = void (*)();
    using cuGraphExecHostNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
    using cuGraphExecKernelNodeSetParams_v2_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
    using cuGraphExecMemcpyNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D* copyParams, CUcontext ctx);
    using cuGraphExecMemsetNodeSetParams_t = CUresult(
        CUDAAPI*)(CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* memsetParams, CUcontext ctx);
    using cuGraphExecNodeSetParams_t = void (*)();
    using cuGraphExecUpdate_v2_t     = CUresult(CUDAAPI*)(CUgraphExec              hGraphExec,
                                                      CUgraph                  hGraph,
                                                      CUgraphNode*             hErrorNode_out,
                                                      CUgraphExecUpdateResult* updateResult_out);
    using cuGraphExternalSemaphoresSignalNodeGetParams_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* params_out);
    using cuGraphExternalSemaphoresSignalNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS* nodeParams);
    using cuGraphExternalSemaphoresWaitNodeGetParams_t = CUresult(CUDAAPI*)(CUgraphNode                    hNode,
                                                                            CUDA_EXT_SEM_WAIT_NODE_PARAMS* params_out);
    using cuGraphExternalSemaphoresWaitNodeSetParams_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS* nodeParams);
    using cuGraphGetEdges_v2_t = CUresult(CUDAAPI*)(CUgraph hGraph, CUgraphNode* from, CUgraphNode* to, size_t* numEdges);
    using cuGraphGetNodes_t          = CUresult(CUDAAPI*)(CUgraph hGraph, CUgraphNode* nodes, size_t* numNodes);
    using cuGraphGetRootNodes_t      = CUresult(CUDAAPI*)(CUgraph hGraph, CUgraphNode* rootNodes, size_t* numRootNodes);
    using cuGraphHostNodeGetParams_t = CUresult(CUDAAPI*)(CUgraphNode hNode, CUDA_HOST_NODE_PARAMS* nodeParams);
    using cuGraphHostNodeSetParams_t = CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_HOST_NODE_PARAMS* nodeParams);
    using cuGraphInstantiate_v2_t    = CUresult(
        CUDAAPI*)(CUgraphExec* phGraphExec, CUgraph hGraph, CUgraphNode* phErrorNode, char* logBuffer, size_t bufferSize);
    using cuGraphInstantiateWithFlags_t       = void (*)();
    using cuGraphInstantiateWithParams_t      = void (*)();
    using cuGraphInstantiateWithParams_ptsz_t = void (*)();
    using cuGraphKernelNodeCopyAttributes_t   = CUresult(CUDAAPI*)(CUgraphNode dst, CUgraphNode src);
    using cuGraphKernelNodeGetAttribute_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue* value_out);
    using cuGraphKernelNodeGetParams_v2_t = CUresult(CUDAAPI*)(CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS* nodeParams);
    using cuGraphKernelNodeSetAttribute_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, CUkernelNodeAttrID attr, CUkernelNodeAttrValue const* value);
    using cuGraphKernelNodeSetParams_v2_t = CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS* nodeParams);
    using cuGraphLaunch_t                = CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUstream hStream);
    using cuGraphLaunch_ptsz_t           = void (*)();
    using cuGraphMemAllocNodeGetParams_t = void (*)();
    using cuGraphMemFreeNodeGetParams_t  = void (*)();
    using cuGraphMemcpyNodeGetParams_t   = CUresult(CUDAAPI*)(CUgraphNode hNode, CUDA_MEMCPY3D* nodeParams);
    using cuGraphMemcpyNodeSetParams_t   = CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_MEMCPY3D* nodeParams);
    using cuGraphMemsetNodeGetParams_t   = CUresult(CUDAAPI*)(CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS* nodeParams);
    using cuGraphMemsetNodeSetParams_t = CUresult(CUDAAPI*)(CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS* nodeParams);
    using cuGraphNodeFindInClone_t = CUresult(CUDAAPI*)(CUgraphNode* phNode, CUgraphNode hOriginalNode, CUgraph hClonedGraph);
    using cuGraphNodeGetDependencies_v2_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, CUgraphNode* dependencies, size_t* numDependencies);
    using cuGraphNodeGetDependentNodes_v2_t =
        CUresult(CUDAAPI*)(CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes);
    using cuGraphNodeGetEnabled_t    = void (*)();
    using cuGraphNodeGetType_t       = CUresult(CUDAAPI*)(CUgraphNode hNode, CUgraphNodeType* type);
    using cuGraphNodeSetEnabled_t    = void (*)();
    using cuGraphNodeSetParams_t     = void (*)();
    using cuGraphReleaseUserObject_t = void (*)();
    using cuGraphRemoveDependencies_v2_t =
        CUresult(CUDAAPI*)(CUgraph hGraph, CUgraphNode const* from, CUgraphNode const* to, size_t numDependencies);
    using cuGraphRetainUserObject_t         = void (*)();
    using cuGraphUpload_t                   = CUresult(CUDAAPI*)(CUgraphExec hGraphExec, CUstream hStream);
    using cuGraphUpload_ptsz_t              = void (*)();
    using cuGraphicsD3D10RegisterResource_t = void (*)();
    using cuGraphicsD3D11RegisterResource_t = void (*)();
    using cuGraphicsD3D9RegisterResource_t  = void (*)();
    using cuGraphicsGLRegisterBuffer_t      = void (*)();
    using cuGraphicsGLRegisterImage_t       = void (*)();
    using cuGraphicsMapResources_t = CUresult(CUDAAPI*)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
    using cuGraphicsMapResources_ptsz_t               = void (*)();
    using cuGraphicsResourceGetMappedMipmappedArray_t = CUresult(CUDAAPI*)(CUmipmappedArray*  pMipmappedArray,
                                                                           CUgraphicsResource resource);
    using cuGraphicsResourceGetMappedPointer_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
    using cuGraphicsResourceSetMapFlags_v2_t = CUresult(CUDAAPI*)(CUgraphicsResource resource, unsigned int flags);
    using cuGraphicsSubResourceGetMappedArray_t =
        CUresult(CUDAAPI*)(CUarray* pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel);
    using cuGraphicsUnmapResources_t = CUresult(CUDAAPI*)(unsigned int count, CUgraphicsResource* resources, CUstream hStream);
    using cuGraphicsUnmapResources_ptsz_t = void (*)();
    using cuGraphicsUnregisterResource_t  = CUresult(CUDAAPI*)(CUgraphicsResource resource);
    using cuGreenCtxCreate_t              = void (*)();
    using cuGreenCtxDestroy_t             = void (*)();
    using cuGreenCtxGetDevResource_t      = void (*)();
    using cuGreenCtxRecordEvent_t         = void (*)();
    using cuGreenCtxStreamCreate_t        = void (*)();
    using cuGreenCtxWaitEvent_t           = void (*)();
    using cuImportExternalMemory_t        = CUresult(CUDAAPI*)(CUexternalMemory*                       extMem_out,
                                                        const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memHandleDesc);
    using cuImportExternalSemaphore_t     = CUresult(CUDAAPI*)(CUexternalSemaphore*                       extSem_out,
                                                           const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semHandleDesc);
    using cuInit_t                        = CUresult(CUDAAPI*)(unsigned int Flags);
    using cuIpcCloseMemHandle_t           = CUresult(CUDAAPI*)(CUdeviceptr dptr);
    using cuIpcGetEventHandle_t           = CUresult(CUDAAPI*)(CUipcEventHandle* pHandle, CUevent event);
    using cuIpcGetMemHandle_t             = CUresult(CUDAAPI*)(CUipcMemHandle* pHandle, CUdeviceptr dptr);
    using cuIpcOpenEventHandle_t          = CUresult(CUDAAPI*)(CUevent* phEvent, CUipcEventHandle handle);
    using cuIpcOpenMemHandle_v2_t  = CUresult(CUDAAPI*)(CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags);
    using cuKernelGetAttribute_t   = void (*)();
    using cuKernelGetFunction_t    = void (*)();
    using cuKernelGetLibrary_t     = void (*)();
    using cuKernelGetName_t        = void (*)();
    using cuKernelGetParamInfo_t   = void (*)();
    using cuKernelSetAttribute_t   = void (*)();
    using cuKernelSetCacheConfig_t = void (*)();
    using cuLaunch_t               = CUresult(CUDAAPI*)(CUfunction f);
    using cuLaunchCooperativeKernel_t = CUresult(CUDAAPI*)(
        CUfunction   f,
        unsigned int gridDimX,
        unsigned int gridDimY,
        unsigned int gridDimZ,
        unsigned int blockDimX,
        unsigned int blockDimY,
        unsigned int blockDimZ,
        unsigned int sharedMemBytes,
        CUstream     hStream,
        void**       kernelParams);
    using cuLaunchCooperativeKernelMultiDevice_t =
        CUresult(CUDAAPI*)(CUDA_LAUNCH_PARAMS* launchParamsList, unsigned int numDevices, unsigned int flags);
    using cuLaunchCooperativeKernel_ptsz_t = void (*)();
    using cuLaunchGrid_t                   = CUresult(CUDAAPI*)(CUfunction f, int grid_width, int grid_height);
    using cuLaunchGridAsync_t     = CUresult(CUDAAPI*)(CUfunction f, int grid_width, int grid_height, CUstream hStream);
    using cuLaunchHostFunc_t      = CUresult(CUDAAPI*)(CUstream hStream, CUhostFn fn, void* userData);
    using cuLaunchHostFunc_ptsz_t = void (*)();
    using cuLaunchKernel_t        = CUresult(CUDAAPI*)(
        CUfunction   f,
        unsigned int gridDimX,
        unsigned int gridDimY,
        unsigned int gridDimZ,
        unsigned int blockDimX,
        unsigned int blockDimY,
        unsigned int blockDimZ,
        unsigned int sharedMemBytes,
        CUstream     hStream,
        void**       kernelParams,
        void**       extra);
    using cuLaunchKernelEx_t            = void (*)();
    using cuLaunchKernelEx_ptsz_t       = void (*)();
    using cuLaunchKernel_ptsz_t         = void (*)();
    using cuLibraryEnumerateKernels_t   = void (*)();
    using cuLibraryGetGlobal_t          = void (*)();
    using cuLibraryGetKernel_t          = void (*)();
    using cuLibraryGetKernelCount_t     = void (*)();
    using cuLibraryGetManaged_t         = void (*)();
    using cuLibraryGetModule_t          = void (*)();
    using cuLibraryGetUnifiedFunction_t = void (*)();
    using cuLibraryLoadData_t           = void (*)();
    using cuLibraryLoadFromFile_t       = void (*)();
    using cuLibraryUnload_t             = void (*)();
    using cuLinkAddData_v2_t            = CUresult(CUDAAPI*)(
        CUlinkState    state,
        CUjitInputType type,
        void*          data,
        size_t         size,
        char const*    name,
        unsigned int   numOptions,
        CUjit_option*  options,
        void**         optionValues);
    using cuLinkAddFile_v2_t = CUresult(CUDAAPI*)(
        CUlinkState    state,
        CUjitInputType type,
        char const*    path,
        unsigned int   numOptions,
        CUjit_option*  options,
        void**         optionValues);
    using cuLinkComplete_t = CUresult(CUDAAPI*)(CUlinkState state, void** cubinOut, size_t* sizeOut);
    using cuLinkCreate_v2_t =
        CUresult(CUDAAPI*)(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
    using cuLinkDestroy_t    = CUresult(CUDAAPI*)(CUlinkState state);
    using cuMemAddressFree_t = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size);
    using cuMemAddressReserve_t =
        CUresult(CUDAAPI*)(CUdeviceptr* ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);
    using cuMemAdvise_v2_t = CUresult(CUDAAPI*)(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device);
    using cuMemAlloc_v2_t  = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t bytesize);
    using cuMemAllocAsync_t      = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t bytesize, CUstream hStream);
    using cuMemAllocAsync_ptsz_t = void (*)();
    using cuMemAllocFromPoolAsync_t = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t bytesize, CUmemoryPool pool, CUstream hStream);
    using cuMemAllocFromPoolAsync_ptsz_t = void (*)();
    using cuMemAllocHost_v2_t            = CUresult(CUDAAPI*)(void** pp, size_t bytesize);
    using cuMemAllocManaged_t            = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t bytesize, unsigned int flags);
    using cuMemAllocPitch_v2_t           = CUresult(
        CUDAAPI*)(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
    using cuMemBatchDecompressAsync_t      = void (*)();
    using cuMemBatchDecompressAsync_ptsz_t = void (*)();
    using cuMemCreate_t                    = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle* handle,
                                             size_t                        size,
                                             CUmemAllocationProp const*    prop,
                                             unsigned long long            flags);
    using cuMemExportToShareableHandle_t   = CUresult(CUDAAPI*)(
        void*                        shareableHandle,
        CUmemGenericAllocationHandle handle,
        CUmemAllocationHandleType    handleType,
        unsigned long long           flags);
    using cuMemFree_v2_t        = CUresult(CUDAAPI*)(CUdeviceptr dptr);
    using cuMemFreeAsync_t      = CUresult(CUDAAPI*)(CUdeviceptr dptr, CUstream hStream);
    using cuMemFreeAsync_ptsz_t = void (*)();
    using cuMemFreeHost_t       = CUresult(CUDAAPI*)(void* p);
    using cuMemGetAccess_t = CUresult(CUDAAPI*)(unsigned long long* flags, CUmemLocation const* location, CUdeviceptr ptr);
    using cuMemGetAddressRange_v2_t       = CUresult(CUDAAPI*)(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
    using cuMemGetAllocationGranularity_t = CUresult(
        CUDAAPI*)(size_t* granularity, CUmemAllocationProp const* prop, CUmemAllocationGranularity_flags option);
    using cuMemGetAllocationPropertiesFromHandle_t = CUresult(CUDAAPI*)(CUmemAllocationProp*         prop,
                                                                        CUmemGenericAllocationHandle handle);
    using cuMemGetHandleForAddressRange_t          = void (*)();
    using cuMemGetInfo_v2_t                        = CUresult(CUDAAPI*)(size_t* free, size_t* total);
    using cuMemHostAlloc_t                         = CUresult(CUDAAPI*)(void** pp, size_t bytesize, unsigned int Flags);
    using cuMemHostGetDevicePointer_v2_t = CUresult(CUDAAPI*)(CUdeviceptr* pdptr, void* p, unsigned int Flags);
    using cuMemHostGetFlags_t            = CUresult(CUDAAPI*)(unsigned int* pFlags, void* p);
    using cuMemHostRegister_v2_t         = CUresult(CUDAAPI*)(void* p, size_t bytesize, unsigned int Flags);
    using cuMemHostUnregister_t          = CUresult(CUDAAPI*)(void* p);
    using cuMemImportFromShareableHandle_t =
        CUresult(CUDAAPI*)(CUmemGenericAllocationHandle* handle, void* osHandle, CUmemAllocationHandleType shHandleType);
    using cuMemMap_t = CUresult(
        CUDAAPI*)(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags);
    using cuMemMapArrayAsync_t = CUresult(CUDAAPI*)(CUarrayMapInfo* mapInfoList, unsigned int count, CUstream hStream);
    using cuMemMapArrayAsync_ptsz_t = void (*)();
    using cuMemPoolCreate_t         = CUresult(CUDAAPI*)(CUmemoryPool* pool, CUmemPoolProps const* poolProps);
    using cuMemPoolDestroy_t        = CUresult(CUDAAPI*)(CUmemoryPool pool);
    using cuMemPoolExportPointer_t  = CUresult(CUDAAPI*)(CUmemPoolPtrExportData* shareData_out, CUdeviceptr ptr);
    using cuMemPoolExportToShareableHandle_t = CUresult(
        CUDAAPI*)(void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handleType, unsigned long long flags);
    using cuMemPoolGetAccess_t = CUresult(CUDAAPI*)(CUmemAccess_flags* flags, CUmemoryPool memPool, CUmemLocation* location);
    using cuMemPoolGetAttribute_t = CUresult(CUDAAPI*)(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
    using cuMemPoolImportFromShareableHandle_t = CUresult(
        CUDAAPI*)(CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handleType, unsigned long long flags);
    using cuMemPoolImportPointer_t =
        CUresult(CUDAAPI*)(CUdeviceptr* ptr_out, CUmemoryPool pool, CUmemPoolPtrExportData* shareData);
    using cuMemPoolSetAccess_t    = CUresult(CUDAAPI*)(CUmemoryPool pool, CUmemAccessDesc const* map, size_t count);
    using cuMemPoolSetAttribute_t = CUresult(CUDAAPI*)(CUmemoryPool pool, CUmemPool_attribute attr, void* value);
    using cuMemPoolTrimTo_t       = CUresult(CUDAAPI*)(CUmemoryPool pool, size_t minBytesToKeep);
    using cuMemPrefetchAsync_v2_t = CUresult(CUDAAPI*)(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream);
    using cuMemPrefetchAsync_ptsz_t = void (*)();
    using cuMemRangeGetAttribute_t  = CUresult(
        CUDAAPI*)(void* data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count);
    using cuMemRangeGetAttributes_t = CUresult(CUDAAPI*)(
        void**                 data,
        size_t*                dataSizes,
        CUmem_range_attribute* attributes,
        size_t                 numAttributes,
        CUdeviceptr            devPtr,
        size_t                 count);
    using cuMemRelease_t                = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle handle);
    using cuMemRetainAllocationHandle_t = CUresult(CUDAAPI*)(CUmemGenericAllocationHandle* handle, void* addr);
    using cuMemSetAccess_t = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size, CUmemAccessDesc const* desc, size_t count);
    using cuMemUnmap_t                = CUresult(CUDAAPI*)(CUdeviceptr ptr, size_t size);
    using cuMemcpy_t                  = CUresult(CUDAAPI*)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
    using cuMemcpy2D_v2_t             = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy);
    using cuMemcpy2DAsync_v2_t        = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy, CUstream hStream);
    using cuMemcpy2DUnaligned_v2_t    = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D* pCopy);
    using cuMemcpy3D_v2_t             = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D* pCopy);
    using cuMemcpy3DAsync_v2_t        = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D* pCopy, CUstream hStream);
    using cuMemcpy3DBatchAsync_t      = void (*)();
    using cuMemcpy3DBatchAsync_ptsz_t = void (*)();
    using cuMemcpy3DPeer_t            = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D_PEER* pCopy);
    using cuMemcpy3DPeerAsync_t       = CUresult(CUDAAPI*)(const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream);
    using cuMemcpy3DPeerAsync_ptsz_t  = void (*)();
    using cuMemcpy3DPeer_ptds_t       = void (*)();
    using cuMemcpyAsync_t = CUresult(CUDAAPI*)(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
    using cuMemcpyAsync_ptsz_t = void (*)();
    using cuMemcpyAtoA_v2_t =
        CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    using cuMemcpyAtoD_v2_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    using cuMemcpyAtoH_v2_t = CUresult(CUDAAPI*)(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
    using cuMemcpyAtoHAsync_v2_t =
        CUresult(CUDAAPI*)(void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
    using cuMemcpyBatchAsync_t      = void (*)();
    using cuMemcpyBatchAsync_ptsz_t = void (*)();
    using cuMemcpyDtoA_v2_t = CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
    using cuMemcpyDtoD_v2_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
    using cuMemcpyDtoDAsync_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
    using cuMemcpyDtoH_v2_t = CUresult(CUDAAPI*)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
    using cuMemcpyDtoHAsync_v2_t = CUresult(CUDAAPI*)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
    using cuMemcpyHtoA_v2_t = CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset, void const* srcHost, size_t ByteCount);
    using cuMemcpyHtoAAsync_v2_t =
        CUresult(CUDAAPI*)(CUarray dstArray, size_t dstOffset, void const* srcHost, size_t ByteCount, CUstream hStream);
    using cuMemcpyHtoD_v2_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, void const* srcHost, size_t ByteCount);
    using cuMemcpyHtoDAsync_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr dstDevice, void const* srcHost, size_t ByteCount, CUstream hStream);
    using cuMemcpyPeer_t = CUresult(
        CUDAAPI*)(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
    using cuMemcpyPeerAsync_t = CUresult(CUDAAPI*)(
        CUdeviceptr dstDevice,
        CUcontext   dstContext,
        CUdeviceptr srcDevice,
        CUcontext   srcContext,
        size_t      ByteCount,
        CUstream    hStream);
    using cuMemcpyPeerAsync_ptsz_t = void (*)();
    using cuMemcpyPeer_ptds_t      = void (*)();
    using cuMemcpy_ptds_t          = void (*)();
    using cuMemsetD16_v2_t         = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned short us, size_t N);
    using cuMemsetD16Async_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream);
    using cuMemsetD16Async_ptsz_t = void (*)();
    using cuMemsetD2D16_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height);
    using cuMemsetD2D16Async_t = CUresult(
        CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream);
    using cuMemsetD2D16Async_ptsz_t = void (*)();
    using cuMemsetD2D32_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height);
    using cuMemsetD2D32Async_t = CUresult(
        CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream);
    using cuMemsetD2D32Async_ptsz_t = void (*)();
    using cuMemsetD2D8_v2_t =
        CUresult(CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height);
    using cuMemsetD2D8Async_t = CUresult(
        CUDAAPI*)(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream);
    using cuMemsetD2D8Async_ptsz_t = void (*)();
    using cuMemsetD32_v2_t         = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned int ui, size_t N);
    using cuMemsetD32Async_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream);
    using cuMemsetD32Async_ptsz_t = void (*)();
    using cuMemsetD8_v2_t         = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
    using cuMemsetD8Async_t = CUresult(CUDAAPI*)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
    using cuMemsetD8Async_ptsz_t    = void (*)();
    using cuMipmappedArrayCreate_t  = CUresult(CUDAAPI*)(CUmipmappedArray*              pHandle,
                                                        const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc,
                                                        unsigned int                   numMipmapLevels);
    using cuMipmappedArrayDestroy_t = CUresult(CUDAAPI*)(CUmipmappedArray hMipmappedArray);
    using cuMipmappedArrayGetLevel_t =
        CUresult(CUDAAPI*)(CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level);
    using cuMipmappedArrayGetMemoryRequirements_t = void (*)();
    using cuMipmappedArrayGetSparseProperties_t   = CUresult(CUDAAPI*)(CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties,
                                                                     CUmipmappedArray              mipmap);
    using cuModuleEnumerateFunctions_t            = void (*)();
    using cuModuleGetFunction_t      = CUresult(CUDAAPI*)(CUfunction* hfunc, CUmodule hmod, char const* name);
    using cuModuleGetFunctionCount_t = void (*)();
    using cuModuleGetGlobal_v2_t = CUresult(CUDAAPI*)(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, char const* name);
    using cuModuleGetLoadingMode_t = void (*)();
    using cuModuleGetSurfRef_t     = CUresult(CUDAAPI*)(CUsurfref* pSurfRef, CUmodule hmod, char const* name);
    using cuModuleGetTexRef_t      = CUresult(CUDAAPI*)(CUtexref* pTexRef, CUmodule hmod, char const* name);
    using cuModuleLoad_t           = CUresult(CUDAAPI*)(CUmodule* module, char const* fname);
    using cuModuleLoadData_t       = CUresult(CUDAAPI*)(CUmodule* module, void const* image);
    using cuModuleLoadDataEx_t     = CUresult(
        CUDAAPI*)(CUmodule* module, void const* image, unsigned int numOptions, CUjit_option* options, void** optionValues);
    using cuModuleLoadFatBinary_t     = CUresult(CUDAAPI*)(CUmodule* module, void const* fatCubin);
    using cuModuleUnload_t            = CUresult(CUDAAPI*)(CUmodule hmod);
    using cuMulticastAddDevice_t      = void (*)();
    using cuMulticastBindAddr_t       = void (*)();
    using cuMulticastBindMem_t        = void (*)();
    using cuMulticastCreate_t         = void (*)();
    using cuMulticastGetGranularity_t = void (*)();
    using cuMulticastUnbind_t         = void (*)();
    using cuOccupancyAvailableDynamicSMemPerBlock_t =
        CUresult(CUDAAPI*)(size_t* dynamicSmemSize, CUfunction func, int numBlocks, int blockSize);
    using cuOccupancyMaxActiveBlocksPerMultiprocessor_t =
        CUresult(CUDAAPI*)(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
    using cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_t =
        CUresult(CUDAAPI*)(int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags);
    using cuOccupancyMaxActiveClusters_t     = void (*)();
    using cuOccupancyMaxPotentialBlockSize_t = CUresult(CUDAAPI*)(
        int*               minGridSize,
        int*               blockSize,
        CUfunction         func,
        CUoccupancyB2DSize blockSizeToDynamicSMemSize,
        size_t             dynamicSMemSize,
        int                blockSizeLimit);
    using cuOccupancyMaxPotentialBlockSizeWithFlags_t = CUresult(CUDAAPI*)(
        int*               minGridSize,
        int*               blockSize,
        CUfunction         func,
        CUoccupancyB2DSize blockSizeToDynamicSMemSize,
        size_t             dynamicSMemSize,
        int                blockSizeLimit,
        unsigned int       flags);
    using cuOccupancyMaxPotentialClusterSize_t = void (*)();
    using cuParamSetSize_t                     = CUresult(CUDAAPI*)(CUfunction hfunc, unsigned int numbytes);
    using cuParamSetTexRef_t                   = CUresult(CUDAAPI*)(CUfunction hfunc, int texunit, CUtexref hTexRef);
    using cuParamSetf_t                        = CUresult(CUDAAPI*)(CUfunction hfunc, int offset, float value);
    using cuParamSeti_t                        = CUresult(CUDAAPI*)(CUfunction hfunc, int offset, unsigned int value);
    using cuParamSetv_t           = CUresult(CUDAAPI*)(CUfunction hfunc, int offset, void* ptr, unsigned int numbytes);
    using cuPointerGetAttribute_t = CUresult(CUDAAPI*)(void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
    using cuPointerGetAttributes_t =
        CUresult(CUDAAPI*)(unsigned int numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
    using cuPointerSetAttribute_t = CUresult(CUDAAPI*)(void const* value, CUpointer_attribute attribute, CUdeviceptr ptr);
    using cuProfilerInitialize_t            = void (*)();
    using cuProfilerStart_t                 = void (*)();
    using cuProfilerStop_t                  = void (*)();
    using cuSignalExternalSemaphoresAsync_t = CUresult(CUDAAPI*)(
        CUexternalSemaphore const*                   extSemArray,
        const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray,
        unsigned int                                 numExtSems,
        CUstream                                     stream);
    using cuSignalExternalSemaphoresAsync_ptsz_t = void (*)();
    using cuStreamAddCallback_t =
        CUresult(CUDAAPI*)(CUstream hStream, CUstreamCallback callback, void* userData, unsigned int flags);
    using cuStreamAddCallback_ptsz_t = void (*)();
    using cuStreamAttachMemAsync_t = CUresult(CUDAAPI*)(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags);
    using cuStreamAttachMemAsync_ptsz_t = void (*)();
    using cuStreamBatchMemOp_v2_t       = CUresult(
        CUDAAPI*)(CUstream stream, unsigned int count, CUstreamBatchMemOpParams* paramArray, unsigned int flags);
    using cuStreamBatchMemOp_ptsz_t          = void (*)();
    using cuStreamBeginCapture_v2_t          = CUresult(CUDAAPI*)(CUstream hStream, CUstreamCaptureMode mode);
    using cuStreamBeginCaptureToGraph_t      = void (*)();
    using cuStreamBeginCaptureToGraph_ptsz_t = void (*)();
    using cuStreamBeginCapture_ptsz_t        = void (*)();
    using cuStreamCopyAttributes_t           = CUresult(CUDAAPI*)(CUstream dst, CUstream src);
    using cuStreamCopyAttributes_ptsz_t      = void (*)();
    using cuStreamCreate_t                   = CUresult(CUDAAPI*)(CUstream* phStream, unsigned int Flags);
    using cuStreamCreateWithPriority_t       = CUresult(CUDAAPI*)(CUstream* phStream, unsigned int flags, int priority);
    using cuStreamDestroy_v2_t               = CUresult(CUDAAPI*)(CUstream hStream);
    using cuStreamEndCapture_t               = CUresult(CUDAAPI*)(CUstream hStream, CUgraph* phGraph);
    using cuStreamEndCapture_ptsz_t          = void (*)();
    using cuStreamGetAttribute_t = CUresult(CUDAAPI*)(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue* value_out);
    using cuStreamGetAttribute_ptsz_t = void (*)();
    using cuStreamGetCaptureInfo_v3_t =
        CUresult(CUDAAPI*)(CUstream hStream, CUstreamCaptureStatus* captureStatus, cuuint64_t* id);
    using cuStreamGetCaptureInfo_ptsz_t = void (*)();
    using cuStreamGetCtx_v2_t           = CUresult(CUDAAPI*)(CUstream hStream, CUcontext* pctx);
    using cuStreamGetCtx_ptsz_t         = void (*)();
    using cuStreamGetDevice_t           = void (*)();
    using cuStreamGetDevice_ptsz_t      = void (*)();
    using cuStreamGetFlags_t            = CUresult(CUDAAPI*)(CUstream hStream, unsigned int* flags);
    using cuStreamGetFlags_ptsz_t       = void (*)();
    using cuStreamGetGreenCtx_t         = void (*)();
    using cuStreamGetId_t               = void (*)();
    using cuStreamGetId_ptsz_t          = void (*)();
    using cuStreamGetPriority_t         = CUresult(CUDAAPI*)(CUstream hStream, int* priority);
    using cuStreamGetPriority_ptsz_t    = void (*)();
    using cuStreamIsCapturing_t         = CUresult(CUDAAPI*)(CUstream hStream, CUstreamCaptureStatus* captureStatus);
    using cuStreamIsCapturing_ptsz_t    = void (*)();
    using cuStreamQuery_t               = CUresult(CUDAAPI*)(CUstream hStream);
    using cuStreamQuery_ptsz_t          = void (*)();
    using cuStreamSetAttribute_t = CUresult(CUDAAPI*)(CUstream hStream, CUstreamAttrID attr, CUstreamAttrValue const* value);
    using cuStreamSetAttribute_ptsz_t              = void (*)();
    using cuStreamSynchronize_t                    = CUresult(CUDAAPI*)(CUstream hStream);
    using cuStreamSynchronize_ptsz_t               = void (*)();
    using cuStreamUpdateCaptureDependencies_v2_t   = void (*)();
    using cuStreamUpdateCaptureDependencies_ptsz_t = void (*)();
    using cuStreamWaitEvent_t      = CUresult(CUDAAPI*)(CUstream hStream, CUevent hEvent, unsigned int Flags);
    using cuStreamWaitEvent_ptsz_t = void (*)();
    using cuStreamWaitValue32_v2_t = CUresult(CUDAAPI*)(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
    using cuStreamWaitValue32_ptsz_t = void (*)();
    using cuStreamWaitValue64_v2_t = CUresult(CUDAAPI*)(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
    using cuStreamWaitValue64_ptsz_t = void (*)();
    using cuStreamWriteValue32_v2_t = CUresult(CUDAAPI*)(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags);
    using cuStreamWriteValue32_ptsz_t = void (*)();
    using cuStreamWriteValue64_v2_t = CUresult(CUDAAPI*)(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags);
    using cuStreamWriteValue64_ptsz_t = void (*)();
    using cuSurfObjectCreate_t  = CUresult(CUDAAPI*)(CUsurfObject* pSurfObject, const CUDA_RESOURCE_DESC* pResDesc);
    using cuSurfObjectDestroy_t = CUresult(CUDAAPI*)(CUsurfObject surfObject);
    using cuSurfObjectGetResourceDesc_t = CUresult(CUDAAPI*)(CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
    using cuSurfRefGetArray_t           = CUresult(CUDAAPI*)(CUarray* phArray, CUsurfref hSurfRef);
    using cuSurfRefSetArray_t           = CUresult(CUDAAPI*)(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags);
    using cuTensorMapEncodeIm2col_t     = void (*)();
    using cuTensorMapEncodeIm2colWide_t = void (*)();
    using cuTensorMapEncodeTiled_t      = void (*)();
    using cuTensorMapReplaceAddress_t   = void (*)();
    using cuTexObjectCreate_t           = CUresult(CUDAAPI*)(CUtexObject*                   pTexObject,
                                                   const CUDA_RESOURCE_DESC*      pResDesc,
                                                   const CUDA_TEXTURE_DESC*       pTexDesc,
                                                   const CUDA_RESOURCE_VIEW_DESC* pResViewDesc);
    using cuTexObjectDestroy_t          = CUresult(CUDAAPI*)(CUtexObject texObject);
    using cuTexObjectGetResourceDesc_t  = CUresult(CUDAAPI*)(CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
    using cuTexObjectGetResourceViewDesc_t = CUresult(CUDAAPI*)(CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
    using cuTexObjectGetTextureDesc_t = CUresult(CUDAAPI*)(CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
    using cuTexRefCreate_t            = CUresult(CUDAAPI*)(CUtexref* pTexRef);
    using cuTexRefDestroy_t           = CUresult(CUDAAPI*)(CUtexref hTexRef);
    using cuTexRefGetAddress_v2_t     = CUresult(CUDAAPI*)(CUdeviceptr* pdptr, CUtexref hTexRef);
    using cuTexRefGetAddressMode_t    = CUresult(CUDAAPI*)(CUaddress_mode* pam, CUtexref hTexRef, int dim);
    using cuTexRefGetArray_t          = CUresult(CUDAAPI*)(CUarray* phArray, CUtexref hTexRef);
    using cuTexRefGetBorderColor_t    = CUresult(CUDAAPI*)(float* pBorderColor, CUtexref hTexRef);
    using cuTexRefGetFilterMode_t     = CUresult(CUDAAPI*)(CUfilter_mode* pfm, CUtexref hTexRef);
    using cuTexRefGetFlags_t          = CUresult(CUDAAPI*)(unsigned int* pFlags, CUtexref hTexRef);
    using cuTexRefGetFormat_t        = CUresult(CUDAAPI*)(CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
    using cuTexRefGetMaxAnisotropy_t = CUresult(CUDAAPI*)(int* pmaxAniso, CUtexref hTexRef);
    using cuTexRefGetMipmapFilterMode_t = CUresult(CUDAAPI*)(CUfilter_mode* pfm, CUtexref hTexRef);
    using cuTexRefGetMipmapLevelBias_t  = CUresult(CUDAAPI*)(float* pbias, CUtexref hTexRef);
    using cuTexRefGetMipmapLevelClamp_t =
        CUresult(CUDAAPI*)(float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
    using cuTexRefGetMipmappedArray_t = CUresult(CUDAAPI*)(CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
    using cuTexRefSetAddress_v2_t = CUresult(CUDAAPI*)(size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
    using cuTexRefSetAddress2D_v3_t =
        CUresult(CUDAAPI*)(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR* desc, CUdeviceptr dptr, size_t Pitch);
    using cuTexRefSetAddressMode_t = CUresult(CUDAAPI*)(CUtexref hTexRef, int dim, CUaddress_mode am);
    using cuTexRefSetArray_t       = CUresult(CUDAAPI*)(CUtexref hTexRef, CUarray hArray, unsigned int Flags);
    using cuTexRefSetBorderColor_t = CUresult(CUDAAPI*)(CUtexref hTexRef, float* pBorderColor);
    using cuTexRefSetFilterMode_t  = CUresult(CUDAAPI*)(CUtexref hTexRef, CUfilter_mode fm);
    using cuTexRefSetFlags_t       = CUresult(CUDAAPI*)(CUtexref hTexRef, unsigned int Flags);
    using cuTexRefSetFormat_t      = CUresult(CUDAAPI*)(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
    using cuTexRefSetMaxAnisotropy_t    = CUresult(CUDAAPI*)(CUtexref hTexRef, unsigned int maxAniso);
    using cuTexRefSetMipmapFilterMode_t = CUresult(CUDAAPI*)(CUtexref hTexRef, CUfilter_mode fm);
    using cuTexRefSetMipmapLevelBias_t  = CUresult(CUDAAPI*)(CUtexref hTexRef, float bias);
    using cuTexRefSetMipmapLevelClamp_t =
        CUresult(CUDAAPI*)(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
    using cuTexRefSetMipmappedArray_t =
        CUresult(CUDAAPI*)(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags);
    using cuThreadExchangeStreamCaptureMode_t = CUresult(CUDAAPI*)(CUstreamCaptureMode* mode);
    using cuUserObjectCreate_t                = void (*)();
    using cuUserObjectRelease_t               = void (*)();
    using cuUserObjectRetain_t                = void (*)();
    using cuWGLGetDevice_t                    = void (*)();
    using cuWaitExternalSemaphoresAsync_t     = CUresult(CUDAAPI*)(
        CUexternalSemaphore const*                 extSemArray,
        const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* paramsArray,
        unsigned int                               numExtSems,
        CUstream                                   stream);
    using cuWaitExternalSemaphoresAsync_ptsz_t = void (*)();
    using cudbgApiAttach_t                     = void (*)();
    using cudbgApiDetach_t                     = void (*)();
    using cudbgApiInit_t                       = void (*)();
    using cudbgGetAPI_t                        = void (*)();
    using cudbgGetAPIVersion_t                 = void (*)();
    using cudbgMain_t                          = void (*)();
    void* m_library;

public:
    cuArray3DCreate_v2_t                                   cuArray3DCreate;
    cuArray3DGetDescriptor_v2_t                            cuArray3DGetDescriptor;
    cuArrayCreate_v2_t                                     cuArrayCreate;
    cuArrayDestroy_t                                       cuArrayDestroy;
    cuArrayGetDescriptor_v2_t                              cuArrayGetDescriptor;
    cuArrayGetMemoryRequirements_t                         cuArrayGetMemoryRequirements;
    cuArrayGetPlane_t                                      cuArrayGetPlane;
    cuArrayGetSparseProperties_t                           cuArrayGetSparseProperties;
    cuCheckpointProcessCheckpoint_t                        cuCheckpointProcessCheckpoint;
    cuCheckpointProcessGetRestoreThreadId_t                cuCheckpointProcessGetRestoreThreadId;
    cuCheckpointProcessGetState_t                          cuCheckpointProcessGetState;
    cuCheckpointProcessLock_t                              cuCheckpointProcessLock;
    cuCheckpointProcessRestore_t                           cuCheckpointProcessRestore;
    cuCheckpointProcessUnlock_t                            cuCheckpointProcessUnlock;
    cuCoredumpGetAttribute_t                               cuCoredumpGetAttribute;
    cuCoredumpGetAttributeGlobal_t                         cuCoredumpGetAttributeGlobal;
    cuCoredumpSetAttribute_t                               cuCoredumpSetAttribute;
    cuCoredumpSetAttributeGlobal_t                         cuCoredumpSetAttributeGlobal;
    cuCtxAttach_t                                          cuCtxAttach;
    cuCtxCreate_v4_t                                       cuCtxCreate;
    cuCtxDestroy_v2_t                                      cuCtxDestroy;
    cuCtxDetach_t                                          cuCtxDetach;
    cuCtxDisablePeerAccess_t                               cuCtxDisablePeerAccess;
    cuCtxEnablePeerAccess_t                                cuCtxEnablePeerAccess;
    cuCtxFromGreenCtx_t                                    cuCtxFromGreenCtx;
    cuCtxGetApiVersion_t                                   cuCtxGetApiVersion;
    cuCtxGetCacheConfig_t                                  cuCtxGetCacheConfig;
    cuCtxGetCurrent_t                                      cuCtxGetCurrent;
    cuCtxGetDevResource_t                                  cuCtxGetDevResource;
    cuCtxGetDevice_t                                       cuCtxGetDevice;
    cuCtxGetExecAffinity_t                                 cuCtxGetExecAffinity;
    cuCtxGetFlags_t                                        cuCtxGetFlags;
    cuCtxGetId_t                                           cuCtxGetId;
    cuCtxGetLimit_t                                        cuCtxGetLimit;
    cuCtxGetSharedMemConfig_t                              cuCtxGetSharedMemConfig;
    cuCtxGetStreamPriorityRange_t                          cuCtxGetStreamPriorityRange;
    cuCtxPopCurrent_v2_t                                   cuCtxPopCurrent;
    cuCtxPushCurrent_v2_t                                  cuCtxPushCurrent;
    cuCtxRecordEvent_t                                     cuCtxRecordEvent;
    cuCtxResetPersistingL2Cache_t                          cuCtxResetPersistingL2Cache;
    cuCtxSetCacheConfig_t                                  cuCtxSetCacheConfig;
    cuCtxSetCurrent_t                                      cuCtxSetCurrent;
    cuCtxSetFlags_t                                        cuCtxSetFlags;
    cuCtxSetLimit_t                                        cuCtxSetLimit;
    cuCtxSetSharedMemConfig_t                              cuCtxSetSharedMemConfig;
    cuCtxSynchronize_t                                     cuCtxSynchronize;
    cuCtxWaitEvent_t                                       cuCtxWaitEvent;
    cuD3D10CtxCreate_v2_t                                  cuD3D10CtxCreate;
    cuD3D10CtxCreateOnDevice_t                             cuD3D10CtxCreateOnDevice;
    cuD3D10GetDevice_t                                     cuD3D10GetDevice;
    cuD3D10GetDevices_t                                    cuD3D10GetDevices;
    cuD3D10GetDirect3DDevice_t                             cuD3D10GetDirect3DDevice;
    cuD3D10MapResources_t                                  cuD3D10MapResources;
    cuD3D10RegisterResource_t                              cuD3D10RegisterResource;
    cuD3D10ResourceGetMappedArray_t                        cuD3D10ResourceGetMappedArray;
    cuD3D10ResourceGetMappedPitch_v2_t                     cuD3D10ResourceGetMappedPitch;
    cuD3D10ResourceGetMappedPointer_v2_t                   cuD3D10ResourceGetMappedPointer;
    cuD3D10ResourceGetMappedSize_v2_t                      cuD3D10ResourceGetMappedSize;
    cuD3D10ResourceGetSurfaceDimensions_v2_t               cuD3D10ResourceGetSurfaceDimensions;
    cuD3D10ResourceSetMapFlags_t                           cuD3D10ResourceSetMapFlags;
    cuD3D10UnmapResources_t                                cuD3D10UnmapResources;
    cuD3D10UnregisterResource_t                            cuD3D10UnregisterResource;
    cuD3D11CtxCreate_v2_t                                  cuD3D11CtxCreate;
    cuD3D11CtxCreateOnDevice_t                             cuD3D11CtxCreateOnDevice;
    cuD3D11GetDevice_t                                     cuD3D11GetDevice;
    cuD3D11GetDevices_t                                    cuD3D11GetDevices;
    cuD3D11GetDirect3DDevice_t                             cuD3D11GetDirect3DDevice;
    cuD3D9Begin_t                                          cuD3D9Begin;
    cuD3D9CtxCreate_v2_t                                   cuD3D9CtxCreate;
    cuD3D9CtxCreateOnDevice_t                              cuD3D9CtxCreateOnDevice;
    cuD3D9End_t                                            cuD3D9End;
    cuD3D9GetDevice_t                                      cuD3D9GetDevice;
    cuD3D9GetDevices_t                                     cuD3D9GetDevices;
    cuD3D9GetDirect3DDevice_t                              cuD3D9GetDirect3DDevice;
    cuD3D9MapResources_t                                   cuD3D9MapResources;
    cuD3D9MapVertexBuffer_v2_t                             cuD3D9MapVertexBuffer;
    cuD3D9RegisterResource_t                               cuD3D9RegisterResource;
    cuD3D9RegisterVertexBuffer_t                           cuD3D9RegisterVertexBuffer;
    cuD3D9ResourceGetMappedArray_t                         cuD3D9ResourceGetMappedArray;
    cuD3D9ResourceGetMappedPitch_v2_t                      cuD3D9ResourceGetMappedPitch;
    cuD3D9ResourceGetMappedPointer_v2_t                    cuD3D9ResourceGetMappedPointer;
    cuD3D9ResourceGetMappedSize_v2_t                       cuD3D9ResourceGetMappedSize;
    cuD3D9ResourceGetSurfaceDimensions_v2_t                cuD3D9ResourceGetSurfaceDimensions;
    cuD3D9ResourceSetMapFlags_t                            cuD3D9ResourceSetMapFlags;
    cuD3D9UnmapResources_t                                 cuD3D9UnmapResources;
    cuD3D9UnmapVertexBuffer_t                              cuD3D9UnmapVertexBuffer;
    cuD3D9UnregisterResource_t                             cuD3D9UnregisterResource;
    cuD3D9UnregisterVertexBuffer_t                         cuD3D9UnregisterVertexBuffer;
    cuDestroyExternalMemory_t                              cuDestroyExternalMemory;
    cuDestroyExternalSemaphore_t                           cuDestroyExternalSemaphore;
    cuDevResourceGenerateDesc_t                            cuDevResourceGenerateDesc;
    cuDevSmResourceSplitByCount_t                          cuDevSmResourceSplitByCount;
    cuDeviceCanAccessPeer_t                                cuDeviceCanAccessPeer;
    cuDeviceComputeCapability_t                            cuDeviceComputeCapability;
    cuDeviceGet_t                                          cuDeviceGet;
    cuDeviceGetAttribute_t                                 cuDeviceGetAttribute;
    cuDeviceGetByPCIBusId_t                                cuDeviceGetByPCIBusId;
    cuDeviceGetCount_t                                     cuDeviceGetCount;
    cuDeviceGetDefaultMemPool_t                            cuDeviceGetDefaultMemPool;
    cuDeviceGetDevResource_t                               cuDeviceGetDevResource;
    cuDeviceGetExecAffinitySupport_t                       cuDeviceGetExecAffinitySupport;
    cuDeviceGetGraphMemAttribute_t                         cuDeviceGetGraphMemAttribute;
    cuDeviceGetLuid_t                                      cuDeviceGetLuid;
    cuDeviceGetMemPool_t                                   cuDeviceGetMemPool;
    cuDeviceGetName_t                                      cuDeviceGetName;
    cuDeviceGetP2PAttribute_t                              cuDeviceGetP2PAttribute;
    cuDeviceGetPCIBusId_t                                  cuDeviceGetPCIBusId;
    cuDeviceGetProperties_t                                cuDeviceGetProperties;
    cuDeviceGetTexture1DLinearMaxWidth_t                   cuDeviceGetTexture1DLinearMaxWidth;
    cuDeviceGetUuid_v2_t                                   cuDeviceGetUuid;
    cuDeviceGraphMemTrim_t                                 cuDeviceGraphMemTrim;
    cuDevicePrimaryCtxGetState_t                           cuDevicePrimaryCtxGetState;
    cuDevicePrimaryCtxRelease_v2_t                         cuDevicePrimaryCtxRelease;
    cuDevicePrimaryCtxReset_v2_t                           cuDevicePrimaryCtxReset;
    cuDevicePrimaryCtxRetain_t                             cuDevicePrimaryCtxRetain;
    cuDevicePrimaryCtxSetFlags_v2_t                        cuDevicePrimaryCtxSetFlags;
    cuDeviceRegisterAsyncNotification_t                    cuDeviceRegisterAsyncNotification;
    cuDeviceSetGraphMemAttribute_t                         cuDeviceSetGraphMemAttribute;
    cuDeviceSetMemPool_t                                   cuDeviceSetMemPool;
    cuDeviceTotalMem_v2_t                                  cuDeviceTotalMem;
    cuDeviceUnregisterAsyncNotification_t                  cuDeviceUnregisterAsyncNotification;
    cuDriverGetVersion_t                                   cuDriverGetVersion;
    cuEventCreate_t                                        cuEventCreate;
    cuEventDestroy_v2_t                                    cuEventDestroy;
    cuEventElapsedTime_v2_t                                cuEventElapsedTime;
    cuEventQuery_t                                         cuEventQuery;
    cuEventRecord_t                                        cuEventRecord;
    cuEventRecordWithFlags_t                               cuEventRecordWithFlags;
    cuEventRecordWithFlags_ptsz_t                          cuEventRecordWithFlags_ptsz;
    cuEventRecord_ptsz_t                                   cuEventRecord_ptsz;
    cuEventSynchronize_t                                   cuEventSynchronize;
    cuExternalMemoryGetMappedBuffer_t                      cuExternalMemoryGetMappedBuffer;
    cuExternalMemoryGetMappedMipmappedArray_t              cuExternalMemoryGetMappedMipmappedArray;
    cuFlushGPUDirectRDMAWrites_t                           cuFlushGPUDirectRDMAWrites;
    cuFuncGetAttribute_t                                   cuFuncGetAttribute;
    cuFuncGetModule_t                                      cuFuncGetModule;
    cuFuncGetName_t                                        cuFuncGetName;
    cuFuncGetParamInfo_t                                   cuFuncGetParamInfo;
    cuFuncIsLoaded_t                                       cuFuncIsLoaded;
    cuFuncLoad_t                                           cuFuncLoad;
    cuFuncSetAttribute_t                                   cuFuncSetAttribute;
    cuFuncSetBlockShape_t                                  cuFuncSetBlockShape;
    cuFuncSetCacheConfig_t                                 cuFuncSetCacheConfig;
    cuFuncSetSharedMemConfig_t                             cuFuncSetSharedMemConfig;
    cuFuncSetSharedSize_t                                  cuFuncSetSharedSize;
    cuGLCtxCreate_v2_t                                     cuGLCtxCreate;
    cuGLGetDevices_v2_t                                    cuGLGetDevices;
    cuGLInit_t                                             cuGLInit;
    cuGLMapBufferObject_v2_t                               cuGLMapBufferObject;
    cuGLMapBufferObjectAsync_v2_t                          cuGLMapBufferObjectAsync;
    cuGLRegisterBufferObject_t                             cuGLRegisterBufferObject;
    cuGLSetBufferObjectMapFlags_t                          cuGLSetBufferObjectMapFlags;
    cuGLUnmapBufferObject_t                                cuGLUnmapBufferObject;
    cuGLUnmapBufferObjectAsync_t                           cuGLUnmapBufferObjectAsync;
    cuGLUnregisterBufferObject_t                           cuGLUnregisterBufferObject;
    cuGetErrorName_t                                       cuGetErrorName;
    cuGetErrorString_t                                     cuGetErrorString;
    cuGetExportTable_t                                     cuGetExportTable;
    cuGetProcAddress_v2_t                                  cuGetProcAddress;
    cuGraphAddBatchMemOpNode_t                             cuGraphAddBatchMemOpNode;
    cuGraphAddChildGraphNode_t                             cuGraphAddChildGraphNode;
    cuGraphAddDependencies_v2_t                            cuGraphAddDependencies;
    cuGraphAddEmptyNode_t                                  cuGraphAddEmptyNode;
    cuGraphAddEventRecordNode_t                            cuGraphAddEventRecordNode;
    cuGraphAddEventWaitNode_t                              cuGraphAddEventWaitNode;
    cuGraphAddExternalSemaphoresSignalNode_t               cuGraphAddExternalSemaphoresSignalNode;
    cuGraphAddExternalSemaphoresWaitNode_t                 cuGraphAddExternalSemaphoresWaitNode;
    cuGraphAddHostNode_t                                   cuGraphAddHostNode;
    cuGraphAddKernelNode_v2_t                              cuGraphAddKernelNode;
    cuGraphAddMemAllocNode_t                               cuGraphAddMemAllocNode;
    cuGraphAddMemFreeNode_t                                cuGraphAddMemFreeNode;
    cuGraphAddMemcpyNode_t                                 cuGraphAddMemcpyNode;
    cuGraphAddMemsetNode_t                                 cuGraphAddMemsetNode;
    cuGraphAddNode_v2_t                                    cuGraphAddNode;
    cuGraphBatchMemOpNodeGetParams_t                       cuGraphBatchMemOpNodeGetParams;
    cuGraphBatchMemOpNodeSetParams_t                       cuGraphBatchMemOpNodeSetParams;
    cuGraphChildGraphNodeGetGraph_t                        cuGraphChildGraphNodeGetGraph;
    cuGraphClone_t                                         cuGraphClone;
    cuGraphConditionalHandleCreate_t                       cuGraphConditionalHandleCreate;
    cuGraphCreate_t                                        cuGraphCreate;
    cuGraphDebugDotPrint_t                                 cuGraphDebugDotPrint;
    cuGraphDestroy_t                                       cuGraphDestroy;
    cuGraphDestroyNode_t                                   cuGraphDestroyNode;
    cuGraphEventRecordNodeGetEvent_t                       cuGraphEventRecordNodeGetEvent;
    cuGraphEventRecordNodeSetEvent_t                       cuGraphEventRecordNodeSetEvent;
    cuGraphEventWaitNodeGetEvent_t                         cuGraphEventWaitNodeGetEvent;
    cuGraphEventWaitNodeSetEvent_t                         cuGraphEventWaitNodeSetEvent;
    cuGraphExecBatchMemOpNodeSetParams_t                   cuGraphExecBatchMemOpNodeSetParams;
    cuGraphExecChildGraphNodeSetParams_t                   cuGraphExecChildGraphNodeSetParams;
    cuGraphExecDestroy_t                                   cuGraphExecDestroy;
    cuGraphExecEventRecordNodeSetEvent_t                   cuGraphExecEventRecordNodeSetEvent;
    cuGraphExecEventWaitNodeSetEvent_t                     cuGraphExecEventWaitNodeSetEvent;
    cuGraphExecExternalSemaphoresSignalNodeSetParams_t     cuGraphExecExternalSemaphoresSignalNodeSetParams;
    cuGraphExecExternalSemaphoresWaitNodeSetParams_t       cuGraphExecExternalSemaphoresWaitNodeSetParams;
    cuGraphExecGetFlags_t                                  cuGraphExecGetFlags;
    cuGraphExecHostNodeSetParams_t                         cuGraphExecHostNodeSetParams;
    cuGraphExecKernelNodeSetParams_v2_t                    cuGraphExecKernelNodeSetParams;
    cuGraphExecMemcpyNodeSetParams_t                       cuGraphExecMemcpyNodeSetParams;
    cuGraphExecMemsetNodeSetParams_t                       cuGraphExecMemsetNodeSetParams;
    cuGraphExecNodeSetParams_t                             cuGraphExecNodeSetParams;
    cuGraphExecUpdate_v2_t                                 cuGraphExecUpdate;
    cuGraphExternalSemaphoresSignalNodeGetParams_t         cuGraphExternalSemaphoresSignalNodeGetParams;
    cuGraphExternalSemaphoresSignalNodeSetParams_t         cuGraphExternalSemaphoresSignalNodeSetParams;
    cuGraphExternalSemaphoresWaitNodeGetParams_t           cuGraphExternalSemaphoresWaitNodeGetParams;
    cuGraphExternalSemaphoresWaitNodeSetParams_t           cuGraphExternalSemaphoresWaitNodeSetParams;
    cuGraphGetEdges_v2_t                                   cuGraphGetEdges;
    cuGraphGetNodes_t                                      cuGraphGetNodes;
    cuGraphGetRootNodes_t                                  cuGraphGetRootNodes;
    cuGraphHostNodeGetParams_t                             cuGraphHostNodeGetParams;
    cuGraphHostNodeSetParams_t                             cuGraphHostNodeSetParams;
    cuGraphInstantiate_v2_t                                cuGraphInstantiate;
    cuGraphInstantiateWithFlags_t                          cuGraphInstantiateWithFlags;
    cuGraphInstantiateWithParams_t                         cuGraphInstantiateWithParams;
    cuGraphInstantiateWithParams_ptsz_t                    cuGraphInstantiateWithParams_ptsz;
    cuGraphKernelNodeCopyAttributes_t                      cuGraphKernelNodeCopyAttributes;
    cuGraphKernelNodeGetAttribute_t                        cuGraphKernelNodeGetAttribute;
    cuGraphKernelNodeGetParams_v2_t                        cuGraphKernelNodeGetParams;
    cuGraphKernelNodeSetAttribute_t                        cuGraphKernelNodeSetAttribute;
    cuGraphKernelNodeSetParams_v2_t                        cuGraphKernelNodeSetParams;
    cuGraphLaunch_t                                        cuGraphLaunch;
    cuGraphLaunch_ptsz_t                                   cuGraphLaunch_ptsz;
    cuGraphMemAllocNodeGetParams_t                         cuGraphMemAllocNodeGetParams;
    cuGraphMemFreeNodeGetParams_t                          cuGraphMemFreeNodeGetParams;
    cuGraphMemcpyNodeGetParams_t                           cuGraphMemcpyNodeGetParams;
    cuGraphMemcpyNodeSetParams_t                           cuGraphMemcpyNodeSetParams;
    cuGraphMemsetNodeGetParams_t                           cuGraphMemsetNodeGetParams;
    cuGraphMemsetNodeSetParams_t                           cuGraphMemsetNodeSetParams;
    cuGraphNodeFindInClone_t                               cuGraphNodeFindInClone;
    cuGraphNodeGetDependencies_v2_t                        cuGraphNodeGetDependencies;
    cuGraphNodeGetDependentNodes_v2_t                      cuGraphNodeGetDependentNodes;
    cuGraphNodeGetEnabled_t                                cuGraphNodeGetEnabled;
    cuGraphNodeGetType_t                                   cuGraphNodeGetType;
    cuGraphNodeSetEnabled_t                                cuGraphNodeSetEnabled;
    cuGraphNodeSetParams_t                                 cuGraphNodeSetParams;
    cuGraphReleaseUserObject_t                             cuGraphReleaseUserObject;
    cuGraphRemoveDependencies_v2_t                         cuGraphRemoveDependencies;
    cuGraphRetainUserObject_t                              cuGraphRetainUserObject;
    cuGraphUpload_t                                        cuGraphUpload;
    cuGraphUpload_ptsz_t                                   cuGraphUpload_ptsz;
    cuGraphicsD3D10RegisterResource_t                      cuGraphicsD3D10RegisterResource;
    cuGraphicsD3D11RegisterResource_t                      cuGraphicsD3D11RegisterResource;
    cuGraphicsD3D9RegisterResource_t                       cuGraphicsD3D9RegisterResource;
    cuGraphicsGLRegisterBuffer_t                           cuGraphicsGLRegisterBuffer;
    cuGraphicsGLRegisterImage_t                            cuGraphicsGLRegisterImage;
    cuGraphicsMapResources_t                               cuGraphicsMapResources;
    cuGraphicsMapResources_ptsz_t                          cuGraphicsMapResources_ptsz;
    cuGraphicsResourceGetMappedMipmappedArray_t            cuGraphicsResourceGetMappedMipmappedArray;
    cuGraphicsResourceGetMappedPointer_v2_t                cuGraphicsResourceGetMappedPointer;
    cuGraphicsResourceSetMapFlags_v2_t                     cuGraphicsResourceSetMapFlags;
    cuGraphicsSubResourceGetMappedArray_t                  cuGraphicsSubResourceGetMappedArray;
    cuGraphicsUnmapResources_t                             cuGraphicsUnmapResources;
    cuGraphicsUnmapResources_ptsz_t                        cuGraphicsUnmapResources_ptsz;
    cuGraphicsUnregisterResource_t                         cuGraphicsUnregisterResource;
    cuGreenCtxCreate_t                                     cuGreenCtxCreate;
    cuGreenCtxDestroy_t                                    cuGreenCtxDestroy;
    cuGreenCtxGetDevResource_t                             cuGreenCtxGetDevResource;
    cuGreenCtxRecordEvent_t                                cuGreenCtxRecordEvent;
    cuGreenCtxStreamCreate_t                               cuGreenCtxStreamCreate;
    cuGreenCtxWaitEvent_t                                  cuGreenCtxWaitEvent;
    cuImportExternalMemory_t                               cuImportExternalMemory;
    cuImportExternalSemaphore_t                            cuImportExternalSemaphore;
    cuInit_t                                               cuInit;
    cuIpcCloseMemHandle_t                                  cuIpcCloseMemHandle;
    cuIpcGetEventHandle_t                                  cuIpcGetEventHandle;
    cuIpcGetMemHandle_t                                    cuIpcGetMemHandle;
    cuIpcOpenEventHandle_t                                 cuIpcOpenEventHandle;
    cuIpcOpenMemHandle_v2_t                                cuIpcOpenMemHandle;
    cuKernelGetAttribute_t                                 cuKernelGetAttribute;
    cuKernelGetFunction_t                                  cuKernelGetFunction;
    cuKernelGetLibrary_t                                   cuKernelGetLibrary;
    cuKernelGetName_t                                      cuKernelGetName;
    cuKernelGetParamInfo_t                                 cuKernelGetParamInfo;
    cuKernelSetAttribute_t                                 cuKernelSetAttribute;
    cuKernelSetCacheConfig_t                               cuKernelSetCacheConfig;
    cuLaunch_t                                             cuLaunch;
    cuLaunchCooperativeKernel_t                            cuLaunchCooperativeKernel;
    cuLaunchCooperativeKernelMultiDevice_t                 cuLaunchCooperativeKernelMultiDevice;
    cuLaunchCooperativeKernel_ptsz_t                       cuLaunchCooperativeKernel_ptsz;
    cuLaunchGrid_t                                         cuLaunchGrid;
    cuLaunchGridAsync_t                                    cuLaunchGridAsync;
    cuLaunchHostFunc_t                                     cuLaunchHostFunc;
    cuLaunchHostFunc_ptsz_t                                cuLaunchHostFunc_ptsz;
    cuLaunchKernel_t                                       cuLaunchKernel;
    cuLaunchKernelEx_t                                     cuLaunchKernelEx;
    cuLaunchKernelEx_ptsz_t                                cuLaunchKernelEx_ptsz;
    cuLaunchKernel_ptsz_t                                  cuLaunchKernel_ptsz;
    cuLibraryEnumerateKernels_t                            cuLibraryEnumerateKernels;
    cuLibraryGetGlobal_t                                   cuLibraryGetGlobal;
    cuLibraryGetKernel_t                                   cuLibraryGetKernel;
    cuLibraryGetKernelCount_t                              cuLibraryGetKernelCount;
    cuLibraryGetManaged_t                                  cuLibraryGetManaged;
    cuLibraryGetModule_t                                   cuLibraryGetModule;
    cuLibraryGetUnifiedFunction_t                          cuLibraryGetUnifiedFunction;
    cuLibraryLoadData_t                                    cuLibraryLoadData;
    cuLibraryLoadFromFile_t                                cuLibraryLoadFromFile;
    cuLibraryUnload_t                                      cuLibraryUnload;
    cuLinkAddData_v2_t                                     cuLinkAddData;
    cuLinkAddFile_v2_t                                     cuLinkAddFile;
    cuLinkComplete_t                                       cuLinkComplete;
    cuLinkCreate_v2_t                                      cuLinkCreate;
    cuLinkDestroy_t                                        cuLinkDestroy;
    cuMemAddressFree_t                                     cuMemAddressFree;
    cuMemAddressReserve_t                                  cuMemAddressReserve;
    cuMemAdvise_v2_t                                       cuMemAdvise;
    cuMemAlloc_v2_t                                        cuMemAlloc;
    cuMemAllocAsync_t                                      cuMemAllocAsync;
    cuMemAllocAsync_ptsz_t                                 cuMemAllocAsync_ptsz;
    cuMemAllocFromPoolAsync_t                              cuMemAllocFromPoolAsync;
    cuMemAllocFromPoolAsync_ptsz_t                         cuMemAllocFromPoolAsync_ptsz;
    cuMemAllocHost_v2_t                                    cuMemAllocHost;
    cuMemAllocManaged_t                                    cuMemAllocManaged;
    cuMemAllocPitch_v2_t                                   cuMemAllocPitch;
    cuMemBatchDecompressAsync_t                            cuMemBatchDecompressAsync;
    cuMemBatchDecompressAsync_ptsz_t                       cuMemBatchDecompressAsync_ptsz;
    cuMemCreate_t                                          cuMemCreate;
    cuMemExportToShareableHandle_t                         cuMemExportToShareableHandle;
    cuMemFree_v2_t                                         cuMemFree;
    cuMemFreeAsync_t                                       cuMemFreeAsync;
    cuMemFreeAsync_ptsz_t                                  cuMemFreeAsync_ptsz;
    cuMemFreeHost_t                                        cuMemFreeHost;
    cuMemGetAccess_t                                       cuMemGetAccess;
    cuMemGetAddressRange_v2_t                              cuMemGetAddressRange;
    cuMemGetAllocationGranularity_t                        cuMemGetAllocationGranularity;
    cuMemGetAllocationPropertiesFromHandle_t               cuMemGetAllocationPropertiesFromHandle;
    cuMemGetHandleForAddressRange_t                        cuMemGetHandleForAddressRange;
    cuMemGetInfo_v2_t                                      cuMemGetInfo;
    cuMemHostAlloc_t                                       cuMemHostAlloc;
    cuMemHostGetDevicePointer_v2_t                         cuMemHostGetDevicePointer;
    cuMemHostGetFlags_t                                    cuMemHostGetFlags;
    cuMemHostRegister_v2_t                                 cuMemHostRegister;
    cuMemHostUnregister_t                                  cuMemHostUnregister;
    cuMemImportFromShareableHandle_t                       cuMemImportFromShareableHandle;
    cuMemMap_t                                             cuMemMap;
    cuMemMapArrayAsync_t                                   cuMemMapArrayAsync;
    cuMemMapArrayAsync_ptsz_t                              cuMemMapArrayAsync_ptsz;
    cuMemPoolCreate_t                                      cuMemPoolCreate;
    cuMemPoolDestroy_t                                     cuMemPoolDestroy;
    cuMemPoolExportPointer_t                               cuMemPoolExportPointer;
    cuMemPoolExportToShareableHandle_t                     cuMemPoolExportToShareableHandle;
    cuMemPoolGetAccess_t                                   cuMemPoolGetAccess;
    cuMemPoolGetAttribute_t                                cuMemPoolGetAttribute;
    cuMemPoolImportFromShareableHandle_t                   cuMemPoolImportFromShareableHandle;
    cuMemPoolImportPointer_t                               cuMemPoolImportPointer;
    cuMemPoolSetAccess_t                                   cuMemPoolSetAccess;
    cuMemPoolSetAttribute_t                                cuMemPoolSetAttribute;
    cuMemPoolTrimTo_t                                      cuMemPoolTrimTo;
    cuMemPrefetchAsync_v2_t                                cuMemPrefetchAsync;
    cuMemPrefetchAsync_ptsz_t                              cuMemPrefetchAsync_ptsz;
    cuMemRangeGetAttribute_t                               cuMemRangeGetAttribute;
    cuMemRangeGetAttributes_t                              cuMemRangeGetAttributes;
    cuMemRelease_t                                         cuMemRelease;
    cuMemRetainAllocationHandle_t                          cuMemRetainAllocationHandle;
    cuMemSetAccess_t                                       cuMemSetAccess;
    cuMemUnmap_t                                           cuMemUnmap;
    cuMemcpy_t                                             cuMemcpy;
    cuMemcpy2D_v2_t                                        cuMemcpy2D;
    cuMemcpy2DAsync_v2_t                                   cuMemcpy2DAsync;
    cuMemcpy2DUnaligned_v2_t                               cuMemcpy2DUnaligned;
    cuMemcpy3D_v2_t                                        cuMemcpy3D;
    cuMemcpy3DAsync_v2_t                                   cuMemcpy3DAsync;
    cuMemcpy3DBatchAsync_t                                 cuMemcpy3DBatchAsync;
    cuMemcpy3DBatchAsync_ptsz_t                            cuMemcpy3DBatchAsync_ptsz;
    cuMemcpy3DPeer_t                                       cuMemcpy3DPeer;
    cuMemcpy3DPeerAsync_t                                  cuMemcpy3DPeerAsync;
    cuMemcpy3DPeerAsync_ptsz_t                             cuMemcpy3DPeerAsync_ptsz;
    cuMemcpy3DPeer_ptds_t                                  cuMemcpy3DPeer_ptds;
    cuMemcpyAsync_t                                        cuMemcpyAsync;
    cuMemcpyAsync_ptsz_t                                   cuMemcpyAsync_ptsz;
    cuMemcpyAtoA_v2_t                                      cuMemcpyAtoA;
    cuMemcpyAtoD_v2_t                                      cuMemcpyAtoD;
    cuMemcpyAtoH_v2_t                                      cuMemcpyAtoH;
    cuMemcpyAtoHAsync_v2_t                                 cuMemcpyAtoHAsync;
    cuMemcpyBatchAsync_t                                   cuMemcpyBatchAsync;
    cuMemcpyBatchAsync_ptsz_t                              cuMemcpyBatchAsync_ptsz;
    cuMemcpyDtoA_v2_t                                      cuMemcpyDtoA;
    cuMemcpyDtoD_v2_t                                      cuMemcpyDtoD;
    cuMemcpyDtoDAsync_v2_t                                 cuMemcpyDtoDAsync;
    cuMemcpyDtoH_v2_t                                      cuMemcpyDtoH;
    cuMemcpyDtoHAsync_v2_t                                 cuMemcpyDtoHAsync;
    cuMemcpyHtoA_v2_t                                      cuMemcpyHtoA;
    cuMemcpyHtoAAsync_v2_t                                 cuMemcpyHtoAAsync;
    cuMemcpyHtoD_v2_t                                      cuMemcpyHtoD;
    cuMemcpyHtoDAsync_v2_t                                 cuMemcpyHtoDAsync;
    cuMemcpyPeer_t                                         cuMemcpyPeer;
    cuMemcpyPeerAsync_t                                    cuMemcpyPeerAsync;
    cuMemcpyPeerAsync_ptsz_t                               cuMemcpyPeerAsync_ptsz;
    cuMemcpyPeer_ptds_t                                    cuMemcpyPeer_ptds;
    cuMemcpy_ptds_t                                        cuMemcpy_ptds;
    cuMemsetD16_v2_t                                       cuMemsetD16;
    cuMemsetD16Async_t                                     cuMemsetD16Async;
    cuMemsetD16Async_ptsz_t                                cuMemsetD16Async_ptsz;
    cuMemsetD2D16_v2_t                                     cuMemsetD2D16;
    cuMemsetD2D16Async_t                                   cuMemsetD2D16Async;
    cuMemsetD2D16Async_ptsz_t                              cuMemsetD2D16Async_ptsz;
    cuMemsetD2D32_v2_t                                     cuMemsetD2D32;
    cuMemsetD2D32Async_t                                   cuMemsetD2D32Async;
    cuMemsetD2D32Async_ptsz_t                              cuMemsetD2D32Async_ptsz;
    cuMemsetD2D8_v2_t                                      cuMemsetD2D8;
    cuMemsetD2D8Async_t                                    cuMemsetD2D8Async;
    cuMemsetD2D8Async_ptsz_t                               cuMemsetD2D8Async_ptsz;
    cuMemsetD32_v2_t                                       cuMemsetD32;
    cuMemsetD32Async_t                                     cuMemsetD32Async;
    cuMemsetD32Async_ptsz_t                                cuMemsetD32Async_ptsz;
    cuMemsetD8_v2_t                                        cuMemsetD8;
    cuMemsetD8Async_t                                      cuMemsetD8Async;
    cuMemsetD8Async_ptsz_t                                 cuMemsetD8Async_ptsz;
    cuMipmappedArrayCreate_t                               cuMipmappedArrayCreate;
    cuMipmappedArrayDestroy_t                              cuMipmappedArrayDestroy;
    cuMipmappedArrayGetLevel_t                             cuMipmappedArrayGetLevel;
    cuMipmappedArrayGetMemoryRequirements_t                cuMipmappedArrayGetMemoryRequirements;
    cuMipmappedArrayGetSparseProperties_t                  cuMipmappedArrayGetSparseProperties;
    cuModuleEnumerateFunctions_t                           cuModuleEnumerateFunctions;
    cuModuleGetFunction_t                                  cuModuleGetFunction;
    cuModuleGetFunctionCount_t                             cuModuleGetFunctionCount;
    cuModuleGetGlobal_v2_t                                 cuModuleGetGlobal;
    cuModuleGetLoadingMode_t                               cuModuleGetLoadingMode;
    cuModuleGetSurfRef_t                                   cuModuleGetSurfRef;
    cuModuleGetTexRef_t                                    cuModuleGetTexRef;
    cuModuleLoad_t                                         cuModuleLoad;
    cuModuleLoadData_t                                     cuModuleLoadData;
    cuModuleLoadDataEx_t                                   cuModuleLoadDataEx;
    cuModuleLoadFatBinary_t                                cuModuleLoadFatBinary;
    cuModuleUnload_t                                       cuModuleUnload;
    cuMulticastAddDevice_t                                 cuMulticastAddDevice;
    cuMulticastBindAddr_t                                  cuMulticastBindAddr;
    cuMulticastBindMem_t                                   cuMulticastBindMem;
    cuMulticastCreate_t                                    cuMulticastCreate;
    cuMulticastGetGranularity_t                            cuMulticastGetGranularity;
    cuMulticastUnbind_t                                    cuMulticastUnbind;
    cuOccupancyAvailableDynamicSMemPerBlock_t              cuOccupancyAvailableDynamicSMemPerBlock;
    cuOccupancyMaxActiveBlocksPerMultiprocessor_t          cuOccupancyMaxActiveBlocksPerMultiprocessor;
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_t cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    cuOccupancyMaxActiveClusters_t                         cuOccupancyMaxActiveClusters;
    cuOccupancyMaxPotentialBlockSize_t                     cuOccupancyMaxPotentialBlockSize;
    cuOccupancyMaxPotentialBlockSizeWithFlags_t            cuOccupancyMaxPotentialBlockSizeWithFlags;
    cuOccupancyMaxPotentialClusterSize_t                   cuOccupancyMaxPotentialClusterSize;
    cuParamSetSize_t                                       cuParamSetSize;
    cuParamSetTexRef_t                                     cuParamSetTexRef;
    cuParamSetf_t                                          cuParamSetf;
    cuParamSeti_t                                          cuParamSeti;
    cuParamSetv_t                                          cuParamSetv;
    cuPointerGetAttribute_t                                cuPointerGetAttribute;
    cuPointerGetAttributes_t                               cuPointerGetAttributes;
    cuPointerSetAttribute_t                                cuPointerSetAttribute;
    cuProfilerInitialize_t                                 cuProfilerInitialize;
    cuProfilerStart_t                                      cuProfilerStart;
    cuProfilerStop_t                                       cuProfilerStop;
    cuSignalExternalSemaphoresAsync_t                      cuSignalExternalSemaphoresAsync;
    cuSignalExternalSemaphoresAsync_ptsz_t                 cuSignalExternalSemaphoresAsync_ptsz;
    cuStreamAddCallback_t                                  cuStreamAddCallback;
    cuStreamAddCallback_ptsz_t                             cuStreamAddCallback_ptsz;
    cuStreamAttachMemAsync_t                               cuStreamAttachMemAsync;
    cuStreamAttachMemAsync_ptsz_t                          cuStreamAttachMemAsync_ptsz;
    cuStreamBatchMemOp_v2_t                                cuStreamBatchMemOp;
    cuStreamBatchMemOp_ptsz_t                              cuStreamBatchMemOp_ptsz;
    cuStreamBeginCapture_v2_t                              cuStreamBeginCapture;
    cuStreamBeginCaptureToGraph_t                          cuStreamBeginCaptureToGraph;
    cuStreamBeginCaptureToGraph_ptsz_t                     cuStreamBeginCaptureToGraph_ptsz;
    cuStreamBeginCapture_ptsz_t                            cuStreamBeginCapture_ptsz;
    cuStreamCopyAttributes_t                               cuStreamCopyAttributes;
    cuStreamCopyAttributes_ptsz_t                          cuStreamCopyAttributes_ptsz;
    cuStreamCreate_t                                       cuStreamCreate;
    cuStreamCreateWithPriority_t                           cuStreamCreateWithPriority;
    cuStreamDestroy_v2_t                                   cuStreamDestroy;
    cuStreamEndCapture_t                                   cuStreamEndCapture;
    cuStreamEndCapture_ptsz_t                              cuStreamEndCapture_ptsz;
    cuStreamGetAttribute_t                                 cuStreamGetAttribute;
    cuStreamGetAttribute_ptsz_t                            cuStreamGetAttribute_ptsz;
    cuStreamGetCaptureInfo_v3_t                            cuStreamGetCaptureInfo;
    cuStreamGetCaptureInfo_ptsz_t                          cuStreamGetCaptureInfo_ptsz;
    cuStreamGetCtx_v2_t                                    cuStreamGetCtx;
    cuStreamGetCtx_ptsz_t                                  cuStreamGetCtx_ptsz;
    cuStreamGetDevice_t                                    cuStreamGetDevice;
    cuStreamGetDevice_ptsz_t                               cuStreamGetDevice_ptsz;
    cuStreamGetFlags_t                                     cuStreamGetFlags;
    cuStreamGetFlags_ptsz_t                                cuStreamGetFlags_ptsz;
    cuStreamGetGreenCtx_t                                  cuStreamGetGreenCtx;
    cuStreamGetId_t                                        cuStreamGetId;
    cuStreamGetId_ptsz_t                                   cuStreamGetId_ptsz;
    cuStreamGetPriority_t                                  cuStreamGetPriority;
    cuStreamGetPriority_ptsz_t                             cuStreamGetPriority_ptsz;
    cuStreamIsCapturing_t                                  cuStreamIsCapturing;
    cuStreamIsCapturing_ptsz_t                             cuStreamIsCapturing_ptsz;
    cuStreamQuery_t                                        cuStreamQuery;
    cuStreamQuery_ptsz_t                                   cuStreamQuery_ptsz;
    cuStreamSetAttribute_t                                 cuStreamSetAttribute;
    cuStreamSetAttribute_ptsz_t                            cuStreamSetAttribute_ptsz;
    cuStreamSynchronize_t                                  cuStreamSynchronize;
    cuStreamSynchronize_ptsz_t                             cuStreamSynchronize_ptsz;
    cuStreamUpdateCaptureDependencies_v2_t                 cuStreamUpdateCaptureDependencies;
    cuStreamUpdateCaptureDependencies_ptsz_t               cuStreamUpdateCaptureDependencies_ptsz;
    cuStreamWaitEvent_t                                    cuStreamWaitEvent;
    cuStreamWaitEvent_ptsz_t                               cuStreamWaitEvent_ptsz;
    cuStreamWaitValue32_v2_t                               cuStreamWaitValue32;
    cuStreamWaitValue32_ptsz_t                             cuStreamWaitValue32_ptsz;
    cuStreamWaitValue64_v2_t                               cuStreamWaitValue64;
    cuStreamWaitValue64_ptsz_t                             cuStreamWaitValue64_ptsz;
    cuStreamWriteValue32_v2_t                              cuStreamWriteValue32;
    cuStreamWriteValue32_ptsz_t                            cuStreamWriteValue32_ptsz;
    cuStreamWriteValue64_v2_t                              cuStreamWriteValue64;
    cuStreamWriteValue64_ptsz_t                            cuStreamWriteValue64_ptsz;
    cuSurfObjectCreate_t                                   cuSurfObjectCreate;
    cuSurfObjectDestroy_t                                  cuSurfObjectDestroy;
    cuSurfObjectGetResourceDesc_t                          cuSurfObjectGetResourceDesc;
    cuSurfRefGetArray_t                                    cuSurfRefGetArray;
    cuSurfRefSetArray_t                                    cuSurfRefSetArray;
    cuTensorMapEncodeIm2col_t                              cuTensorMapEncodeIm2col;
    cuTensorMapEncodeIm2colWide_t                          cuTensorMapEncodeIm2colWide;
    cuTensorMapEncodeTiled_t                               cuTensorMapEncodeTiled;
    cuTensorMapReplaceAddress_t                            cuTensorMapReplaceAddress;
    cuTexObjectCreate_t                                    cuTexObjectCreate;
    cuTexObjectDestroy_t                                   cuTexObjectDestroy;
    cuTexObjectGetResourceDesc_t                           cuTexObjectGetResourceDesc;
    cuTexObjectGetResourceViewDesc_t                       cuTexObjectGetResourceViewDesc;
    cuTexObjectGetTextureDesc_t                            cuTexObjectGetTextureDesc;
    cuTexRefCreate_t                                       cuTexRefCreate;
    cuTexRefDestroy_t                                      cuTexRefDestroy;
    cuTexRefGetAddress_v2_t                                cuTexRefGetAddress;
    cuTexRefGetAddressMode_t                               cuTexRefGetAddressMode;
    cuTexRefGetArray_t                                     cuTexRefGetArray;
    cuTexRefGetBorderColor_t                               cuTexRefGetBorderColor;
    cuTexRefGetFilterMode_t                                cuTexRefGetFilterMode;
    cuTexRefGetFlags_t                                     cuTexRefGetFlags;
    cuTexRefGetFormat_t                                    cuTexRefGetFormat;
    cuTexRefGetMaxAnisotropy_t                             cuTexRefGetMaxAnisotropy;
    cuTexRefGetMipmapFilterMode_t                          cuTexRefGetMipmapFilterMode;
    cuTexRefGetMipmapLevelBias_t                           cuTexRefGetMipmapLevelBias;
    cuTexRefGetMipmapLevelClamp_t                          cuTexRefGetMipmapLevelClamp;
    cuTexRefGetMipmappedArray_t                            cuTexRefGetMipmappedArray;
    cuTexRefSetAddress_v2_t                                cuTexRefSetAddress;
    cuTexRefSetAddress2D_v3_t                              cuTexRefSetAddress2D;
    cuTexRefSetAddressMode_t                               cuTexRefSetAddressMode;
    cuTexRefSetArray_t                                     cuTexRefSetArray;
    cuTexRefSetBorderColor_t                               cuTexRefSetBorderColor;
    cuTexRefSetFilterMode_t                                cuTexRefSetFilterMode;
    cuTexRefSetFlags_t                                     cuTexRefSetFlags;
    cuTexRefSetFormat_t                                    cuTexRefSetFormat;
    cuTexRefSetMaxAnisotropy_t                             cuTexRefSetMaxAnisotropy;
    cuTexRefSetMipmapFilterMode_t                          cuTexRefSetMipmapFilterMode;
    cuTexRefSetMipmapLevelBias_t                           cuTexRefSetMipmapLevelBias;
    cuTexRefSetMipmapLevelClamp_t                          cuTexRefSetMipmapLevelClamp;
    cuTexRefSetMipmappedArray_t                            cuTexRefSetMipmappedArray;
    cuThreadExchangeStreamCaptureMode_t                    cuThreadExchangeStreamCaptureMode;
    cuUserObjectCreate_t                                   cuUserObjectCreate;
    cuUserObjectRelease_t                                  cuUserObjectRelease;
    cuUserObjectRetain_t                                   cuUserObjectRetain;
    cuWGLGetDevice_t                                       cuWGLGetDevice;
    cuWaitExternalSemaphoresAsync_t                        cuWaitExternalSemaphoresAsync;
    cuWaitExternalSemaphoresAsync_ptsz_t                   cuWaitExternalSemaphoresAsync_ptsz;
    cudbgApiAttach_t                                       cudbgApiAttach;
    cudbgApiDetach_t                                       cudbgApiDetach;
    cudbgApiInit_t                                         cudbgApiInit;
    cudbgGetAPI_t                                          cudbgGetAPI;
    cudbgGetAPIVersion_t                                   cudbgGetAPIVersion;
    cudbgMain_t                                            cudbgMain;
};

bool loadNvcudaFunctions(dmt::os::LibraryLoader const& loader, NvcudaLibraryFunctions* funcList);
