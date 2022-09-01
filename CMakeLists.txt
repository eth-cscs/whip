# TODO: License and copyright

cmake_minimum_required(VERSION 3.10) # TODO: What do we actually require?

project(hwh VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(HWH_TYPE "CUDA" CACHE STRING "${description}")
set_property(CACHE HWH_TYPE PROPERTY TYPE STRING)
set_property(CACHE HWH_TYPE PROPERTY HELPSTRING "The backend to use...")
set(hwh_valid_types CUDA HIP)
set_property(CACHE HWH_TYPE PROPERTY STRINGS "${hwh_valid_types}")

if(NOT HWH_TYPE IN_LIST hwh_valid_types)
  set(error_message "hwh requires HWH_TYPE to be set to one of (case sensitive):\n")
  for(hwh_valid_type IN hwh_valid_types)
    set(error_message "${error_message}  ${hwh_valid_type}\n")
  endfor()
  set(error_message "${error_message}HWH_TYPE is currently set to \"${HWH_TYPE}\".")
  message(FATAL_ERROR "${error_message}")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(hwh_helpers)

add_library(hwh INTERFACE)
target_include_directories(
  hwh INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(hwh INTERFACE cxx_std_17) # TODO: Not necessarily required anymore?
target_compile_definitions(hwh INTERFACE "HWH_${HWH_TYPE}")

if(HWH_TYPE STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(hwh INTERFACE CUDA::cudart)
elseif(HWH_TYPE STREQUAL "HIP")
  find_package(hip REQUIRED)
  target_link_libraries(hwh INTERFACE hip::host)
endif()

set(hwh_types_section "// Types")
hwh_add_type(hwh_types_section device_prop DeviceProp_t CUDA_OVERRIDE DeviceProp)
hwh_add_type(hwh_types_section error_t Error_t)
hwh_add_type(hwh_types_section event_t Event_t)
hwh_add_type(hwh_types_section stream_t Stream_t)
hwh_add_type(hwh_types_section stream_callback_t StreamCallback_t)
hwh_add_type(hwh_types_section memcpy_kind MemcpyKind)

set(hwh_constants_section "// Constants")
hwh_add_constant(hwh_constants_section event_disable_timing EventDisableTiming)
hwh_add_constant(hwh_constants_section memcpy_device_to_device MemcpyDeviceToDevice TYPE memcpy_kind)
hwh_add_constant(hwh_constants_section memcpy_device_to_host MemcpyDeviceToHost TYPE memcpy_kind)
hwh_add_constant(hwh_constants_section memcpy_host_to_device MemcpyHostToDevice TYPE memcpy_kind)
hwh_add_constant(hwh_constants_section stream_non_blocking StreamNonBlocking)

set(hwh_errors_section "// Errors")
hwh_add_error(hwh_errors_section success Success)
hwh_add_error(hwh_errors_section error_not_ready ErrorNotReady)

set(hwh_functions_section "// Functions")
hwh_add_function(hwh_functions_section check_last_error "" GetLastError)
hwh_add_function(hwh_functions_section device_synchronize "" DeviceSynchronize)
hwh_add_function(hwh_functions_section event_create "event_t* event" EventCreate)
hwh_add_function(hwh_functions_section event_create_with_flags "event_t* event;unsigned flags" EventCreateWithFlags)
hwh_add_function(hwh_functions_section event_destroy "event_t event" EventDestroy)
hwh_add_function(hwh_functions_section event_query "event_t event" EventQuery NOEXCEPT)
hwh_add_function(hwh_functions_section event_record "event_t event;stream_t stream" EventRecord)
hwh_add_function(hwh_functions_section free "auto* p" Free LAMBDA)
hwh_add_function(hwh_functions_section free_async "auto* p;stream_t stream" FreeAsync LAMBDA) # TODO: Version constraints?
hwh_add_function(hwh_functions_section get_device "int* device" GetDevice)
hwh_add_function(hwh_functions_section get_device_count "int* count" GetDeviceCount)
hwh_add_function(hwh_functions_section get_device_properties "device_prop* prop;int device" GetDeviceProperties)
hwh_add_function(hwh_functions_section get_stream_priority_range "int* least;int* greatest" DeviceGetStreamPriorityRange)
hwh_add_function(hwh_functions_section launch_kernel "const auto* f;dim3 num_blocks;dim3 dim_blocks;void** args;std::size_t shared_mem_bytes;stream_t stream" LaunchKernel LAMBDA)
hwh_add_function(hwh_functions_section malloc "auto** p;std::size_t size" Malloc LAMBDA) # TODO: std::size_t or size_t?
hwh_add_function(hwh_functions_section malloc_async "auto** p;std::size_t size;stream_t stream" MallocAsync LAMBDA) # TODO: std::size_t or size_t?
hwh_add_function(hwh_functions_section malloc_host "auto** p;std::size_t size" MallocHost HIP_OVERRIDE HostMalloc LAMBDA) # TODO: flags? TODO: HIP also has hipMallocHost nowadays, no need for special case?
hwh_add_function(hwh_functions_section mem_get_info "std::size_t* free;std::size_t* total" MemGetInfo)
hwh_add_function(hwh_functions_section memcpy "auto* dst;const auto* src;std::size_t size_bytes;memcpy_kind kind" Memcpy LAMBDA)
hwh_add_function(hwh_functions_section memcpy_async "auto* dst;const auto* src;std::size_t size_bytes;memcpy_kind kind;stream_t stream" MemcpyAsync LAMBDA)
hwh_add_function(hwh_functions_section memset "auto* dst;int value;std::size_t size_bytes" Memset LAMBDA)
hwh_add_function(hwh_functions_section memset_async "auto* dst;int value;std::size_t size_bytes;stream_t stream" MemsetAsync LAMBDA)
hwh_add_function(hwh_functions_section set_device "int device" SetDevice)
hwh_add_function(hwh_functions_section stream_add_callback "stream_t stream;stream_callback_t callback;void* user_data;unsigned int flags" StreamAddCallback)
hwh_add_function(hwh_functions_section stream_create "stream_t* stream" StreamCreate)
hwh_add_function(hwh_functions_section stream_create_with_flags "stream_t* stream;unsigned int flags" StreamCreateWithFlags)
hwh_add_function(hwh_functions_section stream_create_with_priority "stream_t* stream;unsigned int flags;int priority" StreamCreateWithPriority)
hwh_add_function(hwh_functions_section stream_destroy "stream_t stream" StreamDestroy)
hwh_add_function(hwh_functions_section stream_get_flags "stream_t stream;unsigned int* flags" StreamGetFlags)
hwh_add_function(hwh_functions_section stream_query "stream_t stream" StreamQuery NOEXCEPT)
hwh_add_function(hwh_functions_section stream_synchronize "stream_t stream" StreamSynchronize)

write_basic_package_version_file("${PROJECT_BINARY_DIR}/hwh-config-version.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
configure_file("${PROJECT_SOURCE_DIR}/cmake/hwh.hpp.in" "${PROJECT_BINARY_DIR}/include/hwh.hpp")

install(TARGETS hwh EXPORT hwh)
install(
  EXPORT hwh
  FILE hwh-targets.cmake
  NAMESPACE hwh::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hwh")
install(FILES "${PROJECT_BINARY_DIR}/include/hwh.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${PROJECT_BINARY_DIR}/hwh-config-version.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/hwh-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hwh")
