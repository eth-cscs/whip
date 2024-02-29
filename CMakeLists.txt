# whip
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# General project setup
cmake_minimum_required(VERSION 3.22)

project(whip VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(whip_backend_description "Backend to use (CUDA or HIP)")
set(WHIP_BACKEND "CUDA" CACHE STRING "${whip_backend_description}")
set_property(CACHE WHIP_BACKEND PROPERTY TYPE STRING)
set_property(CACHE WHIP_BACKEND PROPERTY HELPSTRING "${whip_backend_description}")
set(whip_valid_backends CUDA HIP)
set_property(CACHE WHIP_BACKEND PROPERTY STRINGS "${whip_valid_backends}")

if(NOT WHIP_BACKEND IN_LIST whip_valid_backends)
  set(error_message "whip requires WHIP_BACKEND to be set to one of (case sensitive):\n")
  foreach(whip_valid_type ${whip_valid_backends})
    set(error_message "${error_message}  ${whip_valid_type}\n")
  endforeach()
  set(error_message "${error_message}WHIP_BACKEND is currently set to \"${WHIP_BACKEND}\".")
  message(FATAL_ERROR "${error_message}")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(whip_helpers)

add_library(whip INTERFACE)
target_include_directories(
  whip INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(whip INTERFACE cxx_std_17)
target_compile_definitions(whip INTERFACE "WHIP_${WHIP_BACKEND}")
target_compile_definitions(whip INTERFACE "WHIP_BACKEND=${WHIP_BACKEND}")

if(WHIP_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(whip INTERFACE CUDA::cudart)
elseif(WHIP_BACKEND STREQUAL "HIP")
  find_package(hip REQUIRED)
  target_link_libraries(whip INTERFACE hip::host)
endif()

# Generate whip header
set(whip_types_section "// Types")
whip_add_type(whip_types_section device_prop DeviceProp_t CUDA_OVERRIDE cudaDeviceProp)
whip_add_type(whip_types_section dim3 dim3 CUDA_OVERRIDE dim3 HIP_OVERRIDE dim3)
whip_add_type(whip_types_section error_t Error_t)
whip_add_type(whip_types_section event_t Event_t)
whip_add_type(whip_types_section memcpy_kind MemcpyKind)
whip_add_type(whip_types_section stream_callback_t StreamCallback_t)
whip_add_type(whip_types_section stream_t Stream_t)

set(whip_constants_section "// Constants")
whip_add_constant(whip_constants_section event_disable_timing EventDisableTiming)
whip_add_constant(whip_constants_section memcpy_host_to_host MemcpyHostToHost TYPE memcpy_kind)
whip_add_constant(whip_constants_section memcpy_host_to_device MemcpyHostToDevice TYPE memcpy_kind)
whip_add_constant(whip_constants_section memcpy_device_to_host MemcpyDeviceToHost TYPE memcpy_kind)
whip_add_constant(whip_constants_section memcpy_device_to_device MemcpyDeviceToDevice TYPE memcpy_kind)
whip_add_constant(whip_constants_section memcpy_default MemcpyDefault TYPE memcpy_kind)
whip_add_constant(whip_constants_section stream_non_blocking StreamNonBlocking)

set(whip_errors_section "// Errors")
whip_add_error(whip_errors_section success Success)
whip_add_error(whip_errors_section error_not_ready ErrorNotReady)

set(whip_functions_section "// Functions")
whip_add_function(whip_functions_section check_last_error "" GetLastError)
whip_add_function(whip_functions_section device_get_stream_priority_range "int* least;int* greatest" DeviceGetStreamPriorityRange)
whip_add_function(whip_functions_section device_synchronize "" DeviceSynchronize)
whip_add_function(whip_functions_section device_reset "" DeviceReset)
whip_add_function(whip_functions_section event_create "event_t* event" EventCreate)
whip_add_function(whip_functions_section event_create_with_flags "event_t* event;unsigned flags" EventCreateWithFlags)
whip_add_function(whip_functions_section event_destroy "event_t event" EventDestroy)
whip_add_function(whip_functions_section event_elapsed_time "float* milliseconds;event_t start_event;event_t stop_event" EventElapsedTime)
whip_add_function(whip_functions_section event_synchronize "event_t event" EventSynchronize)
whip_add_function(whip_functions_section event_ready "event_t event" EventQuery QUERY)
whip_add_function(whip_functions_section event_record "event_t event;stream_t stream" EventRecord)
whip_add_function(whip_functions_section free "auto* p" Free)
whip_add_function(whip_functions_section free_async "auto* p;stream_t stream" FreeAsync)
whip_add_function(whip_functions_section free_host "auto* p" FreeHost)
whip_add_function(whip_functions_section get_device "int* device" GetDevice)
whip_add_function(whip_functions_section get_device_count "int* count" GetDeviceCount)
whip_add_function(whip_functions_section get_device_properties "device_prop* prop;int device" GetDeviceProperties)
whip_add_function(whip_functions_section launch_kernel "const auto* f;dim3 num_blocks;dim3 dim_blocks;void** args;std::size_t shared_mem_bytes;stream_t stream" LaunchKernel)
whip_add_function(whip_functions_section malloc "auto** p;std::size_t size" Malloc)
whip_add_function(whip_functions_section malloc_async "auto** p;std::size_t size;stream_t stream" MallocAsync)
whip_add_function(whip_functions_section malloc_host "auto** p;std::size_t size" MallocHost HIP_OVERRIDE hipHostMalloc)
whip_add_function(whip_functions_section mem_get_info "std::size_t* free;std::size_t* total" MemGetInfo)
whip_add_function(whip_functions_section memcpy "auto* dst;const auto* src;std::size_t size_bytes;memcpy_kind kind" Memcpy)
whip_add_function(whip_functions_section memcpy_2d "auto* dst;std::size_t dpitch;const auto* src;std::size_t spitch;std::size_t width;std::size_t height;memcpy_kind kind" Memcpy2D)
whip_add_function(whip_functions_section memcpy_2d_async "auto* dst;std::size_t dpitch;const auto* src;std::size_t spitch;std::size_t width;std::size_t height;memcpy_kind kind;stream_t stream" Memcpy2DAsync)
whip_add_function(whip_functions_section memcpy_async "auto* dst;const auto* src;std::size_t size_bytes;memcpy_kind kind;stream_t stream" MemcpyAsync)
whip_add_function(whip_functions_section memset "auto* dst;int value;std::size_t size_bytes" Memset)
whip_add_function(whip_functions_section memset_2d "auto* dst;std::size_t pitch;int value;std::size_t width;std::size_t height" Memset2D)
whip_add_function(whip_functions_section memset_2d_async "auto* dst;std::size_t pitch;int value;std::size_t width;std::size_t height;stream_t stream" Memset2DAsync)
whip_add_function(whip_functions_section memset_async "auto* dst;int value;std::size_t size_bytes;stream_t stream" MemsetAsync)
whip_add_function(whip_functions_section set_device "int device" SetDevice)
whip_add_function(whip_functions_section stream_add_callback "stream_t stream;stream_callback_t callback;void* user_data;unsigned int flags" StreamAddCallback)
whip_add_function(whip_functions_section stream_create "stream_t* stream" StreamCreate)
whip_add_function(whip_functions_section stream_create_with_flags "stream_t* stream;unsigned int flags" StreamCreateWithFlags)
whip_add_function(whip_functions_section stream_create_with_priority "stream_t* stream;unsigned int flags;int priority" StreamCreateWithPriority)
whip_add_function(whip_functions_section stream_destroy "stream_t stream" StreamDestroy)
whip_add_function(whip_functions_section stream_get_flags "stream_t stream;unsigned int* flags" StreamGetFlags)
whip_add_function(whip_functions_section stream_ready "stream_t stream" StreamQuery QUERY)
whip_add_function(whip_functions_section stream_synchronize "stream_t stream" StreamSynchronize)

# Install whip
write_basic_package_version_file("${PROJECT_BINARY_DIR}/whip-config-version.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
configure_file("${PROJECT_SOURCE_DIR}/cmake/whip.hpp.in" "${PROJECT_BINARY_DIR}/include/whip.hpp")

install(TARGETS whip EXPORT whip)
install(
  EXPORT whip
  FILE whip-targets.cmake
  NAMESPACE whip::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/whip")
install(FILES "${PROJECT_BINARY_DIR}/include/whip.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${PROJECT_BINARY_DIR}/whip-config-version.cmake"
              "${PROJECT_SOURCE_DIR}/cmake/whip-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/whip")
