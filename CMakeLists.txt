# TODO: License and copyright

cmake_minimum_required(VERSION 3.10) # TODO: What do we actually require?

project(hwh CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(HWH_TYPE cuda)

include(GNUInstallDirs)
include(hwh_helpers)

add_library(hwh INTERFACE)
target_include_directories(hwh INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR})
target_compile_features(hwh INTERFACE cxx_std_17)
target_compile_definitions(hwh INTERFACE HWH_CUDA)

set(hwh_types_section "// Types")
hwh_add_type(hwh_types_section device_prop_t DeviceProp_t)
hwh_add_type(hwh_types_section error_t Error_t)
hwh_add_type(hwh_types_section event_t Event_t)
hwh_add_type(hwh_types_section stream_t Stream_t)

set(hwh_constants_section "// Constants")
hwh_add_constant(hwh_constants_section stream_non_blocking StreamNonBlocking)
hwh_add_constant(hwh_constants_section event_disable_timing EventDisableTiming)

set(hwh_errors_section "// Errors")
hwh_add_error(hwh_errors_section success Success)
hwh_add_error(hwh_errors_section error_not_ready ErrorNotReady)

set(hwh_functions_section "// Functions")
hwh_add_function(hwh_functions_section device_synchronize "int x;double y;std::string<askdjh, askdjh> z" DeviceSynchronize)
hwh_add_function(hwh_functions_section device_get_stream_priority "void * aasd;int e" DeviceGetStreamPriorityRange)
hwh_add_function(hwh_functions_section one_arg "double x" OneArg)

configure_file("${PROJECT_SOURCE_DIR}/cmake/hwh.hpp.in" "${PROJECT_BINARY_DIR}/include/hwh.hpp")
