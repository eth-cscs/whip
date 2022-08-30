# TODO: license and copyright

cmake_policy(VERSION 3.18) # TODO: version...

include("${CMAKE_CURRENT_LIST_DIR}/hwh-targets.cmake")

set(HWH_VERSION_STRING "@HWH_VERSION@")
set(HWH_VERSION_MAJOR @HWH_VERSION_MAJOR@)
set(HWH_VERSION_MINOR @HWH_VERSION_MINOR@)
set(HWH_VERSION_PATCH @HWH_VERSION_PATCH@)
set(HWH_TYPE @HWH_TYPE@)

if(HWH_TYPE STREQUAL "CUDA")
  find_dependency(CUDAToolkit)
elseif(HWH_TYPE STREQUAL "HIP")
  find_package(hip)
endif()
