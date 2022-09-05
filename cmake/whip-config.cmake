# TODO: license and copyright

cmake_policy(VERSION 3.18) # TODO: version...

include("${CMAKE_CURRENT_LIST_DIR}/whip-targets.cmake")

set(WHIP_VERSION_STRING "@WHIP_VERSION@")
set(WHIP_VERSION_MAJOR @WHIP_VERSION_MAJOR@)
set(WHIP_VERSION_MINOR @WHIP_VERSION_MINOR@)
set(WHIP_VERSION_PATCH @WHIP_VERSION_PATCH@)
set(WHIP_TYPE @WHIP_TYPE@)

if(WHIP_TYPE STREQUAL "CUDA")
  find_dependency(CUDAToolkit)
elseif(WHIP_TYPE STREQUAL "HIP")
  find_package(hip)
endif()
