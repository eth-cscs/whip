# whip
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

cmake_policy(VERSION 3.22)

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
