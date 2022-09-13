# whip
#
# Copyright (c) 2018-2022, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

include(CMakeParseArguments)

if("${WHIP_BACKEND}" STREQUAL "CUDA")
  set(whip_wrapped_prefix "cuda")
elseif("${WHIP_BACKEND}" STREQUAL "HIP")
  set(whip_wrapped_prefix "hip")
endif()

function(whip_add_type var wrapper wrapped)
  cmake_parse_arguments(whip_add_type "" "CUDA_OVERRIDE;HIP_OVERRIDE" "" ${ARGN})

  set(prefixed_wrapped "${whip_wrapped_prefix}${wrapped}")
  if("${WHIP_BACKEND}" STREQUAL "CUDA" AND NOT "${whip_add_type_CUDA_OVERRIDE}" STREQUAL "")
    set(prefixed_wrapped "${whip_add_type_CUDA_OVERRIDE}")
  elseif("${WHIP_BACKEND}" STREQUAL "HIP" AND NOT "${whip_add_type_HIP_OVERRIDE}" STREQUAL "")
    set(prefixed_wrapped "${whip_add_type_HIP_OVERRIDE}")
  endif()

  set(type "using ${wrapper} = ${prefixed_wrapped};")
  set(${var} "${${var}}\n${type}" PARENT_SCOPE)
endfunction()

function(whip_add_constant var wrapper wrapped)
  cmake_parse_arguments(whip_add_constant "" "TYPE" "" ${ARGN})
  set(type "int")
  if(NOT "${whip_add_constant_TYPE}" STREQUAL "")
    set(type "${whip_add_constant_TYPE}")
  endif()

  set(constant "inline constexpr ${type} ${wrapper} = ${whip_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${constant}" PARENT_SCOPE)
endfunction()

function(whip_add_error var wrapper wrapped)
  set(error "inline constexpr error_t ${wrapper} = ${whip_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${error}" PARENT_SCOPE)
endfunction()

function(whip_add_function var wrapper params wrapped)
  cmake_parse_arguments(whip_add_function "QUERY" "CUDA_OVERRIDE;HIP_OVERRIDE" "" ${ARGN})

  set(prefixed_wrapped "${whip_wrapped_prefix}${wrapped}")
  if("${WHIP_BACKEND}" STREQUAL "CUDA" AND NOT "${whip_add_function_CUDA_OVERRIDE}" STREQUAL "")
    set(prefixed_wrapped "${whip_add_function_CUDA_OVERRIDE}")
  elseif("${WHIP_BACKEND}" STREQUAL "HIP" AND NOT "${whip_add_function_HIP_OVERRIDE}" STREQUAL "")
    set(prefixed_wrapped "${whip_add_function_HIP_OVERRIDE}")
  endif()

  # Transform parameter list into something usable in the actual parameter list:
  # replace semicolons with commas
  string(REPLACE ";" ", " wrapper_definition_params "${params}")
  set(wrapper_definition "${wrapper}(${wrapper_definition_params})")

  # Transform parameter list into somethig usable in the call to the wrapped
  # function: replace semicolons with commas and keep only the variable names
  set(wrapped_call_args)
  foreach(param ${params})
    string(REGEX REPLACE ".* ([^ ]+)[  ]*$" "\\1" param_name "${param}")
    set(wrapped_call_args "${wrapped_call_args}, ${param_name}")
  endforeach()
  string(REGEX REPLACE "^, " "" wrapped_call_args "${wrapped_call_args}")
  set(wrapped_call "${prefixed_wrapped}(${wrapped_call_args})")

  set(checker "whip::check_error")
  if(whip_add_function_QUERY)
    set(checker "whip::impl::check_error_query")
  endif()
  set(fun "inline constexpr auto ${wrapper} = [](${wrapper_definition_params}) { return ${checker}(${wrapped_call}); };")
  set(${var} "${${var}}\n${fun}" PARENT_SCOPE)
endfunction()
