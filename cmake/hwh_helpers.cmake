# TODO: license and copyright
include(CMakeParseArguments)

if("${HWH_TYPE}" STREQUAL "CUDA")
  set(hwh_wrapped_prefix "cuda")
elseif("${HWH_TYPE}" STREQUAL "HIP")
  set(hwh_wrapped_prefix "hip")
endif()

function(hwh_add_type var wrapper wrapped)
  cmake_parse_arguments(hwh_add_type "" "CUDA_OVERRIDE;HIP_OVERRIDE" "" ${ARGN})

  if("${HWH_TYPE}" STREQUAL "CUDA" AND NOT "${hwh_add_type_CUDA_OVERRIDE}" STREQUAL "")
    set(wrapped "${hwh_add_type_CUDA_OVERRIDE}")
  elseif("${HWH_TYPE}" STREQUAL "HIP" AND NOT "${hwh_add_type_HIP_OVERRIDE}" STREQUAL "")
    set(wrapped "${hwh_add_type_HIP_OVERRIDE}")
  endif()

  set(type "using ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${type}" PARENT_SCOPE)
endfunction()

function(hwh_add_constant var wrapper wrapped)
  cmake_parse_arguments(hwh_add_constant "" "TYPE" "" ${ARGN})
  set(type "int")
  if(NOT "${hwh_add_constant_TYPE}" STREQUAL "")
    set(type "${hwh_add_constant_TYPE}")
  endif()

  set(constant "inline constexpr ${type} ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${constant}" PARENT_SCOPE)
endfunction()

function(hwh_add_error var wrapper wrapped)
  set(error "inline constexpr int ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${error}" PARENT_SCOPE)
endfunction()

function(hwh_add_function var wrapper params wrapped)
  cmake_parse_arguments(hwh_add_function "NOEXCEPT" "CUDA_OVERRIDE;HIP_OVERRIDE" "" ${ARGN})

  if("${HWH_TYPE}" STREQUAL "CUDA" AND NOT "${hwh_add_function_CUDA_OVERRIDE}" STREQUAL "")
    set(wrapped "${hwh_add_function_CUDA_OVERRIDE}")
  elseif("${HWH_TYPE}" STREQUAL "HIP" AND NOT "${hwh_add_function_HIP_OVERRIDE}" STREQUAL "")
    set(wrapped "${hwh_add_function_HIP_OVERRIDE}")
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
  set(wrapped_call "${hwh_wrapped_prefix}${wrapped}(${wrapped_call_args})")

  set(wrapper_noexcept)
  set(wrapped_call_with_check "check_error(${wrapped_call})")
  if(hwh_add_function_NOEXCEPT)
    set(wrapper_noexcept " noexcept")
    set(wrapped_call_with_check "return ${wrapped_call}")
  endif()

  set(fun "inline auto ${wrapper} = [](${wrapper_definition_params})${wrapper_noexcept} { ${wrapped_call_with_check}; };")
  set(${var} "${${var}}\n${fun}" PARENT_SCOPE)
endfunction()
