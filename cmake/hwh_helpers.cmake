# TODO: license and copyright
include(CMakeParseArguments)

if("${HWH_TYPE}" STREQUAL "CUDA")
  set(hwh_wrapped_prefix "cuda")
elseif("${HWH_TYPE}" STREQUAL "HIP")
  set(hwh_wrapped_prefix "hip")
endif()

function(hwh_add_type var wrapper wrapped)
  set(type "using ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${type}" PARENT_SCOPE)
endfunction()

function(hwh_add_constant var wrapper wrapped)
  set(constant "inline constexpr int ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${constant}" PARENT_SCOPE)
endfunction()

function(hwh_add_error var wrapper wrapped)
  set(error "inline constexpr int ${wrapper} = ${hwh_wrapped_prefix}${wrapped};")
  set(${var} "${${var}}\n${error}" PARENT_SCOPE)
endfunction()

function(hwh_add_function var wrapper params wrapped)
  set(one_value_args HIP_OVERRIDE CUDA_OVERRIDE)
  cmake_parse_arguments(hwh_add_function "" "CUDA_OVERRIDE HIP_OVERRIDE" "" ${ARGN})

  if("${HWH_TYPE}" STREQUAL "CUDA" AND "${hwh_add_function_CUDA_OVERRIDE}")
    set(wrapped "${hwh_add_function_CUDA_OVERRIDE}")
  elseif("${HWH_TYPE}" STREQUAL "HIP" AND "${hwh_add_function_HIP_OVERRIDE}")
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

  string(CONCAT fun
    "inline void ${wrapper_definition} {\n"
    "    check_error(${wrapped_call});\n"
    "}\n"
    )
  set(${var} "${${var}}\n${fun}" PARENT_SCOPE)
endfunction()
