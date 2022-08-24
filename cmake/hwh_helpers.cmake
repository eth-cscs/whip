# TODO: license and copyright

function(hwh_add_type var wrapper wrapped)
  set(type "using ${wrapper} = ${HWH_TYPE}${wrapped};")
  set(${var} "${${var}}\n${type}" PARENT_SCOPE)
endfunction()

function(hwh_add_constant var wrapper wrapped)
  set(constant "inline constexpr int ${wrapper} = ${HWH_TYPE}${wrapped};")
  set(${var} "${${var}}\n${constant}" PARENT_SCOPE)
endfunction()

function(hwh_add_error var wrapper wrapped)
  set(error "inline constexpr int ${wrapper} = ${HWH_TYPE}${wrapped};")
  set(${var} "${${var}}\n${error}" PARENT_SCOPE)
endfunction()

function(hwh_add_function var wrapper params wrapped)
  # Transform parameter list into something usable in the actual parameter list:
  # replace semicolons with commas
  string(REPLACE ";" ", " wrapper_definition_params "${params}")
  set(wrapper_definition "${wrapper}(${wrapper_definition_params})")

  # Transform parameter list into somethig usable in the call to the wrapped
  # function: replace semicolons with commas and keep only the variable names
  string(REGEX REPLACE ";[^;]+ ([^ ]+);" ";\\1;" wrapped_call_args "${params}")
  string(REGEX REPLACE "^[^;]+ ([^ ]+);" "\\1;" wrapped_call_args "${wrapped_call_args}")
  string(REGEX REPLACE ";[^;]+ ([^ ]+)$" ";\\1" wrapped_call_args "${wrapped_call_args}")
  string(REGEX REPLACE "^[^;]+ ([^ ]+)$" "\\1" wrapped_call_args "${wrapped_call_args}")
  string(REPLACE ";" ", " wrapped_call_args "${wrapped_call_args}")
  set(wrapped_call "${HWH_TYPE}${wrapped}(${wrapped_call_args})")

  string(CONCAT fun
    "void ${wrapper_definition} noexcept {\n"
    "    if (error_t e = ${wrapped_call}; e != success) { throw exception(e); }\n"
    "}\n"
    )
  set(${var} "${${var}}\n${fun}" PARENT_SCOPE)
endfunction()
