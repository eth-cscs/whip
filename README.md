# whip

whip is a C++ library that abstracts the CUDA and HIP APIs into a single API
with minimal additions.

HIP essentially covers the use case of whip. However, it requires the relatively
bulky installation of HIP even when targeting CUDA. whip provides the same type
of abstraction as HIP, but only depends on HIP when targeting HIP and only
depends on CUDA when targeting CUDA. A common way to work around this is to
define macros of the type `#define cudaMemcpy hipMemcpy`. Doing so risks
conflicting definitions. whip instead provides wrappers in a separate namespace.

# whip wrappers

whip generally tries to wrap CUDA and HIP as minimally as possible. This means
that:

- Types in whip are `typedef`s of the corresponding types in CUDA and HIP to
  allow interoperability with other libraries using the CUDA or HIP types
  directly.
- Errors and constants in whip use the same types as in CUDA and HIP, but with
  new names.

whip lightly wraps CUDA and HIP functions. Function wrappers:

- Are C++ lambdas to easily allow passing templated functions (like
  `cuda/hipMemcpy`) into higher order functions.
- Check the return codes from wrapped functions and throw a `whip::exception` on
  failure.

Wrapped types, errors, and functions have the same names as in CUDA and HIP,
with the `cuda/hip` prefix removed and the name changed to `snake_case`. For
example `cuda/hipMemcpy` is wrapped by a `whip::error_t whip::memcpy(auto* dst,
const auto*, std::size_t size_bytes, whip::memcpy_kind)` function. All
functionality is available in the `whip` namespace.

There are two main exceptions to the above rules:

- The functions `cuda/hipEventQuery` and `cuda/hipStreamQuery` are expected to
  fail under normal circumstances. The wrappers in whip (respectively
  `whip::event_ready` and `whip::stream_ready`) return instead a bool indicating
  readiness.
  The wrappers throw an exception in all other cases.
- `cuda/hipGetLastError` is called `whip::check_last_error` instead of
  `whip::get_last_error` because it throws on failure as other wrappers do.
  
# Usage

whip requires a C++17-capable compiler, CMake 3.22, and one of CUDA or HIP. The
CMake variable `WHIP_BACKEND` controls whether to target CUDA or HIP, and should
be set to `CUDA` or `HIP`.  The default value is `CUDA`.

whip can be included in CMake projects with `find_package(whip)` after which a
CMake target `whip::whip` will be available.

All functionality is available through the `whip.hpp` header. whip is a
header-only library.

# API coverage

We have made no effort to try to cover 100% of the available CUDA and HIP APIs.
whip currently covers a useful subset of CUDA and HIP that is used by a few
projects. Adding more functions can be done through the helper functions in
`CMakeLists.txt` and PRs are welcome.
