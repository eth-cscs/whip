# TODO

TODO is a C++ library that abstracts the CUDA and HIP APIs into a single API
with minimal additions.

HIP essentially covers the use case of TODO. However, it requires the relatively
bulky installation of HIP even when targeting CUDA. TODO provides the same type
of abstraction as HIP, but only depends on HIP when targeting HIP and only
depends on CUDA when targeting CUDA. A common way to work around this is to
define macros of the type `#define cudaMemcpy hipMemcpy`. Doing so risks
conflicting definitions. TODO instead provides wrappers in a separate namespace.

# TODO wrappers

TODO generally tries to wrap CUDA and HIP as minimally as possible. This means
that:

- Types in TODO are `typedef`s of the corresponding types in CUDA and HIP to
  allow interoperability with other libraries using the CUDA or HIP types
  directly.
- Errors and constants in TODO use the same types as in CUDA and HIP, but with
  new names.

TODO lightly wraps CUDA and HIP functions. Function wrappers:

- Are C++ lambdas to easily allow passing templated functions (like
  `cuda/hipMemcpy`) into higher order functions.
- Check the return codes from wrapped functions and throw a `TODO::exception` on
  failure.

Wrapped types, errors, and functions have the same names as in CUDA and HIP,
with the `cuda/hip` prefix removed and the name changed to `snake_case`. For
example `cuda/hipMemcpy` is wrapped by a `TODO::error_t TODO::memcpy(auto* dst,
const auto*, std::size_t size_bytes, TODO::memcpy_kind)` function. All
functionality is available in the `TODO` namespace.

There are two main exceptions to the above rules:

- The functions `TODO::event_query` and `TODO::stream_query` are expected to
  fail under normal circumstances so they instead return a `TODO::error_t` to
  allow the caller to check the status of the event or stream.
- `cuda/hipGetLastError` is called `TODO::check_last_error` instead of
  `TODO::get_last_error` because it throws on failure as other wrappers do.
  
# Usage

TODO requires CMake TODO, and one of CUDA or HIP. The CMake variable `TODO_TYPE`
controls whether to target CUDA or HIP, and should be set to `CUDA` or `HIP`.
The default value is `CUDA`.

TODO can be included in CMake projects with `find_package(TODO)` after which a
CMake target `TODO::todo` will be available.

All functionality is available through the `TODO.hpp` header. TODO is a
header-only library.

# API coverage

We have made no effort to try to cover 100% of the available CUDA and HIP APIs.
TODO currently covers a useful subset of CUDA and HIP that is used by a few
projects. Adding more functions can be done through the helper functions in
`CMakeLists.txt` and PRs are welcome.
