# TODO

This currently generates something like this through CMake:

``` c++
namespace hwh {
inline constexpr std::size_t version_major = 0;
inline constexpr std::size_t version_minor = 1;
inline constexpr std::size_t version_patch = 0;
inline constexpr const char *version_string = "0.1.0";

// Types
using device_prop_t = cudaDeviceProp_t;
using error_t = cudaError_t;
using event_t = cudaEvent_t;
using stream_t = cudaStream_t;

// Constants
inline constexpr int stream_non_blocking = cudaStreamNonBlocking;
inline constexpr int event_disable_timing = cudaEventDisableTiming;

// Errors
inline constexpr int success = cudaSuccess;
inline constexpr int error_not_ready = cudaErrorNotReady;

// Custom exception which wraps a CUDA/HIP error
class exception : std::exception {
public:
  exception(error_t e) : e(e), msg(/*TODO*/ cudaGetErrorString(e)) {}
  error_t get_error() const noexcept { return e; }
  const char *what() const noexcept { return msg.c_str(); }

private:
  error_t e;
  std::string msg;
};

// Check an error and throw an exception on failure.
inline void check_error(error_t e) {
  if (e != success) { throw exception(e); }
}

// Functions
inline void device_synchronize(int x, double y, std::string<askdjh, askdjh> z) noexcept {
    if (error_t e = cudaDeviceSynchronize(x, y, z); e != success) { throw exception(e); }
}

inline void device_get_stream_priority(void * aasd, int e) noexcept {
    if (error_t e = cudaDeviceGetStreamPriorityRange(aasd, e); e != success) { throw exception(e); }
}

inline void one_arg(double x) noexcept {
    if (error_t e = cudaOneArg(x); e != success) { throw exception(e); }
}

} // namespace hwh
```
