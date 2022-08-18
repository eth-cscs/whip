// TODO: License and copyright
#pragma once

#if !(defined(HWH_WITH_CUDA) || defined(HWH_WITH_HIP))
#error "HWH requires exactly one of HWH_WITH_CUDA and HWH_WITH_HIP to be defined. Neither is defined."
#endif

#if defined(HWH_WITH_CUDA) && defined(HWH_WITH_HIP)
#error "HWH requires exactly one of HWH_WITH_CUDA and HWH_WITH_HIP to be defined. Both are defined."
#endif

#define HWH_IMPL_CAT_IMPL(x, y) x##y
#define HWH_IMPL_CAT(x, y) HWH_IMPL_CAT_IMPL(x, y)

#if defined(HWH_WITH_CUDA)
#include <cuda_runtime.h>
#define HWH_IMPL_PREFIX cuda
#elif defined(HWH_WITH_HIP)
#include <hip/hip_runtime.h>
#define HWH_IMPL_PREFIX hip
#endif

#define HWH_IMPL_ADD_PREFIX(x) HWH_IMPL_CAT(HWH_IMPL_PREFIX, x)

namespace hwh {
inline constexpr std::size_t version_major = 0;
inline constexpr std::size_t version_minor = 1;
inline constexpr std::size_t version_patch = 0;
inline constexpr const char *version_string = "0.1.0";

// Types
//
// Types are translated using simple typedefs.
#define HWH_IMPL_TYPE(t, t_wrapped) using t = HWH_IMPL_ADD_PREFIX(t_wrapped)
HWH_IMPL_TYPE(device_prop_t,
              DeviceProp); // TODO: cudaDeviceProp vs hipDeviceProp
HWH_IMPL_TYPE(error_t, Error_t);
HWH_IMPL_TYPE(event_t, Event_t);
HWH_IMPL_TYPE(stream_t, Stream_t);
#undef HWH_IMPL_TYPE

// Constants
//
// Constants are inline constexpr ints. TODO: Is int ok?
#define HWH_IMPL_CONSTANT(c, c_wrapped) inline constexpr int c = HWH_IMPL_ADD_PREFIX(c_wrapped)
HWH_IMPL_CONSTANT(stream_non_blocking, StreamNonBlocking);
HWH_IMPL_CONSTANT(event_disable_timing, EventDisableTiming);
#undef HWH_IMPL_CONSTANT

// Errors
//
// Errors are inline constexpr variables of type error_t.
#define HWH_IMPL_ERROR(e, e_wrapped) inline constexpr error_t e = HWH_IMPL_ADD_PREFIX(e_wrapped)
HWH_IMPL_ERROR(success, Success);
HWH_IMPL_ERROR(error_not_ready, ErrorNotReady);
#undef HWH_IMPL_ERROR

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
  if (e != success) {
    throw exception(e);
  }
}

// Functions
//
// Functions are wrapped in lambdas in two variants:
// - One which returns a regular error code
// - One which checks the error code and throws an exception on failure
#define HWH_IMPL_FUNC_IMPL(f, f_wrapped)                                                                \
  inline constexpr auto f = [](auto &&...ts) noexcept -> decltype(HWH_IMPL_ADD_PREFIX(f_wrapped)(       \
                                                          static_cast<decltype(ts) &&>(ts)...)) {       \
    return HWH_IMPL_ADD_PREFIX(f_wrapped)(static_cast<decltype(ts) &&>(ts)...);                         \
  };                                                                                                    \
                                                                                                        \
  inline constexpr auto f##_ex =                                                                        \
      [](auto &&...ts) -> decltype(HWH_IMPL_ADD_PREFIX(f_wrapped)(static_cast<decltype(ts) &&>(ts)...), \
                                   void()) {                                                            \
    HWH_IMPL_ADD_PREFIX(f_wrapped)(static_cast<decltype(ts) &&>(ts)...);                                \
  }
#if defined(HWH_WITH_CUDA)
#define HWH_IMPL_FUNC_SELECT(f, f_wrapped_cuda, f_wrapped_hip) HWH_IMPL_FUNC_IMPL(f, f_wrapped_cuda)
#else
#define HWH_IMPL_FUNC_SELECT(f, f_wrapped_cuda, f_wrapped_hip) HWH_IMPL_FUNC_IMPL(f, f_wrapped_hip)
#endif
#define HWH_IMPL_FUNC(f, f_wrapped) HWH_IMPL_FUNC_IMPL(f, f_wrapped)

HWH_IMPL_FUNC(device_synchronize, DeviceSynchronize);
HWH_IMPL_FUNC(device_get_stream_priority_range, DeviceGetStreamPriorityRange);

HWH_IMPL_FUNC(event_create_with_flags, EventCreateWithFlags);
HWH_IMPL_FUNC(event_destroy, EventDestroy);
HWH_IMPL_FUNC(event_query, EventQuery);
HWH_IMPL_FUNC(event_record, EventRecord);

HWH_IMPL_FUNC(get_device, GetDevice);
HWH_IMPL_FUNC(get_device_count, GetDeviceCount);
HWH_IMPL_FUNC(get_device_properties, GetDeviceProperties);
HWH_IMPL_FUNC(get_error_string, GetErrorString);
HWH_IMPL_FUNC(get_last_error, GetLastError);
HWH_IMPL_FUNC(get_parameter_buffer, GetParameterBuffer);

HWH_IMPL_FUNC(launch_device, LaunchDevice);
HWH_IMPL_FUNC(launch_kernel, LaunchKernel);

HWH_IMPL_FUNC(free, Free);
HWH_IMPL_FUNC(malloc, Malloc);
HWH_IMPL_FUNC_SELECT(malloc_host, MallocHost, HostMalloc);

HWH_IMPL_FUNC(memcpy, Memcpy);
HWH_IMPL_FUNC(memcpy_async, Memcpy);
HWH_IMPL_FUNC(memcpy_device_to_device, Memcpy);
HWH_IMPL_FUNC(memcpy_device_to_host, Memcpy);
HWH_IMPL_FUNC(memcpy_host_to_device, Memcpy);
HWH_IMPL_FUNC(mem_get_info, MemGetInfo);
HWH_IMPL_FUNC(memset_async, MemsetAsync);

HWH_IMPL_FUNC(stream_add_callback, StreamAddCallback);
HWH_IMPL_FUNC(stream_create, StreamCreate);
HWH_IMPL_FUNC(stream_create_with_flags, StreamCreateWithFlags);
HWH_IMPL_FUNC(stream_create_with_priority, StreamCreateWithPriority);
HWH_IMPL_FUNC(stream_destroy, StreamDestroy);
HWH_IMPL_FUNC(stream_get_flags, StreamGetFlags);
HWH_IMPL_FUNC(stream_synchronize, StreamSynchronize);
HWH_IMPL_FUNC(stream_query, StreamQuery);

#undef HWH_IMPL_FUNC
#undef HWH_IMPL_FUNC_SELECT
#undef HWH_IMPL_FUNC_IMPL
} // namespace hwh

#undef HWH_IMPL_ADD_PREFIX
#undef HWH_IMPL_PREFIX
#undef HWH_IMPL_CAT
#undef HWH_IMPL_CAT_IMPL
