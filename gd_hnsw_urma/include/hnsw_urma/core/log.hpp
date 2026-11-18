// Copyright (c) ub_hnsw alignment contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Self-contained logging facade for the ub_hnsw algorithm library.
//
// The algorithm library must not depend on the host's logging stack (Milvus
// glog + Go zap sink + conan Log.h wrapper). This header provides the
// LOG_INFO / LOG_WARN / LOG_ERROR / LOG_DEBUG / LOG_TRACE macros the algorithm
// source already uses, with the same fmt "{}" placeholder syntax, backed by a
// default stderr sink that the host (Milvus) may override via
// hnsw_urma::log::SetSink to route into its own logger.
//
// Formatting uses the fmt library (fmt::format with a runtime format string,
// matching the prior conan Log.h macros) so all existing call sites —
// including enum-class types with fmt::formatter specializations — compile
// unchanged. The algorithm CMake target links fmt::fmt (PUBLIC).
//
// Design notes:
//   - Macro form mirrors the prior conan Log.h wrapper so algorithm .cpp
//     files need only an include swap (log/Log.h -> hnsw_urma/core/log.hpp).
//   - Severity filtering is process-global and thread-safe.
//   - The sink is a single function pointer; the host overrides it once at
//     startup. A null sink drops messages.

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "fmt/format.h"

namespace hnsw_urma {
namespace log {

enum class Severity : int {
    kTrace = 0,
    kDebug = 1,
    kInfo = 2,
    kWarn = 3,
    kError = 4,
    kFatal = 5,
};

// Sink signature: (severity, fully-formatted message). Implementations must be
// thread-safe; the default sink writes one line to stderr.
using Sink = void (*)(Severity, std::string_view);

namespace detail {

inline Sink& GlobalSink() {
    static Sink sink = +[](Severity sev, std::string_view msg) {
        const char* tag = "DEBUG";
        switch (sev) {
            case Severity::kTrace: tag = "TRACE"; break;
            case Severity::kDebug: tag = "DEBUG"; break;
            case Severity::kInfo:  tag = "INFO";  break;
            case Severity::kWarn:  tag = "WARN";  break;
            case Severity::kError: tag = "ERROR"; break;
            case Severity::kFatal: tag = "FATAL"; break;
        }
        // std::fwrite is thread-safe at the byte level; keep one call to avoid
        // interleaving. stderr is unbuffered by default.
        std::fprintf(stderr, "[ub_hnsw %s] %.*s\n", tag,
                     static_cast<int>(msg.size()), msg.data());
    };
    return sink;
}

inline std::atomic<int>& GlobalLevel() {
    // Default: Info and above. Host may raise/lower via SetLevel.
    static std::atomic<int> level{static_cast<int>(Severity::kInfo)};
    return level;
}

template <typename... Args>
inline void Dispatch(Severity sev, std::string_view fmt_str, Args&&... args) {
    if (static_cast<int>(sev) < GlobalLevel().load(std::memory_order_relaxed)) {
        return;
    }
    Sink s = GlobalSink();
    if (s == nullptr) return;
    // fmt::runtime lets the format string be a runtime value (the macros pass
    // __VA_ARGS__ which may be a non-literal), matching the prior conan Log.h
    // behavior under fmt 11.
    std::string msg = fmt::format(fmt::runtime(fmt_str), std::forward<Args>(args)...);
    s(sev, msg);
}

}  // namespace detail

// Host integration points.
inline void SetSink(Sink sink) {
    detail::GlobalSink() = sink;
}

inline void SetLevel(Severity sev) {
    detail::GlobalLevel().store(static_cast<int>(sev), std::memory_order_relaxed);
}

}  // namespace log
}  // namespace hnsw_urma

// ---- Public macros (drop-in for the prior conan Log.h wrapper) ----
// Usage unchanged: LOG_INFO("UB knobs: {}", x);
// The format string is wrapped with fmt::runtime so non-literal first
// arguments (e.g. LOG_INFO(some_prebuilt_string)) keep working under fmt 11.
#define LOG_TRACE(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kTrace, fmt_str, ##__VA_ARGS__)
#define LOG_DEBUG(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kDebug, fmt_str, ##__VA_ARGS__)
#define LOG_INFO(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kInfo, fmt_str, ##__VA_ARGS__)
#define LOG_WARN(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kWarn, fmt_str, ##__VA_ARGS__)
#define LOG_WARNING(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kWarn, fmt_str, ##__VA_ARGS__)
#define LOG_ERROR(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kError, fmt_str, ##__VA_ARGS__)
#define LOG_FATAL(fmt_str, ...) \
    ::hnsw_urma::log::detail::Dispatch(::hnsw_urma::log::Severity::kFatal, fmt_str, ##__VA_ARGS__)
