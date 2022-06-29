#pragma once

#include <fastba/version.h>

#include <cstdint>
#include <cstdio>

#include <spdlog/spdlog.h>

#include <spdlog/fmt/ostr.h>

#if defined(__ANDROID__)
#include <spdlog/sinks/android_sink.h>
#elif defined(__EMSCRIPTEN__)
#include <spdlog/sinks/stdout_sinks.h>
#else
#include <spdlog/sinks/stdout_color_sinks.h>
#endif

enum LogLevel : int32_t {
    LOG_OFF = 0,
    LOG_ERROR = 1,
    LOG_WARN = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4,
    LOG_TRACE = 5,
};

class LoggingSupport {
public:
    LoggingSupport() = default;
    ~LoggingSupport() = default;

    static int32_t &level() {
        static int32_t lvl = LOG_TRACE;
        return lvl;
    }

    static void set_log_level(int32_t t_lvl) {
        switch (t_lvl) {
            case LOG_OFF: {
                logger().set_level(spdlog::level::off);
            } break;
            case LOG_ERROR: {
                logger().set_level(spdlog::level::err);
            } break;
            case LOG_WARN: {
                logger().set_level(spdlog::level::warn);
            } break;
            case LOG_INFO: {
                logger().set_level(spdlog::level::info);
            } break;
            case LOG_DEBUG: {
                logger().set_level(spdlog::level::debug);
            } break;
            case LOG_TRACE: {
                logger().set_level(spdlog::level::trace);
            } break;
        }

        level() = t_lvl;
    }

    static spdlog::logger &logger() {
        static spdlog::logger s_logger = []() {
#if defined(__ANDROID__)
            auto logger =
                spdlog::logger {"lvo", std::make_shared<spdlog::sinks::android_sink_mt>("lvo")};
#elif defined(__EMSCRIPTEN__)
            auto logger = spdlog::logger {"lvo", std::make_shared<spdlog::sinks::stdout_sink_mt>()};
#else
            auto logger =
                spdlog::logger {"fastba", std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};
#endif
            logger.set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n][%^%=5l%$] %v");
            logger.set_level(spdlog::level::trace);
            return logger;
        }();
        return s_logger;
    }
};


#if PROJECTION_BUILD_LOGGING
#if !defined(log_trace)
#define log_trace(...) LoggingSupport::logger().trace(__VA_ARGS__)
#endif
#else
#define log_trace(...)
#endif

#if PROJECTION_BUILD_LOGGING
#if !defined(log_debug)
#define log_debug(...) LoggingSupport::logger().debug(__VA_ARGS__)
#endif
#else
#define log_debug(...)
#endif

#if PROJECTION_BUILD_LOGGING
#if !defined(log_info)
#define log_info(...) LoggingSupport::logger().info(__VA_ARGS__)
#endif
#else
#define log_info(...)
#endif

#if PROJECTION_BUILD_LOGGING
#if !defined(log_warn)
#define log_warn(...) LoggingSupport::logger().warn(__VA_ARGS__)
#endif
#else
#define log_warn(...)
#endif

#if PROJECTION_BUILD_LOGGING
#if !defined(log_error)
#define log_error(...) LoggingSupport::logger().error(__VA_ARGS__)
#endif
#else
#define log_error(...)
#endif

#if PROJECTION_BUILD_LOGGING
#define MACRO_STRINGIFY2(X) #X
#define MACRO_STRINGIFY(X) MACRO_STRINGIFY2(X)
#define runtime_assert(condition, ...)                                                             \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            LoggingSupport::logger().critical(                                                     \
                "Assertion failed at {}:{} when testing condition: {}", __FILE__, __LINE__,        \
                #condition);                                                                       \
            LoggingSupport::logger().critical(__VA_ARGS__);                                        \
        }                                                                                          \
    } while (0)
#else
#define runtime_assert(...)
#endif

#if !defined(log_critical_info)
#define log_critical_info(...) LoggingSupport::logger().info(__VA_ARGS__)
#endif
#if !defined(log_critical_warn)
#define log_critical_warn(...) LoggingSupport::logger().warn(__VA_ARGS__)
#endif
#if !defined(log_critical_error)
#define log_critical_error(...) LoggingSupport::logger().error(__VA_ARGS__)
#endif
