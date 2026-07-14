/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_log.h
 * Purpose : Leveled, timestamped logging to stderr and an optional file.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_LOG_H
#define KWS_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KWS_LOG_DEBUG = 0,
    KWS_LOG_INFO  = 1,
    KWS_LOG_WARN  = 2,
    KWS_LOG_ERROR = 3
} kws_log_level_t;

void kws_log_init(kws_log_level_t min_level, const char *file_path /*or NULL*/);
void kws_log_close(void);
void kws_log(kws_log_level_t level, const char *fmt, ...);

#define KWS_DEBUG(...) kws_log(KWS_LOG_DEBUG, __VA_ARGS__)
#define KWS_INFO(...)  kws_log(KWS_LOG_INFO,  __VA_ARGS__)
#define KWS_WARN(...)  kws_log(KWS_LOG_WARN,  __VA_ARGS__)
#define KWS_ERROR(...) kws_log(KWS_LOG_ERROR, __VA_ARGS__)

#ifdef __cplusplus
}
#endif
#endif /* KWS_LOG_H */
