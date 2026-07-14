/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : log.c
 * Purpose : Logging implementation (see kws_log.h).
 * ---------------------------------------------------------------------------*/
#include "kws_log.h"

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

static kws_log_level_t g_min   = KWS_LOG_INFO;
static FILE           *g_file  = 0;

static const char *level_str(kws_log_level_t l)
{
    switch (l) {
    case KWS_LOG_DEBUG: return "DBG";
    case KWS_LOG_INFO:  return "INF";
    case KWS_LOG_WARN:  return "WRN";
    default:            return "ERR";
    }
}

void kws_log_init(kws_log_level_t min_level, const char *file_path)
{
    g_min = min_level;
    if (file_path && file_path[0]) {
        g_file = fopen(file_path, "a");
        if (!g_file) fprintf(stderr, "kws: cannot open log file '%s'\n", file_path);
    }
}

void kws_log_close(void)
{
    if (g_file) { fclose(g_file); g_file = 0; }
}

void kws_log(kws_log_level_t level, const char *fmt, ...)
{
    if (level < g_min) return;

    char stamp[32];
    time_t t = time(0);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    strftime(stamp, sizeof(stamp), "%H:%M:%S", &tmv);

    va_list ap;
    fprintf(stderr, "[%s %s] ", stamp, level_str(level));
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);

    if (g_file) {
        fprintf(g_file, "[%s %s] ", stamp, level_str(level));
        va_start(ap, fmt);
        vfprintf(g_file, fmt, ap);
        va_end(ap);
        fputc('\n', g_file);
        fflush(g_file);
    }
}
