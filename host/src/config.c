/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : config.c
 * Purpose : Configuration defaults and INI loader (see kws_config.h).
 * ---------------------------------------------------------------------------*/
#include "kws_config.h"
#include "kws_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void kws_config_default(kws_config_t *c)
{
    memset(c, 0, sizeof(*c));
#ifdef _WIN32
    strcpy(c->port, "COM7");
#else
    strcpy(c->port, "/dev/ttyUSB1");
#endif
    c->baud             = 115200;
    strcpy(c->input, "mic");
    c->duration_s       = 0;
    c->stats_interval_s = 10;
    c->check            = 1;
    c->verbose          = 0;
    c->quant_scale      = 20.0f;
    strcpy(c->labels, "silence,unknown,yes,no");
}

static void trim(char *s)
{
    char *e;
    while (*s == ' ' || *s == '\t') memmove(s, s + 1, strlen(s));
    e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r'
                     || e[-1] == '\n')) *--e = 0;
}

int kws_config_load(kws_config_t *c, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    int  lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = 0;
        char *key = line, *val = eq + 1;
        trim(key);
        trim(val);
        if (!key[0]) continue;

        if      (!strcmp(key, "port"))           snprintf(c->port, sizeof(c->port), "%s", val);
        else if (!strcmp(key, "baud"))           c->baud = atoi(val);
        else if (!strcmp(key, "input"))          snprintf(c->input, sizeof(c->input), "%s", val);
        else if (!strcmp(key, "duration_s"))     c->duration_s = atoi(val);
        else if (!strcmp(key, "stats_interval_s")) c->stats_interval_s = atoi(val);
        else if (!strcmp(key, "check"))          c->check = atoi(val);
        else if (!strcmp(key, "verbose"))        c->verbose = atoi(val);
        else if (!strcmp(key, "log_file"))       snprintf(c->log_file, sizeof(c->log_file), "%s", val);
        else if (!strcmp(key, "quant_scale"))    c->quant_scale = (float)atof(val);
        else if (!strcmp(key, "labels"))         snprintf(c->labels, sizeof(c->labels), "%s", val);
        else KWS_WARN("config %s:%d: unknown key '%s'", path, lineno, key);
    }
    fclose(f);
    return 0;
}

const char *kws_config_label(const kws_config_t *c, int class_id)
{
    static char out[48];
    const char *p = c->labels;
    for (int i = 0; *p; i++) {
        const char *comma = strchr(p, ',');
        size_t len = comma ? (size_t)(comma - p) : strlen(p);
        if (i == class_id) {
            if (len >= sizeof(out)) len = sizeof(out) - 1;
            memcpy(out, p, len);
            out[len] = 0;
            return out;
        }
        if (!comma) break;
        p = comma + 1;
    }
    snprintf(out, sizeof(out), "class%d", class_id);
    return out;
}
