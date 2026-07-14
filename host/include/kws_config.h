/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_config.h
 * Purpose : Host configuration: defaults, INI-style file loader ('key = value'
 *           lines, '#' comments), CLI overrides applied on top by main().
 * ---------------------------------------------------------------------------*/
#ifndef KWS_CONFIG_H
#define KWS_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_CFG_STR 128

typedef struct {
    char  port[KWS_CFG_STR];       /* serial port                            */
    int   baud;                    /* UART baud rate                         */
    char  input[KWS_CFG_STR];      /* mic | wav:<path> | synth               */
    int   duration_s;              /* stop after N seconds (0 = forever)     */
    int   stats_interval_s;        /* READ_STATS poll period (0 = off)       */
    int   check;                   /* run software reference cross-check     */
    int   verbose;                 /* debug logging                          */
    char  log_file[KWS_CFG_STR];   /* optional log file                      */
    float quant_scale;             /* MFCC INT8 units per std dev            */
    char  labels[KWS_CFG_STR];     /* comma-separated class names            */
} kws_config_t;

void kws_config_default(kws_config_t *c);

/* Returns 0 on success, -1 if the file could not be opened. Unknown keys are
 * warnings, not errors. */
int kws_config_load(kws_config_t *c, const char *path);

/* Class label for id (from cfg->labels), safe for any id. */
const char *kws_config_label(const kws_config_t *c, int class_id);

#ifdef __cplusplus
}
#endif
#endif /* KWS_CONFIG_H */
