/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_ref.h
 * Purpose : Bit-accurate software reference model of the FPGA inference and
 *           decision pipeline. Every arithmetic step matches the RTL exactly
 *           (same rounding, saturation, tie-breaking and update ordering), so
 *           hardware outputs can be compared for equality, not similarity.
 *
 * Used by the host application (--check mode: live cross-check against the
 * FPGA) and by every Verilator testbench as the golden model.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_REF_H
#define KWS_REF_H

#include <stdint.h>
#include "kws_weights.h"   /* generated: dimensions, weights, requant params */

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_POOL_MAX 0
#define KWS_POOL_AVG 1

/* Layer-by-layer API (used by the per-module testbenches) ------------------*/

/* y = sat8((acc * m + 2^(s-1)) >> s), arithmetic shift; optional ReLU. */
int8_t kws_requant(int32_t acc, int32_t m, int32_t s, int relu);

/* Conv over one window: feat[KWS_WINDOW_LEN][KWS_NUM_MFCC] ->
 * act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH] (requantized, ReLU applied). */
void kws_ref_conv1d(const int8_t feat[KWS_WINDOW_LEN][KWS_NUM_MFCC],
                    int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH]);

/* Temporal pooling: act -> pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH]. */
void kws_ref_pool(const int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH],
                  int8_t pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH],
                  int mode);

/* Dense layer: pooled -> logits[KWS_NUM_CLASSES]. */
void kws_ref_dense(const int8_t pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH],
                   int8_t logits[KWS_NUM_CLASSES]);

/* argmax with lowest-index tie-break. */
int kws_ref_argmax(const int8_t *v, int n);

/* Full window inference: features -> logits (+ optional winner index). */
void kws_ref_infer(const int8_t feat[KWS_WINDOW_LEN][KWS_NUM_MFCC],
                   int8_t logits[KWS_NUM_CLASSES], int *winner, int pool_mode);

/* Temporal smoothing (mirror of rtl/temporal_smoothing.sv) -----------------*/

#define KWS_SMOOTH_DEPTH 4   /* smoothing defaults: must match
                              * kws_pkg::SMOOTH_DEPTH and kws_quant.py */

typedef struct {
    int8_t  thresh;        /* smoothed-score threshold             */
    uint8_t vote_min;      /* majority votes required              */
    uint8_t min_consec;    /* consecutive candidate evaluations    */
    uint8_t debounce;      /* refractory period, in inferences     */
    uint8_t target_mask;   /* bit c: class c may trigger           */
    uint8_t enable;
} kws_smooth_cfg_t;

typedef struct {
    kws_smooth_cfg_t cfg;
    int8_t  hist[KWS_SMOOTH_DEPTH][KWS_NUM_CLASSES];
    int16_t sums[KWS_NUM_CLASSES];
    uint8_t win_hist[KWS_SMOOTH_DEPTH];
    int     head, whead, fill;
    int     consec, last_cand, debounce_cnt;
} kws_smooth_t;

typedef struct {
    uint8_t class_id;
    uint8_t confidence;   /* smoothed winner score, clamped >= 0 */
    uint8_t votes;
} kws_ref_event_t;

/* Default configuration = kws_pkg defaults (thresh 25, votes 2, consec 1,
 * debounce 12, mask 0b1100, enabled - the tuned operating point). */
void kws_smooth_init(kws_smooth_t *s, const kws_smooth_cfg_t *cfg /*or NULL*/);

/* Fold one inference result; returns 1 and fills evt on detection. */
int kws_smooth_step(kws_smooth_t *s, const int8_t logits[KWS_NUM_CLASSES],
                    int winner, kws_ref_event_t *evt);

#ifdef __cplusplus
}
#endif
#endif /* KWS_REF_H */
