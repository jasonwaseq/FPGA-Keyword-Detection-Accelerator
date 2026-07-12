/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : ref_model.c
 * Purpose : Bit-accurate reference implementation of the accelerator's
 *           arithmetic. Any divergence between this file and the RTL is a bug
 *           in one of them by definition; the Verilator benches enforce
 *           equality on every value.
 *
 * Portability note: right shifts of negative values are implemented as
 * explicit floor shifts (kws_asr*) rather than relying on the C
 * implementation-defined behaviour of '>>' on signed integers.
 * ---------------------------------------------------------------------------*/
#include "kws_ref.h"
#include <string.h>

/* Arithmetic (floor) shift right for signed 64-bit. */
static int64_t kws_asr64(int64_t v, int s)
{
    return (v < 0) ? ~((~v) >> s) : (v >> s);
}

static int32_t kws_asr32(int32_t v, int s)
{
    return (v < 0) ? ~((~v) >> s) : (v >> s);
}

static int8_t sat8(int64_t v)
{
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

int8_t kws_requant(int32_t acc, int32_t m, int32_t s, int relu)
{
    int64_t prod = (int64_t)acc * (int64_t)m + ((int64_t)1 << (s - 1));
    int8_t  y    = sat8(kws_asr64(prod, s));
    if (relu && y < 0) y = 0;
    return y;
}

void kws_ref_conv1d(const int8_t feat[KWS_WINDOW_LEN][KWS_NUM_MFCC],
                    int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH])
{
    for (int t = 0; t < KWS_CONV_OUT_LEN; t++) {
        for (int oc = 0; oc < KWS_CONV_OUT_CH; oc++) {
            int32_t acc = kws_conv_b[oc];
            for (int k = 0; k < KWS_CONV_K; k++) {
                for (int ic = 0; ic < KWS_NUM_MFCC; ic++) {
                    acc += (int32_t)feat[t + k][ic] * (int32_t)kws_conv_w[oc][k][ic];
                }
            }
            act[t][oc] = kws_requant(acc, KWS_M_CONV, KWS_S_CONV, 1);
        }
    }
}

void kws_ref_pool(const int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH],
                  int8_t pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH],
                  int mode)
{
    for (int p = 0; p < KWS_POOL_OUT_LEN; p++) {
        for (int c = 0; c < KWS_CONV_OUT_CH; c++) {
            if (mode == KWS_POOL_AVG) {
                int32_t sum = 0;
                for (int q = 0; q < KWS_POOL_SIZE; q++) {
                    sum += act[p * KWS_POOL_SIZE + q][c];
                }
                /* POOL_SIZE is a power of two; log2 shift with floor. */
                int sh = 0;
                while ((1 << sh) < KWS_POOL_SIZE) sh++;
                pooled[p][c] = (int8_t)kws_asr32(sum, sh);
            } else {
                int8_t mx = act[p * KWS_POOL_SIZE][c];
                for (int q = 1; q < KWS_POOL_SIZE; q++) {
                    int8_t v = act[p * KWS_POOL_SIZE + q][c];
                    if (v > mx) mx = v;
                }
                pooled[p][c] = mx;
            }
        }
    }
}

void kws_ref_dense(const int8_t pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH],
                   int8_t logits[KWS_NUM_CLASSES])
{
    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        int32_t acc = kws_dense_b[c];
        for (int t = 0; t < KWS_POOL_OUT_LEN; t++) {
            for (int ch = 0; ch < KWS_CONV_OUT_CH; ch++) {
                int i = t * KWS_CONV_OUT_CH + ch;
                acc += (int32_t)pooled[t][ch] * (int32_t)kws_dense_w[c][i];
            }
        }
        logits[c] = kws_requant(acc, KWS_M_DENSE, KWS_S_DENSE, 0);
    }
}

int kws_ref_argmax(const int8_t *v, int n)
{
    int idx = 0;
    int8_t mx = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > mx) { mx = v[i]; idx = i; }   /* strict >: lowest-index tie */
    }
    return idx;
}

void kws_ref_infer(const int8_t feat[KWS_WINDOW_LEN][KWS_NUM_MFCC],
                   int8_t logits[KWS_NUM_CLASSES], int *winner, int pool_mode)
{
    int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH];
    int8_t pooled[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH];

    kws_ref_conv1d(feat, act);
    kws_ref_pool(act, pooled, pool_mode);
    kws_ref_dense(pooled, logits);
    if (winner) *winner = kws_ref_argmax(logits, KWS_NUM_CLASSES);
}

/* --- temporal smoothing (mirror of rtl/temporal_smoothing.sv) --------------*/

void kws_smooth_init(kws_smooth_t *s, const kws_smooth_cfg_t *cfg)
{
    memset(s, 0, sizeof(*s));
    if (cfg) {
        s->cfg = *cfg;
    } else {
        s->cfg.thresh      = 40;
        s->cfg.vote_min    = 5;
        s->cfg.min_consec  = 2;
        s->cfg.debounce    = 12;
        s->cfg.target_mask = 0x0C;   /* classes 2 and 3 */
        s->cfg.enable      = 1;
    }
    s->last_cand = -1;
}

int kws_smooth_step(kws_smooth_t *s, const int8_t logits[KWS_NUM_CLASSES],
                    int winner, kws_ref_event_t *evt)
{
    /* 1. Fold the new sample into moving sums and winner history (RTL cycle 0). */
    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        s->sums[c] = (int16_t)(s->sums[c] + logits[c] - s->hist[s->head][c]);
        s->hist[s->head][c] = logits[c];
    }
    s->head = (s->head + 1) % KWS_SMOOTH_DEPTH;

    s->win_hist[s->whead] = (uint8_t)winner;
    s->whead = (s->whead + 1) % KWS_SMOOTH_DEPTH;
    if (s->fill < KWS_SMOOTH_DEPTH) s->fill++;

    /* 2. Evaluate (RTL cycle 1). Averages are floor shifts of the sums. */
    int sh = 0;
    while ((1 << sh) < KWS_SMOOTH_DEPTH) sh++;

    int8_t avg[KWS_NUM_CLASSES];
    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        avg[c] = (int8_t)kws_asr32(s->sums[c], sh);
    }

    int    sm_idx = kws_ref_argmax(avg, KWS_NUM_CLASSES);
    int8_t sm_val = avg[sm_idx];

    int votes = 0;
    for (int d = 0; d < s->fill; d++) {
        if (s->win_hist[d] == sm_idx) votes++;
    }

    int candidate = s->cfg.enable
                 && ((s->cfg.target_mask >> sm_idx) & 1)
                 && (sm_val >= s->cfg.thresh)
                 && (votes >= s->cfg.vote_min);

    int run;
    if (!candidate)                                        run = 0;
    else if (s->consec != 0 && s->last_cand == sm_idx)     run = (s->consec >= 15) ? s->consec : s->consec + 1;
    else                                                   run = 1;

    int fire = candidate && (s->debounce_cnt == 0) && (run >= s->cfg.min_consec);

    if (candidate) s->last_cand = sm_idx;

    if (fire) {
        if (evt) {
            evt->class_id   = (uint8_t)sm_idx;
            evt->confidence = (sm_val < 0) ? 0u : (uint8_t)sm_val;
            evt->votes      = (uint8_t)votes;
        }
        s->debounce_cnt = s->cfg.debounce;
        s->consec       = 0;
        return 1;
    }

    s->consec = run;
    if (s->debounce_cnt > 0) s->debounce_cnt--;
    return 0;
}
