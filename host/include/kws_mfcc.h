/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_mfcc.h
 * Purpose : Streaming MFCC front end: pre-emphasis, Hamming window, 512-point
 *           real FFT, 40-band mel filterbank (20..7600 Hz), log, orthonormal
 *           DCT-II, per-coefficient running normalization and INT8
 *           quantization. No external DSP dependencies.
 *
 * One call per 10 ms hop with the most recent 25 ms (400-sample) frame;
 * produces the KWS_NUM_MFCC INT8 coefficients streamed to the FPGA.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_MFCC_H
#define KWS_MFCC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_MFCC_SAMPLE_RATE 16000
#define KWS_MFCC_FRAME_LEN   400     /* 25 ms */
#define KWS_MFCC_HOP_LEN     160     /* 10 ms */
#define KWS_MFCC_FFT_LEN     512
#define KWS_MFCC_N_BINS      (KWS_MFCC_FFT_LEN / 2 + 1)
#define KWS_MFCC_N_MELS      40      /* must equal KWS_NUM_MFCC */

typedef struct {
    float preemph;      /* pre-emphasis coefficient (default 0.97)          */
    float fmin, fmax;   /* mel filterbank edges in Hz (default 20 / 7600)   */
    float norm_alpha;   /* EMA factor for mean/var tracking (default 0.995) */
    float quant_scale;  /* INT8 units per standard deviation (default 20.0) */
} kws_mfcc_cfg_t;

typedef struct {
    kws_mfcc_cfg_t cfg;
    float window[KWS_MFCC_FRAME_LEN];                    /* Hamming        */
    float mel_w[KWS_MFCC_N_MELS][KWS_MFCC_N_BINS];       /* filterbank     */
    float dct[KWS_MFCC_N_MELS][KWS_MFCC_N_MELS];         /* DCT-II matrix  */
    float mean[KWS_MFCC_N_MELS], var[KWS_MFCC_N_MELS];   /* running stats  */
    float preemph_state;
    long  frames_done;
} kws_mfcc_t;

/* cfg may be NULL for defaults. */
void kws_mfcc_init(kws_mfcc_t *m, const kws_mfcc_cfg_t *cfg);

/* samples: KWS_MFCC_FRAME_LEN mono PCM16 values (the current 25 ms frame).
 * out: KWS_MFCC_N_MELS quantized INT8 coefficients. */
void kws_mfcc_frame(kws_mfcc_t *m, const int16_t *samples,
                    int8_t out[KWS_MFCC_N_MELS]);

#ifdef __cplusplus
}
#endif
#endif /* KWS_MFCC_H */
