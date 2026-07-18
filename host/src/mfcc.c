/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : mfcc.c
 * Purpose : Self-contained MFCC extraction (see kws_mfcc.h). The FFT is an
 *           iterative in-place radix-2 implementation; tables (window, mel
 *           filterbank, DCT) are computed once at init.
 * ---------------------------------------------------------------------------*/
#include "kws_mfcc.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- iterative radix-2 complex FFT (in place) -------------------------------*/
static void fft(double *re, double *im, int n)
{
    /* bit reversal */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) {
            double t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                double ur = re[a], ui = im[a];
                double vr = re[b] * cr - im[b] * ci;
                double vi = re[b] * ci + im[b] * cr;
                re[a] = ur + vr; im[a] = ui + vi;
                re[b] = ur - vr; im[b] = ui - vi;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

static double hz_to_mel(double hz)  { return 2595.0 * log10(1.0 + hz / 700.0); }
static double mel_to_hz(double mel) { return 700.0 * (pow(10.0, mel / 2595.0) - 1.0); }

void kws_mfcc_init(kws_mfcc_t *m, const kws_mfcc_cfg_t *cfg)
{
    memset(m, 0, sizeof(*m));
    if (cfg) {
        m->cfg = *cfg;
    } else {
        m->cfg.preemph     = 0.97f;
        m->cfg.fmin        = 20.0f;
        m->cfg.fmax        = 7600.0f;
        m->cfg.norm_alpha  = 0.995f;
        m->cfg.quant_scale = 20.0f;
    }

    for (int i = 0; i < KWS_MFCC_FRAME_LEN; i++) {
        m->window[i] = 0.54f - 0.46f
                     * (float)cos(2.0 * M_PI * i / (KWS_MFCC_FRAME_LEN - 1));
    }

    /* Triangular mel filterbank */
    double mlo = hz_to_mel(m->cfg.fmin), mhi = hz_to_mel(m->cfg.fmax);
    double centers[KWS_MFCC_N_MELS + 2];
    for (int i = 0; i < KWS_MFCC_N_MELS + 2; i++) {
        double mel = mlo + (mhi - mlo) * i / (KWS_MFCC_N_MELS + 1);
        centers[i] = mel_to_hz(mel);
    }
    double bin_hz = (double)KWS_MFCC_SAMPLE_RATE / KWS_MFCC_FFT_LEN;
    for (int f = 0; f < KWS_MFCC_N_MELS; f++) {
        double lo = centers[f], mid = centers[f + 1], hi = centers[f + 2];
        for (int b = 0; b < KWS_MFCC_N_BINS; b++) {
            double hz = b * bin_hz, w = 0.0;
            if (hz > lo && hz < mid)       w = (hz - lo) / (mid - lo);
            else if (hz >= mid && hz < hi) w = (hi - hz) / (hi - mid);
            m->mel_w[f][b] = (float)w;
        }
    }

    /* Orthonormal DCT-II */
    for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
        double s = (k == 0) ? sqrt(1.0 / KWS_MFCC_N_MELS)
                            : sqrt(2.0 / KWS_MFCC_N_MELS);
        for (int n = 0; n < KWS_MFCC_N_MELS; n++) {
            m->dct[k][n] = (float)(s * cos(M_PI * (n + 0.5) * k / KWS_MFCC_N_MELS));
        }
    }

    for (int i = 0; i < KWS_MFCC_N_MELS; i++) { m->mean[i] = 0.0f; m->var[i] = 1.0f; }
    m->frozen = 0;
}

void kws_mfcc_set_stats(kws_mfcc_t *m, const float *mean, const float *std,
                        int frozen)
{
    for (int i = 0; i < KWS_MFCC_N_MELS; i++) {
        m->mean[i] = mean[i];
        m->var[i]  = std[i] * std[i];
    }
    m->frozen = frozen;
}

void kws_mfcc_frame_f(kws_mfcc_t *m, const int16_t *samples,
                      float coef[KWS_MFCC_N_MELS])
{
    double re[KWS_MFCC_FFT_LEN], im[KWS_MFCC_FFT_LEN];

    /* Pre-emphasis + window; zero-pad to FFT length. The pre-emphasis state
     * carries the last sample of the previous hop for continuity. */
    float prev = m->preemph_state;
    for (int i = 0; i < KWS_MFCC_FRAME_LEN; i++) {
        float x = (float)samples[i] / 32768.0f;
        float y = x - m->cfg.preemph * prev;
        prev = x;
        re[i] = (double)(y * m->window[i]);
        im[i] = 0.0;
    }
    /* State advances by one hop, not one frame (frames overlap). */
    m->preemph_state = (float)samples[KWS_MFCC_HOP_LEN - 1] / 32768.0f;

    for (int i = KWS_MFCC_FRAME_LEN; i < KWS_MFCC_FFT_LEN; i++) {
        re[i] = 0.0; im[i] = 0.0;
    }

    fft(re, im, KWS_MFCC_FFT_LEN);

    double power[KWS_MFCC_N_BINS];
    for (int b = 0; b < KWS_MFCC_N_BINS; b++) {
        power[b] = re[b] * re[b] + im[b] * im[b];
    }

    /* Mel energies -> log -> DCT */
    double logmel[KWS_MFCC_N_MELS];
    for (int f = 0; f < KWS_MFCC_N_MELS; f++) {
        double e = 0.0;
        for (int b = 0; b < KWS_MFCC_N_BINS; b++) {
            if (m->mel_w[f][b] != 0.0f) e += m->mel_w[f][b] * power[b];
        }
        logmel[f] = log(e + 1e-10);
    }

    for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
        double c = 0.0;
        for (int n = 0; n < KWS_MFCC_N_MELS; n++) c += m->dct[k][n] * logmel[n];
        coef[k] = (float)c;
    }
}

void kws_mfcc_frame(kws_mfcc_t *m, const int16_t *samples,
                    int8_t out[KWS_MFCC_N_MELS])
{
    float coef[KWS_MFCC_N_MELS];
    kws_mfcc_frame_f(m, samples, coef);

    /* Normalization + INT8 quantization. Frozen mode uses the corpus
     * statistics installed via kws_mfcc_set_stats (trained deployments);
     * otherwise a running EMA adapts mean and variance online. */
    float a = m->cfg.norm_alpha;
    for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
        if (!m->frozen) {
            float d = coef[k] - m->mean[k];
            m->mean[k] = a * m->mean[k] + (1.0f - a) * coef[k];
            m->var[k]  = a * m->var[k]  + (1.0f - a) * d * d;
        }
        float std = sqrtf(m->var[k] + 1e-6f);
        float q   = (coef[k] - m->mean[k]) / std * m->cfg.quant_scale;
        long  qi  = lroundf(q);
        if (qi > 127)  qi = 127;
        if (qi < -128) qi = -128;
        out[k] = (int8_t)qi;
    }
    m->frames_done++;
}
