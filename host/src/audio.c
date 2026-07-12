/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : audio.c
 * Purpose : Audio capture backends (see kws_audio.h).
 * ---------------------------------------------------------------------------*/
#include "kws_audio.h"
#include "kws_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#  include <mmsystem.h>
#else
#  include <time.h>
#  include <unistd.h>
#  ifdef KWS_USE_ALSA
#    include <alsa/asoundlib.h>
#  endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum backend { BK_MIC, BK_WAV, BK_SYNTH };

#ifdef _WIN32
#define MIC_N_BUF   8
#define MIC_BUF_LEN 320   /* 20 ms at 16 kHz */
#endif

struct kws_audio {
    enum backend kind;
    int sample_rate;
    int realtime;
    /* wav */
    FILE *fp;
    int   wav_channels;
    long  wav_data_left;   /* bytes */
    /* synth */
    long long synth_pos;
    /* pacing */
    long long paced_samples;
    long long t0_us;
#ifdef _WIN32
    HWAVEIN  hwi;
    WAVEHDR  hdr[MIC_N_BUF];
    int16_t  mic_buf[MIC_N_BUF][MIC_BUF_LEN];
    int      cur_buf, cur_off;
#endif
#if !defined(_WIN32) && defined(KWS_USE_ALSA)
    snd_pcm_t *pcm;
#endif
};

/* --- monotonic microseconds -------------------------------------------------*/
static long long now_us(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (long long)((double)t.QuadPart * 1e6 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
#endif
}

static void sleep_us(long long us)
{
    if (us <= 0) return;
#ifdef _WIN32
    Sleep((DWORD)((us + 999) / 1000));
#else
    usleep((useconds_t)us);
#endif
}

/* Pace non-realtime sources to wall clock. */
static void pace(kws_audio_t *a, int n)
{
    if (!a->realtime) return;
    a->paced_samples += n;
    long long due = a->t0_us
                  + a->paced_samples * 1000000LL / a->sample_rate;
    sleep_us(due - now_us());
}

/* --- WAV --------------------------------------------------------------------*/
static int wav_open(kws_audio_t *a, const char *path, char *err, size_t errlen)
{
    a->fp = fopen(path, "rb");
    if (!a->fp) { snprintf(err, errlen, "cannot open '%s'", path); return -1; }

    uint8_t h[12];
    if (fread(h, 1, 12, a->fp) != 12 || memcmp(h, "RIFF", 4) != 0
        || memcmp(h + 8, "WAVE", 4) != 0) {
        snprintf(err, errlen, "'%s' is not a RIFF/WAVE file", path);
        return -1;
    }

    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    /* Walk chunks until 'data' */
    for (;;) {
        uint8_t ch[8];
        if (fread(ch, 1, 8, a->fp) != 8) {
            snprintf(err, errlen, "no data chunk in '%s'", path);
            return -1;
        }
        uint32_t sz = (uint32_t)ch[4] | (uint32_t)ch[5] << 8
                    | (uint32_t)ch[6] << 16 | (uint32_t)ch[7] << 24;
        if (memcmp(ch, "fmt ", 4) == 0) {
            uint8_t f[16];
            if (sz < 16 || fread(f, 1, 16, a->fp) != 16) {
                snprintf(err, errlen, "bad fmt chunk"); return -1;
            }
            fmt      = (uint16_t)(f[0] | f[1] << 8);
            channels = (uint16_t)(f[2] | f[3] << 8);
            rate     = (uint32_t)f[4] | (uint32_t)f[5] << 8
                     | (uint32_t)f[6] << 16 | (uint32_t)f[7] << 24;
            bits     = (uint16_t)(f[14] | f[15] << 8);
            if (sz > 16) fseek(a->fp, (long)(sz - 16), SEEK_CUR);
        } else if (memcmp(ch, "data", 4) == 0) {
            a->wav_data_left = (long)sz;
            break;
        } else {
            fseek(a->fp, (long)((sz + 1) & ~1u), SEEK_CUR);
        }
    }

    if (fmt != 1 || bits != 16 || (int)rate != a->sample_rate
        || (channels != 1 && channels != 2)) {
        snprintf(err, errlen,
                 "'%s': need PCM16 %d Hz mono/stereo (got fmt=%u ch=%u %u Hz %u bit)",
                 path, a->sample_rate, fmt, channels, rate, bits);
        return -1;
    }
    a->wav_channels = channels;
    return 0;
}

static int wav_read(kws_audio_t *a, int16_t *buf, int n)
{
    int got = 0;
    while (got < n && a->wav_data_left > 0) {
        int16_t s[2];
        size_t need = (size_t)a->wav_channels * 2;
        if ((long)need > a->wav_data_left) { a->wav_data_left = 0; break; }
        if (fread(s, 1, need, a->fp) != need) { a->wav_data_left = 0; break; }
        a->wav_data_left -= (long)need;
        buf[got++] = (a->wav_channels == 2)
                   ? (int16_t)(((int)s[0] + (int)s[1]) / 2) : s[0];
    }
    pace(a, got);
    return got;  /* 0 => EOF */
}

/* --- synth ------------------------------------------------------------------*/
static int synth_read(kws_audio_t *a, int16_t *buf, int n)
{
    /* 500 ms tone burst (alternating 700/1400 Hz) every 3 s over low noise. */
    for (int i = 0; i < n; i++) {
        long long t   = a->synth_pos++;
        double    sec = (double)t / a->sample_rate;
        double    ph  = fmod(sec, 3.0);
        double    v   = 0.02 * ((double)rand() / RAND_MAX - 0.5);
        if (ph < 0.5) {
            double f = (fmod(sec, 6.0) < 3.0) ? 700.0 : 1400.0;
            v += 0.5 * sin(2.0 * M_PI * f * sec);
        }
        buf[i] = (int16_t)(v * 30000.0);
    }
    pace(a, n);
    return n;
}

/* --- microphone -------------------------------------------------------------*/
#ifdef _WIN32
static int mic_open(kws_audio_t *a, char *err, size_t errlen)
{
    WAVEFORMATEX wf;
    memset(&wf, 0, sizeof(wf));
    wf.wFormatTag      = WAVE_FORMAT_PCM;
    wf.nChannels       = 1;
    wf.nSamplesPerSec  = (DWORD)a->sample_rate;
    wf.wBitsPerSample  = 16;
    wf.nBlockAlign     = 2;
    wf.nAvgBytesPerSec = wf.nSamplesPerSec * wf.nBlockAlign;

    if (waveInOpen(&a->hwi, WAVE_MAPPER, &wf, 0, 0, CALLBACK_NULL)
        != MMSYSERR_NOERROR) {
        snprintf(err, errlen, "waveInOpen failed (no microphone?)");
        return -1;
    }
    for (int i = 0; i < MIC_N_BUF; i++) {
        memset(&a->hdr[i], 0, sizeof(WAVEHDR));
        a->hdr[i].lpData         = (LPSTR)a->mic_buf[i];
        a->hdr[i].dwBufferLength = MIC_BUF_LEN * 2;
        waveInPrepareHeader(a->hwi, &a->hdr[i], sizeof(WAVEHDR));
        waveInAddBuffer(a->hwi, &a->hdr[i], sizeof(WAVEHDR));
    }
    a->cur_buf = 0;
    a->cur_off = 0;
    waveInStart(a->hwi);
    return 0;
}

static int mic_read(kws_audio_t *a, int16_t *buf, int n)
{
    int got = 0;
    while (got < n) {
        WAVEHDR *h = &a->hdr[a->cur_buf];
        if (!(h->dwFlags & WHDR_DONE)) { Sleep(1); continue; }
        int avail = (int)(h->dwBytesRecorded / 2) - a->cur_off;
        int take  = (n - got < avail) ? (n - got) : avail;
        if (take > 0) {
            memcpy(&buf[got], &a->mic_buf[a->cur_buf][a->cur_off],
                   (size_t)take * 2);
            got        += take;
            a->cur_off += take;
        }
        if (a->cur_off >= (int)(h->dwBytesRecorded / 2)) {
            h->dwFlags &= ~WHDR_DONE;
            waveInAddBuffer(a->hwi, h, sizeof(WAVEHDR));
            a->cur_buf = (a->cur_buf + 1) % MIC_N_BUF;
            a->cur_off = 0;
        }
    }
    return got;
}
#elif defined(KWS_USE_ALSA)
static int mic_open(kws_audio_t *a, char *err, size_t errlen)
{
    if (snd_pcm_open(&a->pcm, "default", SND_PCM_STREAM_CAPTURE, 0) < 0) {
        snprintf(err, errlen, "snd_pcm_open failed");
        return -1;
    }
    if (snd_pcm_set_params(a->pcm, SND_PCM_FORMAT_S16_LE,
                           SND_PCM_ACCESS_RW_INTERLEAVED, 1,
                           (unsigned)a->sample_rate, 1, 100000) < 0) {
        snprintf(err, errlen, "snd_pcm_set_params failed");
        return -1;
    }
    return 0;
}

static int mic_read(kws_audio_t *a, int16_t *buf, int n)
{
    int got = 0;
    while (got < n) {
        snd_pcm_sframes_t r = snd_pcm_readi(a->pcm, buf + got,
                                            (snd_pcm_uframes_t)(n - got));
        if (r < 0) {
            if (snd_pcm_recover(a->pcm, (int)r, 1) < 0) return -1;
            continue;
        }
        got += (int)r;
    }
    return got;
}
#else
static int mic_open(kws_audio_t *a, char *err, size_t errlen)
{
    (void)a;
    snprintf(err, errlen,
             "mic backend not built (rebuild with KWS_USE_ALSA=1 on Linux)");
    return -1;
}
static int mic_read(kws_audio_t *a, int16_t *buf, int n)
{
    (void)a; (void)buf; (void)n;
    return -1;
}
#endif

/* --- public API ---------------------------------------------------------------*/
kws_audio_t *kws_audio_open(const char *spec, int sample_rate, int realtime,
                            char *err, size_t errlen)
{
    kws_audio_t *a = (kws_audio_t *)calloc(1, sizeof(*a));
    if (!a) { snprintf(err, errlen, "out of memory"); return NULL; }
    a->sample_rate = sample_rate;
    a->realtime    = realtime;
    a->t0_us       = now_us();

    if (strcmp(spec, "mic") == 0) {
        a->kind = BK_MIC;
        if (mic_open(a, err, errlen) != 0) { free(a); return NULL; }
    } else if (strncmp(spec, "wav:", 4) == 0) {
        a->kind = BK_WAV;
        if (wav_open(a, spec + 4, err, errlen) != 0) {
            if (a->fp) fclose(a->fp);
            free(a);
            return NULL;
        }
    } else if (strcmp(spec, "synth") == 0) {
        a->kind = BK_SYNTH;
        srand(1);   /* deterministic */
    } else {
        snprintf(err, errlen, "unknown audio spec '%s'", spec);
        free(a);
        return NULL;
    }
    return a;
}

int kws_audio_read(kws_audio_t *a, int16_t *buf, int n)
{
    switch (a->kind) {
    case BK_MIC:   return mic_read(a, buf, n);
    case BK_WAV:   return wav_read(a, buf, n);
    case BK_SYNTH: return synth_read(a, buf, n);
    default:       return -1;
    }
}

void kws_audio_close(kws_audio_t *a)
{
    if (!a) return;
#ifdef _WIN32
    if (a->kind == BK_MIC && a->hwi) {
        waveInStop(a->hwi);
        waveInReset(a->hwi);
        for (int i = 0; i < MIC_N_BUF; i++) {
            waveInUnprepareHeader(a->hwi, &a->hdr[i], sizeof(WAVEHDR));
        }
        waveInClose(a->hwi);
    }
#endif
#if !defined(_WIN32) && defined(KWS_USE_ALSA)
    if (a->kind == BK_MIC && a->pcm) snd_pcm_close(a->pcm);
#endif
    if (a->fp) fclose(a->fp);
    free(a);
}
