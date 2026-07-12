/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_audio.h
 * Purpose : Audio capture abstraction with three backends:
 *             "mic"        - live microphone (WinMM on Windows, ALSA on Linux
 *                            when built with -DKWS_USE_ALSA)
 *             "wav:<path>" - 16 kHz mono PCM16 WAV file, paced in real time
 *                            (or as fast as possible with realtime=0)
 *             "synth"      - deterministic synthetic signal (tone bursts over
 *                            noise) for soak testing without hardware audio
 * ---------------------------------------------------------------------------*/
#ifndef KWS_AUDIO_H
#define KWS_AUDIO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kws_audio kws_audio_t;

/* spec: "mic", "wav:<path>" or "synth". Returns NULL on failure with a
 * message in err. realtime: pace file/synth sources to wall-clock rate. */
kws_audio_t *kws_audio_open(const char *spec, int sample_rate, int realtime,
                            char *err, size_t errlen);

/* Blocking read of exactly n mono PCM16 samples.
 * Returns n, 0 on end-of-stream (file sources), or -1 on error. */
int kws_audio_read(kws_audio_t *a, int16_t *buf, int n);

void kws_audio_close(kws_audio_t *a);

#ifdef __cplusplus
}
#endif
#endif /* KWS_AUDIO_H */
