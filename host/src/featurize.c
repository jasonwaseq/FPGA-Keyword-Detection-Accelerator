/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : featurize.c
 * Purpose : Training featurizer. Converts speech WAV clips into INT8 feature
 *           windows using the EXACT deployment MFCC front end (mfcc.c), so
 *           the model trains on precisely what the hardware will see at run
 *           time - no train/deploy feature mismatch by construction.
 *
 * Modes:
 *   --stats-out F   pass 1: accumulate global per-coefficient mean/std over
 *                   all frames of the manifest's clips, write to F (text).
 *   --stats-in F    pass 2: load frozen stats and emit quantized features.
 *   --all-frames    emit all 98 frames per clip (for stream construction /
 *                   selftest selection) instead of one energy-centered
 *                   32-frame training window (+/-4-frame jitter crops with
 *                   --jitter).
 *   --mix-dir D     augmentation: mix a random background-noise crop from
 *                   directory D into speech clips (labels 1..3) at a random
 *                   5..20 dB SNR with probability 0.7 (deterministic per
 *                   clip). Use for the training emission only; the stats
 *                   pass and eval sets stay clean.
 *   --whole         treat each manifest entry as one continuous recording
 *                   (up to 120 s): emit a single record with all its frames.
 *                   Used to featurize long validation streams exactly as the
 *                   live host would.
 *
 * Manifest: one clip per line:  <label 0..3> <wav path> [offset_samples]
 * Clips are read as 16 kHz mono PCM16, cropped/zero-padded to 1 s from the
 * optional offset (offsets carve silence examples out of the long
 * _background_noise_ recordings).
 *
 * Output (--out): "KWSF" magic, u32 n_records, u32 frames_per_record, then
 * records of { u8 label, int8 data[frames*40] }, all little-endian.
 * ---------------------------------------------------------------------------*/
#include "kws_mfcc.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLIP_SAMPLES 16000
#define CLIP_FRAMES  98        /* (16000 - 400) / 160 + 1 */
#define WIN_FRAMES   32

/* --- minimal WAV reader (PCM16, 16 kHz, mono/stereo) -------------------------*/
static int wav_read_clip(const char *path, long offset, int16_t *out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint8_t h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fclose(f);
        return -1;
    }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    long data_pos = -1, data_len = 0;
    for (;;) {
        uint8_t c[8];
        if (fread(c, 1, 8, f) != 8) break;
        uint32_t sz = (uint32_t)c[4] | (uint32_t)c[5] << 8
                    | (uint32_t)c[6] << 16 | (uint32_t)c[7] << 24;
        if (!memcmp(c, "fmt ", 4)) {
            uint8_t b[16];
            if (sz < 16 || fread(b, 1, 16, f) != 16) break;
            fmt  = (uint16_t)(b[0] | b[1] << 8);
            ch   = (uint16_t)(b[2] | b[3] << 8);
            rate = (uint32_t)b[4] | (uint32_t)b[5] << 8
                 | (uint32_t)b[6] << 16 | (uint32_t)b[7] << 24;
            bits = (uint16_t)(b[14] | b[15] << 8);
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
        } else if (!memcmp(c, "data", 4)) {
            data_pos = ftell(f);
            data_len = (long)sz;
            break;
        } else {
            fseek(f, (long)((sz + 1) & ~1u), SEEK_CUR);
        }
    }
    if (data_pos < 0 || fmt != 1 || bits != 16 || rate != 16000
        || (ch != 1 && ch != 2)) {
        fclose(f);
        return -1;
    }

    memset(out, 0, CLIP_SAMPLES * sizeof(int16_t));
    long total = data_len / (2 * ch);
    long start = offset < 0 ? 0 : offset;
    long avail = total - start;
    if (avail > CLIP_SAMPLES) avail = CLIP_SAMPLES;
    if (avail > 0) {
        fseek(f, data_pos + start * 2 * ch, SEEK_SET);
        for (long i = 0; i < avail; i++) {
            int16_t s[2] = {0, 0};
            if (fread(s, 2, ch, f) != ch) break;
            out[i] = (ch == 2) ? (int16_t)(((int)s[0] + (int)s[1]) / 2) : s[0];
        }
    }
    fclose(f);
    return 0;
}

/* --- noise-mix augmentation ----------------------------------------------------*/
#define MAX_NOISE_FILES 16
static int16_t *g_noise[MAX_NOISE_FILES];
static long     g_noise_len[MAX_NOISE_FILES];
static int      g_n_noise = 0;
static uint32_t g_mix_rng = 0x4D584E1u;

static uint32_t mix_rand(void)
{
    g_mix_rng = g_mix_rng * 1664525u + 1013904223u;
    return g_mix_rng;
}

static int load_noise_dir(const char *dir)
{
    /* Load every 16 kHz PCM16 WAV in the directory whole. */
    char path[512];
    static const char *names[] = {
        "doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav",
        "pink_noise.wav", "running_tap.wav", "white_noise.wav"
    };
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, names[i]);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        long bytes = ftell(f) - 44;
        if (bytes <= 32000) { fclose(f); continue; }
        long n = bytes / 2;
        int16_t *buf = (int16_t *)malloc((size_t)n * 2);
        fseek(f, 44, SEEK_SET);
        if (fread(buf, 2, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); continue; }
        fclose(f);
        g_noise[g_n_noise] = buf;
        g_noise_len[g_n_noise] = n;
        g_n_noise++;
        if (g_n_noise == MAX_NOISE_FILES) break;
    }
    return g_n_noise > 0 ? 0 : -1;
}

static void mix_noise(int16_t *clip)
{
    if (g_n_noise == 0 || (mix_rand() % 100) >= 70) return;

    int   ni  = (int)(mix_rand() % (uint32_t)g_n_noise);
    long  off = (long)(mix_rand() % (uint32_t)(g_noise_len[ni] - CLIP_SAMPLES));
    const int16_t *nz = g_noise[ni] + off;

    double sp = 0, np_ = 0;
    for (int i = 0; i < CLIP_SAMPLES; i++) {
        sp  += (double)clip[i] * clip[i];
        np_ += (double)nz[i] * nz[i];
    }
    if (np_ < 1.0 || sp < 1.0) return;

    double snr_db = 5.0 + (double)(mix_rand() % 1500) / 100.0;   /* 5..20 dB */
    double gain = sqrt(sp / (np_ * pow(10.0, snr_db / 10.0)));
    for (int i = 0; i < CLIP_SAMPLES; i++) {
        long v = clip[i] + (long)(gain * nz[i]);
        clip[i] = (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
    }
}

/* --- continuous (whole-file) mode ---------------------------------------------*/
#define WHOLE_MAX_SAMPLES (120 * 16000)

static int wav_read_whole(const char *path, int16_t *out, long *n_out)
{
    /* Reuse the clip reader chunk-wise would truncate; do a direct read. */
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t h[12];
    if (fread(h, 1, 12, f) != 12 || memcmp(h, "RIFF", 4) || memcmp(h + 8, "WAVE", 4)) {
        fclose(f);
        return -1;
    }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    long data_pos = -1, data_len = 0;
    for (;;) {
        uint8_t c[8];
        if (fread(c, 1, 8, f) != 8) break;
        uint32_t sz = (uint32_t)c[4] | (uint32_t)c[5] << 8
                    | (uint32_t)c[6] << 16 | (uint32_t)c[7] << 24;
        if (!memcmp(c, "fmt ", 4)) {
            uint8_t b[16];
            if (sz < 16 || fread(b, 1, 16, f) != 16) break;
            fmt  = (uint16_t)(b[0] | b[1] << 8);
            ch   = (uint16_t)(b[2] | b[3] << 8);
            rate = (uint32_t)b[4] | (uint32_t)b[5] << 8
                 | (uint32_t)b[6] << 16 | (uint32_t)b[7] << 24;
            bits = (uint16_t)(b[14] | b[15] << 8);
            if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
        } else if (!memcmp(c, "data", 4)) {
            data_pos = ftell(f);
            data_len = (long)sz;
            break;
        } else {
            fseek(f, (long)((sz + 1) & ~1u), SEEK_CUR);
        }
    }
    if (data_pos < 0 || fmt != 1 || bits != 16 || rate != 16000 || ch != 1) {
        fclose(f);
        return -1;
    }
    long n = data_len / 2;
    if (n > WHOLE_MAX_SAMPLES) n = WHOLE_MAX_SAMPLES;
    if (fread(out, 2, (size_t)n, f) != (size_t)n) { fclose(f); return -1; }
    fclose(f);
    *n_out = n;
    return 0;
}

/* --- per-clip feature extraction ---------------------------------------------*/
static kws_mfcc_t g_mfcc;

static void clip_frames_f(const int16_t *clip, float coefs[CLIP_FRAMES][KWS_MFCC_N_MELS])
{
    g_mfcc.preemph_state = 0.0f;
    for (int fr = 0; fr < CLIP_FRAMES; fr++) {
        kws_mfcc_frame_f(&g_mfcc, clip + fr * KWS_MFCC_HOP_LEN, coefs[fr]);
    }
}

static void clip_frames_q(const int16_t *clip, int8_t q[CLIP_FRAMES][KWS_MFCC_N_MELS])
{
    g_mfcc.preemph_state = 0.0f;
    for (int fr = 0; fr < CLIP_FRAMES; fr++) {
        kws_mfcc_frame(&g_mfcc, clip + fr * KWS_MFCC_HOP_LEN, q[fr]);
    }
}

/* energy-centered window start (sum |q| per frame, peak of 32-frame sum) */
static int best_window(const int8_t q[CLIP_FRAMES][KWS_MFCC_N_MELS])
{
    long best = -1;
    int  best_s = 0;
    for (int s = 0; s <= CLIP_FRAMES - WIN_FRAMES; s++) {
        long e = 0;
        for (int fr = s; fr < s + WIN_FRAMES; fr++)
            for (int k = 0; k < KWS_MFCC_N_MELS; k++)
                e += q[fr][k] < 0 ? -q[fr][k] : q[fr][k];
        if (e > best) { best = e; best_s = s; }
    }
    return best_s;
}

int main(int argc, char **argv)
{
    const char *manifest = 0, *out_path = 0, *stats_in = 0, *stats_out = 0;
    const char *mix_dir = 0;
    int all_frames = 0, jitter = 0, whole = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest") && i + 1 < argc)   manifest  = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)   out_path  = argv[++i];
        else if (!strcmp(argv[i], "--stats-in") && i + 1 < argc)  stats_in  = argv[++i];
        else if (!strcmp(argv[i], "--stats-out") && i + 1 < argc) stats_out = argv[++i];
        else if (!strcmp(argv[i], "--mix-dir") && i + 1 < argc)   mix_dir   = argv[++i];
        else if (!strcmp(argv[i], "--all-frames"))            all_frames = 1;
        else if (!strcmp(argv[i], "--jitter"))                jitter = 1;
        else if (!strcmp(argv[i], "--whole"))                 whole = 1;
        else { fprintf(stderr, "unknown arg '%s'\n", argv[i]); return 2; }
    }
    if (mix_dir && load_noise_dir(mix_dir) != 0) {
        fprintf(stderr, "no noise files found in '%s'\n", mix_dir);
        return 1;
    }
    if (!manifest || (!stats_out && !out_path)) {
        fprintf(stderr,
            "usage: kws_featurize --manifest M (--stats-out F | "
            "--stats-in F --out B [--all-frames] [--jitter])\n");
        return 2;
    }

    kws_mfcc_init(&g_mfcc, 0);

    if (stats_in) {
        float mean[KWS_MFCC_N_MELS], std[KWS_MFCC_N_MELS];
        FILE *sf = fopen(stats_in, "r");
        if (!sf) { fprintf(stderr, "cannot open stats '%s'\n", stats_in); return 1; }
        for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
            if (fscanf(sf, "%f %f", &mean[k], &std[k]) != 2) {
                fprintf(stderr, "bad stats file\n");
                fclose(sf);
                return 1;
            }
        }
        fclose(sf);
        kws_mfcc_set_stats(&g_mfcc, mean, std, 1);
    }

    FILE *mf = fopen(manifest, "r");
    if (!mf) { fprintf(stderr, "cannot open manifest '%s'\n", manifest); return 1; }

    FILE *of = 0;
    uint32_t n_records = 0;
    uint32_t rec_frames = all_frames ? CLIP_FRAMES : WIN_FRAMES;
    if (out_path) {
        of = fopen(out_path, "wb");
        if (!of) { fprintf(stderr, "cannot open output '%s'\n", out_path); return 1; }
        uint32_t zero = 0;
        fwrite("KWSF", 1, 4, of);
        fwrite(&zero, 4, 1, of);          /* n_records patched at the end */
        fwrite(&rec_frames, 4, 1, of);
    }

    /* stats accumulators */
    double s_sum[KWS_MFCC_N_MELS] = {0}, s_sq[KWS_MFCC_N_MELS] = {0};
    long long s_n = 0;

    static int16_t clip[CLIP_SAMPLES];
    static float   coefs[CLIP_FRAMES][KWS_MFCC_N_MELS];
    static int8_t  q[CLIP_FRAMES][KWS_MFCC_N_MELS];

    char line[512];
    long clips = 0, skipped = 0;
    while (fgets(line, sizeof(line), mf)) {
        int label;
        char path[440];
        long offset = 0;
        int n = sscanf(line, "%d %439s %ld", &label, path, &offset);
        if (n < 2) continue;

        if (whole) {
            /* Continuous mode: one record per file, all frames, exactly as
             * the live host front end would process the stream. */
            static int16_t wbuf[WHOLE_MAX_SAMPLES];
            long wn = 0;
            if (wav_read_whole(path, wbuf, &wn) != 0) { skipped++; continue; }
            clips++;
            long wframes = (wn - KWS_MFCC_FRAME_LEN) / KWS_MFCC_HOP_LEN + 1;
            if (wframes < 1) { skipped++; continue; }
            g_mfcc.preemph_state = 0.0f;
            uint8_t lab = (uint8_t)label;
            fwrite(&lab, 1, 1, of);
            for (long fr = 0; fr < wframes; fr++) {
                int8_t qf[KWS_MFCC_N_MELS];
                kws_mfcc_frame(&g_mfcc, wbuf + fr * KWS_MFCC_HOP_LEN, qf);
                fwrite(qf, 1, KWS_MFCC_N_MELS, of);
            }
            rec_frames = (uint32_t)wframes;   /* single-record files only */
            n_records++;
            continue;
        }

        if (wav_read_clip(path, offset, clip) != 0) { skipped++; continue; }
        clips++;
        if (mix_dir && label != 0) mix_noise(clip);   /* speech labels only */

        if (stats_out) {
            clip_frames_f(clip, coefs);
            for (int fr = 0; fr < CLIP_FRAMES; fr++)
                for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
                    s_sum[k] += coefs[fr][k];
                    s_sq[k]  += (double)coefs[fr][k] * coefs[fr][k];
                }
            s_n += CLIP_FRAMES;
            continue;
        }

        clip_frames_q(clip, q);
        uint8_t lab = (uint8_t)label;
        if (all_frames) {
            fwrite(&lab, 1, 1, of);
            fwrite(q, 1, CLIP_FRAMES * KWS_MFCC_N_MELS, of);
            n_records++;
        } else {
            int s0 = best_window(q);
            int starts[3] = { s0, s0 - 4, s0 + 4 };
            int n_crops = jitter ? 3 : 1;
            for (int c = 0; c < n_crops; c++) {
                int s = starts[c];
                if (s < 0) s = 0;
                if (s > CLIP_FRAMES - WIN_FRAMES) s = CLIP_FRAMES - WIN_FRAMES;
                fwrite(&lab, 1, 1, of);
                fwrite(q[s], 1, WIN_FRAMES * KWS_MFCC_N_MELS, of);
                n_records++;
            }
        }
    }
    fclose(mf);

    if (stats_out) {
        FILE *sf = fopen(stats_out, "w");
        if (!sf) { fprintf(stderr, "cannot open stats out\n"); return 1; }
        for (int k = 0; k < KWS_MFCC_N_MELS; k++) {
            double mean = s_sum[k] / (double)s_n;
            double var  = s_sq[k] / (double)s_n - mean * mean;
            fprintf(sf, "%.8f %.8f\n", mean, sqrt(var > 0 ? var : 1e-6));
        }
        fclose(sf);
        fprintf(stderr, "stats over %lld frames from %ld clips -> %s "
                "(%ld skipped)\n", s_n, clips, stats_out, skipped);
    }
    if (of) {
        fseek(of, 4, SEEK_SET);
        fwrite(&n_records, 4, 1, of);
        if (whole) fwrite(&rec_frames, 4, 1, of);   /* patch true frame count */
        fclose(of);
        fprintf(stderr, "%u records (%u frames each) from %ld clips -> %s "
                "(%ld skipped)\n", n_records, rec_frames, clips, out_path,
                skipped);
    }
    return 0;
}
