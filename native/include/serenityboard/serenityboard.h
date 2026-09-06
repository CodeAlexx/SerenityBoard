/* SerenityBoard native writer — C ABI for the Diffusion Compiler tools
 * (or any C/C++ program). Link libsbcore. Thread-safe per writer handle. */
#ifndef SERENITYBOARD_H
#define SERENITYBOARD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sb_writer sb_writer;

/* hparams_json may be NULL. resume_step < 0 means "new run". Returns NULL on
 * error; sb_last_error() carries the message. */
sb_writer *sb_writer_open(const char *logdir, const char *run_name, const char *hparams_json, long long resume_step,
                          int system_metrics);
const char *sb_last_error(void);
int sb_add_scalar(sb_writer *w, const char *tag, double value, long long step);
int sb_add_text(sb_writer *w, const char *tag, const char *text, long long step);
int sb_add_histogram(sb_writer *w, const char *tag, const double *values, size_t count, long long step, int bins);
/* HWC uint8 pixels, channels 1/3/4 */
int sb_add_image(sb_writer *w, const char *tag, uint32_t width, uint32_t height, uint32_t channels,
                 const uint8_t *pixels, long long step);
int sb_add_image_file(sb_writer *w, const char *tag, const char *png_path, long long step);
int sb_add_trace(sb_writer *w, long long step, const char *phase, double duration_ms, const char *details_json);
int sb_add_eval(sb_writer *w, const char *suite, const char *case_id, long long step, const char *score_name,
                double score_value, const char *artifact_key, const char *details_json);
int sb_add_hparams(sb_writer *w, const char *hparams_json, const char *metrics_json);
int sb_add_audio_pcm16(sb_writer *w, const char *tag, const int16_t *samples, size_t count, uint32_t channels,
                       uint32_t sample_rate, long long step);
int sb_flush(sb_writer *w);
/* Marks the session complete and releases the handle. */
int sb_writer_close(sb_writer *w);

#ifdef __cplusplus
}
#endif
#endif
