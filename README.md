Batch transcription of video/audio files with speaker diarization, powered by
[whisperX](https://github.com/m-bain/whisperX). Produces one Markdown file per
input, with speech grouped by speaker and timestamped.

Designed for multi-speaker recordings (2-5 speakers), Russian by default, but
the language is configurable.

## Features

- Batch processing: pass individual files or a whole directory
- Automatic audio extraction from video via `ffmpeg`
- Word-level timestamp alignment
- Speaker diarization (who said what)
- Markdown output, grouped by consecutive speaker turns
- Per-file and total run timing printed to the console
- Skips files that already have a matching transcript in the output
  directory (use `--force` to re-transcribe them)
- Hugging Face token loaded from a `.env` file, an environment variable, or a
  CLI flag
- Continues processing remaining files if one file fails, and reports errors
  at the end

## Requirements

- macOS (tested on Apple Silicon) with Python 3.12+
- [Homebrew](https://brew.sh/) `ffmpeg`
- A Hugging Face account and access token (see [Setup](#setup))

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install whisperx python-dotenv
brew install ffmpeg
```

## Setup

### 1. Hugging Face token

The diarization model requires a Hugging Face token.

1. Create an account at [huggingface.co](https://huggingface.co)
2. Generate a token: Settings -> Access Tokens -> New token (a **read** token
   is enough)
3. Accept the license terms for the diarization model whisperX uses. On the
   first run, whisperX will log which model it is loading and, if access is
   missing, will point you to the exact page(s) to accept on huggingface.co.

### 2. Store the token in `.env`

Copy the example file and fill in your token:

```bash
cp .env.example .env
```

Edit `.env`:

```
HF_TOKEN=hf_your_token_here
```

The script automatically loads `.env` from the same directory as
`transcribe.py`. Add `.env` to `.gitignore` if this project is under version
control -- never commit your token.

Token resolution order: `--hf-token` flag > `.env` file > `HF_TOKEN`
environment variable.

## Usage

**Single file:**
```bash
python transcribe.py meeting.mp4
```

**All supported files in a directory:**
```bash
python transcribe.py --input-dir ./videos --output-dir ./transcripts
```

**Exact number of speakers (recommended when known -- improves accuracy):**
```bash
python transcribe.py meeting.mp4 --num-speakers 3
```

**Multiple specific files, keeping the extracted audio:**
```bash
python transcribe.py video1.mp4 video2.mp4 --keep-wav
```

**Re-transcribe a file even if its .md output already exists:**
```bash
python transcribe.py video2.mp4 --output-dir ./transcripts --force
```

### Options

| Flag | Default | Description |
|---|---|---|
| `files` | -- | Positional list of video/audio files |
| `--input-dir` | -- | Process all supported files in a directory |
| `--output-dir` | `./transcripts` | Where to write `.md` files |
| `--model` | `medium` | Whisper model: `tiny`/`base`/`small`/`medium`/`large-v3` |
| `--language` | `ru` | Language code |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--compute-type` | `int8` | `int8`/`float16`/`float32` |
| `--hf-token` | -- | Hugging Face token (overrides `.env` / env var) |
| `--num-speakers` | -- | Exact speaker count, if known |
| `--min-speakers` | `2` | Lower bound for diarization (used if `--num-speakers` is not set) |
| `--max-speakers` | `5` | Upper bound for diarization (used if `--num-speakers` is not set) |
| `--keep-wav` | off | Keep the extracted WAV file next to the output |
| `--force` | off | Re-transcribe even if a matching `.md` file already exists in `--output-dir` |

### Supported input formats

- Video: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`

## Output format

Each input file produces a `.md` file named after it, e.g. `meeting.mp4` ->
`meeting.md`:

```markdown
# Transcript: meeting.mp4

**SPEAKER_00** _[00:00:02.15]_

Hello everyone, let's get started...

**SPEAKER_01** _[00:00:15.40]_

Sure, I have a couple of questions...
```

Console output also reports per-file and total processing time, e.g.:

```
=== Done: 2 file(s) processed, 0 error(s) ===
  transcripts/meeting1.md  (12m 4s)
  transcripts/meeting2.md  (8m 41s)

Total time: 20m 45s
```

## Skipping already-transcribed files

Before processing each input file, the script checks whether a matching
`.md` file already exists in `--output-dir` (based on the filename stem,
e.g. `video2.mp4` -> `video2.md`). If it exists, the file is skipped and
listed separately in the final summary. Pass `--force` to disable this and
re-transcribe every input file regardless of existing output.

## Notes

- On Apple Silicon without CUDA, processing runs on CPU. A one-hour recording
  with diarization can take roughly 15-40 minutes depending on the model
  size.
- If diarization accuracy is poor with similar-sounding voices, providing
  `--num-speakers` explicitly (rather than a min/max range) usually helps.
- You may see non-fatal warnings in the console about `torchcodec` failing to
  load `libavutil`, or about a Lightning checkpoint being upgraded. Both are
  harmless -- whisperX automatically falls back to an alternate audio
  decoding path, and the checkpoint is simply re-converted on each run. They
  do not affect transcription quality.
