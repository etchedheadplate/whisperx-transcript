#!/usr/bin/env python3
"""
transcribe.py -- Batch video/audio transcription with speaker diarization via whisperX.

Usage:
    python transcribe.py video1.mp4 video2.mp4 ...
    python transcribe.py --input-dir ./videos --output-dir ./transcripts
    python transcribe.py video.mp4 --num-speakers 3
    python transcribe.py video.mp4 --min-speakers 2 --max-speakers 5

Requirements:
    pip install whisperx python-dotenv
    brew install ffmpeg

    A Hugging Face token is required for the diarization model.
    Provide it via a .env file next to this script (HF_TOKEN=...),
    an HF_TOKEN environment variable, or the --hf-token flag.
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Keep NLTK's downloaded data inside the project folder instead of ~/nltk_data.
# Must be set before whisperx (and its dependencies) import nltk.
os.environ.setdefault("NLTK_DATA", str(Path(__file__).resolve().parent / ".nltk_data"))

try:
    import whisperx
except ImportError:
    print("Error: whisperx is not installed. Run: pip install whisperx", file=sys.stderr)
    sys.exit(1)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio(input_path: Path, tmp_dir: str) -> Path:
    """Extract audio to 16kHz mono WAV via ffmpeg (required input format for whisperX)."""
    out_path = Path(tmp_dir) / (input_path.stem + ".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to process {input_path}:\n{result.stderr}")
    return out_path


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss for display in the transcript."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:05.2f}"


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a short human-readable string (e.g. 4m 12s)."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def build_markdown(result: dict, source_name: str) -> str:
    """Build the markdown content: a title plus turns grouped by consecutive speaker."""
    lines = [f"# Transcript: {source_name}", ""]

    segments = result.get("segments", [])
    if not segments:
        lines.append("_No speech detected in this file._")
        return "\n".join(lines)

    current_speaker = None
    buffer = []
    start_ts = None

    def flush():
        if buffer:
            text = " ".join(buffer).strip()
            lines.append(f"**{current_speaker}** _[{start_ts}]_")
            lines.append("")
            lines.append(text)
            lines.append("")

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            flush()
            buffer = []
            current_speaker = speaker
            start_ts = format_timestamp(seg["start"])
        buffer.append(text)

    flush()
    return "\n".join(lines)


def process_file(
    input_path: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
    hf_token: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    keep_wav: bool = False,
) -> tuple[Path, float]:
    """Run the full pipeline (transcribe -> align -> diarize -> write markdown) for one file.

    Returns the output path and the elapsed time in seconds.
    """
    print(f"\n=== Processing: {input_path.name} ===")
    file_start = time.monotonic()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Prepare audio
        if input_path.suffix.lower() == ".wav":
            audio_path = input_path
        else:
            print("  -> extracting audio (ffmpeg)...")
            audio_path = extract_audio(input_path, tmp_dir)

        audio = whisperx.load_audio(str(audio_path))

        # 2. Transcription
        print(f"  -> transcribing (model: {model_name})...")
        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
        result = model.transcribe(audio, batch_size=8)
        del model
        gc.collect()

        # 3. Word-level alignment
        print("  -> aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        del model_a
        gc.collect()

        # 4. Speaker diarization
        print("  -> diarizing speakers...")
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)  # type: ignore[reportAttributeAccessIssue]
        except TypeError:
            # Newer whisperx/pyannote versions renamed use_auth_token -> token
            diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)  # type: ignore[reportAttributeAccessIssue]
        diarize_kwargs = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        else:
            diarize_kwargs["min_speakers"] = min_speakers or 2
            diarize_kwargs["max_speakers"] = max_speakers or 5
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        del diarize_model
        gc.collect()

        # 5. Assign speakers to words/segments
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if keep_wav and input_path.suffix.lower() != ".wav":
            saved_path = output_dir / (input_path.stem + ".wav")
            os.replace(audio_path, saved_path)
            print(f"  -> audio saved: {saved_path}")

    # 6. Write markdown output
    output_dir.mkdir(parents=True, exist_ok=True)
    md_content = build_markdown(result, input_path.name)
    out_path = output_dir / (input_path.stem + ".md")
    out_path.write_text(md_content, encoding="utf-8")

    elapsed = time.monotonic() - file_start
    print(f"  -> done: {out_path} ({format_duration(elapsed)})")
    return out_path, elapsed


def collect_input_files(args) -> list[Path]:
    """Gather the list of input files from positional args and/or --input-dir."""
    files = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {input_dir} does not exist or is not a directory", file=sys.stderr)
            sys.exit(1)
        for p in sorted(input_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
        if not files:
            print(f"No supported files found in {input_dir} ({', '.join(SUPPORTED_EXTENSIONS)})", file=sys.stderr)
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"File not found, skipped: {p}", file=sys.stderr)
            continue
        files.append(p)
    return files


def load_hf_token(cli_token: str | None) -> str | None:
    """Resolve the Hugging Face token, in priority order: --hf-token, .env file, environment."""
    if cli_token:
        return cli_token

    if load_dotenv is not None:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

    return os.environ.get("HF_TOKEN")


def main():
    parser = argparse.ArgumentParser(
        description="Batch video/audio transcription with speaker diarization (whisperX -> Markdown)"
    )
    parser.add_argument("files", nargs="*", help="Paths to video/audio files")
    parser.add_argument("--input-dir", help="Process all supported files from a directory")
    parser.add_argument("--output-dir", default="./transcripts", help="Output directory for .md files (default: ./transcripts)")
    parser.add_argument("--model", default="medium", help="Whisper model: tiny/base/small/medium/large-v3 (default: medium)")
    parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--compute-type", default="int8", help="int8/float16/float32 (default: int8)")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (overrides .env and HF_TOKEN env var)")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers, if known")
    parser.add_argument("--min-speakers", type=int, default=2, help="Minimum number of speakers (default: 2)")
    parser.add_argument("--max-speakers", type=int, default=5, help="Maximum number of speakers (default: 5)")
    parser.add_argument("--keep-wav", action="store_true", help="Keep the extracted WAV file alongside the output")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if an output .md file already exists")

    args = parser.parse_args()

    if not args.files and not args.input_dir:
        parser.error("Provide files directly or use --input-dir")

    hf_token = load_hf_token(args.hf_token)
    if not hf_token:
        print(
            "Error: no Hugging Face token found.\n"
            "Provide one via --hf-token YOUR_TOKEN, a .env file (HF_TOKEN=...) next to this script,\n"
            "or the HF_TOKEN environment variable. The token is required to access the pyannote\n"
            "diarization model (make sure you've accepted its license terms on huggingface.co).",
            file=sys.stderr,
        )
        sys.exit(1)

    input_files = collect_input_files(args)
    if not input_files:
        print("No files to process.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    results = []
    errors = []
    skipped = []
    run_start = time.monotonic()

    for f in input_files:
        existing_md = output_dir / (f.stem + ".md")
        if existing_md.exists() and not args.force:
            print(f"\n=== Skipping: {f.name} (transcript already exists: {existing_md.name}) ===")
            skipped.append(f)
            continue
        try:
            out_path, elapsed = process_file(
                input_path=f,
                output_dir=output_dir,
                model_name=args.model,
                device=args.device,
                compute_type=args.compute_type,
                language=args.language,
                hf_token=hf_token,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                keep_wav=args.keep_wav,
            )
            results.append((out_path, elapsed))
        except Exception as e:
            print(f"  !! Error processing {f.name}: {e}", file=sys.stderr)
            errors.append(f)

    total_elapsed = time.monotonic() - run_start

    print(
        f"\n=== Done: {len(results)} file(s) processed, "
        f"{len(skipped)} skipped, {len(errors)} error(s) ==="
    )
    for r, elapsed in results:
        print(f"  {r}  ({format_duration(elapsed)})")
    if skipped:
        print("Skipped (transcript already exists):")
        for s in skipped:
            print(f"  {s}")
    if errors:
        print("Files with errors:")
        for e in errors:
            print(f"  {e}")
    print(f"\nTotal time: {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
