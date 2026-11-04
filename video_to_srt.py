"""Video -> .srt subtitles: ffmpeg audio extraction, 30 s chunking, Whisper, cue building."""
import argparse
import os
import subprocess
import tempfile
import textwrap

import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SR = 16000
CHUNK_SECONDS = 30
MIN_CHUNK_SECONDS = 1.0
STOPWORDS = {"the", "a", "an", "of", "to", "and", "its", "his", "in", "that", "with", "for", "at", "by"}


def extract_audio(video_path, wav_path, sr=SR):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vn", "-ac", "1", "-ar", str(sr), wav_path],
        check=True,
    )


def make_chunks(audio, sr, chunk_seconds=CHUNK_SECONDS):
    size = int(chunk_seconds * sr)
    chunks = [(start / sr, audio[start:start + size]) for start in range(0, len(audio), size)]
    return [(t, a) for t, a in chunks if len(a) >= MIN_CHUNK_SECONDS * sr]


def transcribe(chunks, model_name, language, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    segments = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = processor([a for _, a in batch], sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(inputs.input_features, language=language,
                                 task="transcribe", return_timestamps=True)
        for (chunk_start, arr), seq in zip(batch, ids):
            chunk_len = len(arr) / SR
            decoded = processor.tokenizer.decode(seq, output_offsets=True)
            for seg in decoded["offsets"]:
                s, e = seg["timestamp"]
                if e is None:
                    e = chunk_len
                text = seg["text"].strip()
                if text:
                    segments.append({"start": chunk_start + s, "end": chunk_start + e, "text": text})
    return segments


def fix_overlaps(segments):
    segs = [dict(s) for s in sorted(segments, key=lambda x: x["start"])]
    for a, b in zip(segs, segs[1:]):
        if a["end"] > b["start"]:
            a["end"] = b["start"]
    return segs


def split_text(text, n):
    words = text.split()
    if n <= 1 or len(words) < n:
        return [text]
    ends, pos = [], 0
    for w in words:
        pos += len(w) + 1
        ends.append(pos - 1)
    cuts, prev = [], -1
    for k in range(1, n):
        ideal = len(text) * k / n
        best, best_score = None, None
        for i in range(prev + 1, len(words) - (n - k)):
            score = abs(ends[i] - ideal) - (10 if words[i][-1] in ",.;:?!" else 0)
            if best is None or score < best_score:
                best, best_score = i, score
        cuts.append(best)
        prev = best
    parts, start = [], 0
    for c in cuts:
        parts.append(" ".join(words[start:c + 1]))
        start = c + 1
    parts.append(" ".join(words[start:]))
    return parts


def fit_lines(p, width=42):
    return [p] if len(p) <= width else split_text(p, 2)


def make_cues(segs, width=42):
    out = []
    for seg in segs:
        n_words = len(seg["text"].split())
        n = 1
        while True:
            parts = split_text(seg["text"], n)
            lines = [fit_lines(p, width) for p in parts]
            ok = all(len(l) <= width for ls in lines for l in ls)
            if ok or n >= n_words:
                break
            n += 1
        total = sum(len(p) for p in parts)
        t, dur = seg["start"], seg["end"] - seg["start"]
        for p, ls in zip(parts, lines):
            end = t + dur * len(p) / total
            out.append({"start": t, "end": end, "text": "\n".join(ls)})
            t = end
    return out


def fmt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues):
    return "\n".join(
        f"{i}\n{fmt_time(c['start'])} --> {fmt_time(c['end'])}\n{c['text']}\n"
        for i, c in enumerate(cues, start=1)
    )


def main():
    ap = argparse.ArgumentParser(description="Make an .srt subtitle file from a video.")
    ap.add_argument("video")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--language", default="en", help="Whisper language code, e.g. en, ta, hi")
    args = ap.parse_args()

    out_path = args.output or os.path.splitext(args.video)[0] + ".srt"
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "audio.wav")
        extract_audio(args.video, wav)
        audio, rate = sf.read(wav, dtype="float32")

    chunks = make_chunks(audio, rate)
    segments = transcribe(chunks, args.model, args.language)
    cues = make_cues(fix_overlaps(segments))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(to_srt(cues))

    dangling = sum(c["text"].split()[-1].lower() in STOPWORDS for c in cues)
    print(f"chunks: {len(chunks)} | segments: {len(segments)} | cues: {len(cues)} | dangling: {dangling}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
