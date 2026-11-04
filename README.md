# AI Localization Pipeline

Video in, subtitles out. A small open-source pipeline built on free tools (Kaggle GPU, Hugging Face, ffmpeg).

## What it does
video → ffmpeg audio extraction (16 kHz mono) → 30 s chunks → Whisper → timestamp shifting → overlap fixing → subtitle cues (max 2 lines × 42 characters) → `.srt`

## Usage
```
python video_to_srt.py my_video.mp4 -o my_video.srt
python video_to_srt.py my_video.mp4 --model openai/whisper-medium --language ta
```
`--language` takes a Whisper language code (en, ta, hi, ...).

## Results (136.5 s test video, 12 read-aloud English clips)
| Run | WER |
|---|---|
| Pipeline (30 s fixed chunks) | 0.066 |
| Each clip transcribed on its own | 0.063 |
### Tamil test (21.6 s sung clip, same pipeline, `--language ta`)
| Model | CER vs correct reference | WER |
|---|---|---|
| whisper-small | 0.216 | 0.929 |
| whisper-medium | 0.135 | 1.000 |

- CER computed with spaces removed; WER is unreliable for Tamil because words are long and are split differently by the model.
- Scoring against a reference of the wrong length gives CER above 1.0, so reference and audio must cover the same material.
- One short sung clip: a demo, not a benchmark.

- 303 reference words; the 0.003 gap is about one word, so fixed chunking cost almost nothing here.
- 36 cues, 0 overlaps, longest line 42 characters, 2 cues ending on a function word.
- WER computed after normalizing case, punctuation and "mister"/"mr".

## Other experiments (notebooks)
- Day 1–2: PyTorch basics, CNN (88.7% on FashionMNIST), transfer learning with ResNet18 (peak 93.5% on 244 images vs ~61% from scratch)
- Day 3: Hugging Face basics, English→Tamil translation with NLLB, chrF/BLEU scoring
- Day 4–5: Whisper transcription, WER and CER; Tamil sung-audio test (CER 0.286 small, 0.19 medium)

## Known limitations
- Fixed 30 s cuts can split a word or sentence; silence-aware chunking is next.
- Cue timing inside a segment is estimated from text length, not word timestamps.
- The shortest cue is 0.84 s, under the usual 1 s guideline.
- Tested on clean read speech; noisy, sung or multi-speaker audio will score worse.

## Next
Translation to subtitle files in other languages, LLM-based line breaking and QA, Docker + a small API.


