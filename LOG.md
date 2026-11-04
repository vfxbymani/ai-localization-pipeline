# Learning Log

## Day 1: PyTorch basics
**What I built:**
- Tensors, tensor math, and autograd
- A linear model that learned y = 3x + 2 (weight 3.00, bias 2.03)
- A small CNN on FashionMNIST: 88.7% test accuracy after 3 epochs

**What I learned:**
- (one sentence about the training loop: guess, measure error, find slopes, nudge)
- (one sentence about what a CNN does)

**What confused me:**
- (one thing that was unclear, for example why the loss printed went up and down)

**Tools:** Kaggle free GPU, GitHub

## Day 2: Transfer learning
**What I built:**
- Loaded a pretrained ResNet18, froze it, and replaced the last layer (1,026 trainable parameters out of 11.2M)
- Ants vs bees, 244 training images: 93.5% peak validation accuracy (89.5% at epoch 5)
- Same model trained from scratch: 61.4% after 5 epochs

**What I learned:**
- (one sentence on why the pretrained model did so much better)
- (one sentence on what freezing layers does)

**What confused me:**
- (one thing, for example why validation accuracy moved up and down between epochs)

**Tools:** Kaggle free GPU, torchvision, GitHub

## Day 3: Hugging Face basics
**What I built:**
- Sentiment pipeline, then rebuilt it by hand: tokenizer → IDs → model → logits → softmax → label
- English-to-Tamil translation with NLLB-200 (distilled 600M), using the model directly because the old `translation` pipeline was removed in transformers v5
- Scored the output with chrF and BLEU against my own Tamil references (chrF 49.3, BLEU 15.6 on 3 sentences)

**What I learned:**
- (one sentence on tokenizers and what the model actually sees)
- (one sentence on why a translation score can be low for a correct translation)

## Day 4: Speech-to-text with Whisper
**What I built:**
- Loaded a 13 s audio clip and found its sample rate was 22,050 Hz, so I resampled to the 16,000 Hz Whisper expects
- Transcribed it with whisper-small by hand (processor → log-mel input [1, 80, 3000] → generate → decode)
- Got segment timestamps and wrote a real .srt file with a function that formats times and wraps lines at 42 characters

**What I learned:**
- (one sentence on why audio must be resampled to 16 kHz)
- (one sentence on what Whisper's input looks like)

**What confused me:**
- (one thing, for example why the whole clip came back as a single 13 s cue)

**Next problem to solve:**
- Long cues break subtitle rules (max 2 lines, short duration), so the pipeline needs a segment-splitting step

**Tools:** Kaggle free GPU, Hugging Face transformers, soundfile, scipy, GitHub
**What confused me:**
- (one thing, for example why the sanity check scores a reference against itself)

**Tools:** Kaggle free GPU, Hugging Face transformers, sacrebleu, GitHub

## Day 5: Measuring speech accuracy (WER and CER)
**What I built:**
- Transcribed 8 English clips with whisper-small and scored them with WER: 0.10 raw, 0.07 after normalizing "mister" vs "mr" (186 correct, 12 substitutions, 2 deletions, 0 insertions)
- Transcribed a 21.6 s Tamil clip (sung) with `language="ta"`
- Compared whisper-small and whisper-medium on it: CER 0.286 → 0.19 (0.216 → 0.108 ignoring spaces), WER 1.0 for both

**What I learned:**
- (one sentence on why text normalization changes WER)
- (one sentence on why CER is fairer than WER for Tamil)

**What confused me:**
- (one thing, for example why WER stayed at 1.0 although the output got closer)

**Caveats:**
- Tamil test was one short sung clip, so it is a demo, not a benchmark

**Tools:** Kaggle free GPU, whisper-small/medium, jiwer, librosa, GitHub
**Post-processing step:**
- Snapped Whisper-medium output to known reference text using fuzzy matching (rapidfuzz ratio 89.2, threshold 75), with a flag-for-human-review fallback below the threshold
- Wrote tamil_clip.srt (one cue, 0 to 21.6 s)
- Note: the final text matches because the reference was supplied, not because Whisper got it right (Whisper-medium CER was 0.19)
