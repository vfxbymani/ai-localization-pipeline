# Learning Log

## Day 1: PyTorch basics
**What I built:**
- Tensors, tensor math, and autograd
- A linear model that learned y = 3x + 2 (weight 3.00, bias 2.03)
- A small CNN on FashionMNIST: 88.7% test accuracy after 3 epochs




**Tools:** Kaggle free GPU, GitHub

## Day 2: Transfer learning
**What I built:**
- Loaded a pretrained ResNet18, froze it, and replaced the last layer (1,026 trainable parameters out of 11.2M)
- Ants vs bees, 244 training images: 93.5% peak validation accuracy (89.5% at epoch 5)
- Same model trained from scratch: 61.4% after 5 epochs



**Tools:** Kaggle free GPU, torchvision, GitHub

## Day 3: Hugging Face basics
**What I built:**
- Sentiment pipeline, then rebuilt it by hand: tokenizer → IDs → model → logits → softmax → label
- English-to-Tamil translation with NLLB-200 (distilled 600M), using the model directly because the old `translation` pipeline was removed in transformers v5
- Scored the output with chrF and BLEU against my own Tamil references (chrF 49.3, BLEU 15.6 on 3 sentences)



## Day 4: Speech-to-text with Whisper
**What I built:**
- Loaded a 13 s audio clip and found its sample rate was 22,050 Hz, so I resampled to the 16,000 Hz Whisper expects
- Transcribed it with whisper-small by hand (processor → log-mel input [1, 80, 3000] → generate → decode)
- Got segment timestamps and wrote a real .srt file with a function that formats times and wraps lines at 42 characters



**Next problem to solve:**
- Long cues break subtitle rules (max 2 lines, short duration), so the pipeline needs a segment-splitting step

## Day 5: Measuring speech accuracy (WER and CER)
**What I built:**
- Transcribed 8 English clips with whisper-small and scored them with WER: 0.10 raw, 0.07 after normalizing "mister" vs "mr" (186 correct, 12 substitutions, 2 deletions, 0 insertions)
- Transcribed a 21.6 s Tamil clip (sung) with `language="ta"`
- Compared whisper-small and whisper-medium on it: CER 0.286 → 0.19 (0.216 → 0.108 ignoring spaces), WER 1.0 for both





**Caveats:**
- Tamil test was one short sung clip, so it is a demo, not a benchmark

**Tools:** Kaggle free GPU, whisper-small/medium, jiwer, librosa, GitHub

**Post-processing step:**
- Snapped Whisper-medium output to known reference text using fuzzy matching (rapidfuzz ratio 89.2, threshold 75), with a flag-for-human-review fallback below the threshold
- Wrote tamil_clip.srt (one cue, 0 to 21.6 s)
- Note: the final text matches because the reference was supplied, not because Whisper got it right (Whisper-medium CER was 0.19)

- ## Day 8: Translating subtitles and keeping timings
**What I built:**
- A parser that reads an .srt back into cues, and a translator that swaps only the text so timings stay identical
- English → Tamil with NLLB-200 (600M): 36 cues, Tamil 1.13× longer than English
- Two fixes for cut sentences: group cues into 14 whole sentences before translating (step 3), then cut at clause boundaries (step 4)

**Measured:**
- Step 3: 36 cues, 23/36 end at punctuation, longest 6.68 s, shortest 0.84 s
- Step 4: 32 cues, 26/32 end at punctuation, longest 7.25 s, shortest 0.69 s (both guideline breaks reported)

**Still wrong:**
- Name written two ways, a missing word ("similes"), text drifting from audio because Tamil word order differs. Next: glossary + LLM post-edit, and word-level timestamps.

## Day 9: Glossary and LLM post-edit
**What I built:**
- Gemini API set up with the key in Kaggle Secrets (never in the notebook)
- A glossary (names and terms with fixed Tamil forms) and an editing prompt
- Post-edited 14 whole sentences with gemini-3.5-flash-lite and gemini-3.5-flash; both followed the glossary 13/13
- A review report that flags sentences for a human: 3 of 14 flagged

**Measured (Day 8 NLLB-only vs Day 9 post-edit):**
- Cues with Latin letters: 2 → 0
- Cues over 84 characters: 2 → 0
- Longest cue: 7.25 s → 6.66 s
- Ending at punctuation: unchanged at 81%; shortest cue still under 1 s

**What I learned:**
- A glossary check done by exact string match fails for Tamil because suffixes change the end of a word, so I matched the stem
- A model flagged for "big change" can be an improvement; a flag means "look here", not "wrong"
- The "wall" error came from Whisper's chunk cut on Day 6; the LLM translated it faithfully, so source errors must be fixed at the source



**Still wrong:**
- Sentence 4 (source error), cue timing drift because Tamil word order differs, two spellings of "Ithaca" not yet in the glossary
