---
language: de
license: mit
library_name: mlx
pipeline_tag: text-generation
tags:
  - text-generation
  - mlx
  - nanogpt
  - german
  - kafka
---

# Posaidon

Posaidon is a small (~4.2M-parameter) GPT language model **built from scratch in
[Apple MLX](https://github.com/ml-explore/mlx)**, trained on a corpus of
public-domain German Franz Kafka texts. It is a learning project — a nanoGPT-style
transformer grown step by step into a Llama-style stack — not a general-purpose
assistant. It writes early-20th-century German prose in Kafka's bureaucratic,
unsettling register.

## Intended use

Educational and creative: studying a transformer end-to-end, and generating
Kafka-flavoured German text. It is **not** instruction-tuned, multilingual, or
factual, and should not be used for anything that needs correctness.

## How to use

The model uses a custom byte-level BPE tokenizer and architecture, so it ships with
a small inference script rather than `transformers` auto-classes:

```bash
git clone https://github.com/Tsuskov/Posaidon && cd Posaidon
python3 -m venv .venv && source .venv/bin/activate
pip install mlx tokenizers
# place the checkpoint files in out/  (model.safetensors, config.json, tokenizer.json)
python minigpt_mlx.py --generate --prompt "K. saß " --max_new_tokens 200
```

Architecture and tokenizer are read from `out/config.json`, so no model flags are
needed at generation time.

It also runs under [Ollama](https://ollama.com) via the included GGUF export:

```bash
python export_gguf.py --gguf posaidon.gguf
ollama create posaidon -f Modelfile && ollama run posaidon "K. "
```

## Architecture

Llama-like decoder-only transformer:

| | |
| --- | --- |
| parameters | 4.2M |
| layers / heads / d_model | 4 / 4 / 256 |
| normalization | RMSNorm (pre-norm) |
| positional encoding | RoPE (base 10000) |
| feed-forward | SwiGLU (hidden = 8/3·d_model) |
| attention bias | none (Llama-exact, so it exports cleanly to GGUF) |
| context length | 256 tokens |
| tokenizer | byte-level BPE, vocab 2048 (~3.38 chars/token) |

## Training data

A ~887K-character corpus assembled by `build_kafka_corpus.py` from Project Gutenberg
(public domain): *Der Prozess, Die Verwandlung, Ein Landarzt, Das Urteil, In der
Strafkolonie, Ein Hungerkünstler, Betrachtung, Der Heizer*. A 90/10 train/val split
gives ~237K training tokens.

## Training procedure

| | |
| --- | --- |
| hardware | Apple Silicon (16GB), MLX |
| optimizer | AdamW, lr 3e-4, weight decay 0.1 |
| regularization | dropout 0.2, early stopping (patience 10) |
| batch / steps | batch 32 × 256 tokens, early-stopped at 3,250 iters (~7 min) |
| checkpoint | best validation loss, not last step |

## Evaluation

Cross-entropy on the held-out 10% split (BPE tokens; not comparable to char-level
loss): **best val loss 4.42**.

The corpus is tiny (~237K tokens), so model size matters more than usual. A 27M-param
model drove training loss to ~0 while **validation loss climbed to ~8** — it memorized
the corpus and recited it verbatim. This 4.2M model, with dropout + weight decay +
early stopping, reaches a **lower** val loss (4.42 vs 4.54/4.79), keeps a much smaller
train/val gap, and actually *generates* rather than reciting. The trade-off is that
its samples read rougher — see Limitations.

## Limitations and biases

- **Tiny and narrow.** Trained only on early-1900s German literary prose; it knows
  nothing else and is not factual.
- **Invents words.** Because it generalizes rather than memorizes, it produces
  plausible-looking but non-existent German words ("Rausengrat", "Schwerkehrer") and
  loses grammatical coherence over long spans.
- **No safety tuning.** No alignment, instruction-following, or content filtering.
- **Reflects its source.** Vocabulary, gender roles, and worldview are those of its
  early-20th-century source texts.

## Sample

Prompt `"\n"`:

> Während seien. Dann hätte man ihn K. schon während des Dieners nichts auffallend
> sein Schreien. […] Der plötzliche war eine gewisse Unterbrechung des Zuspruches
> begreifens und mit Zach Farbe bedeckte er mit der Tasse ein wenig aufgefunden.

## Attribution

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat). Training texts from
[Project Gutenberg](https://www.gutenberg.org/) (public domain).
