---
language: en
license: mit
library_name: mlx
pipeline_tag: text-generation
tags:
  - text-generation
  - mlx
  - nanogpt
  - greek-mythology
---

# Posaidon

Posaidon is a small (~15.7M-parameter) GPT language model **built from scratch in
[Apple MLX](https://github.com/ml-explore/mlx)**, trained on a corpus of
public-domain English retellings and prose translations of Greek mythology and the
Homeric epics. It is a learning project — a nanoGPT-style transformer grown step by
step into a Llama-style stack — not a general-purpose assistant. The name fits the
data: Posaidon (Poseidon) trained on the sea-god's own myths. It writes
19th/early-20th-century English myth-prose in the register of its sources.

## Intended use

Educational and creative: studying a transformer end-to-end, and generating
myth-flavoured English text. It is **not** instruction-tuned, factual, or safe for
anything that needs correctness.

## How to use

The model uses a custom byte-level BPE tokenizer and architecture, so it ships with
a small inference script rather than `transformers` auto-classes:

```bash
git clone https://github.com/Tsuskov/Posaidon && cd Posaidon
python3 -m venv .venv && source .venv/bin/activate
pip install mlx tokenizers
# place the checkpoint files in out/  (model.safetensors, config.json, tokenizer.json)
python minigpt_mlx.py --generate --prompt "Zeus " --max_new_tokens 200
```

Architecture and tokenizer are read from `out/config.json`, so no model flags are
needed at generation time.

It also runs under [Ollama](https://ollama.com) via the included GGUF export:

```bash
python export_gguf.py --gguf posaidon.gguf
ollama create posaidon -f Modelfile && ollama run posaidon "Zeus "
```

## Architecture

Llama-like decoder-only transformer:

| | |
| --- | --- |
| parameters | 15.74M |
| layers / heads / d_model | 8 / 8 / 384 |
| normalization | RMSNorm (pre-norm) |
| positional encoding | RoPE (base 10000) |
| feed-forward | SwiGLU (hidden = 8/3·d_model) |
| attention bias | none (Llama-exact, so it exports cleanly to GGUF) |
| context length | 256 tokens |
| tokenizer | byte-level BPE, vocab 2048 (~3.12 chars/token) |

## Training data

A ~4.93M-character corpus assembled by `build_greek_corpus.py` from Project Gutenberg
(public domain): Bulfinch's *The Age of Fable*; Berens' *Myths and Legends of Greece
and Rome*; Kingsley's *The Heroes*; Hawthorne's *Tanglewood Tales*; Homer's *Iliad*
(Butler) and *Odyssey* (Butler, and Butcher & Lang); and *Hesiod, the Homeric Hymns,
and Homerica*. A 90/10 train/val split gives ~1.42M training tokens (1.58M total).

## Training procedure

| | |
| --- | --- |
| hardware | Apple Silicon (16GB), MLX |
| optimizer | AdamW, lr 3e-4, weight decay 0.1 |
| regularization | dropout 0.1, early stopping (patience 10) |
| batch / steps | batch 32 × 256 tokens, early-stopped at 4,000 iters (best at 1,500) |
| throughput | 17,499 tokens/s, peak memory 5.25 GB |
| checkpoint | best validation loss, not last step |

## Evaluation

Cross-entropy on the held-out 10% split (BPE tokens; not comparable to char-level
loss): **best val loss 3.97**.

Validation bottomed out at iter 1,500 and then rose while training loss kept falling
(to ~1.77 by iter 4,000) — the model begins to overfit the corpus, and early stopping
keeps the best-val checkpoint rather than the last step.

## Limitations and biases

- **Small and narrow.** Trained only on 19th/early-20th-century English myth-prose; it
  knows nothing else and is not factual.
- **Invents and conflates.** It produces fluent-looking sentences that mangle names and
  facts (mixing up gods, epics, and titles) and loses coherence over long spans.
- **No safety tuning.** No alignment, instruction-following, or content filtering.
- **Reflects its source.** Vocabulary, gender roles, and worldview are those of its
  19th/early-20th-century source texts.

## Sample

Prompt `"\n"`:

> called the Trojans in order, and declared heartily to prosper with some purpose.
> Pentheus and Uranus and Gaea were renowned for their great fruit, that Cronion had
> putting home like Artemis to enter homage. […] the “Odyssey” was a goddess bestowed
> with the honour of the Argonauts as the _Iliad_ and “Odyssey” to this description is
> describing the _Works and Days_, by Dodona and CAICTOLIA.

## Attribution

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and
[nanochat](https://github.com/karpathy/nanochat). Training texts from
[Project Gutenberg](https://www.gutenberg.org/) (public domain).
