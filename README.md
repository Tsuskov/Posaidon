# Posaidon

Posaidon is a small GPT language model built from scratch in [MLX](https://github.com/ml-explore/mlx), grown step by step from a char-level toy into a Llama-style model trained on a corpus of its own.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx
python build_kafka_corpus.py   # downloads public-domain Kafka into input.txt
python minigpt_mlx.py
```

Trains a tiny transformer on `input.txt` and prints a sample of generated text at the end. Tweak the model with flags, e.g. `--n_layer 8 --n_head 8`.

By default it tokenizes per character. Pass `--tokenizer bpe` (needs `pip install tokenizers`) to learn a byte-level BPE vocab from the corpus instead — common letter sequences become single tokens, so a fixed `--block_size` covers ~3-4× more text and the samples form real words:

```bash
python minigpt_mlx.py --tokenizer bpe --vocab_size 2048
```

### Architecture

The defaults are a GPT-2-style baseline. Three flags modernize it toward Llama, and can be toggled one at a time to measure each effect:

```bash
python minigpt_mlx.py --norm rmsnorm --pos rope --mlp swiglu   # Llama-like
```

| Flag | baseline → modern | effect |
| --- | --- | --- |
| `--norm` | `layernorm` → `rmsnorm` | cheaper normalization (no mean/bias) |
| `--pos` | `learned` → `rope` | rotary position encoding inside attention |
| `--mlp` | `gelu` → `swiglu` | gated feed-forward (hidden width kept param-matched) |

### Dataset

The corpus gives Posaidon its voice: `build_kafka_corpus.py` assembles German Kafka works (Der Prozess, Die Verwandlung, …) from Project Gutenberg. Swap in any `input.txt` to retrain on a different style. For the original toy run, use tinyshakespeare instead:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

## Attribution

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat).
