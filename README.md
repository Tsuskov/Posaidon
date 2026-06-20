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

The defaults are a Llama-like stack (RMSNorm + RoPE + SwiGLU). Each piece can be flipped back to the GPT-2-style baseline to measure its effect:

| Flag | modern (default) → baseline | what it is |
| --- | --- | --- |
| `--norm` | `rmsnorm` → `layernorm` | cheaper normalization (no mean/bias) |
| `--pos` | `rope` → `learned` | rotary position encoding inside attention |
| `--mlp` | `swiglu` → `gelu` | gated feed-forward (hidden width kept param-matched) |

### Scaled training

The Kafka corpus is tiny (~263K BPE tokens), so a big model just memorizes it: a
27M-param run drives train loss to ~0 while val loss *climbs* to ~8. A ~4M-param
model with light regularization generalizes better (best val 4.42 vs 4.54/4.79),
trains in ~7 min, and actually *generates* text instead of reciting it:

```bash
python minigpt_mlx.py --tokenizer bpe --vocab_size 2048 \
  --n_layer 4 --n_head 4 --n_embd 256 --block_size 256 --batch_size 32 \
  --max_iters 10000 --eval_interval 250 \
  --dropout 0.2 --weight_decay 0.1 --early_stop_patience 10 --no_attn_bias
```

`--dropout`/`--weight_decay` fight overfitting; `--early_stop_patience N` stops once
val loss hasn't improved for N evals. The checkpoint saved is always the best-val
one, not the last. `--no_attn_bias` drops the attention biases (Llama-exact, no
measurable loss cost) so the result exports cleanly to GGUF — see below.

### Generate from a checkpoint

`--generate` skips training and samples from a saved checkpoint. The architecture
and tokenizer are read from `<out_dir>/config.json`, so you don't repeat the model
flags — just point at the directory and give a prompt:

```bash
python minigpt_mlx.py --generate --prompt "K. saß " --max_new_tokens 200
python minigpt_mlx.py --generate --out_dir out_27m_reg --prompt "Der Prozess "
```

Generated text goes to stdout (the load info line to stderr), so it pipes cleanly.

### Publish: Hugging Face + Ollama

`publish_hf.py` stages the checkpoint into a Hugging Face repo layout (weights +
config + tokenizer + `MODEL_CARD.md` as the README + the loader) and can push it:

```bash
python publish_hf.py                              # stage into hf_repo/
python publish_hf.py --push --repo you/Posaidon   # needs `huggingface-cli login`
```

For [Ollama](https://ollama.com), `export_gguf.py` converts the checkpoint to GGUF.
It requires a **biasless** model (llama.cpp's `llama` arch has no attention bias),
which is what the scaled-training recipe above produces with `--no_attn_bias`:

```bash
python export_gguf.py --gguf posaidon.gguf
ollama create posaidon -f Modelfile && ollama run posaidon "K. "
```

The exporter permutes the q/k weights from MLX's RoPE layout to GGUF's, so greedy
(`temperature 0`) output is token-identical to `--generate` until floating-point
drift between the two engines diverges it.

Writes a checkpoint (`model.safetensors` + `config.json` + `tokenizer.json`), a `loss_curve.png`, and a `report_card.md` to `--out_dir` (default `out/`).

### Dataset

The corpus gives Posaidon its voice: `build_kafka_corpus.py` assembles German Kafka works (Der Prozess, Die Verwandlung, …) from Project Gutenberg. Swap in any `input.txt` to retrain on a different style. For the original toy run, use tinyshakespeare instead:

```bash
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

## Attribution

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat).
