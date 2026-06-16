"""minigpt_mlx.py — a tiny char-level GPT in MLX.

Posaidon, Phase 0. A nanoGPT-style transformer that trains on a plain-text
input.txt (tinyshakespeare by default) and prints a sample of generated text
at the end.

The five building blocks (the focus of Phase 1):
  1. Embeddings        — token + position lookup tables
  2. Self-attention    — causal, multi-head (see the mask in CausalSelfAttention)
  3. MLP               — per-token feed-forward
  4. Block             — attention + MLP wrapped in residuals + LayerNorm
  5. GPT               — the full stack + training loop + sampling

Run:  python minigpt_mlx.py
"""

import argparse
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ----------------------------------------------------------------------------
# Config — Phase 1 experiments live here (halve/double n_layer, change n_head).
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="tiny char-level GPT in MLX")
    p.add_argument("--data", default="input.txt")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char",
                   help="char = one token per character; bpe = learned subwords")
    p.add_argument("--vocab_size", type=int, default=2048,
                   help="target vocab for the bpe tokenizer (ignored for char)")
    p.add_argument("--block_size", type=int, default=64, help="context length (in tokens)")
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    # Architecture — defaults are the Llama-like stack (Phase 4). Flip any one
    # back to the GPT-2-style baseline with --norm layernorm / --pos learned / --mlp gelu.
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    p.add_argument("--pos", choices=["learned", "rope"], default="rope",
                   help="learned position embeddings vs rotary (RoPE)")
    p.add_argument("--mlp", choices=["gelu", "swiglu"], default="swiglu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", default="out",
                   help="where to write checkpoints, loss curve and report card")
    return p.parse_args()


# ----------------------------------------------------------------------------
# 1+2+3+4. The model
# ----------------------------------------------------------------------------
def make_norm(cfg, dims):
    """LayerNorm (GPT-2) vs RMSNorm (Llama). RMSNorm drops the mean-centering
    and the bias — it just rescales by the root-mean-square, which is cheaper
    and works as well in practice."""
    return nn.RMSNorm(dims) if cfg.norm == "rmsnorm" else nn.LayerNorm(dims)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token may only attend to itself
    and earlier tokens. The "causal" part is the upper-triangular mask of
    -inf added to the attention scores before softmax, so future positions
    get zero weight."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # q, k, v in one matmul
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        head_size = cfg.n_embd // cfg.n_head
        # RoPE rotates q/k by position instead of adding a position embedding.
        self.rope = nn.RoPE(head_size, base=10000) if cfg.pos == "rope" else None

    def __call__(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        q, k, v = mx.split(self.c_attn(x), 3, axis=-1)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.reshape(B, T, self.n_head, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, hs).transpose(0, 2, 1, 3)

        if self.rope is not None:  # position enters here, not at the embeddings
            q, k = self.rope(q), self.rope(k)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
        mask = mx.triu(mx.full((T, T), -mx.inf), k=1)  # future positions -> -inf
        att = mx.softmax(att + mask, axis=-1)
        y = att @ v  # (B, nh, T, head_size)

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # reassemble heads
        return self.c_proj(y)


class MLP(nn.Module):
    """Position-wise feed-forward: expand 4x, GELU, project back (GPT-2 style)."""

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def __call__(self, x):
        return self.c_proj(nn.gelu(self.c_fc(x)))


class SwiGLU(nn.Module):
    """Gated feed-forward (Llama style): out = w2( silu(w1 x) * w3 x ).
    Hidden width is 8/3*n_embd so the parameter count matches the GELU MLP,
    making the loss comparison about the architecture, not the size."""

    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 / 3 * cfg.n_embd)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


def make_mlp(cfg):
    return SwiGLU(cfg) if cfg.mlp == "swiglu" else MLP(cfg)


class Block(nn.Module):
    """One transformer block: pre-norm attention + MLP, each with a residual."""

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = make_norm(cfg, cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg, cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.block_size = cfg.block_size
        self.use_wpe = cfg.pos == "learned"          # RoPE adds position inside attention
        self.wte = nn.Embedding(vocab_size, cfg.n_embd)          # token embeddings
        if self.use_wpe:
            self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # position embeddings
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = make_norm(cfg, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size, bias=False)

    def __call__(self, idx):
        B, T = idx.shape
        x = self.wte(idx)                            # (B, T, n_embd)
        if self.use_wpe:
            x = x + self.wpe(mx.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))  # (B, T, vocab_size)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.block_size:])[:, -1, :]  # last step's logits
            next_idx = mx.random.categorical(logits)            # sample from softmax
            idx = mx.concatenate([idx, next_idx[:, None]], axis=1)
            mx.eval(idx)
        return idx


# ----------------------------------------------------------------------------
# Tokenizer — turn text into a sequence of integer ids (and back).
#   char: one id per character (the Phase 0-2 toy).
#   bpe:  byte-level BPE learned from the corpus, so common letter sequences
#         ("ung", "der", " Kommandant") become single ids -> fewer tokens,
#         so a fixed block_size covers far more text.
# ----------------------------------------------------------------------------
def build_tokenizer(text, kind, vocab_size, out_dir=None):
    if kind == "char":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda ids: "".join(itos[i] for i in ids)
        return encode, decode, len(chars)

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # any UTF-8
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    tok.train_from_iterator([text], trainer)  # learn merges from our corpus
    if out_dir is not None:
        tok.save(os.path.join(out_dir, "tokenizer.json"))  # so we can decode later
    encode = lambda s: tok.encode(s).ids
    decode = lambda ids: tok.decode(ids)
    return encode, decode, tok.get_vocab_size()


# ----------------------------------------------------------------------------
# Phase 5 — checkpoints, loss curve, report card.
# ----------------------------------------------------------------------------
def save_checkpoint(model, cfg, vocab_size, out_dir):
    """Save weights + the config needed to rebuild and reload the model."""
    model.save_weights(os.path.join(out_dir, "model.safetensors"))
    meta = {**vars(cfg), "vocab_size": vocab_size}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def plot_losses(history, out_dir):
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just write a file
    import matplotlib.pyplot as plt

    its = [h[0] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(its, [h[1] for h in history], label="train")
    plt.plot(its, [h[2] for h in history], label="val")
    plt.xlabel("iteration"); plt.ylabel("cross-entropy loss")
    plt.title("Posaidon training"); plt.legend(); plt.grid(alpha=0.3)
    path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    return path


def write_report(out_dir, cfg, vocab_size, n_params, history, peak_gb,
                 toks_per_s, elapsed, chars_per_tok, sample):
    """A nanochat-style report card: config + numbers + a sample."""
    best_val = min(h[2] for h in history)
    lines = [
        "# Posaidon — report card", "",
        f"- **architecture**: norm={cfg.norm}, pos={cfg.pos}, mlp={cfg.mlp} (Llama-like)",
        f"- **size**: {n_params/1e6:.1f}M params "
        f"(n_layer={cfg.n_layer}, n_head={cfg.n_head}, n_embd={cfg.n_embd})",
        f"- **tokenizer**: {cfg.tokenizer}, vocab={vocab_size} "
        f"({chars_per_tok:.2f} chars/token, block_size={cfg.block_size})",
        f"- **training**: {cfg.max_iters:,} iters, batch={cfg.batch_size}, "
        f"lr={cfg.learning_rate}, {elapsed/60:.1f} min",
        f"- **hardware**: {peak_gb:.2f} GB peak, {toks_per_s:,.0f} tokens/s",
        f"- **loss**: final train {history[-1][1]:.4f} / val {history[-1][2]:.4f}, "
        f"best val {best_val:.4f}", "",
        "![loss curve](loss_curve.png)", "",
        "## sample", "", "```", sample, "```", "",
    ]
    with open(os.path.join(out_dir, "report_card.md"), "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------------
# 5. Data + training loop
# ----------------------------------------------------------------------------
def main():
    cfg = get_args()
    mx.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    with open(cfg.data, "r", encoding="utf-8") as f:
        text = f.read()
    encode, decode, vocab_size = build_tokenizer(
        text, cfg.tokenizer, cfg.vocab_size, cfg.out_dir)

    data = mx.array(encode(text))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    chars_per_tok = len(text) / len(data)
    print(f"tokenizer={cfg.tokenizer}  vocab_size={vocab_size}  "
          f"{len(text):,d} chars -> {len(data):,d} tokens  "
          f"({chars_per_tok:.2f} chars/token)")
    print(f"block_size={cfg.block_size} tokens  =>  ~{cfg.block_size * chars_per_tok:.0f} "
          f"chars of context per window")
    print(f"train={len(train_data)}  val={len(val_data)}")

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = mx.random.randint(0, len(d) - cfg.block_size, (cfg.batch_size,))
        x = mx.stack([d[i:i + cfg.block_size] for i in ix.tolist()])
        y = mx.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix.tolist()])
        return x, y

    model = GPT(vocab_size, cfg)
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"arch: norm={cfg.norm} pos={cfg.pos} mlp={cfg.mlp}  |  params: {n_params/1e6:.2f}M")

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="mean"
        )

    def eval_loss(split, batches=20):
        return mx.mean(mx.array([loss_fn(model, *get_batch(split)) for _ in range(batches)])).item()

    optimizer = optim.AdamW(learning_rate=cfg.learning_rate)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    history = []  # (iter, train_loss, val_loss) for the loss curve
    mx.reset_peak_memory()
    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            tr, va = eval_loss("train"), eval_loss("val")
            history.append((it, tr, va))
            print(f"iter {it:5d} | train {tr:.4f} | val {va:.4f} | {time.time()-t0:.1f}s")
            save_checkpoint(model, cfg, vocab_size, cfg.out_dir)  # crash-safe
        x, y = get_batch("train")
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    elapsed = time.time() - t0
    toks_per_s = cfg.max_iters * cfg.batch_size * cfg.block_size / elapsed
    peak_gb = mx.get_peak_memory() / 1e9
    save_checkpoint(model, cfg, vocab_size, cfg.out_dir)
    print(f"peak memory: {peak_gb:.2f} GB  |  throughput: {toks_per_s:,.0f} tokens/s")

    print("\n--- sample ---")
    start = mx.array([encode("\n")])
    sample = decode(model.generate(start, max_new_tokens=300)[0].tolist())
    print(sample)

    plot_losses(history, cfg.out_dir)
    write_report(cfg.out_dir, cfg, vocab_size, n_params, history, peak_gb,
                 toks_per_s, elapsed, chars_per_tok, sample)
    print(f"\nwrote checkpoint, loss_curve.png and report_card.md to {cfg.out_dir}/")


if __name__ == "__main__":
    main()
