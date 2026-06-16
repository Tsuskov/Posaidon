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
import math
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
    # Phase 4 — modernize toward Llama (defaults reproduce the GPT-2-style baseline)
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="layernorm")
    p.add_argument("--pos", choices=["learned", "rope"], default="learned",
                   help="learned position embeddings vs rotary (RoPE)")
    p.add_argument("--mlp", choices=["gelu", "swiglu"], default="gelu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1337)
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
def build_tokenizer(text, kind, vocab_size):
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
    encode = lambda s: tok.encode(s).ids
    decode = lambda ids: tok.decode(ids)
    return encode, decode, tok.get_vocab_size()


# ----------------------------------------------------------------------------
# 5. Data + training loop
# ----------------------------------------------------------------------------
def main():
    cfg = get_args()
    mx.random.seed(cfg.seed)

    with open(cfg.data, "r", encoding="utf-8") as f:
        text = f.read()
    encode, decode, vocab_size = build_tokenizer(text, cfg.tokenizer, cfg.vocab_size)

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

    t0 = time.time()
    for it in range(cfg.max_iters + 1):
        if it % cfg.eval_interval == 0:
            tr, va = eval_loss("train"), eval_loss("val")
            print(f"iter {it:5d} | train {tr:.4f} | val {va:.4f} | {time.time()-t0:.1f}s")
        x, y = get_batch("train")
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    print("\n--- sample ---")
    start = mx.array([encode("\n")])
    out = model.generate(start, max_new_tokens=300)[0].tolist()
    print(decode(out))


if __name__ == "__main__":
    main()
