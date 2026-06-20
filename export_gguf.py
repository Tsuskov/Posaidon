"""export_gguf.py — convert a (biasless) Posaidon checkpoint to GGUF (Phase 6).

Writes a llama-architecture GGUF that llama.cpp / Ollama can run, mapping our
tensor names to the GGUF llama names and embedding the byte-level BPE tokenizer.

    python export_gguf.py --out_dir out_llama --gguf posaidon.gguf

Requires the checkpoint to be trained with --no_attn_bias (llama arch has no
attention bias). RoPE: MLX nn.RoPE rotates the two halves of each head (HF style),
while GGUF's llama arch expects the interleaved layout, so q/k weights are permuted
exactly as llama.cpp's HF converter does. Verify with a side-by-side sample.
"""
import argparse
import json
import os

import numpy as np
import mlx.core as mx
import gguf


def permute(w, n_head):
    """HF rotate-half RoPE layout -> GGUF llama interleaved layout (q/k only)."""
    d0 = w.shape[0]
    return (w.reshape(n_head, 2, d0 // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2)
             .reshape(w.shape))


def main():
    ap = argparse.ArgumentParser(description="export a Posaidon checkpoint to GGUF")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--gguf", default="posaidon.gguf")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.out_dir, "config.json")))
    if not cfg.get("no_attn_bias"):
        raise SystemExit("checkpoint has attention bias; retrain with --no_attn_bias "
                         "(llama arch in GGUF has no attn bias)")
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    head_dim = n_embd // n_head
    w = mx.load(os.path.join(args.out_dir, "model.safetensors"))
    w = {k: np.array(v, dtype=np.float32) for k, v in w.items()}

    gw = gguf.GGUFWriter(args.gguf, "llama")
    gw.add_name("Posaidon")
    gw.add_context_length(cfg["block_size"])
    gw.add_embedding_length(n_embd)
    gw.add_block_count(cfg["n_layer"])
    gw.add_feed_forward_length(int(8 / 3 * n_embd))   # SwiGLU hidden width
    gw.add_head_count(n_head)
    gw.add_head_count_kv(n_head)                       # no GQA
    gw.add_layer_norm_rms_eps(1e-5)                    # MLX RMSNorm default
    gw.add_rope_dimension_count(head_dim)
    gw.add_rope_freq_base(10000.0)
    gw.add_file_type(gguf.LlamaFileType.ALL_F32)

    # --- tokenizer (byte-level BPE -> GGUF gpt2 model) ---
    tok = json.load(open(os.path.join(args.out_dir, "tokenizer.json")))
    vocab = tok["model"]["vocab"]                      # token -> id
    id_to_tok = {i: t for t, i in vocab.items()}
    tokens = [id_to_tok[i] for i in range(len(id_to_tok))]
    special = {a["content"] for a in tok.get("added_tokens", []) if a.get("special")}
    types = [gguf.TokenType.UNKNOWN if t == "<unk>" else
             (gguf.TokenType.CONTROL if t in special else gguf.TokenType.NORMAL)
             for t in tokens]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]]
    gw.add_tokenizer_model("gpt2")
    gw.add_tokenizer_pre("default")   # llama.cpp's "default" = the GPT-2 byte-level regex
    gw.add_token_list(tokens)
    gw.add_token_types(types)
    gw.add_token_merges(merges)
    gw.add_unk_token_id(vocab["<unk>"])
    gw.add_bos_token_id(vocab["<unk>"])
    gw.add_eos_token_id(vocab["<unk>"])
    gw.add_add_bos_token(False)

    # --- tensors: our names -> GGUF llama names ---
    gw.add_tensor("token_embd.weight", w["wte.weight"])
    gw.add_tensor("output_norm.weight", w["ln_f.weight"])
    gw.add_tensor("output.weight", w["lm_head.weight"])
    for i in range(cfg["n_layer"]):
        p = f"blocks.{i}."
        q, k, v = np.split(w[p + "attn.c_attn.weight"], 3, axis=0)
        gw.add_tensor(f"blk.{i}.attn_norm.weight", w[p + "ln_1.weight"])
        gw.add_tensor(f"blk.{i}.attn_q.weight", permute(q, n_head))
        gw.add_tensor(f"blk.{i}.attn_k.weight", permute(k, n_head))
        gw.add_tensor(f"blk.{i}.attn_v.weight", v)
        gw.add_tensor(f"blk.{i}.attn_output.weight", w[p + "attn.c_proj.weight"])
        gw.add_tensor(f"blk.{i}.ffn_norm.weight", w[p + "ln_2.weight"])
        gw.add_tensor(f"blk.{i}.ffn_gate.weight", w[p + "mlp.w1.weight"])
        gw.add_tensor(f"blk.{i}.ffn_up.weight", w[p + "mlp.w3.weight"])
        gw.add_tensor(f"blk.{i}.ffn_down.weight", w[p + "mlp.w2.weight"])

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"wrote {args.gguf} ({os.path.getsize(args.gguf)/1e6:.1f} MB, "
          f"llama arch, {cfg['n_layer']}L/{n_head}H/{n_embd}d, vocab {len(tokens)})")


if __name__ == "__main__":
    main()
