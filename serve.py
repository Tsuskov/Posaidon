"""serve.py — a tiny web frontend for the Posaidon oracle.

Loads the trained checkpoint once and serves a single-page UI that streams
generated text token-by-token over Server-Sent Events. Stdlib only (no Flask /
FastAPI), so it runs in the same venv that trained the model:

    source .venv/bin/activate && python serve.py            # serves out_greek/
    python serve.py --out_dir out_other --port 8080

Then open http://localhost:8000.
"""

import argparse
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mlx.core as mx

from minigpt_mlx import GPT, build_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(out_dir):
    """Rebuild the model from out_dir/config.json so it matches the weights,
    and return (model, encode, decode, block_size)."""
    with open(os.path.join(out_dir, "config.json")) as f:
        ckpt = argparse.Namespace(**json.load(f))
    model = GPT(ckpt.vocab_size, ckpt)
    model.load_weights(os.path.join(out_dir, "model.safetensors"))
    model.eval()
    n = sum(p.size for _, p in __import__("mlx.nn", fromlist=["utils"]).utils.tree_flatten(model.parameters()))
    if ckpt.tokenizer == "bpe":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(out_dir, "tokenizer.json"))
        encode, decode = (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(ids))
    else:  # char tokenizer is rebuilt deterministically from the corpus
        with open(ckpt.data, encoding="utf-8") as f:
            encode, decode, _ = build_tokenizer(f.read(), "char", 0)
    print(f"loaded {out_dir}: {n/1e6:.1f}M params, vocab={ckpt.vocab_size}, "
          f"block_size={ckpt.block_size}", file=sys.stderr)
    return model, encode, decode, ckpt.block_size


def sample_step(logits, temperature, top_k):
    """Apply temperature + top-k to a (1, vocab) logits row and sample one id."""
    if temperature <= 0:                       # temp 0 == greedy
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k and top_k > 0:
        kth = mx.sort(logits, axis=-1)[:, -top_k:][:, 0:1]   # k-th largest value
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits)


class Oracle:
    """Owns the model and serializes generation (MLX is happiest single-stream)."""

    def __init__(self, out_dir):
        self.model, self.encode, self.decode, self.block_size = load_model(out_dir)
        self.lock = threading.Lock()

    def stream(self, prompt, temperature, top_k, max_new_tokens):
        """Yield text deltas (str) as each token is generated."""
        with self.lock:
            ids = mx.array([self.encode(prompt or "\n")])
            gen, prev = [], ""
            for _ in range(max_new_tokens):
                logits = self.model(ids[:, -self.block_size:])[:, -1, :]
                nxt = sample_step(logits, temperature, top_k)
                mx.eval(nxt)
                ids = mx.concatenate([ids, nxt[:, None]], axis=1)
                gen.append(nxt.item())
                text = self.decode(gen)          # re-decode so byte-level BPE joins cleanly
                if text != prev:
                    yield text[len(prev):]
                    prev = text


ORACLE = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                  # quiet the per-request console spam
        pass

    def _file(self, name, ctype):
        try:
            with open(os.path.join(HERE, name), "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._file("index.html", "text/html; charset=utf-8")
        elif self.path == "/api/art":
            self._file("ascii-art.txt", "text/plain; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or "{}")
        prompt = req.get("prompt", "")
        temperature = float(req.get("temperature", 0.8))
        top_k = int(req.get("top_k", 40))
        max_new_tokens = max(1, min(int(req.get("max_new_tokens", 240)), 600))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        try:
            for delta in ORACLE.stream(prompt, temperature, top_k, max_new_tokens):
                send({"delta": delta})
            send({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            pass                                 # client navigated away mid-stream


def main():
    global ORACLE
    p = argparse.ArgumentParser(description="web frontend for the Posaidon oracle")
    p.add_argument("--out_dir", default="out_greek")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    ORACLE = Oracle(args.out_dir)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  Posaidon oracle awake at  \033[36m{url}\033[0m  (Ctrl-C to silence)\n",
          file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nthe deep falls quiet.", file=sys.stderr)


if __name__ == "__main__":
    main()
