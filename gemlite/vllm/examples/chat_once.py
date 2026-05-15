# SPDX-License-Identifier: Apache-2.0
# What this script does:
#   One-shot non-streaming call to a local vLLM /v1/chat/completions endpoint.
#   temperature=0 so both backends produce deterministic text for comparison.
#
# Usage:
#   python3 chat_once.py --prompt "..." --max-tokens 128 > out.txt

import argparse
import json
import sys
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="baseten/Qwen3-4B-NVFP4-PTQ")
    ap.add_argument("--prompt", default="In one paragraph, explain what a GPU kernel is and why it matters for LLMs.")
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    args = ap.parse_args()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"]
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
