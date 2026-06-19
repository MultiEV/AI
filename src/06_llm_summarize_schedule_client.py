#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-input-csv", default=None)
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    input_json = Path(args.input_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    runtime_input_csv = (
        str(Path(args.runtime_input_csv).expanduser().resolve())
        if args.runtime_input_csv
        else None
    )

    payload = {
        "input_json": str(input_json),
        "output_dir": str(output_dir),
        "runtime_input_csv": runtime_input_csv,
        "max_new_tokens": args.max_new_tokens,
        "use_llm": not args.skip_llm,
    }

    url = args.server_url.rstrip("/") + "/summarize_schedule"

    try:
        with httpx.Client(timeout=300.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"[ERROR] LLM summary server request failed: {e}", file=sys.stderr)
        sys.exit(1)

    if data.get("summary_status") != "success":
        print(json.dumps(data, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)

    print("✅ LLM server summary completed")
    print(f"SUMMARY MODE : {data.get('summary_mode')}")
    print(f"COMPACT JSON : {data.get('compact_json')}")
    print(f"SUMMARY JSON : {data.get('summary_json')}")
    print(f"MERGED JSON  : {data.get('merged_json')}")
    print()
    print("[BACKEND DISPLAY TEXT]")
    print(data.get("backend_display_text", ""))


if __name__ == "__main__":
    main()
