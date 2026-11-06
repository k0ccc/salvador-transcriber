#!/usr/bin/env python3
"""Explicit real-OpenAI scenario runner; excluded from the regular test suite."""

from salvador_transcriber.demo.live import main

if __name__ == "__main__":
    raise SystemExit(main())
