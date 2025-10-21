"""Entry point forwarding to the existing RAG client CLI."""
from __future__ import annotations

import sys
from typing import List

from .client import main as client_main


def _translate_args(args: List[str]) -> List[str]:
    """Map shorthand flags to the client module's expected arguments."""

    translated: List[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--q":
            translated.append("--question")
        else:
            translated.append(arg)
        idx += 1
    return translated


def main() -> None:
    forwarded = _translate_args(sys.argv[1:])
    client_main(forwarded)


if __name__ == "__main__":
    main()
