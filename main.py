"""Punto de entrada CLI limpio para el proyecto de optimización WSN (Etapa 3)."""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_path()

from wsn.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
