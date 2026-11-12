#!/usr/bin/env python3
"""Entrypoint compatible con Arduino App Lab.

App Lab ejecuta este archivo (`python/main.py`). Se limita a arrancar Uvicorn
con la configuración leída de `.env`, usando la app FastAPI de `app.main`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_settings  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    try:
        import uvicorn
    except ImportError:
        print(
            "Falta uvicorn. Ejecuta scripts/setup_board.sh para crear el venv "
            "e instalar requirements.txt.",
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level="info",
        proxy_headers=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
