"""Viser gallery entry — delegates to G1MovesWebGalleryFinal (v2)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.dashboard import AppConfig, G1MovesWebGalleryFinal


def main() -> None:
    config = AppConfig()
    app = G1MovesWebGalleryFinal(config)
    app.run_forever()


if __name__ == "__main__":
    main()
