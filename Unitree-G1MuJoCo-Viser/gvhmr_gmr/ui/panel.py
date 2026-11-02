from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import viser

from gvhmr_gmr.config import IntegrationConfig


@dataclass
class GVHMRGMRPanelState:
    reference_mode: str = "npz"
    action_mode: str = "residual"
    gmr_path: str = ""
    obs_status: str = "unknown"
    latency_ms: float = 0.0


class GVHMRGMRPanel:
    """Viser UI controls for GVHMR/GMR reference integration (Phase A)."""

    def __init__(
        self,
        gui: viser.GuiApi,
        config: IntegrationConfig | None = None,
        *,
        on_apply: Callable[[GVHMRGMRPanelState], None] | None = None,
    ) -> None:
        self.config = config or IntegrationConfig()
        self.state = GVHMRGMRPanelState(
            reference_mode=self.config.reference_mode,
            action_mode=self.config.action_mode,
            gmr_path=self.config.default_gmr_demo,
        )
        self._on_apply = on_apply

        with gui.add_folder("🎥 GVHMR / GMR 接入", order=-1):
            self.ref_source = gui.add_dropdown(
                "Reference Source",
                options=["npz", "pkl", "gmr"],
                initial_value=self.state.reference_mode,
            )
            self.action_mode = gui.add_dropdown(
                "Action Mode",
                options=["teacher", "residual", "raw_policy"],
                initial_value=self.state.action_mode,
            )
            self.gmr_path = gui.add_text(
                "GMR / PKL Path",
                initial_value=self.state.gmr_path,
            )
            self.obs_status = gui.add_markdown("**Obs Status** · `unknown`")
            self.latency_md = gui.add_markdown("**Latency** · `-- ms`")
            self.apply_btn = gui.add_button("✓ 应用参考源配置", color="teal")

            @self.apply_btn.on_click
            def _(_: viser.GuiEvent) -> None:
                self._sync_state()
                if self._on_apply:
                    self._on_apply(self.state)

            @self.ref_source.on_update
            def _(_: viser.GuiEvent) -> None:
                self._sync_state()

    def _sync_state(self) -> None:
        self.state.reference_mode = str(self.ref_source.value)
        self.state.action_mode = str(self.action_mode.value)
        self.state.gmr_path = str(self.gmr_path.value)

    def set_obs_status(self, status: str, *, valid: bool = True) -> None:
        icon = "✅" if valid else "⚠️"
        self.state.obs_status = status
        self.obs_status.content = f"**Obs Status** · {icon} `{status}`"

    def set_latency_ms(self, ms: float) -> None:
        self.state.latency_ms = ms
        self.latency_md.content = f"**Latency** · `{ms:.1f} ms`"

    def resolve_path(self, fallback: Path | None = None) -> Path:
        p = Path(self.state.gmr_path)
        if p.exists():
            return p
        if fallback and fallback.exists():
            return fallback
        raise FileNotFoundError(f"GMR/PKL path not found: {p}")
