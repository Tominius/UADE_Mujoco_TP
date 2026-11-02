from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

PREFERRED_STANDING_CSV = "dance_B_DadDance_web.csv"


@dataclass(frozen=True)
class SceneConfig:
    model_path: str
    host: str = "0.0.0.0"
    port: int = 9878
    use_mj_step: bool = False
    default_qpos: tuple[float, ...] | None = None


def load_standing_qpos_from_csv(csv_dir: str | Path, model: mujoco.MjModel) -> np.ndarray:
    """Use the same standing reference as the original gallery script (DadDance frame 0)."""
    csv_path = Path(csv_dir) / PREFERRED_STANDING_CSV
    if not csv_path.exists():
        return neutral_standing_qpos(model)
    import pandas as pd

    q = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)[0]
    if q.shape != (model.nq,):
        return neutral_standing_qpos(model)
    return q.copy()


def neutral_standing_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Fallback: identity quaternion + root z so feet touch z=0."""
    q = np.zeros(model.nq, dtype=np.float64)
    q[3] = 1.0
    foot_ids = [
        i
        for i in range(model.ngeom)
        if "foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        and "collision" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
    ]
    if not foot_ids:
        q[2] = 0.793
        return q

    data = mujoco.MjData(model)
    lo, hi = -0.5, 1.5
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        data.qpos[:] = 0.0
        data.qpos[2] = mid
        data.qpos[3] = 1.0
        mujoco.mj_forward(model, data)
        foot_z = min(float(data.geom_xpos[g][2]) for g in foot_ids)
        if foot_z > 0.0:
            hi = mid
        else:
            lo = mid
    q[2] = 0.5 * (lo + hi)
    return q


class MujocoViserScene:
    """
    MuJoCo + Viser bridge using mjviser body-level mesh rendering (same strategy as
    the original play_g1moves_qpos_viser / mjviser examples).
    """

    def __init__(self, config: SceneConfig) -> None:
        model_path = Path(config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo XML 不存在: {model_path}")

        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
        self.data = mujoco.MjData(self.model)
        self.server = viser.ViserServer(
            host=config.host,
            port=config.port,
            label="G1 Moves Web Gallery",
        )

        self._default_qpos = (
            np.asarray(config.default_qpos, dtype=np.float64)
            if config.default_qpos is not None
            else neutral_standing_qpos(self.model)
        )

        self._viser_scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self._viser_scene.camera_tracking_enabled = False

        self.apply_qpos(self._default_qpos)

    @property
    def handles(self) -> dict:
        """Legacy accessor; mjviser manages handles internally."""
        return {}

    def reload_model(self, model_path: str, default_qpos: np.ndarray | None = None) -> None:
        """Hot-swap MuJoCo model (e.g. qpos display vs policy physics XML)."""
        resolved = str(Path(model_path).resolve())
        if resolved == str(Path(self.config.model_path).resolve()):
            return
        self.config = SceneConfig(
            model_path=model_path,
            host=self.config.host,
            port=self.config.port,
            use_mj_step=self.config.use_mj_step,
            default_qpos=tuple(default_qpos) if default_qpos is not None else tuple(self._default_qpos),
        )
        self.model = mujoco.MjModel.from_xml_path(resolved)
        self.data = mujoco.MjData(self.model)
        if default_qpos is not None:
            self._default_qpos = np.asarray(default_qpos, dtype=np.float64).copy()
        self._viser_scene = ViserMujocoScene(self.server, self.model, num_envs=1)
        self._viser_scene.camera_tracking_enabled = False
        self.apply_qpos(self._default_qpos)

    def set_default_qpos(self, qpos: np.ndarray) -> None:
        self._default_qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.apply_qpos(self._default_qpos)

    def reset_pose(self) -> None:
        self.apply_qpos(self._default_qpos)

    def apply_qpos(self, qpos: np.ndarray) -> None:
        arr = np.asarray(qpos, dtype=np.float64)
        if arr.shape != (self.model.nq,):
            raise ValueError(
                f"单帧 qpos 维度不匹配: got={arr.shape}, expected=({self.model.nq},)"
            )

        self.data.qpos[:] = arr
        self.data.qvel[:] = 0.0
        if self.model.na > 0:
            self.data.act[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        if self.config.use_mj_step:
            mujoco.mj_step(self.model, self.data)

        self._viser_scene.update_from_mjdata(self.data)
