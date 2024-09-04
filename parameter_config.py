from pydantic import BaseModel
from typing import List


class ExperimentConfig(BaseModel):
    path: str = ""
    grid_num: int = 0
    first_sec: int = 0
    last_sec: int = 0
    grid_shape: List[int] = []
    acq_dir: str = ""


class AcquisitionConfig(BaseModel):
    sbem_root_dir: str = ""
    acquisition: str = "run_0"
    tile_grid: str = "g0001"
    thickness: float = 25
    resolution_xy: float = 10

class ProjectConfig(BaseModel):
    path: str = ""

class SystemConfig(BaseModel):
    root: str = ""
    projects: List[ProjectConfig]


class FlowFieldEstimationConfig(BaseModel):
    patch_size: int = 160
    stride: int = 40
    batch_size: int = 256
    min_peak_ratio: float = 1.6
    min_peak_sharpness: float = 1.6
    max_magnitude: float = 80
    max_deviation: float = 20
    max_gradient: float = 0
    min_patch_size: int = 400


class MeshIntegrationConfig(BaseModel):
    dt: float = 0.001
    gamma: float = 0.0
    k0: float = 0.01
    k: float = 0.1
    num_iters: int = 1000
    max_iters: int = 100000
    stop_v_max: float = 0.005
    dt_max: float = 1000
    start_cap: float = 0.01
    final_cap: float = 10
    prefer_orig_order: bool = True
    block_size: int = 50


class WarpConfig(BaseModel):
    target_volume_name: str = "warped_zyx.zarr"
    start_section: int = 0
    end_section: int = 199
    yx_start: list[int] = list([1000, 2000])
    yx_size: list[int] = list([1000, 1000])
    parallelization: int = 16
