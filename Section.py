import collections
import glob
import os
import pickle
import time
import jax
import skimage.io

import sys

from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

sof_path1 = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\packages\sofima-forked-dev"
sof_path2 = r"/tungstenfs/scratch/gmicro_sem/gfriedri/tgan/packages/sofima-forked-dev"
sys.path.append(sof_path1)
sys.path.append(sof_path2)

from sofima import stitch_rigid, stitch_elastic, flow_utils, mesh, warp
from Tile import Tile
import inspection_utils as utils
from inspection_utils import Num
import mask_utils as mutils
import cv2
import csv
from contextlib import suppress
import functools as ft
import jax.numpy as jnp
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray, dtype
from pathlib import Path
import re
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage import filters
from typing import Tuple, List, Set, Union, Optional, Dict, Mapping, Iterable, Any

from ruyaml import YAML
from ruyaml.scalarfloat import ScalarFloat
yaml = YAML(typ="rt")

import experiment_configs as cfg

sof_path1 = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\scripts\pythonProject"
sof_path2 = r"/tungstenfs/scratch/gmicro_sem/gfriedri/tgan/scripts/pythonProject"
sys.path.append(sof_path1)
sys.path.append(sof_path2)
from SOFIMA import sofima_files as sutils


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

# Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

Vector = Union[Tuple[int, int], Tuple[int, int, int], Union[Tuple[int], Tuple[Any, ...]]]  # [z]yx order
# Vector = Union[tuple[int, int], tuple[int, int, int]]
UniPath = Union[str, Path]
TileXY = Tuple[int, int]
ShapeXYZ = Tuple[int, int, int]
TileFlow = Dict[TileXY, np.ndarray]
TileOffset = Dict[TileXY, Vector]
TileMap = Dict[TileXY, np.ndarray]
FineFlow = Union[Tuple[TileFlow, TileOffset], None]
FineFlows = Tuple[Optional[FineFlow], Optional[FineFlow]]
MaskMap = Dict[TileXY, Optional[np.ndarray]]
TileFlowData = Tuple[np.ndarray, TileFlow, TileOffset]
MarginOverrides = Dict[TileXY, Tuple[int, int, int, int]]
GridXY = Tuple[Any, Any, Any]
ExpConfig = collections.namedtuple('ExpConfig', ['path', 'grid_num', 'first_sec', 'last_sec', 'grid_shape'])


class Section:
    def __init__(self, path: Union[Path, str]):

        path = Path(utils.cross_platform_path(str(path)))
        if not path.is_dir():
            m = f"The path '{path}' is not a directory or does not exist."
            raise NotADirectoryError(m)

        self.path: Path = path
        self.path_stitched: Path = self.resolve_dir_stitched()
        self.path_stitched_custom = self.path.parent.parent / 'stitched-sections-derotated' / (str(self.path.name) + ".zarr")
        self.path_section_yaml = str(self.path / 'section.yaml')
        self.path_stats_yaml = str(self.path / 'stats.yaml')
        self.path_margin_masks = str(self.path / 'margin_masks.npz')
        self.path_cmesh = str(self.path / 'coarse_mesh.pkl')
        self.path_fmesh = str(self.path / 'meshes.npz')
        self.path_fflows = str(self.path / 'fflows.pkl')
        self.path_shift_to_prev = str(self.path_stitched / 'shift_to_previous.json')

        self.section_num = int(str(self.path.name).split("_g")[0][1:])
        self._grid_num = int(self.path.name[-1])
        self.tile_shape = utils.get_tile_shape(self.path_section_yaml)
        self.tile_id_map: Optional[np.ndarray[int]] = None
        self.tile_dicts: Optional[Dict[int, str]] = None

        self.cxy: Optional[np.ndarray[float]] = None
        self.coarse_mesh: Optional[np.ndarray[float]] = None
        self.mesh_offsets: Optional[np.ndarray[float]] = None
        self.fflows: Optional[FineFlows] = None
        self.fflows_clean: Optional[FineFlows] = None
        self.fflows_recon: Optional[FineFlows] = None
        self.fmesh: Optional[Dict[TileXY, np.ndarray]] = None

        self.tile_map: Optional[TileMap] = None
        self.mask_map: MaskMap = {}  # TODO: make None as default (?)
        self.roi_mask_map: MaskMap = {}
        self.smr_mask_map: MaskMap = {}
        self.margin_overrides: Optional[Dict[TileXY, Tuple[int, int, int, int]]] = None
        self.margin_masks: Optional[Dict[TileXY, np.ndarray]] = None

        self.path_thumb = self.resolve_path_thumb()
        self.thumb: Optional[np.ndarray] = None
        self.roi_inspect = None
        self.roi_conv = None
        self.filtered_image = None
        self.image: Optional[np.ndarray] = None
        self.height: Optional[int] = None  # tile_shape already in attributes, refactor.
        self.width: Optional[int] = None
        self._stitched: bool = False

    @property
    def stitched(self) -> bool:

        MIN_ZARR_SIZE = 1000  # in kb

        # Verify that stitched folder has valid size
        zarr_size = utils.get_folder_size(str(self.path_stitched))
        if zarr_size is None:
            self._stitched = False
            return False

        zarr_size_valid = True if zarr_size > MIN_ZARR_SIZE else False
        if not zarr_size_valid:
            self._stitched = False
            return False

        zarr_file_valid = utils.check_zarr_integrity(self.path_stitched)
        if not zarr_file_valid:
            self._stitched = False
            return False

        self._stitched = True
        return True

    @property
    def grid_num(self) -> int:
        return self._grid_num

    # @property
    # def stitched(self) -> bool:
    #     return self.path_stitched.exists()

    @property
    def cross_aligned(self) -> bool:
        return Path(self.path_shift_to_prev).exists()

    def clear_section(self):
        del self.fflows
        del self.fflows_recon
        del self.fflows_clean
        del self.tile_map
        del self.mask_map
        del self.roi_mask_map
        del self.smr_mask_map
        del self.margin_masks
        return


    def load_fmesh(self) -> None:
        self.fmesh = np.load(self.path_fmesh)
        return

    def fix_coarse_offset(self,
                          tid_a: int,
                          tid_b: int,
                          est_vec_orig: bool,
                          skip_xcorr: bool,
                          refine: bool,
                          xcorr_kwargs: dict,
                          refine_kwargs: dict,
                          plot_ov=False,
                          ) -> None:
        """ Recomputes specific coarse offset vector by SOFIMA or refine pyramid

        Recompute either by both methods or refinement only. Refinement can be done
        with respect to original coarse shift vector or specific vector. Optionally,
        plot resulting overlap image to section folder. Specify SOFIMA registration
        and/or refinement parameters and pass it ot the function.

        :param tid_a:
        :param tid_b:
        :param est_vec_orig:
        :param skip_xcorr:
        :param refine:
        :param xcorr_kwargs:
        :param refine_kwargs:
        :param plot_ov:
        :return:
        """

        def opt_coo(section: Section) -> None:
            if not skip_xcorr:
                shift_vec = section.compute_coarse_offset(**xcorr_kwargs)
                logging.info(f'computed coarse shift: {shift_vec}')
                print(f'computed coarse shift: {shift_vec}')
            if refine:
                best_vec = self.refine_pyramid(**refine_kwargs)
                print(f'refined vector: {best_vec}')
            return

        self.feed_section_data()

        # Get original coarse shift vector if requested,
        # otherwise average will be estimated
        is_vert = utils.pair_is_vertical(self.tile_id_map, tid_a, tid_b)
        axis = 1 if is_vert else 0
        if est_vec_orig:
            refine_kwargs['est_vec'] = self.get_coarse_offset(tid_a, axis)

        # Perform optimization
        opt_coo(self)

        # Plot OV
        if plot_ov:
            self.plot_ov(
                tid_a, tid_b, self.get_coarse_offset(tid_a, axis),
                dir_out=self.path, clahe=True, blur=1.3, show_plot=False,
                rotate_vert=True, store_to_root=True
            )
        return

    def verify_tile_id_map(self, print_ids: bool = False) -> bool:

        # Get tile IDs from section yaml file
        yaml_tile_ids: Set[int] = set(utils.get_tile_ids(self.path))
        if not yaml_tile_ids:
            logging.warning(f'Verify tile_id_map: No tile IDs found in section .yaml file.')
            return False

        # Get tile IDs form section tile_id_map.json
        if not self.tile_id_map:
            self.read_tile_id_map()

        if not isinstance(self.tile_id_map, np.ndarray):
            logging.warning(f'Verify tile_id_map: no tile IDs found in section tile_id_map.json')
            return False

        tile_id_map_ids = set(self.tile_id_map.flatten())
        if -1 in tile_id_map_ids:
            tile_id_map_ids.remove(-1)

        # Perform comparison
        eq = yaml_tile_ids == tile_id_map_ids

        # Print info if tile IDs are not the same in both sets
        if not eq and print_ids:
            sec_num = self.section_num
            ids = yaml_tile_ids.symmetric_difference(tile_id_map_ids)
            logging.warning(f'section s{sec_num} yaml tile IDs: {sorted(list(yaml_tile_ids))}')
            logging.warning(f'section s{sec_num} tile_id_map IDs: {sorted(list(tile_id_map_ids))}')
            logging.warning(f'missing s{sec_num} tile ids: {sorted(list(ids))}')

        return eq

    def rotate_and_store_stitched(self, rot_angle: float, dst_dir: UniPath) -> None:
        stitched_img = self.load_image()
        assert isinstance(stitched_img, np.ndarray), "rotate_stitched failed to load stitched .zarr file"
        img_rot = utils.rotate_image(stitched_img, rot_angle)
        utils.store_section_zarr(img_rot, self.path.name + '.zarr', dst_dir)

    def warp_section(self,
                     stride: int,
                     margin: int = 0,
                     use_clahe: bool = False,
                     clahe_kwargs: ... = None,
                     zarr_store=True,
                     rescale_fct: Optional[float] = None,
                     parallelism: int = 1,
                     rot_angle: float = 0
                     ) -> None:

        # Load mesh
        if self.fmesh is None:
            self.fmesh = utils.load_mapped_npz(self.path_fmesh)
            if self.fmesh is None:
                logging.warning(
                    f'Warping s{self.section_num} failed: {Path(self.path_fmesh).name} could not be loaded.')
                return

        # Load tile-data
        if self.tile_dicts is None:
            self.feed_section_data()

        if self.tile_map is None:
            self.load_tile_map(clahe=use_clahe)
            if self.tile_map is None:
                logging.warning(f'Warping s{self.section_num} failed: tile-map could not be loaded.')
                return

        # Load margin masks
        if self.margin_masks is None:
            self.margin_masks = utils.load_mapped_npz(self.path_margin_masks)
        # self.margin_masks = None

        # Warp the tiles into a single image
        stitched, mask = warp.render_tiles(
            tiles=self.tile_map,
            coord_maps=self.fmesh,
            stride=(stride, stride),
            margin=margin,
            use_clahe=use_clahe,
            clahe_kwargs=clahe_kwargs,
            tile_masks=self.margin_masks,
            parallelism=parallelism
        )

        if rot_angle != 0:
            stitched = utils.rotate_image(stitched, rot_angle)

        path_stitched = self.path_stitched.parent
        if zarr_store:
            utils.store_section_zarr(stitched, self.path.name + '.zarr', path_stitched)

        # # Downscale warped image and save image data to disk
        # if rescale_fct is not None:
        #     thumb_img = utils.downscale_image(stitched, rescale_fct)
        #     ext = f'_thumb_{rescale_fct}.png'
        #     name_end = str(self.path.name).split("_")[1]
        #     zfilled = str(self.section_num).zfill(5)
        #     new_name = "s" + zfilled + name_end
        #     thumb_fn = str(self.path / (new_name + ext))
        #     cv2.imwrite(thumb_fn, cv2.convertScaleAbs(thumb_img))

        print(f'Section {self.section_num} stitched and warped.')
        return

    def compute_fine_mesh(self, config, stride: int, store=True) -> None:

        if self.tile_dicts is None:
            self.feed_section_data()
        if self.tile_map is None:
            self.load_tile_map()
        if self.coarse_mesh is None:
            self.load_coarse_mesh()
        if self.coarse_mesh is None:
            logging.warning(f'compute_fine_mesh s{self.section_num} failed (coarse mesh not loaded.)')
            return

        # Prepare data for mesh computation
        if self.fflows is None:
            self.load_fflows()
        self.clean_fflows()
        self.reconcile_fflows()

        cx, cy = np.squeeze(self.cxy)
        ffx, ffxo = self.fflows_recon[0]
        ffy, ffyo = self.fflows_recon[1]
        data_x: Tuple[np.ndarray, TileFlow, TileOffset] = (cx, ffx, ffxo)
        data_y: Tuple[np.ndarray, TileFlow, TileOffset] = (cy, ffy, ffyo)

        fx, fy, nds, nbors, key_to_idx = stitch_elastic.aggregate_arrays(
            data_x, data_y, list(self.tile_map.keys()),
            self.coarse_mesh[:, 0, ...], stride=(stride, stride),
            tile_shape=next(iter(self.tile_map.values())).shape)

        @jax.jit
        def prev_fn(nds):
            target_fn = ft.partial(stitch_elastic.compute_target_mesh, x=nds, fx=fx,
                                   fy=fy, stride=(stride, stride))
            nds = jax.vmap(target_fn)(nbors)
            return jnp.transpose(nds, [1, 0, 2, 3])

        if config is None:
            config = mesh.IntegrationConfig(
                dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                num_iters=1000, max_iters=20000, stop_v_max=0.001,
                dt_max=100, prefer_orig_order=True,
                start_cap=0.1, final_cap=10., remove_drift=True
            )

        # Compute fine mesh
        res, _, _ = mesh.relax_mesh(nds, None, config, prev_fn=prev_fn)

        # Unpack meshes into a dictionary.
        idx_to_key = {v: k for k, v in key_to_idx.items()}
        self.fmesh = {idx_to_key[i]: np.array(res[:, i:i + 1:, :]) for i in range(res.shape[1])}

        # Save mesh for later processing
        if store:
            meshes_to_save = {str(k): v for k, v in self.fmesh.items()}
            np.savez(self.path_fmesh, **meshes_to_save)

        return

    def clean_fflows(self,
                     min_pr: float = 1.4,
                     min_ps: float = 1.4,
                     max_mag: float = 0.,
                     max_dev: float = 5., ) -> None:

        if self.fflows is None:
            logging.warning(f's{self.section_num} clean fflows failed: fine flows not available.')
            return

        fine_x, offsets_x = self.fflows[0]
        fine_y, offsets_y = self.fflows[1]

        kwargs = {"min_peak_ratio": min_pr, "min_peak_sharpness": min_ps, "max_deviation": max_dev,
                  "max_magnitude": max_mag}
        fine_x = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_x.items()}
        fine_y = {k: flow_utils.clean_flow(v[:, np.newaxis, ...], **kwargs)[:, 0, :, :] for k, v in fine_y.items()}

        ffx = fine_x, offsets_x
        ffy = fine_y, offsets_y
        self.fflows_clean = (ffx, ffy)
        return

    def reconcile_fflows(self,
                         max_gradient: float = -1.,
                         max_deviation: float = -1.,
                         min_patch_size: int = 10) -> None:

        fine_x, offsets_x = self.fflows_clean[0]
        fine_y, offsets_y = self.fflows_clean[1]

        kwargs = {"min_patch_size": min_patch_size, "max_gradient": max_gradient, "max_deviation": max_deviation}
        fine_x = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in
                  fine_x.items()}
        fine_y = {k: flow_utils.reconcile_flows([v[:, np.newaxis, ...]], **kwargs)[:, 0, :, :] for k, v in
                  fine_y.items()}

        ffx = fine_x, offsets_x
        ffy = fine_y, offsets_y
        self.fflows_recon = (ffx, ffy)
        return

    def compute_fine_flows(self,
                           patch_size: int, stride: int, masking=False,
                           store=True, overwrite: bool = False,
                           ext: Optional[str] = None) -> None:

        def load_infra() -> bool:
            if self.cxy is None:
                self.feed_section_data()

            if self.cxy is None:
                logging.warning(f'compute_fine_flows section s{self.section_num}: coarse offset array not loaded!')
                return False

            if self.tile_map is None:
                self.load_tile_map(clahe=False)

            if self.tile_map is None:
                return False

            if not self.mask_map and masking:
                self.load_masks()

            return True

        def iter_compute_flows(patch_size=patch_size, ff_iter=0, max_iter=5,
                               min_patch_size=10, step=5) -> None:
            """Compute fine flows with iterative decrease if patch size in case of SOFIMA negative dimension error"""

            def compute_flows(ps: int, axis: int) -> Tuple[TileFlow, TileOffset]:
                return stitch_elastic.compute_flow_map(self.tile_map, self.cxy[axis], axis,
                                                       patch_size=(ps, ps),
                                                       stride=(stride, stride), batch_size=256,
                                                       tile_masks=self.mask_map)

            logging.info(f'Computing fine flows for section s{self.section_num}')
            while self.fflows is None or ff_iter == max_iter:
                try:
                    logging.info(f's{self.section_num} fflows params: iter={ff_iter}, patch_size={patch_size}')
                    self.fflows = (compute_flows(patch_size, axis=0),
                                   compute_flows(patch_size, axis=1))

                except ValueError as _:
                    logging.warning(
                        f'fine flows s{self.section_num} decreasing patch size ({patch_size} -> {patch_size - step})')
                    patch_size -= step
                    ff_iter += 1
                    if patch_size < min_patch_size:
                        ff_iter = max_iter

            logging.info(
                f's{self.section_num} fine flows computed after {ff_iter + 1} iterations. Final patch size: {patch_size}')
            return

        def store_fflows(fine_flows: FineFlows, ext: Optional[str]):
            ext = '' if ext is None else ext
            fname = f'fflows{ext}.pkl'
            with open(self.path / fname, 'wb') as f:
                pickle.dump(fine_flows, f)
            return

        # Load necessary data first
        infra_loaded = load_infra()
        if not infra_loaded:
            return

        if not overwrite:
            self.load_fflows()  # Load if present
            if self.fflows is not None:
                logging.info(f'compute_fine_flows skipping s{self.section_num} (fflows already exists and overwrite is disabled).')
                return

        # Compute fine flows and fine offsets
        iter_compute_flows()

        # Store results
        if store and self.fflows is not None:
            store_fflows(self.fflows, ext)

        return


    def load_fflows(self, ext: Optional[str] = None) -> None:
        ext = '' if ext is None else ext
        fp_fflows = self.path / f'fflows{ext}.pkl'
        logging.info(f's{self.section_num}: loading fine flows from {fp_fflows}.')

        try:
            with open(fp_fflows, 'rb') as f:
                self.fflows = pickle.load(f)

            if (not isinstance(self.fflows, tuple) or len(self.fflows) != 2
                    or not all(isinstance(item, (dict, type(None))) for item in self.fflows)):
                logging.info(
                    f's{self.section_num} loaded fine flows from {fp_fflows}, but data appears to be corrupted or invalid.')
                # print(len(self.fflows))
                # print(self.fflows)
                # self.fflows = None  # Reset to default value
            else:
                logging.info(f's{self.section_num}: successfully loaded fine flows from {fp_fflows}.')

        except FileNotFoundError:
            logging.warning(f's{self.section_num}: fine flows file {fp_fflows} not found!')
        except EOFError:
            logging.warning(f's{self.section_num}: EOFError - Ran out of input while reading {fp_fflows}.')
        except pickle.UnpicklingError as e:
            logging.error(f's{self.section_num}: Error while unpickling {fp_fflows}: {e}')
        except Exception as e:
            logging.error(f"An error occurred while reading '{fp_fflows}': {e}")


    def check_and_load_coarse_mesh(self) -> bool:
        try:
            if Path(self.path_cmesh).exists():
                self.load_coarse_mesh()
                return self.coarse_mesh is not None
            else:
                print(f"File '{self.path_cmesh}' does not exist.")
                return False
        except Exception as e:
            print(f"An error occurred while checking and loading '{self.path_cmesh}': {e}")
            return False


    def compute_coarse_mesh(self, cfg: Optional[mesh.IntegrationConfig] = None, store=True, overwrite=False) -> None:

        if cfg is None:
            cfg = mesh.IntegrationConfig(
                dt=0.001,
                gamma=0.0,
                k0=0.0,  # unused
                k=0.1,
                stride=(1, 1),  # unused
                num_iters=1000,
                max_iters=100000,
                stop_v_max=0.001,
                dt_max=100,
            )

        def store_cmesh(data):
            with open(self.path_cmesh, 'wb') as f:
                pickle.dump(data, f)
            return

        logging.info('Computing coarse mesh ...')
        if self.check_and_load_coarse_mesh() and not overwrite:
            return

        try:
            cx, cy = self.get_coarse_mat()
        except TypeError as _:
            logging.warning(f's{self.section_num} coarse mesh not computed')
            return

        if cx.ndim != 4:
            cx = cx[:, np.newaxis, ...]
            cy = cy[:, np.newaxis, ...]

        self.coarse_mesh = stitch_rigid.optimize_coarse_mesh(cx, cy, cfg)

        if self.coarse_mesh is None:
            logging.warning(f'Section s{self.section_num} coarse mesh not computed.')
        elif store:
            logging.info(f'Storing coarse mesh.')
            store_cmesh(self.coarse_mesh)

        return

    def load_coarse_mesh(self) -> None:
        try:
            with open(self.path_cmesh, 'rb') as f:
                self.coarse_mesh = pickle.load(f)
            logging.info(f"s{self.section_num} coarse mesh loaded")
        except EOFError:
            print("EOFError: Ran out of input while reading the pickled data.")
        except FileNotFoundError:
            print(f"File '{self.path_cmesh}' not found.")
        except Exception as e:
            print(f"An error occurred while reading cmesh {self.path_cmesh}: {e}")


    def create_masks(self,
                     roi_thresh: int,
                     max_vert_ext: int,
                     edge_only: bool,
                     n_lines: int,
                     store: bool = False,
                     filter_size: int = 20,
                     range_limit: int = 0
                     ) -> Optional[MaskMap]:
        """
        Create ROI and smearing masks for tiles and store them if specified.

        Args:
            roi_thresh: parameter influencing detection sensitivity of silver particles in resin
            max_vert_ext: Mask only specified number of lines from top of the image
            edge_only: If True, mask only top N lines of tile-data and do not compute smearing mask
            n_lines: Number of lines from top of the image top be fully masked
            store (bool, optional): Whether to store the masks. Defaults to False.
        Returns:
            Optional[MaskMap]: A map of masks.

        """

        def get_masks(tid: int, tile_coords: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            m_roi = None
            m_smr = None
            if tid == -1:
                return m_roi, m_smr

            tile_path = self.tile_dicts.get(tid, None)
            if tile_path is not None and Path(tile_path).exists():
                tile = Tile(tile_path)
                if self.tile_map is not None:
                    tile.img_data = self.tile_map[tile_coords]
                m_smr = tile.create_smearing_mask(max_vert_ext, n_lines, edge_only)
                if tid not in inner_ids:
                    m_roi = tile.create_roi_mask(roi_thresh, filter_size, range_limit)
            else:
                logging.debug(f'tile_path does not exist')
            try:
                logging.debug(f'shapes: {m_roi.shape, m_smr.shape}')
            except AttributeError as e:
                logging.debug(f'WARNING: {tid} attribute error!')
            return m_roi, m_smr

        # Ensure necessary data is loaded
        if self.tile_id_map is None:
            self.read_tile_id_map()

        if self.tile_id_map is None:
            return None

        if self.tile_dicts is None:
            self.feed_section_data()

        # Initialize returned values
        mask_map = {}
        roi_mask_map = {}
        smr_mask_map = {}

        # Get tile_ids that are at the edge of ROI and thus may contain resin in image-data
        inner_ids: List[int] = mutils.tile_ids_with_all_neighbors(self.tile_id_map)

        h, w = self.tile_id_map.shape
        for y in range(h):
            for x in range(w):
                tile_id = int(self.tile_id_map[y, x])
                roi_mask, smr_mask = get_masks(tile_id, (x, y))
                roi_mask_map[(x, y)] = roi_mask
                smr_mask_map[(x, y)] = smr_mask

                comb_mask = None
                if roi_mask is not None and smr_mask is not None:
                    h1, w1 = np.shape(roi_mask)
                    h2, w2 = np.shape(smr_mask)
                    pad_smr = np.pad(smr_mask, ((0, h1 - h2), (0, w1 - w2)), 'constant')
                    comb_mask = np.logical_or(roi_mask, pad_smr)
                    # comb_mask = roi_mask.copy()
                    # comb_mask[:roi_mask.shape[0]] |= smr_mask
                if roi_mask is not None and smr_mask is None:
                    comb_mask = roi_mask
                if smr_mask is not None and roi_mask is None:
                    img = skimage.io.imread(self.tile_dicts[tile_id])
                    tile_shape = np.shape(img)
                    comb_mask = np.full(tile_shape, fill_value=False)
                    comb_mask[:np.shape(smr_mask)[0], :] = smr_mask

                mask_map[(x, y)] = comb_mask

        self.smr_mask_map = smr_mask_map
        self.roi_mask_map = roi_mask_map
        self.mask_map = mask_map

        # Store masks if requested
        if store:
            fns = ('roi_masks.npz', 'smr_masks.npz', 'tile_masks.npz')
            maps = (roi_mask_map, smr_mask_map, mask_map)

            for fn, i_map in zip(fns, maps):
                data = {str(k): v for k, v in i_map.items()}
                fp = self.path / fn
                logging.debug(f'saving to: {fp}')
                np.savez_compressed(fp, **data)

        return None

    def load_masks(self):
        """Loads binary masks associated to each tile within section"""
        fns = ('roi_masks.npz', 'smr_masks.npz', 'tile_masks.npz')
        maps = (self.roi_mask_map, self.smr_mask_map, self.mask_map)
        for fn, i_map in zip(fns, maps):
            path_mask = self.path / fn
            if not path_mask.exists():
                logging.warning(f's{self.section_num} {fn} does not exist!')
                continue
            try:
                data = np.load(path_mask, allow_pickle=True)
                for key, item in data.items():
                    i_map[eval(key)] = item if item.size != 1 else None
            except FileNotFoundError:
                logging.info(f"s{self.section_num}: {path_mask} not loaded.")
            except Exception as e:
                print(f"An error occurred: s{self.section_num} {fn}.", e)
        # self.margin_masks = utils.load_mapped_npz(self.path_margin_masks)
        return

    def build_mesh_offsets(self, mesh_config: Optional[mesh.IntegrationConfig] = None,
                           overwrite: Optional[bool] = True
                           ) -> None:
        """Creates coarse offset matrix from coarse mesh values

        Create coarse mesh offset array in form of individual tile offsets
        in same notation as coarse offsets. These values will be used to create
        margin overrides for section warping.
        """

        def diff_mat(mat: np.ndarray, row_mode=False) -> np.ndarray:
            if row_mode:
                result = [[np.round(mat[i][j] - mat[i - 1][j]) for j in range(len(mat[i]))] for i in range(1, len(mat))]
            else:
                result = [[np.round(mat[i][j] - mat[i][j - 1]) for j in range(1, len(mat[i]))] for i in range(len(mat))]
            return np.array(result)

        def make_mesh_offsets() -> Optional[np.ndarray]:

            if self.cxy is None:
                _ = self.get_coarse_mat()

            if Path(self.path_cmesh).exists():
                self.load_coarse_mesh()
            else:
                self.compute_coarse_mesh(mesh_config, overwrite=overwrite)

            if self.coarse_mesh is None:
                return

            cxx = diff_mat(self.coarse_mesh[0, 0, ...], row_mode=False)
            cxy = diff_mat(self.coarse_mesh[1, 0, ...], row_mode=False)
            cyx = diff_mat(self.coarse_mesh[0, 0, ...], row_mode=True)
            cyy = diff_mat(self.coarse_mesh[1, 0, ...], row_mode=True)

            mo = np.full_like(self.cxy, fill_value=np.nan)
            nr, nc = cxx.shape
            mo[0, 0, 0:nr, 0:nc] = cxx
            mo[0, 1, 0:nr, 0:nc] = cxy

            nr, nc = cyx.shape
            mo[1, 0, 0:nr, 0:nc] = cyx
            mo[1, 1, 0:nr, 0:nc] = cyy

            mask = np.isnan(self.cxy)
            mo[mask] = np.nan
            return mo

        self.mesh_offsets = make_mesh_offsets()
        return

    def feed_section_data(self):
        self.tile_dicts = utils.get_tile_dicts(self.path)
        self.read_tile_id_map()
        _ = self.get_coarse_mat()
        return

    def build_margin_masks(self,
                           grid_shape: List[int],
                           margin: int = 20,
                           rim_size: int = 60,
                           overwrite: bool = False,
                           mesh_config: Optional[mesh.IntegrationConfig] = None
                           ) -> None:
        """ Creates masks for section rendering.

        Margin masks allow to render overlap regions with better quality. Charging
        and deformations are most often related to multiple-exposed regions. Margin
        masks build on this fact and assign higher rendering priority to image-data
        acquired on fresh sample surface. Procedure requires coarse offsets to be
        computed and saved in advance.

        grid_shape: Total number of rows and columns in the SBEMimage grid
        margin: masks deformed borders of tiles due to elastic transformation
        rim_size: Size of safety margin added to the coarse offset
                to avoid holes in warped image. Defaults to 60 pixels.
        """

        def create_sbem_grid() -> np.ndarray:
            # Create virtual SBEMimage grid of active tiles
            grid = np.full(shape=grid_shape, fill_value=-1)
            rows, cols = grid_shape
            for row_pos in range(rows):
                for col_pos in range(cols):
                    tile_index = row_pos * cols + col_pos
                    if tile_index in self.tile_id_map:
                        grid[row_pos, col_pos] = tile_index
            return grid

        def tile_mask_junction(
                tile_id: int,
                row_is_odd: bool,
                grid: np.ndarray,
                rim: int,
                min_rim: int = 5,
                n_smr_lines: int = 20
        ) -> Optional[np.ndarray[bool]]:
            """Create tile mask for rendering

            :param n_smr_lines:
            :param rim: Size of safety margin added to the coarse offset
                        to avoid holes in warped image.
            :param min_rim: minimal masked extent from each edge of a tile
            """

            def eval_(xo):
                dx_new = xo + rim + margin
                dx_new = -min_rim if dx_new >= 0 else dx_new
                return xo if abs(dx_new) > abs(xo) else dx_new

            mask = np.full(self.tile_shape, fill_value=True)
            y, x = np.where(tile_id == grid)
            y, x = int(y[0]), int(x[0])

            # Determine the next tile ID based on row parity
            try:
                tid = grid[y, x + (1 if row_is_odd else -1)]
            except IndexError:
                tid = -1

            if row_is_odd:
                # if margin != 0:
                #     mask[:, :margin] = False
                if tid != -1:
                    offset = self.get_coarse_mesh_offset(tile_id)
                    dx, dy = eval_(offset[0]), offset[1]
                    if dy >= 0:
                        dyr = dy + rim if dy + rim > min_rim else min_rim
                        # print(f'tile_id: {tile_id} dyr+: {dyr} dx: {dx}')
                        mask[dyr:, dx:] = False
                    else:
                        dyr = dy - rim if abs(dy - rim) > min_rim else -min_rim
                        # print(f'tile_id: {tile_id} dyr-: {dyr} dx: {dx}')
                        mask[:dyr, dx:] = False

            # Even rows
            else:
                # if margin != 0:
                #     mask[:, -margin:] = False
                if tid != -1:
                    offset = self.get_coarse_mesh_offset(tile_id - 1)
                    dx, dy = eval_(offset[0]), offset[1]
                    if dy >= 0:
                        dyy = int(-dy - rim)
                        dyr = dyy if abs(dyy) > min_rim else -min_rim
                        mask[:dyr, :abs(dx)] = False
                    else:
                        dyy = int(abs(dy) + rim)
                        dyr = dyy if dyy > min_rim else min_rim
                        mask[dyr:, :abs(dx)] = False

            # Set the top tile-edges mask

            # # Mask elastic deformation on bottom edge  (WHY?)
            # if margin != 0:
            #     try:
            #         tid_nn_y = int(grid[y + 1, x])
            #     except IndexError:
            #         tid_nn_y = -1
            #     if tid_nn_y != -1:
            #         mask[-margin:, :] = False

            # Mask top tile-edges
            try:
                tid_nn_y = int(grid[y - 1, x])
            except IndexError:
                tid_nn_y = -1

            if tid_nn_y != -1:
                offset = self.get_coarse_mesh_offset(tid_nn_y, axis=1)
                dx, dy = offset[0], eval_(offset[1])
                dy = min(offset[1] + rim, 0)  # Testing phase
                # dy = offset[1]

                mask[:n_smr_lines] = False  # Testing phase

                if dx >= 0:
                    dxr = int(dx + rim) if int(dx + rim) != 0 else min_rim
                    # print(f'tile_id: {tile_id} dxr-: {dxr} dxy: {dx},{dy}')
                    mask[:abs(dy), :-dxr] = False

                else:
                    dxr = int(abs(dx) + rim) if int(abs(dx) + rim) != 0 else min_rim
                    # print(f'tile_id: {tile_id} dxr-: {dxr} dxy: {dx},{dy}')
                    mask[:abs(dy), dxr:] = False

            return mask

        def create_margin_masks(tile_space: Tuple[TileXY], rim: int):
            sbem_grid = create_sbem_grid()
            self.margin_masks = {}

            for tile_xy in tile_space:
                x, y = tile_xy
                tile_id = int(self.tile_id_map[y, x])

                # Find the indices of the tile_id in sbem_grid
                indices = np.where(tile_id == sbem_grid)
                if len(indices[0]) == 0:  # Tile not found in sbem_grid
                    continue

                row, col = indices[0][0], indices[1][0]
                odd_row = row % 2 != 0  # Checking for odd row directly

                self.margin_masks[tile_xy] = tile_mask_junction(tile_id, odd_row, sbem_grid, rim)
            return

        def store_margin_masks():
            if self.margin_masks is None:
                logging.warning(f's{self.section_num} skipping storing None margin masks.')
                return
            data = {str(k): v for k, v in self.margin_masks.items()}
            logging.debug(f'Storing margin masks to: {self.path_margin_masks}')
            np.savez_compressed(self.path_margin_masks, **data)
            return

        if Path(self.path_margin_masks).exists() and not overwrite:
            print('Skipping margin mask computation. File exists and overwriting is disabled.')
            self.margin_masks = utils.load_mapped_npz(self.path_margin_masks)
            return

        if self.tile_id_map is None:
            self.feed_section_data()

        if self.mesh_offsets is None:
            self.build_mesh_offsets(mesh_config=mesh_config)

        if self.mesh_offsets is None:
            logging.warning(f'Section s{self.section_num} mesh offsets could not be computed.')
            return

        tiles_xy = utils.build_tiles_coords(self.tile_id_map)
        create_margin_masks(tiles_xy, rim_size)
        store_margin_masks()
        return

    def build_margin_overrides(self,
                               grid_shape: Tuple[int, int],
                               rim: int = 10
                               ) -> Optional[MarginOverrides]:
        """Builds margin overrides for each tile coordinate in section.

         Args:
             grid_shape: Total number of rows and columns in the SBEMimage grid
             rim (int): Size of safety margin added to the coarse offset
                        to avoid holes in warped image. Defaults to 10 pixels.

         Returns:
             Optional[Dict[Tuple[int, int], Tuple[int, int, int, int]]]: A dictionary
                    mapping tile coordinates to their corresponding margin overrides
                    (top, bottom, left, right).
         """

        def create_sbem_grid() -> np.ndarray:
            # Create virtual SBEMimage grid of active tiles
            grid = np.full(shape=grid_shape, fill_value=-1)
            rows, cols = grid_shape
            for row_pos in range(rows):
                for col_pos in range(cols):
                    tile_index = row_pos * cols + col_pos
                    if tile_index in self.tile_id_map:
                        grid[row_pos, col_pos] = tile_index
            return grid

        def eval_margin(offset: Vector, axis: int) -> int:
            offset = offset[axis]
            if not np.isinf(offset):
                res = abs(offset) - rim
                return res if res > 0 else 0
            return 0

        def build_tile_margins(tile_id, grid):
            # Get current margin override
            mo = list(overrides.get((mox, moy)))

            # Modify margin overrides
            y, x = np.where(tile_id == grid)
            y, x = int(y[0]), int(x[0])

            # Left tile-edges
            if y % 2 == 0:
                if grid[y, x - 1] != -1:
                    axis = 0
                    cx_nn = self.get_coarse_mesh_offset(tile_id - 1, axis)
                    mo[2] = eval_margin(cx_nn, axis)

            # Right tile-edges
            elif grid[y, x + 1] != -1:
                axis = 0
                cx_nn = self.get_coarse_mesh_offset(tile_id, axis)
                mo[3] = eval_margin(cx_nn, axis)

            # Top tile-edges
            tile_id_y_nn = int(grid[y - 1, x])
            if tile_id_y_nn != -1:
                axis = 1
                cy_nn = self.get_coarse_mesh_offset(tile_id_y_nn, axis)
                mo[0] = eval_margin(cy_nn, axis)

            return tuple(mo)

        if self.tile_id_map is None:
            self.read_tile_id_map()

        coords = utils.build_tiles_coords(self.tile_id_map)
        if coords is None:
            print('no coords were build')
            return None

        # Create dictionary of default margins
        def_margin = 500
        margins = (def_margin,) * 4
        overrides = {coord: margins for coord in coords}

        if self.cxy is None:
            # _ = self.get_coarse_mat()
            pass

        # In case of warping without coarse offsets use default margins
        if self.cxy is None:
            print(f'No cxy loaded, returning only overrides defined by rim size ({rim})')
            margins = (rim,) * 4
            overrides = {coord: margins for coord in coords}
            self.margin_overrides = overrides

        # Compute tile offsets from optimized coarse mesh
        self.build_mesh_offsets()

        # Modify overrides based on coarse mesh offsets
        sbem_grid = create_sbem_grid()
        for tid in np.unique(self.tile_id_map):
            if tid == -1:
                continue

            # Get TileXY coordinate and compute new margin
            moy, mox = np.where(tid == self.tile_id_map)
            moy, mox = int(moy[0]), int(mox[0])
            overrides[mox, moy] = build_tile_margins(tid, sbem_grid)

        # self.margin_overrides = verify(overrides)
        self.margin_overrides = overrides

        return overrides

    def loc_inf(self) -> Optional[Tuple[List[Tuple[int, Any, Any]], List[Any]]]:
        """
        # TODO test returned values
        Locate Inf values within the tile ID map.

        Returns:
            Optional[Tuple[List[Tuple[int, Any, Any]], List[Any]]]:
                    A tuple containing lists of coordinates and tile IDs if Inf values
                    are found, otherwise None.
        """

        # Ensure tile ID map is available
        if self.tile_id_map is None:
            self.feed_section_data()

        # Locate Inf values
        try:
            inf_coords, inf_tile_ids = utils.locate_inf(
                cxy=self.get_coarse_mat(),
                tile_id_map=self.tile_id_map,
                section_num=self.section_num
            )
            return inf_coords, inf_tile_ids

        except Exception as e:
            logging.warning(e)
            return None

    def read_tile_id_map(self) -> None:
        fp = self.path / 'tile_id_map.json'
        self.tile_id_map = utils.get_tile_id_map(fp) if fp.exists() else None
        return

    # TODO refactor get_coarse_mat using commented code

    # def get_coarse_mat(self) -> Optional[np.ndarray]:
    #     # Read the coarse matrix file
    #     cx_cy = self.read_coarse_offsets()
    #     if cx_cy is not None:
    #         self.cxy = cx_cy
    #         return cx_cy
    #     return None
    #
    # def read_coarse_offsets(self) -> Optional[np.ndarray]:
    #     fn_cxcy = self.path / 'cx_cy.json'
    #     if not fn_cxcy.exists():
    #         logging.warning(f"Section {self.section_num}: Coarse offsets file does not exist.")
    #         return None
    #     try:
    #         return self.parse_coarse_offsets(fn_cxcy)
    #     except (json.JSONDecodeError, FileNotFoundError) as e:
    #         logging.warning(f"Section {self.section_num}: Error reading coarse offsets - {str(e)}")
    #         return None
    #
    # def parse_coarse_offsets(self, filepath: Path) -> np.ndarray:
    #     with open(filepath, 'r') as file:
    #         data = json.load(file)
    #     cx = data.get('cx', 0)  # Provide default values if not found
    #     cy = data.get('cy', 0)
    #     return np.array([cx, cy])


    def get_coarse_mat(self) -> Optional[np.ndarray]:
        # Read the coarse matrix file
        fn_cxcy = self.path / 'cx_cy.json'
        if fn_cxcy.exists():
            try:
                _, cx, cy = utils.read_coarse_mat(fn_cxcy)
                cx_cy = np.array((cx, cy))
                self.cxy = cx_cy
                return cx_cy
            except (TypeError, FileNotFoundError) as e:
                logging.warning(f's{self.section_num}: reading coarse offsets file failed!')
                return
        else:
            return None

    def load_tile_map(self, clahe: bool = False) -> None:
        """
           Get a tile-data-map mapping tile (x, y) coordinates to the loaded
           image data.

           :return: tile-data-map
           """
        if self.tile_id_map is None:
            self.read_tile_id_map()

        self.tile_map = {}
        h, w = self.tile_id_map.shape

        for y in range(h):
            for x in range(w):
                tile_id = int(self.tile_id_map[y, x])
                if tile_id != -1:
                    tile_path = self.tile_dicts[tile_id]
                    if tile_path is not None and Path(tile_path).exists():
                        logging.debug(f'Loading tile: {Path(tile_path).name}')
                        img = skimage.io.imread(tile_path)
                        img = utils.apply_clahe(img) if clahe else img
                        self.tile_map[(x, y)] = img
                    else:
                        logging.warning(f'Missing tile in raw section data (!): {tile_path}')
                        # missing_tile_path = str(
                        #     self.path / f"20220429_scr_Ruth_Run0001_g0000_t0{tile_id}_s0{self.section_num}.tif")
                        missing_tile_path = ''
                        if Path(missing_tile_path).exists():
                            print('loading missing tile path')
                            img = skimage.io.imread(missing_tile_path)
                            img = utils.apply_clahe(img) if clahe else img
                            self.tile_map[(x, y)] = img
                        else:
                            logging.warning(f"section s{self.section_num}: loading tile-map failed.")
                            self.tile_map = None
                            return
        return

    def get_dummy_coarse_mat(self) -> np.ndarray:
        if self.tile_id_map is None:
            self.read_tile_id_map()
        return utils.create_coarse_mat(self.tile_id_map)

    def save_coarse_mat(self, new_cxy: Optional[np.ndarray] = None):
        if new_cxy is not None and isinstance(new_cxy, np.ndarray):
            utils.save_coarse_mat(new_cxy, self.path, file_format='json')
        elif self.cxy is not None:
            utils.save_coarse_mat(self.cxy, self.path, file_format='json')
        else:
            print(f'Saving coarse mat not performed: cxy not initialized nor specified!')

    def neighbour_path(self, rng_n: Optional[int] = None) -> Optional[str]:
        """
        Get the section path of the current section's neighbor.

        Look to lower or higher section numbers defined by the input parameter.

        :param rng_n: Section numbers range towards smaller (negative rng_n) or higher (positive) section numbers.
        :return: The path of the neighboring section, or None if not found.
        """

        if rng_n is not None:
            step = -1 if rng_n < 0 else 1
            for i in range(step, rng_n, step):
                sec_name = f's{self.section_num + i}_g{self.grid_num}'
                dir_adj = self.path.parent / sec_name
                if dir_adj.exists():
                    return str(dir_adj)
        else:
            print('Neighbour section distance not defined. Please define range rng_n.')

        return None



    def load_image(self) -> Optional[np.ndarray]:

        if self.path.resolve().suffix == '.tif':
            data = skimage.io.imread(self.path)

        elif self.path.resolve().suffix == '.zarr':
            fp = self.path / '0'
            logging.info(f'reading: {fp}')
            data = utils.read_zarr_volume(fp)
        else:
            logging.info(f'reading: {self.path_stitched}')
            data = utils.read_zarr_volume(self.path_stitched)

        if data is None:
            logging.warning(f's{self.section_num} image not loaded!')
            return

        self.image = np.asarray(data['0'])

        try:
            self.height, self.width = self.image.shape
        except ValueError as _:
            logging.warning(f'Loading s{self.section_num} image-data failed: Wrong image dimensionality.')
            return None

        print(f'Image of section s{self.section_num} loaded.')
        return self.image

    def get_tile_parameters(self, tile_id: int, parameters: Union[str, Iterable[str]]) -> Union[str, Dict[str, str]]:
        """
            Get the specified parameters from stats.yaml for a given tile ID.

            :param tile_id: Tile ID to retrieve the parameters for.
            :param parameters: A set of parameter keys to extract values,
                                or a single parameter key (default is 'ssim').
            :return: Value of specified parameter for given tile ID
                     if a single parameter is provided, or a dictionary
                     containing the requested parameters and their values
                     for the given tile ID.
            """

        if not Path(self.path_stats_yaml).exists():
            raise FileNotFoundError("stats.yaml file not found. Initialize it first.")

        data = yaml.load(Path(self.path_stats_yaml))
        tile_data = next((tile for tile in data['tiles'] if tile['tile_id'] == tile_id), None)

        if tile_data is None:
            raise ValueError(f"TileID {tile_id} not found in the 'stats.yaml' file.")

        if isinstance(parameters, str):
            v = tile_data.get(parameters, None)
            if isinstance(v, ScalarFloat):
                v = float(v)  # Convert ScalarFloat to float
            return v
        elif isinstance(parameters, Iterable):
            result = {key: tile_data.get(key, None) for key in parameters}
            # Retype ScalarFloat object to float
            for k, v in result.items():
                if isinstance(v, ScalarFloat):
                    result[k] = float(v)  # Convert ScalarFloat to float
            return result
        else:
            raise ValueError("Invalid type for 'parameters'. Provide a single parameter key or an iterable of keys.")

    def read_tile_shift(self, tile_id: int) -> List[Num]:
        parameters = ('shift_to_prev_tile_x', 'shift_to_prev_tile_y')
        shift_dict = self.get_tile_parameters(tile_id, parameters)
        if None in shift_dict.values():
            raise ValueError(f"TileID {tile_id} or shift data not found in the 'stats.yaml' file.")
        shift = list(map(float, shift_dict.values()))
        return shift



    def downscale_section(self, fct: float) -> Optional[np.ndarray]:
        """Rescale section by specified factor"""

        if self.image is None:
            self.load_image()

        if self.image is None:
            return None

        resized = utils.downscale_image(self.image, fct)
        logging.info(f'shape mini: {np.shape(resized)}')
        self.thumb = resized
        return resized

    def display_image(self, *args):
        mode = args[0]
        if mode == "thumb":
            if self.thumb is not None:
                utils.plot_img(data=self.thumb)
            else:
                print("No thumbnail was found. Downscale the original image first.")
        elif mode == "regular":
            if self.image is not None:
                utils.plot_img(self.image)
            else:
                print("No image was found. Load the original image first.")
        elif mode == "roi":
            if self.image is not None:
                utils.plot_img(self.roi_inspect)
            else:
                print("Can't display non-existing roi_inspect image.")
        elif mode == "conv":
            if self.image is not None:
                utils.plot_img(self.roi_conv)
            else:
                print("Can't display non-existing roi_inspect image.")

    def find_contours(self, level: float):
        contours = []
        if self.filtered_image is not None:
            contours = skimage.measure.find_contours(self.filtered_image / 255.0, level)
        return contours

    def identify_gaps(self, data: np.ndarray, kernel_length=6):
        """
        Identify presence and location of gaps between stitched tiles
        Perform low-pass filtering to remove false positives before convolution
        :return:  tuple of gaps' xy location(s) and image showing the gaps
        """
        data = utils.filter_low_pass(data, 0)
        coordinates, conv = utils.find_black_stripe(data, kernel_length)
        conv = 255 * np.asarray(conv)
        self.roi_conv = conv
        return coordinates, conv

    def display_contours(self, contours):
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(self.filtered_image, cmap=plt.cm.gray)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def process_mssim(self, ssim_value=None) -> Optional[float]:
        """
        Write new mean_ssim value into stats.yaml file.
        Return mean_ssim value computed from tiles' ssim entries
        if nothing is passed to 'ssim_value' function argument.

        :param ssim_value: value to be written into .yaml file
        :return: mean SSIM computed as an average of all tiles' SSIM
        """
        if not Path(self.path_stats_yaml).exists():
            raise FileNotFoundError("stats.yaml file not found. Initialize it first.")

        with open(self.path_stats_yaml, 'r') as file:
            data = yaml.load(file)
            keys = list(data)
            if 'mean_ssim' not in keys:
                to_ind = keys.index('tile_overlap')
                data.insert(to_ind + 1, 'mean_ssim', None)

        if ssim_value is None:
            # Compute mean_ssim
            if 'tiles' in data and isinstance(data['tiles'], list):
                # Compute the mean SSIM value from all tiles
                ssim_values = [tile.get('ssim', None) for tile in data['tiles']]
                ssim_values = [value for value in ssim_values if value is not None]
                ssim_value = sum(ssim_values) / len(ssim_values) if ssim_values else None
                ssim_value = round(ssim_value, 3)

        data['mean_ssim'] = ssim_value

        # Write the updated YAML back to the file
        with open(self.path_stats_yaml, 'w') as file:
            yaml.dump(data, file)

        return ssim_value

    def mssim(self) -> Optional[float]:
        stats_yaml_path = Path(self.path_stats_yaml)

        if not stats_yaml_path.exists():
            raise FileNotFoundError("stats.yaml file not found. Initialize it first.")

        with open(stats_yaml_path, 'r') as file:
            data = yaml.load(file)
            return data.get('mean_ssim')

    def plot_ov(self,
                tid_a: int,
                tid_b: int,
                shift_vec: Optional[Vector] = None,
                dir_out: Optional[UniPath] = None,
                blur: float = 0,
                show_plot=False,
                clahe=False,
                rotate_vert=False,
                store_to_root=False) -> None:

        """Visualize overlap region of a tile-pair. Shift vector must be
        computed in advance.
        
        :param dir_out: parent directory where overlap image will be stored
        :param shift_vec: Optional tuple of two ints
                - if None, do not plot anything
                - if any of values in the input is None, shift vector will be
                  read form cx_cy.json
                - if shift_vec is specified, stitch tile-pair using it
        :param tid_a: tile_id of the first tile
        :param tid_b: tile_id of the second tile
        :param show_plot: visualize output
        :param clahe: apply CLAHE before visualization
        :param blur: apply Gaussian blur to output image
        :param rotate_vert: rotate stored overlap of horizontal tile-pair by 90 deg
                            clockwise
        :param store_to_root: If True, save resulting image into a dir_out folder
                                otherwise create a sub-folder in dir_out.
        :return: None
        """
        assert tid_a != tid_b

        if self.tile_dicts is None:
            self.feed_section_data()

        if tid_a not in self.tile_dicts or tid_b not in self.tile_dicts:
            logging.info('plot_ov: wrong tile_ids specification')
            return

        # Fix ordering of tiles
        tid_a, tid_b = min(tid_a, tid_b), max(tid_a, tid_b)

        path_a = self.tile_dicts[tid_a]
        path_b = self.tile_dicts[tid_b]

        # Load image data
        if Path(path_a).exists() and Path(path_b).exists():
            img_a = skimage.io.imread(path_a)
            img_b = skimage.io.imread(path_b)
            if clahe:
                img_a, img_b = [utils.apply_clahe(img) for img in (img_a, img_b)]
            is_vert = utils.pair_is_vertical(self.tile_id_map, tid_a, tid_b)

            tile_map = {(0, 0): img_a}
            if is_vert:
                axis = 1
                tile_map[(0, 1)] = img_b
            else:
                tile_map[(1, 0)] = img_b
                axis = 0
        else:
            logging.warning("Image files could not be loaded")
            return

        if not tile_map:
            logging.info("Tile_map is empty!")
            return

        # Get shift vector if not specified in input
        if shift_vec is None or None in shift_vec:
            shift_vec = utils.get_shift(self.cxy, self.tile_id_map, tid_a, axis)
            logging.info(f's{self.section_num} loaded coarse offset: {shift_vec}')

        # Visualize and store overlap image
        if shift_vec is None:
            logging.info(f"t{tid_a}: nothing to plot")
            return

        if dir_out is not None:
            dir_ov = Path(dir_out)
            str_tid_a, str_tid_b = f't{tid_a:04d}', f't{tid_b:04d}'
            if not store_to_root:
                dir_ov = Path(dir_out) / f'{str_tid_a}_{str_tid_b}'
                utils.create_directory(dir_ov)

            # Create plot filename
            plot_name = f's{self.section_num:04d}_{str_tid_a}_{str_tid_b}_ov.jpg'
            path_plot = str(dir_ov / plot_name)
            logging.info(f'plotting: {path_plot}')
        else:
            path_plot = None  # Do not store the OV image to HDD

        # Get stitched image
        img_pair = sutils.plot_tile_pair(
            tile_map, shift_vec, show_plot=False,
            path_plot=None, blur=1.0, img_only=True
        )

        # Crop overlap from stitched image and save it
        if img_pair is not None:
            _ = sutils.plot_thin_image(img_pair, is_vert, path_plot, show_plot, blur, rotate_vert)

        return

    def plot_tile_pair(self, tid_a: int, tid_b: int, clahe: bool = True,
                       shift_vec: Optional[Vector] = None, masking=False,
                       blur: float = 1.0, img_only=False) -> None:

        logging.info(f'Plotting t{tid_a}-t{tid_b} tiles')

        assert tid_a != tid_b
        if self.tile_dicts is None:
            self.feed_section_data()

        tiles = self.load_image_pair(tid_a, tid_b, clahe=clahe)
        if tiles is None:
            logging.warning('Tile reading failed')
            return None

        a, b = tiles
        is_vert = utils.pair_is_vertical(self.tile_id_map, tid_a, tid_b)
        axis = 1 if is_vert else 0

        # Get shift vector if not specified in input
        if shift_vec is None or None in shift_vec:
            shift_vec = utils.get_shift(self.cxy, self.tile_id_map, tid_a, axis)
            logging.info(f's{self.section_num} loaded coarse offset: {shift_vec}')

        if shift_vec is None:
            print(f's{self.section_num} t{tid_a}_t{tid_b}: shift vector contains Inf value.')
            shift_vec = (np.inf, np.inf)

        # Load and apply masks
        if masking:
            def apply_mask(image, mask):
                modified = image.copy()
                modified = modified.astype(np.float32)
                modified[mask] = np.nan
                return modified

            self.load_masks()
            self.read_tile_id_map()

            default_mask = np.full_like(a, fill_value=False, dtype=np.bool_)

            y, x = np.where(tid_a == self.tile_id_map)
            mask_a = self.mask_map.get((int(x[0]), int(y[0])), default_mask)

            y, x = np.where(tid_b == self.tile_id_map)
            mask_b = self.mask_map.get((int(x[0]), int(y[0])), default_mask)

            a.img_data = apply_mask(a.img_data, mask_a)
            b.img_data = apply_mask(b.img_data, mask_b)

        # Get tile map
        tile_map = {(0, 0): a.img_data}
        if is_vert:
            tile_map[(0, 1)] = b.img_data
        else:
            tile_map[(1, 0)] = b.img_data

        # Create plot filename
        str_sec = 's' + str(self.section_num).zfill(4)
        plot_name = str_sec + '_' + str(tid_a) + '_' + str(tid_b) + '_ov.jpg'
        path_plot = str(self.path / plot_name)

        _ = sutils.plot_tile_pair(tile_map, shift_vec, show_plot=False,
                                  blur=blur, path_plot=path_plot)
        return


    def plot_masks(self) -> None:

        def get_mask_shapes(masks: MaskMap) -> Optional[Dict[TileXY, Optional[Tuple[int, int]]]]:
            if masks == {}:
                logging.warning('WARNING: masks dict is empty!')
                return None

            msk_shapes = {}
            for key, msk in masks.items():
                msk_shapes[key] = msk.shape if isinstance(msk, np.ndarray) else None
            return msk_shapes

        def get_msk_shape(mapped_shapes: Optional[Dict[TileXY, Optional[Tuple[int, int]]]]
                          ) -> Optional[Tuple[int, int]]:
            if mapped_shapes is None:
                return None

            res = None
            for k, mask_shape in mapped_shapes.items():
                if isinstance(mask_shape, tuple):
                    res = mask_shape
                    break
            return res

        def plot_any_masks(mask_map: MaskMap, mask_shape: Tuple[int, int], ext: str):
            path_plot = self.path / f'mask_map_{ext}.jpg'
            mutils.plot_tile_masks(
                mask_map=mask_map,
                path_plot=path_plot,
                show=False,
                tid_map=self.tile_id_map,
                mask_shape=mask_shape)
            return

        def plot_main():
            graph_modes = 'margin', 'smr', 'roi', 'comb'

            if self.margin_masks is None:
                self.margin_masks = {}
            else:
                margin_mask_with_none = {k: v for k, v in self.roi_mask_map.items()}
                for k in self.margin_masks.keys():
                    margin_mask_with_none[k] = self.margin_masks[k]
                    self.margin_masks = margin_mask_with_none

            data = (self.margin_masks,
                    self.smr_mask_map,
                    self.roi_mask_map,
                    self.mask_map)

            graph_shapes = [get_msk_shape(get_mask_shapes(m)) for m in data]
            for i, (mask_map, shp) in enumerate(zip(data, graph_shapes)):
                if mask_map != {} and shp is not None:
                    # pass
                    plot_any_masks(mask_map, shp, graph_modes[i])
                else:
                    continue

        self.load_masks()
        self.read_tile_id_map()
        plot_main()
        return

    def get_masked_img_pair(self, tid_a: int, tid_b: int,
                            masking: bool,
                            clahe: bool = True,
                            blur_fct: float = 1.0,
                            shift_vec: Optional[Vector] = None,
                            custom_params: Tuple[int, int, int, int] = (0, 0, 0, 0)
                            ) -> Optional[Tuple[TileMap, TileXY, bool, Vector]]:

        def get_mask(x, y, mask_map, default_mask, custom_mask):
            if custom_mask:
                return np.copy(default_mask)
            ma = mask_map.get((x, y), default_mask)
            return default_mask if ma is None else ma

        def apply_mask(image, mask):
            modified = image.copy()
            modified = modified.astype(np.float32)
            modified[mask] = np.nan
            return modified

        def make_custom_mask(input_mask: np.ndarray,
                             top: int,
                             bottom: int,
                             left: int,
                             right: int):
            """Mask lines or columns (for 'eval_ov' procedure)"""
            if all(custom_params):
                if tid_b == utils.get_vert_tile_id(self.tile_id_map, tid_a):
                    top, bottom = 0, 0
                else:
                    left, right = 0, 0

            if top == 0:
                input_mask[:, -right:] = True
                input_mask[:, :left] = True
            if left == 0:
                input_mask[:top] = True
                input_mask[-bottom:] = True
            return input_mask

        # Determine tile-pair orientation
        msg = f'Specified tile_ids {tid_a}, {tid_b} are not neighbors!'
        assert tid_a != tid_b, msg
        is_vert = utils.pair_is_vertical(self.tile_id_map, tid_a, tid_b)
        if is_vert is None:
            logging.warning(msg)
            return

        if self.tile_dicts is None:
            self.feed_section_data()

        tiles = self.load_image_pair(tid_a, tid_b, clahe, blur_fct)
        if tiles is None:
            logging.warning(f'Unable to load tile image data (s{self.section_num}: t{tid_a}, t{tid_b}).')
            return

        # Get shift vector if not specified in input
        axis = 1 if is_vert else 0
        if shift_vec is None or None in shift_vec:
            shift_vec = self.get_coarse_offset(tid_a, axis)
            logging.info(f's{self.section_num} t{tid_a}-t{tid_b} loaded coarse offset: {shift_vec}')

        if shift_vec is None:
            print(f's{self.section_num} t{tid_a}_t{tid_b}: shift vector contains Inf value.')
            shift_vec = (np.inf, np.inf)

        a, b = tiles
        # Load and apply masks
        if masking:
            custom_mask = any(custom_params)
            if len(self.mask_map) == 0 and not custom_mask:
                self.load_masks()

            # Get masks
            default_mask = np.full_like(a.img_data, fill_value=False, dtype=np.bool_)
            ya, xa = np.where(tid_a == self.tile_id_map)
            yb, xb = np.where(tid_b == self.tile_id_map)
            mask_a = get_mask(xa[0], ya[0], self.roi_mask_map, default_mask, custom_mask)
            mask_b = get_mask(xb[0], yb[0], self.roi_mask_map, default_mask, custom_mask)

            # Custom mask definition
            if custom_mask:
                mask_a = make_custom_mask(mask_a, *custom_params)
                mask_b = make_custom_mask(mask_b, *custom_params)

            a.img_data = apply_mask(a.img_data, mask_a)
            b.img_data = apply_mask(b.img_data, mask_b)

        # Get tile map
        tile_map = {(0, 0): a.img_data}
        if is_vert:
            tile_map[(0, 1)] = b.img_data
            pair_id_map = np.array([[tid_a], [tid_b]])
        else:
            tile_map[(1, 0)] = b.img_data
            pair_id_map = np.array([[tid_a, tid_b]])

        tile_space: Tuple[int, int] = np.shape(pair_id_map)

        return tile_map, tile_space, is_vert, shift_vec

    def eval_seam(self,
                  tile_map: TileMap,
                  tid_a: int,
                  tid_b: int,
                  shift_vec: Vector,
                  is_vert: bool,
                  dir_out: Optional[UniPath] = None,
                  plot=False,
                  show_plot=False,
                  **kwargs
                  ) -> Tuple[Optional[float], List[str]]:

        # Get stitched image (without plotting)
        img_pair = sutils.plot_tile_pair(tile_map,
                                         shift_vec,
                                         path_plot=None,
                                         show_plot=False,
                                         blur=1.0,
                                         img_only=True)
        log_mes = []
        if img_pair is None:
            return np.nan, log_mes

        # Crop overlap from stitched image and save it
        tile_shape: Tuple[int, int] = tile_map[(0, 0)].shape
        ov_img = utils.crop_ov(ov_img=img_pair, pad=700, offset=shift_vec,
                               is_vert=is_vert, tile_shape=tile_shape)
        # Create plot name
        if not plot:
            path_plot = None
        else:
            str_tid_a = 't' + str(tid_a).zfill(4)
            str_tid_b = 't' + str(tid_b).zfill(4)
            dir_name = str_tid_a + '_' + str_tid_b
            dir_ov = Path(dir_out) / dir_name
            utils.create_directory(dir_ov)

            # Create plot filename
            str_sec = '_s' + str(self.section_num).zfill(4)
            plot_name = str_sec + '_' + str_tid_a + '_' + str_tid_b + '_ov.jpg'
            path_plot = str(dir_ov / plot_name)

        args = dict(
            ov_img=ov_img,
            is_vert=is_vert,
            plot=plot,
            path_plot=path_plot,
            show_plot=show_plot)
        res = utils.detect_bad_seam(**args)

        if res is None:
            return np.nan, log_mes

        mes = f'grid vector, seam inaccuracy: {shift_vec}, {res:.3f}'
        return res, log_mes

    def load_masked_pair(self,
                         tid_a: int,
                         tid_b: int,
                         roi: bool,
                         smr: bool,
                         gauss_blur=False,
                         sigma=1.5
                         ) -> Optional[Tuple[Tile, Tile]]:
        """Load a pair of masked tiles.

            Args:
                tid_a (int): Tile ID of the first tile.
                tid_b (int): Tile ID of the second tile.
                roi (bool): Whether to apply the ROI mask.
                smr (bool): Whether to apply the SMR mask.
                :param sigma:
                :param gauss_blur:

            Returns:
                Optional[Tuple[Tile, Tile]]: A tuple containing two Tile objects if the tiles were loaded successfully,
                otherwise None.

            """

        def apply_mask(image, mask):
            modified = image.copy()
            modified = modified.astype(np.float32)
            modified[mask] = np.nan
            return modified

        # Load masked tile pair
        if self.tile_dicts is None:
            self.feed_section_data()

        tiles = self.load_image_pair(tid_a, tid_b, clahe=True)
        if tiles is None:
            logging.warning(f'Unable to load tile image data (s{self.section_num}: t{tid_a}, t{tid_b}).')
            return

        if None in tiles:
            logging.warning(f'Unable to load s{self.section_num} t{tid_a} or t{tid_b}.')
            return

        if gauss_blur:
            for tile in tiles:
                if isinstance(tile.img_data, np.ndarray):
                    tile.img_data = filters.gaussian(tile.img_data, sigma=sigma)
                    continue
                logging.warning(f'Unable to load s{self.section_num} t{tile.tile_id}.')
                return

        # Load and apply ROI masks
        if not any((roi, smr)):
            return tiles

        if len(self.mask_map) == 0:
            self.load_masks()

        if self.tile_id_map is None:
            self.read_tile_id_map()

        a, b = tiles
        default_mask = np.full_like(a.img_data, fill_value=False, dtype=np.bool_)
        mask_a = np.copy(default_mask)
        mask_b = np.copy(default_mask)

        ya, xa = np.where(tid_a == self.tile_id_map)
        yb, xb = np.where(tid_b == self.tile_id_map)
        ya, xa = int(ya[0]), int(xa[0])
        yb, xb = int(yb[0]), int(xb[0])

        if roi:
            ma = self.roi_mask_map.get((xa, ya), default_mask)
            mb = self.roi_mask_map.get((xb, yb), default_mask)
            if ma is not None:
                mask_a |= ma
            if mb is not None:
                mask_b |= mb
        if smr:
            ma = self.smr_mask_map.get((xa, ya), default_mask)
            mb = self.smr_mask_map.get((xb, yb), default_mask)
            if ma is not None:
                mask_a |= ma
            if mb is not None:
                mask_b |= mb

        a.img_data = apply_mask(a.img_data, mask_a)
        b.img_data = apply_mask(b.img_data, mask_b)
        return a, b

    def eval_ov(self,
                tile_pair: Tuple[Tile, Tile],
                offset: Vector,
                plot_pair=False,
                half_width=50
                ) -> float:

        def check_zero_dimension(image_array):
            return any(dim == 0 for dim in image_array.shape)

        def plot_eval_pair(a, b):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(a, cmap='gray')
            ax1.set_title('Image 1')
            ax2.imshow(b, cmap='gray')
            ax2.set_title('Image 2')
            plt.show()
            return

        MIN_OV_WIDTH = 15

        tile_a, tile_b = tile_pair
        logging.debug(f'eval_ov tile-pair: {tile_a.tile_id, tile_b.tile_id}')

        is_vert = utils.pair_is_vertical(
            self.tile_id_map, tile_a.tile_id, tile_b.tile_id)

        if tile_a.img_data is None:
            tile_a.load_image(clahe=True)
        if tile_b.img_data is None:
            tile_b.load_image(clahe=True)

        # Rotate vertical tile-pair to work with horizontal stripe
        if is_vert:
            img_a = np.rot90(tile_a.img_data, k=1)
            img_b = np.rot90(tile_b.img_data, k=1)
            offset = (-offset[0], offset[1])
            axis = 1
        else:
            axis = 0
            img_a = tile_a.img_data
            img_b = tile_b.img_data

        # Crop common area
        logging.debug(f'image a shape: {np.shape(img_a)}')
        logging.debug(f'image b shape: {np.shape(img_b)}')
        h, _ = np.shape(img_a)

        ov_a = img_a[max(0, offset[1 - axis]):min(h, h + offset[1 - axis])]
        ov_a = ov_a[:, -abs(offset[axis]):]

        ov_b = img_b[max(0, -offset[1 - axis]):min(h, h - offset[1 - axis])]
        ov_b = ov_b[:, :abs(offset[axis])]

        if check_zero_dimension(ov_a):
            logging.warning('Cropping first overlap overlap resulted in error')
            return np.nan
        if check_zero_dimension(ov_b):
            logging.warning('Cropping second overlap overlap resulted in error')
            return np.nan

        # Remove masked regions
        ov_stacked = np.rot90(np.hstack((ov_a, ov_b)), k=-1)
        ov_stacked = utils.crop_nan(ov_stacked)
        logging.debug(f'eval_ov: cropped stack ov shape {ov_stacked.shape}')

        if any((size < MIN_OV_WIDTH for size in ov_stacked.shape)):
            logging.info(f'eval_ov: s{self.section_num} cropped ov-shape is under limit ({MIN_OV_WIDTH} pixels)')
            return np.nan

        # Split stacked ov-images
        ov_ac = ov_stacked[:ov_a.shape[1]]
        ov_bc = ov_stacked[ov_a.shape[1]:]
        # utils.plot_images_with_overlay(ov_ac, ov_bc)

        # Perform crop around intended seam position
        ov_h = np.shape(ov_ac)[0]
        seam_pos = int(ov_h / 2)
        hw_ov = min(half_width, ov_h // 2)
        if ov_h > 2 * MIN_OV_WIDTH:
            ov_ac = ov_ac[seam_pos - hw_ov:seam_pos + hw_ov]
            ov_bc = ov_bc[seam_pos - hw_ov:seam_pos + hw_ov]

        # Compute SSIM over area
        ov_ac, ov_bc = map(utils.norm_img, (ov_ac, ov_bc))
        mssim = ssim(ov_ac, ov_bc)
        mssim = (mssim + 1) / 2
        logging.debug(f'mssim: {mssim:.3f}')
        if plot_pair:
            plot_eval_pair(ov_ac, ov_bc)

        return mssim


    def eval_section_overlaps(self, roi_masking=False, smr_masking=False) -> np.ndarray:

        def process_ov(tid_a, tid_b) -> Optional[float]:
            is_vert = utils.pair_is_vertical(self.tile_id_map, tid_a, tid_b)
            if is_vert is None:
                logging.warning(f'Specified tile_ids are not neighbors!')
                return

            tile_pair = self.load_masked_pair(tid_a, tid_b, roi_masking, smr_masking)
            if tile_pair is None:
                return

            axis = 1 if is_vert else 0
            shift_vec = self.get_coarse_offset(tid_a, axis)
            val = self.eval_ov(tile_pair, shift_vec)

            return val

        # Compute
        logging.info(f'eval_section_overlaps: s{self.section_num}, ROI: {roi_masking}, smr: {smr_masking}')
        if not self.tile_dicts:
            self.feed_section_data()
        self.load_tile_map()
        self.load_masks()
        yx_shape = np.shape(self.tile_id_map)

        # Refine horizontal neighbors
        eov_x = np.full((yx_shape[0], yx_shape[1]), np.nan)
        for x in range(0, yx_shape[1] - 1):
            for y in range(0, yx_shape[0]):
                tile_id_a = int(self.tile_id_map[y, x])
                tile_id_b = int(self.tile_id_map[y, x + 1])
                if tile_id_a == -1 or tile_id_b == -1:
                    continue
                eov_x[y, x] = process_ov(tile_id_a, tile_id_b)

        # Refine vertical neighbors
        eov_y = np.full((yx_shape[0], yx_shape[1]), np.nan)
        for y in range(0, yx_shape[0] - 1):
            for x in range(0, yx_shape[1]):
                tile_id_a = int(self.tile_id_map[y, x])
                tile_id_b = int(self.tile_id_map[y + 1, x])
                if tile_id_a == -1 or tile_id_b == -1:
                    continue
                eov_y[y, x] = process_ov(tile_id_a, tile_id_b)

        return np.array((eov_x, eov_y))


    def replace_coarse_offset(self,
                              coord: Tuple[int, ...],
                              offset: Union[float, Tuple[float, float]],
                              store: bool) -> None:
        """
        Replace coarse offsets in cx_cy.json at the specified coordinate
        with the given offset vector.

        Parameters:
        - coord (Tuple[int, int, int, int]): The 4D coordinate (C, Z, X, Y).
        - offset (Union[float, Tuple[float, float]]): The offset value or a tuple of offsets.

        Returns:
        - None
        """

        path_cx_cy = self.path / 'cx_cy.json'
        assert len(coord) == 4

        c, z, y, x = coord

        if c == 0:
            tile_id_a = self.tile_id_map[y, x]
            tile_id_b = self.tile_id_map[y, x + 1]
        else:
            tile_id_a = self.tile_id_map[y, x]
            tile_id_b = self.tile_id_map[y + 1, x]

        logging.info(f"Section {self.section_num} tile-pair IDs: {int(tile_id_a), int(tile_id_b)}")

        # Read coarse mat if not already loaded
        if self.cxy is None and path_cx_cy.exists():
            _, cx, cy = utils.read_coarse_mat(path_cx_cy)
            self.cxy = np.array((cx, cy))

        # Check if there's anything to replace
        if self.cxy is None:
            print('Nothing to replace. Coarse offsets .json is missing.')
            return

        if len(offset) == 2:
            self.cxy[c, :, y, x] = offset
            logging.info(f'Replacing original coarse offset with {offset}')
        else:
            self.cxy[coord] = offset
            logging.info(f'Replacing coarse vector {self.cxy[coord]} at '
                         f'coordinate: {coord}')

        # Save coarse mat
        if store:
            utils.save_coarse_mat(self.cxy, path_cx_cy.parent, file_format="json")
        else:
            logging.warning('Storing is disabled. Coarse offset will not be written out.')
        return

    def load_image_pair(self,
                        id_a: int,
                        id_b: int,
                        clahe: bool = True,
                        blur_fct: float = 1.0) -> Optional[Tuple[Tile, Tile]]:

        if self.tile_dicts is None:
            self.feed_section_data()

        tiles = []
        for tid in (id_a, id_b):
            try:
                tile = Tile(self.tile_dicts[tid])
                tile.load_image(clahe)
                if blur_fct > 1:
                    tile.img_data = filters.gaussian(tile.img_data, sigma=blur_fct)
                tiles.append(tile)
            except KeyError:
                logging.warning(f'Tile t{tid} not present in s{self.section_num}')
                return None

        return tuple(tiles)

    def compute_coarse_offset(self,
                              id_a: int,
                              id_b: int,
                              refine: bool,
                              store: bool,
                              clahe: bool,
                              overlaps_xy=((200, 300), (200, 300)),
                              min_range=(10, 100, 0),
                              min_overlap=10,
                              filter_size=10,
                              masking=False,
                              max_valid_offset=350,
                              est_vec: Optional[Vector] = None,
                              custom_mask_params: Tuple[int, int, int, int] = (0, 0, 0, 0),
                              co_score_lim: Optional[float] = None,
                              co_lim_dist: Optional[float] = None
                              ) -> Optional[Union[tuple[int, ...], ndarray[Any, dtype[Any]]]]:

        def co_score_valid(co: Optional[Vector] = None,
                           lim_val: Optional[float] = None
                           ) -> bool:
            # Compute current seam quality and skip refinement if meets criteria
            if lim_val is None:
                return False

            if np.nan not in co or np.inf not in co and co is not None:
                tile_pair = self.load_image_pair(id_a, id_b, blur_fct=6.0)
                co_score = self.eval_ov(tile_pair, co)
                if co_score > co_score_lim:
                    msg = f's{self.section_num} t{id_a}-t{id_b} coarse offset vector quality already sufficient ({co_score}>{co_score_lim})'
                    print(msg), logging.warning(msg)
                    return True
            return False

        def co_dist_valid(co: Optional[Vector] = None,
                          est_vec: Optional[Vector] = None,
                          co_lim_dist: Optional[float] = 15.
                          ) -> bool:
            # Verifies that the absolute pixel distance of current coarse offset is within limits from estimated vector

            if est_vec is None or co is None:
                return False

            abs_diff = np.abs(np.asarray(co) - np.asarray(est_vec))
            abs_dist = np.linalg.norm(abs_diff)
            vec_valid = abs_dist < co_lim_dist
            if not vec_valid:
                msg = f's{self.section_num:04d} t{id_a}-t{id_b}: coarse offset vector deviation from est. vec. too large ({int(abs_dist)} pix., limit={int(co_lim_dist)} pix.)'
                print(msg), logging.warning(msg)
            return vec_valid

        logging.info(f'Section s{self.section_num:04d} - computing coarse offset t{id_a} - t{id_b}')

        # Load image data
        if self.tile_dicts is None:
            self.feed_section_data()

        # Get tile-map and tile-space of a tile-pair
        is_vert = utils.pair_is_vertical(self.tile_id_map, id_a, id_b)
        if is_vert is None:
            logging.info(f'Specified tile_ids are not neighbors or one of the tile-ids are missing!')
            return

        axis = 1 if is_vert else 0

        # Refine any coarse shifts using refine pyramid method
        if refine:

            # offset = self.get_coarse_offset(id_a, axis)

            # if co_score_valid(offset, co_score_lim):
            #     return offset
            #
            # if co_dist_valid(offset, est_vec, co_lim_dist):
            #     return offset

            shift_vec = self.refine_pyramid(
                id_a, id_b, masking=masking, levels=1, max_ext=30, stride=6,
                clahe=True, store=store, plot=False, show_plot=False,
                est_vec=est_vec, custom_mask_params=custom_mask_params
            )

        else:
            tiles = self.load_image_pair(id_a, id_b, clahe)
            if tiles is None:
                logging.debug(f'Unable to load tile image data (s{self.section_num:04d}: t{id_a}, t{id_b}).')
                return

            tile_a, tile_b = tiles
            tile_map = {(0, 0): tile_a.img_data}
            if is_vert:
                tile_map[(0, 1)] = tile_b.img_data
                pair_id_map = np.array([[id_a], [id_b]])
            else:
                tile_map[(1, 0)] = tile_b.img_data
                pair_id_map = np.array([[id_a, id_b]])

            if not masking:
                mask_map = None
            else:
                self.load_masks()

                # For dummy mask map indices
                k0: (int, int) = tuple(np.squeeze(np.where(self.tile_id_map == id_a))[::-1])
                k1: (int, int) = tuple(np.squeeze(np.where(self.tile_id_map == id_b))[::-1])
                mask_map = {(0, 0): self.mask_map[k0]}

                top, bottom = 80, 80
                left, right = 80, 80

                assert 0 not in (top, bottom, left, right)

                mask_map[(0, 0)][:top] = True  # Custom mask
                mask_map[(0, 0)][-bottom:] = True  # Custom mask
                #
                # mask_map[(0, 0)][:, :left] = True  # Custom mask
                # mask_map[(0, 0)][:, -right:] = True  # Custom mask

                if is_vert:
                    mask_map[(0, 1)] = self.mask_map[k1]
                    mask_map[(0, 1)][:, :left] = True  # Custom mask
                    mask_map[(0, 1)][:, -right:] = True  # Custom mask
                else:
                    mask_map[(1, 0)] = self.mask_map[k1]
                    mask_map[(1, 0)][:top] = True
                    mask_map[(1, 0)][-bottom:] = True

            # for mm in mask_map.values():
            #     plt.imshow(mm*255, cmap='gray')
            #     plt.show()

            tile_space: Union[Tuple[int, int], Tuple[int, ...]]
            tile_space = np.shape(pair_id_map)

            # Read coarse offset if refining is desired
            co = self.get_coarse_offset(id_a, axis) if refine else None
            if co is not None:
                logging.info(f'Original coarse offset (to be refined): {co}')

            # Check if co is not None and does not contain NaN
            co = co if co is None or not any(np.isnan(x) for x in co) else None

            # Compute offset
            cx, cy = stitch_rigid.compute_coarse_offsets(
                yx_shape=tile_space,
                tile_map=tile_map,
                mask_map=mask_map,
                co=co,
                overlaps_xy=overlaps_xy,
                min_range=min_range,
                min_overlap=min_overlap,
                filter_size=filter_size,
                max_valid_offset=max_valid_offset,
            )

            # # Compute offset (Original)
            # cx, cy = stitch_rigid.compute_coarse_offsets(
            #     yx_shape=tile_space,
            #     tile_map=tile_map,
            #     mask_map=mask_map,
            #     overlaps_xy=overlaps_xy,
            #     min_range=min_range,
            #     min_overlap=min_overlap,
            #     filter_size=filter_size
            # )

            # Get final shift vector
            cx, cy = map(np.squeeze, (cx, cy))
            shift_vec = cy[:, 0] if is_vert else cx[:, 0]

        try:
            shift_vec = tuple(map(int, shift_vec))
        except OverflowError as _:
            logging.info(f's{self.section_num:04d} t{id_a}-t{id_b} Inf value in coarse shift shift_vector.')
            return np.inf, np.inf
        except ValueError as _:
            logging.info(f's{self.section_num:04d} t{id_a}-t{id_b} NaN value in coarse shift shift_vector.')
            return np.inf, np.inf

        if store:
            y, x = np.where(self.tile_id_map == id_a)
            id_a_coord = (axis, 0, int(y[0]), int(x[0]))
            self.replace_coarse_offset(id_a_coord, shift_vec, store)

        return shift_vec

    def estimate_offset(self,
                        tile_id: int,
                        axis: int,
                        before: int = 10,
                        after: int = 10
                        ) -> Optional[Tuple[Vector, Vector]]:

        """Compute mean coarse offset vector from neighboring sections.

        Args:
            tile_id (int): Tile for which the mean coarse shift should be computed.
            axis (int): Orientation of a tile pair (to be used for slicing cxy).
            before (int, optional): Number of sections before the current section
                                    to consider. Defaults to 10.
            after (int, optional): Number of sections after the current section
                                    to consider. Defaults to 10.

        Returns:
            Optional[Tuple[int, int]]: Mean coarse offset vector if computed, else None.
        """

        def get_sec_dirs() -> List[str]:
            exp_dir = str(Path(self.path).parent)
            exp_dir = Path(utils.cross_platform_path(exp_dir))
            grid_num = int(Path(self.path).name[-1])

            # Select sections to be used for mean vector estimation
            sec_num = utils.get_section_num(self.path)
            start = max(sec_num - before, 0)
            end = sec_num + after
            sec_nums = set(range(start, end))

            dirs = [str(exp_dir / f"s{num}_g{grid_num}") for num in sec_nums]
            return dirs

        def aggregate_offsets(sec_dirs: List[str]) -> List[np.ndarray[float]]:
            shift_vecs = []
            for sec_dir in sec_dirs:
                if Path(sec_dir).exists():
                    sec = Section(sec_dir)
                    sec.read_tile_id_map()
                    _ = sec.get_coarse_mat()
                    try:
                        coord = np.where(sec.tile_id_map == tile_id)
                        y, x = int(coord[0][0]), int(coord[1][0])
                    except IndexError:
                        continue

                    try:
                        shift_vec = sec.cxy[axis, :, y, x]
                    except TypeError:
                        continue

                    # Omit Inf offsets to be able to compute mean vector
                    if shift_vec is not None and not any(np.isinf(shift_vec)):
                        shift_vecs.append(shift_vec)
            return shift_vecs

        section_dirs = get_sec_dirs()
        vecs = aggregate_offsets(section_dirs)

        if len(vecs) == 0:
            logging.warning(f's{self.section_num} t{tile_id}: average coarse offset could not be estimated.')
            return None

        # np.nanmean is necessary due to missing offset values if tile has no neighbor
        if np.all(np.isnan(np.array(vecs))):
            logging.warning(f's{self.section_num} t{tile_id}: only NaN values found during mean vec. estimation')
            return None

        mean = np.nanmean(np.array(vecs), axis=0)
        std = np.nanstd(np.array(vecs), axis=0)
        mean = tuple(map(int, np.round(mean)))
        std = tuple(map(int, np.round(std)))

        return mean, std

    def analyze_offset(self, offset: Vector, tile_id: int, axis: int,
                       before=15, after=15, std_band=6) -> Optional[Vector]:

        try:
            mean_offset, std = self.estimate_offset(tile_id, axis, before, after)
        except TypeError:
            return None

        # Check if offset is within bounds
        mean_offset, std = map(np.array, (mean_offset, std_band))
        low_bound = mean_offset - std * std_band
        up_bound = mean_offset + std * std_band

        if np.any((offset > up_bound) | (offset < low_bound)):
            m1 = f"Original coarse offset is out of stddev. bounds."
            b1 = (up_bound[0] - mean_offset[0], low_bound[0] - mean_offset[0])
            b2 = (up_bound[1] - mean_offset[1], low_bound[1] - mean_offset[1])
            vec1 = f"{mean_offset[0]} +/- {b1}"
            vec2 = f"{mean_offset[1]} +/- {b2}"
            m2 = f"Average estimated offset x: {vec1}, y: {vec2}"
            m12 = (m1, m2) if np.inf not in offset else (m2,)
            for m in m12:
                print(m)
                logging.info(m)

        return mean_offset

    def refine_coarse_offset(self,
                             offset: Vector,
                             tid_a: int,
                             tid_b: int,
                             tile_pair: Tuple[Tile, Tile],
                             is_vert,
                             max_ext,
                             stride,
                             orig_co: Optional[Vector] = None,
                             tile_map: Optional[TileMap] = None,
                             ) -> Optional[Tuple[Vector, Tuple[GridXY, GridXY]]]:

        # Create set of shift vectors
        shift_grid, gx, gy = utils.get_shift_grid(max_ext, stride, offset, is_vert)

        # Refine offset and return best result
        gz = []
        for offset in shift_grid:
            # # Method 1
            seam_ssim = self.eval_ov(tile_pair=tile_pair, offset=offset)
            if seam_ssim is None:
                break
            seam_res = 1 / seam_ssim

            # # # Method 2
            # seam_res = self.eval_seam(tile_map, tid_a, tid_b, offset, is_vert)
            # seam_res = seam_res[0]

            gz.append(seam_res)
        if not gz:
            logging.warning(f'Refining offset not successful. Check image masks and consider disabling them.')
            return None

        # Select best offset from computed offset field
        best_offset, refine_data = utils.interp_coarse_grid((gx, gy), gz)
        logging.debug(f'estimated offset: {best_offset}')

        return best_offset, refine_data

    def refine_pyramid(self, tid_a: int, tid_b: int, masking: bool,
                       levels: int, max_ext: int, stride: int, clahe: bool, store: bool,
                       plot: bool, show_plot: bool, est_vec: Optional[Vector] = None,
                       custom_mask_params=(0, 0, 0, 0)
                       ) -> Optional[Vector]:

        """Refine the coarse offset vector between two tiles.

        Args:
            tid_a (int): Tile ID of the first tile.
            tid_b (int): Tile ID of the second tile.
            masking (bool): Load and apply ROI and smearing masks to input images.
            levels (int): number of pyramid search levels for coarse offset refinement.
            max_ext (int): Maximum extent of coarse shift vector search space.
            stride (int): Stride for search.
            clahe (bool): Whether to use CLAHE.
            store (bool): Whether to store the refined vector in cx_cy.json.
            plot (bool): Whether to plot the seam quality maps.
            show_plot (bool): Whether to show the plot.
            est_vec (Vector): Refine shift vector in the vicinity of predefined coarse offset.
            custom_mask_params (Tuple): Each number defines custom mask for ov in eval_ov

        Returns:
            Optional[Vector]: Refined offset vector if successful, else None.
            
        """
        fmsg = f's{self.section_num:04d} t{tid_a}-t{tid_b}'

        # Load image data and original coarse shift vector
        try:
            tile_map, _, is_vert, orig_co = self.get_masked_img_pair(
                tid_a, tid_b, masking=masking, blur_fct=6.0, clahe=clahe,
                shift_vec=None, custom_params=custom_mask_params
            )
            axis = 1 if is_vert else 0
        except TypeError as _:
            print(f'Refine pyramid for {fmsg} failed: masked tile-pair could not be loaded.')
            return

        t1, t2 = Tile(self.tile_dicts[tid_a]), Tile(self.tile_dicts[tid_b])
        t1.img_data, t2.img_data = tile_map.values()

        msg = f'{fmsg}: original shift vector: {orig_co}'
        logging.info(msg), print(msg)

        # Treat non-reliable offsets and estimate mean shift vector from neighboring sections
        # est_vec = None  # !!!
        # est_vec = orig_co
        if est_vec is None:
            est_vec = self.analyze_offset(orig_co, tid_a, axis, before=10, after=10, std_band=4)

            shift_vec = est_vec if est_vec is not None else orig_co
            if est_vec is None:
                msg = 'Mean shift vector from neighbors could not be estimated. Trying original one:'
                print(f'{msg} {orig_co}'), logging.info(msg + f' {orig_co}')
            else:
                msg = f'{fmsg}: mean shift vector from neighbors: {tuple(est_vec)}'
                print(msg), logging.info(msg)
        else:
            shift_vec = est_vec
            msg = f'{fmsg}: refining based on custom coarse offset: {shift_vec}'
            print(msg), logging.info(msg)

        # Terminate in case no mean shift vector could be estimated for Inf orig. coarse offset
        if np.inf in shift_vec:
            logging.warning(f'{fmsg}: coarse offset refinement not performed.')
            return None

        # Run refining using pyramidal search
        for i, (max_ext, stride) in enumerate(utils.get_pyramid(levels, max_ext, stride)):
            try:
                shift_vec, interp_data = self.refine_coarse_offset(shift_vec, tid_a, tid_b, (t1, t2), is_vert, max_ext, stride, orig_co, tile_map)
            except TypeError as _:
                shift_vec = (np.nan, np.nan)  # Refining offset failed for some reason
                continue
            if plot:
                path_plot = str(self.path / str(f'refined_offsets_{fmsg}_{i}.jpg'))
                utils.plot_refined_grid(interp_data, path_plot, show_plot)

        # Evaluate new seam quality
        kwargs = dict(tile_map=tile_map, img_a=t1.img_data, img_b=t2.img_data,
                      tid_a=tid_a, tid_b=tid_b, orig_co=orig_co, new_co=shift_vec,
                      is_vert=is_vert)

        quality_passed: bool = self.ov_quality_check(utils.eval_ov_static, kwargs)
        # quality_passed: bool = self.ov_quality_check(self.eval_seam, kwargs)

        # Store refined vector into cx_cy.json only if vector does not contain Inf values
        if store and np.inf not in shift_vec and quality_passed:
            y, x = np.where(self.tile_id_map == tid_a)
            coord4d = (axis, 0, int(y[0]), int(x[0]))
            self.replace_coarse_offset(coord4d, shift_vec, store)

        print(f'{fmsg}: orig. shift vec.: {orig_co}, new shift vec.: {shift_vec}')
        return shift_vec

    def refine_coarse_offset_section(
            self,
            masking=True,
            levels=3,
            max_ext=50,
            stride=10,
            clahe=True,
            store=True,
            plot=True,
            show_plot=False
    ) -> None:

        if self.tile_id_map is None:
            self.feed_section_data()

        if masking:
            self.load_masks()

        yx_shape = np.shape(self.tile_id_map)

        args = dict(masking=masking, levels=levels, max_ext=max_ext, stride=stride,
                    clahe=clahe, store=store, plot=plot, show_plot=show_plot)

        # Refine horizontal neighbors
        conn_x = np.full((2, 1, yx_shape[0], yx_shape[1]), np.nan)
        for x in range(0, yx_shape[1] - 1):
            for y in range(0, yx_shape[0]):
                tid_a = int(self.tile_id_map[y, x])
                tid_b = int(self.tile_id_map[y, x + 1])
                if tid_a == -1 or tid_b == -1:
                    continue

                _ = self.refine_pyramid(tid_a=tid_a, tid_b=tid_b, **args)

        # Refine vertical neighbors
        conn_y = np.full((2, 1, yx_shape[0], yx_shape[1]), np.nan)
        for y in range(0, yx_shape[0] - 1):
            for x in range(0, yx_shape[1]):
                tid_a = int(self.tile_id_map[y, x])
                tid_b = int(self.tile_id_map[y + 1, x])
                if tid_a == -1 or tid_b == -1:
                    continue

                _ = self.refine_pyramid(tid_a=tid_a, tid_b=tid_b, **args)

        return

    def compute_coarse_offset_section(self,
                                      co: Optional[Vector],
                                      store=bool,
                                      overlaps_xy=((200, 300), (200, 300)),
                                      min_range=((10, 100, 0), (10, 100, 0)),
                                      min_overlap=160,
                                      filter_size=10,
                                      max_valid_offset=400,
                                      clahe: bool = True,
                                      masking: bool = False,
                                      overwrite_cxcy: bool = False,
                                      rewrite_masks: bool = False,
                                      kwargs_masks: Optional[Dict] = None
                                      ) -> None:

        """
        Computes coarse offsets for entire section.

        :param co: Optional vector
        :param store: Boolean flag to store results
        :param overlaps_xy: Tuple for overlaps in x and y directions
        :param min_range: Tuple for minimum range values
        :param min_overlap: Minimum overlap
        :param filter_size: Filter size
        :param max_valid_offset: Maximum valid offset
        :param clahe: Boolean flag for CLAHE
        :param masking: Boolean flag for masking
        :param overwrite_cxcy: Boolean flag to overwrite cx_cy.json
        :param rewrite_masks: Boolean flag to write masks
        :param kwargs_masks: Optional dictionary for mask parameters
        :return: None
        """

        if kwargs_masks is None:
            kwargs_masks = dict(roi_thresh=20,
                                max_vert_ext=200,
                                edge_only=True,
                                n_lines=20,
                                store=True,
                                filter_size=50,
                                range_limit=0)

        def process_masks():
            if not masking:
                self.mask_map = None
                # self.mask_map = self.roi_mask_map  # Disable smearing masks
            else:
                self.load_masks()
                if not self.mask_map or rewrite_masks:
                    self.create_masks(**kwargs_masks)
            return

        if (self.path / 'cx_cy.json').exists() and not overwrite_cxcy:
            sec_num = utils.get_section_num(self.path)
            msg = f's{sec_num} cx_cy.json already exists. Skipping coarse offsets computation.'
            logging.info(msg)
            return

        if self.tile_id_map is None:
            self.feed_section_data()

        if self.tile_map is None:
            self.load_tile_map(clahe=clahe)

        if self.tile_map is None:
            logging.warning(f'Computing coarse offsets s{self.section_num} failed.')
            return

        # Load or compute tile masks
        process_masks()

        cx, cy = stitch_rigid.compute_coarse_offsets(
            yx_shape=np.shape(self.tile_id_map),
            tile_map=self.tile_map,
            mask_map=self.mask_map,
            co=co,
            overlaps_xy=overlaps_xy,
            min_range=min_range,
            min_overlap=min_overlap,
            filter_size=filter_size,
            max_valid_offset=max_valid_offset
        )

        # Save coarse offsets
        if store:
            cx = np.squeeze(cx)
            cy = np.squeeze(cy)
            cx_cy = np.array((cx, cy))
            self.cxy = cx_cy
            self.save_coarse_mat(cx_cy)

        self.clear_section()
        return

    def get_coarse_offset(self, tile_id: int, axis: int) -> Optional[Vector]:
        if self.tile_id_map is None:
            self.feed_section_data()

        coord = np.where(self.tile_id_map == tile_id)

        try:
            y, x = int(coord[0][0]), int(coord[1][0])
            co = self.cxy[axis, :, y, x]
        except TypeError:
            logging.warning(f'Coarse offset s{self.section_num} t{tile_id} not defined!')
            return None
        except IndexError:
            logging.warning(f'Coarse offset could not be retrieved: s{self.section_num} t{tile_id} axis: {axis}')
            return None

        # Convert to a tuple of integers, handling np.nans
        co = tuple(int(x) if not (np.isinf(x) | np.isnan(x)) else x for x in co)

        logging.debug(f'tile_id_map: {self.tile_id_map}')
        logging.debug(f'y, x: {y, x}')
        logging.info(f'Loaded coarse offset: {co}')
        return co

    def get_coarse_mesh_offset(self, tile_id: int, axis: int = 0) -> Optional[Vector]:
        if self.tile_id_map is None:
            self.feed_section_data()

        coord = np.where(self.tile_id_map == tile_id)
        try:
            y, x = int(coord[0][0]), int(coord[1][0])
        except IndexError:
            return None

        co = self.mesh_offsets[axis, :, y, x]

        # Convert to a tuple of integers, handling np.nans
        co = tuple(int(x) if not (np.isinf(x) | np.isnan(x)) else x for x in co)

        logging.debug(f'tile_id_map: {self.tile_id_map}')
        logging.debug(f'y, x: {y, x}')
        logging.info(f'loaded offset: {co}')
        return co

    def resolve_dir_stitched(self) -> Path:
        return self.path.parent.parent / 'stitched-sections' / (str(self.path.name) + ".zarr")

    def resolve_path_thumb(self) -> str:
        ext = f'_mini.jpg'
        name_end = "_" + str(self.path.name).split("_")[1]
        zfilled = str(self.section_num).zfill(5)
        new_name = "s" + zfilled + name_end
        thumb_fn = str(self.path.parent.parent / '_inspect' / 'downscaled' / (new_name + ext))
        return thumb_fn


    def ov_quality_check(self, eval_func, kwargs) -> bool:

        def evaluate_seam_quality(**kwargs):
            seam_score, log_msgs = eval_func(**kwargs)
            for msg in log_msgs:
                print(msg)
            return seam_score

        # Extract tile IDs
        tid_a, tid_b = kwargs['tid_a'], kwargs['tid_b']
        fmsg = f"s{self.section_num} t{tid_a}-t{tid_b}:"

        # Check if vector is valid at all
        if np.nan in kwargs['new_co']:
            msg = f"{fmsg} new coarse offset invalid (contains NaN value)."
            print(msg), logging.warning(msg)
            return False

        # Evaluate original seam quality
        kwargs.update({'shift_vec': kwargs['orig_co'], 'offset': kwargs['orig_co']})
        orig_seam_score = evaluate_seam_quality(**kwargs)

        # Evaluate new seam quality
        kwargs.update({'shift_vec': kwargs['new_co'], 'offset': kwargs['new_co']})
        new_seam_score = evaluate_seam_quality(**kwargs)

        # Compare seam quality and print results
        try:
            if new_seam_score < orig_seam_score:
                msg = f"{fmsg} new coarse offset quality worse than original {new_seam_score:.3f}/{orig_seam_score:.3f}."
                print(msg), logging.warning(msg)
                return False
            else:
                msg = f"{fmsg} new coarse offset quality better than original {new_seam_score:.3f}/{orig_seam_score:.3f}."
                print(msg), logging.warning(msg)
                return True
        except TypeError:
            print(f"{fmsg} encountered an issue with seam quality comparison.")
            return False


def repl_coarse_offset():
    # Replace coarse offset value in cx_cy.json by specified vector or scalar

    # Parse Inf log file
    inf_vals: List[np.ndarray]
    fp_infs = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702\stitched-sections\_inspect_01_orig_sofima\inf_vals.txt"
    fp_infs = utils.cross_platform_path(fp_infs)
    infs = utils.parse_inf_log(str(fp_infs))  # TODO fix reading multiple Inf in log file

    # Initialize section
    print(list(infs.keys()))
    nums = list(infs.keys())
    nums = [4874, ]
    path_exp = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702"
    path_exp = utils.cross_platform_path(path_exp)

    for sec_num in nums:
        if sec_num in nums:
            path_my_sec = Path(path_exp) / 'sections' / f's{sec_num}_g0'
            my_sec = Section(path_my_sec)
            my_sec.feed_section_data()

            # Save new coarse offset
            coord = infs[sec_num]
            new_val = 0.0, 232.0
            my_sec.replace_coarse_offset(coord, offset=new_val, store=True)

            # Plot multiple OVs
            dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702\stitched-sections\_inspect_01_orig_sofima\overlaps"
            dir_out = utils.cross_platform_path(dir_out)

            c, z, y, x = coord
            if c == 0:
                tid_a = int(my_sec.tile_id_map[y, x])
                tid_b = int(my_sec.tile_id_map[y, x + 1])
            else:
                tid_a = int(my_sec.tile_id_map[y - 1, x])
                tid_b = int(my_sec.tile_id_map[y, x])

            my_sec.plot_ov(
                tid_a=tid_a,
                tid_b=tid_b,
                shift_vec=None,
                dir_out=str(dir_out),
                show_plot=False,
                clahe=True,
                blur=1.0
            )
    # EOF replace coarse offset


def run_plot_ov(config: cfg.ExpConfig, sec_num: int, tid_a: int, tid_b: int, shift_vec: Optional[Vector] = None):

    ps = Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'
    my_sec = Section(ps)
    my_sec.feed_section_data()

    if shift_vec is None:
        axis = 1 if utils.pair_is_vertical(my_sec.tile_id_map, tid_a, tid_b) else 0
        shift_vec = my_sec.get_coarse_offset(tid_a, axis)

    print(f"Plotting overlap with coarse offset: {shift_vec}")
    my_sec.plot_ov(tid_a=tid_a, tid_b=tid_b, shift_vec=shift_vec, dir_out=str(my_sec.path),
                   show_plot=True, clahe=True, blur=1.1, store_to_root=False)
    return


def run_compute_coarse_offset(sec_path: str):
    my_sec = Section(sec_path)
    my_sec.feed_section_data()

    tile_id_a = 379
    tile_id_b = 380

    # Ruth4 01
    overlaps_xy = ((200, 250, 300), (200, 270, 340))
    min_range = ((35, 50, 65), (0, 10, 100))

    args = dict(overlaps_xy=overlaps_xy,
                min_range=min_range,
                min_overlap=2,
                filter_size=10,
                max_valid_offset=300)

    _ = my_sec.compute_coarse_offset(tile_id_a, tile_id_b, store=True, **args)


def main_eval_seam():
    path_my_sec = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\sections\s2886_g0"
    path_my_sec_2 = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\sections\s2887_g0"
    dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\_inspect\tmp"

    path_my_sec = utils.cross_platform_path(path_my_sec)
    path_my_sec_2 = utils.cross_platform_path(path_my_sec_2)
    dir_out = utils.cross_platform_path(dir_out)

    my_sec = Section(path_my_sec)
    args = dict(tid_a=861, tid_b=862, dir_out=dir_out, shift_vec=None,
                masking=True, plot=True, show_plot=False)
    r1 = my_sec.eval_seam(**args)

    my_sec = Section(path_my_sec_2)
    args = dict(tid_a=661, tid_b=662, dir_out=dir_out, shift_vec=None,
                masking=True, plot=True, show_plot=False)
    r2 = my_sec.eval_seam(**args)

    print(r1, r2)
    # my_sec.refine_coarse_offset_section()
    return


def main_refine_tilepair():

    path_my_sec = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\gitrepos\gfriedri_dataset-alignment\processed_data\roli-1\sections\s7200_g1"

    path_my_sec = utils.cross_platform_path(path_my_sec)

    # dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240217\_inspect\overlaps"
    # dir_out = utils.cross_platform_path(dir_out)

    my_sec = Section(path_my_sec)
    my_sec.feed_section_data()

    # vec = my_sec.estimate_offset(tile_id=861, axis=1)
    # print(f'mean vec: {vec}')

    tid_a = 983
    tid_b = 984

    args = dict(
        tid_a=tid_a,
        tid_b=tid_b,
        masking=False,
        levels=1,
        max_ext=30,
        stride=6,
        clahe=True,
        store=False,
        plot=True,
        show_plot=False
    )

    best_vec = my_sec.refine_pyramid(**args)
    print(f'refined vector: {best_vec}')

    # shift_vec = (-192, -40)
    # my_sec.plot_tile_pair(tid_a, tid_b, shift_vec=best_vec)

    my_sec.plot_ov(tid_a, tid_b, best_vec, my_sec.path, clahe=True)

    return


def coarse_align_section():
    sec_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\dp-2\2024_06_12\sections\s1000_g1"
    sec_path = utils.cross_platform_path(sec_path)

    section = Section(sec_path)

    overlaps_xy = ((200, 150, 300), (200, 150, 350))
    min_range = ((0, 70, 140), (0, 70, 140))
    store = True

    kwargs = dict(
        co=None,
        overlaps_xy=overlaps_xy,
        min_range=min_range,
        store=store,
        min_overlap=2,
        filter_size=10,
        max_valid_offset=450,
        clahe=True,
        masking=True
    )
    section.compute_coarse_offset_section(**kwargs)

    # REFINE
    args = dict(masking=True,
                levels=3,
                max_ext=50,
                stride=10,
                clahe=True,
                store=True,
                plot=True,
                show_plot=False)

    # section.refine_coarse_offset_section(**args)
    return


def main_eval_ov(path_section: str,
                 tid_a: int,
                 tid_b: int,
                 half_width: int,
                 plot_pair=False,
                 blur=False,
                 sigma=1.5
                 ) -> Optional[float]:
    path_section = utils.cross_platform_path(path_section)
    section = Section(path_section)
    section.read_tile_id_map()

    # Determine tile-pair orientation
    is_vert = utils.pair_is_vertical(section.tile_id_map, tid_a, tid_b)
    if is_vert is None:
        logging.warning(f'Specified tile_ids are not neighbors!')
        return

    # Load tile-pair
    tile_pair = section.load_masked_pair(
        tid_a, tid_b, roi=True, smr=False, gauss_blur=blur, sigma=sigma
    )

    if tile_pair is None:
        return

    axis = 1 if is_vert else 0
    shift_vec = section.get_coarse_offset(tid_a, axis)
    # shift_vec = (50, -156)

    mssim = section.eval_ov(tile_pair, shift_vec, plot_pair, half_width)
    if mssim is not None:
        logging.info(f's{section.section_num} t{tid_a}-t{tid_b} ov mssim: {mssim:.2f}')
        return mssim
    else:
        logging.warning(f'eval_ov (t{tid_a}-{tid_b}): estimation of overlap quality failed.')
        return


def main_eval_section_ovs(section: Section):
    res = section.eval_section_overlaps(roi_masking=True, smr_masking=False)
    fp_out = section.path / 'overlap_quality_smr.npz'
    np.savez(fp_out, arr=res)
    return


def main_create_margin_overrides(section: Section, grid_shape: Tuple[int, int]):
    grid_shape: Tuple[int, int]  # rows, columns

    def tst_mo():

        def store_margins(path_margins: str, margins: MarginOverrides):
            # Saves margin overrides to a .npz file
            # Note that yaml stores tuple as list. Mind during loading.
            with open(path_margins, 'w') as json_file:
                m_out = {str(k): v for k, v in margins.items()}
                json.dump(m_out, json_file, indent=4)
            return

        def load_margins(path_margins: str) -> Optional[MarginOverrides]:
            try:
                with open(path_margins, 'r') as yaml_file:
                    loaded_data = dict(yaml.load(yaml_file))
                    fmt_data = {eval(k): tuple(v) for k, v in loaded_data.items()}
                return fmt_data

            except FileNotFoundError:
                print(f"Error: File '{path_margins}' not found.")
                return None

        # Build test margin override dict
        tile_id_map = utils.read_tile_id_map(section.path)
        tile_xy = utils.build_tiles_coords(tile_id_map)
        margins = (0,) * 4
        overrides = {coord: margins for coord in tile_xy}
        overrides[(0, 0)] = (0, 0, 0, 239)
        overrides[(1, 0)] = (0, 0, 0, 183)
        overrides[(2, 0)] = (0, 0, 0, 217)
        overrides[(3, 0)] = (0, 0, 0, 296)
        overrides[(4, 0)] = (0, 0, 0, 0)
        overrides[(0, 1)] = (156, 0, 0, 0)
        overrides[(1, 1)] = (104, 0, 262, 0)
        overrides[(2, 1)] = (100, 0, 207, 0)
        overrides[(3, 1)] = (88, 0, 187, 0)
        overrides[(4, 1)] = (145, 0, 192, 0)
        overrides[(5, 1)] = (0, 0, 215, 0)
        overrides[(6, 1)] = (0, 0, 223, 0)
        overrides[(5, 3)] = (215, 0, 203, 0)  # t0985
        overrides[(3, 4)] = (263, 0, 0, 247)  # t1023
        overrides[(4, 5)] = (190, 0, 0, 0)  # t1064
        overrides[(5, 5)] = (182, 0, 0, 190)  # t1065

        res = section.build_margin_overrides(grid_shape, rim=0)
        print(res)
        # desired = (0,)
        # assert section.margin_overrides == desired

        # Store margins
        path_out = str(section.path / 'margin_overrides.json')
        store_margins(path_out, res)

        # Load margins
        loaded_margins = load_margins(path_out)

        return

    tst_mo()
    return


def main_build_margin_masks(exp_config: cfg.ExpConfig,
                            sec_num: int,
                            mesh_config: Optional[mesh.IntegrationConfig] = None
                            ):

    # Creates margin masks for warping
    section = Section(Path(exp_config.path) / 'sections' / f's{sec_num}_g{exp_config.grid_num}')
    rim_size = 60  # Safety margin to custom overlaps
    margin = max(10, rim_size // 3)  # Cut all tile edges by 'margin'

    if mesh_config is None:
        mesh_config = mesh.IntegrationConfig(
            dt=0.001, gamma=0.1, k0=0.01, k=0.1, stride=20,
            num_iters=1000, max_iters=20000, stop_v_max=0.001,
            dt_max=100, prefer_orig_order=True,
            start_cap=0.1, final_cap=10., remove_drift=True
        )

    section.build_margin_masks(exp_config.grid_shape, margin, rim_size, overwrite=True, mesh_config=mesh_config)
    return


# def compute_fflows(section: Section):
#     # Compute fine-flows
#     section.compute_fine_flows(patch_size=100, stride=40, store=True, masking=True)
#     # section.load_fflows()
#     section.clean_fflows()
#     section.reconcile_fflows()
#     return


def get_fine_mesh(section: Section, stride: int, overwrite=False) -> None:
    if Path(section.path_fmesh).exists() and not overwrite:
        section.load_fmesh()
        return

    config = mesh.IntegrationConfig(
        dt=0.001, gamma=0.01, k0=0.01, k=0.05, stride=stride,
        num_iters=1000, max_iters=20000, stop_v_max=0.001,
        dt_max=100, prefer_orig_order=True,
        start_cap=0.1, final_cap=10., remove_drift=True
    )

    start = time.time()
    section.compute_fine_mesh(config, stride)
    end = time.time()
    logging.info(f's{section.section_num} fine mesh computed in {int((end - start) / 60)} minutes')
    return


def visualize_fflow(config: cfg.ExpConfig,
                    sec_num: int,
                    tile_id: int,
                    xy: Optional[Tuple[int, int]] = None,
                    fname_ext: Optional[str] = None):
    def _interp_nan(array, key: Tuple[int, int], gauss: float = 4.):

        if np.isnan(array).sum() / array.size > 0.9:
            logging.warning(f'interp_tile_flow: s{sec_num} key={key} interpolation skipped (not enough valid points for interpolation)')
            return array

        if gauss > 1:
            array = filters.gaussian(array, sigma=gauss)

        mask = np.isnan(array)
        known_coords = np.array(np.nonzero(~mask)).T
        known_vals = array[~mask]
        unknown_coords = np.array(np.nonzero(mask)).T
        interp_vals = griddata(known_coords, known_vals, unknown_coords, method='linear')
        array[mask] = interp_vals
        return array

    def apply_gaussian_filter(array, sigma, key):
        # Temporarily replace np.nan with zero for filtering
        if np.isnan(array).sum() / array.size > 0.9:
            logging.warning(
                f'interp_tile_flow: s{sec_num} key={key} interpolation skipped (not enough valid points for interpolation)')
            return array

        temp_array = np.copy(array)
        nan_mask = np.isnan(temp_array)
        temp_array[nan_mask] = 0
        filtered_array = gaussian_filter(temp_array, sigma=sigma)
        return filtered_array

    def interp_nan(array, key):
        mask = np.isnan(array)

        if mask.sum() / array.size > 0.9:
            logging.warning(f'interp_tile_flow: s{sec_num} key={key} interpolation skipped (not enough valid points for interpolation)')
            return array

        known_coords = np.array(np.nonzero(~mask)).T
        known_vals = array[~mask]
        unknown_coords = np.array(np.nonzero(mask)).T
        interp_vals = griddata(known_coords, known_vals, unknown_coords, method='linear')
        array[mask] = interp_vals
        return array

    def interp_tile_flow(t_flow: Optional[TileFlow], keys: Optional[Iterable[Tuple[int, int]]] = None) -> Optional[TileFlow]:
        if t_flow is not None:
            int_flow = {}
            if keys is None:
                keys = list(t_flow.keys())
            for k in keys:
                if k not in t_flow.keys():
                    logging.warning(f'key {k} not present in TileFLow')
                    continue
                try:
                    flow = t_flow[k]
                    flow = apply_gaussian_filter(flow, 3, k)
                    int_flow[k] = interp_nan(flow, k)

                except ValueError as e:
                    x, y = k
                    tid = section.tile_id_map[y][x]
                    print(f"Error with key {k}, t{tid}: {e}")
                    print(f"Error data {t_flow[k]}: {e}")
                    int_flow[k] = None
            return int_flow
        return None

    def interp_fine_flows(fine_flows: FineFlows, keys: Optional[Iterable[Tuple[int, int]]] = None) -> Optional[FineFlows]:
        """Interpolates both tile flows in the input fine flow object"""

        if fine_flows is None:
            print(f"interp_fine_flows: s{sec_num} t{tile_id} incorrect input")
            return

        fflow_x: Optional[FineFlow] = fine_flows[0]
        fflow_y: Optional[FineFlow] = fine_flows[1]

        ffx = None
        if fflow_x is not None:
            tile_flow_x: Optional[TileFlow] = fflow_x[0]
            tile_offsets_y: Optional[TileOffset] = fflow_x[1]
            tile_flow_x = interp_tile_flow(tile_flow_x, keys)
            ffx = (tile_flow_x, tile_offsets_y)

        ffy = None
        if fflow_y is not None:
            tile_flow_x: Optional[TileFlow] = fflow_y[0]
            tile_offsets_y: Optional[TileOffset] = fflow_y[1]
            tile_flow_x = interp_tile_flow(tile_flow_x, keys)
            ffy = (tile_flow_x, tile_offsets_y)

        return ffx, ffy

    path_section = Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'
    section = Section(path_section)
    section.feed_section_data()

    # Identify tile_id
    if xy is None:
        coord = np.where(section.tile_id_map == tile_id)
        try:
            y, x = int(coord[0][0]), int(coord[1][0])
            xy = (x, y)
        except TypeError as _:
            print(f'visualize_fflow failed (tile_id not present in tile_id_map)')
            return None
    else:
        x, y = xy
        tile_id = section.tile_id_map[y][x]
        print(f'visualize_fflow: s{sec_num} tile_id = {tile_id}, xy={xy}')

    section.load_fflows(ext=fname_ext)
    if section.fflows is None:
        logging.warning(f's{section.section_num} clean fflows failed: fine flows not available.')
        return

    # # No filtering - original fine flows
    # section.clean_fflows(min_pr=1.0, min_ps=1.0, max_mag=0., max_dev=0.)
    # section.reconcile_fflows(max_gradient=0, max_deviation=0, min_patch_size=0)
    # fine_x, offsets_x = section.fflows_recon[0]
    # # utils.plot_flow_components(fine_x, xy, 'Fine Flow X', transpose=True)
    # fine_y, offsets_y = section.fflows_recon[1]
    # # utils.plot_flow_components(fine_y, xy, 'Fine Flow Y')
    # utils.plot_all_flow_components(fine_x, fine_y, xy)

    # keys = [(0, 9), (1, 9), (3, 9), (0, 2), (2, 2), (4, 2), (0, 3)]
    keys = [xy,]
    # keys = None  # For interpolation testing

    # # Filtered
    for xy in keys:
        section.load_fflows(ext=fname_ext)
        section.clean_fflows(min_pr=1.0, min_ps=1.0, max_mag=0., max_dev=6.)
        section.reconcile_fflows(max_gradient=6, max_deviation=0, min_patch_size=2)
        # section.fflows_recon = interp_fine_flows(section.fflows_recon, keys)
        fine_x, offsets_x = section.fflows_recon[0]
        fine_y, offsets_y = section.fflows_recon[1]
        utils.plot_all_flow_components(fine_x, fine_y, xy)

    return


def fine_align_section(section: Section, grid_shape: List[int], masking=False) -> None:
    stride = 10  # Fine resolution
    patch_size = 30  # For fine-flows
    rim_size = 15  # Safety margin to custom overlaps
    margin = max(10, rim_size // 3)  # Cut all tile edges by 'margin'
    # margin = 0
    use_clahe = True
    zarr_store = True  # Store .zarr into stitched-sections folder
    rescale_fct = 0.2  # Store .jpg thumbnail into section folder
    # rescale_fct = None  # Store .jpg thumbnail into section folder
    parallelism: int = 1
    overwrite = True
    rot_angle = 0

    section.feed_section_data()

    # # Pass if section already stitched
    # if section.stitched:
    #     logging.info(f'Skipping s{section.section_num} Section is already stitched.')
    #     print(f'Skipping s{section.section_num} Section is already stitched.')
    #     section.clear_data()
    #     return

    # if section.path_stitched.exists():
    #     logging.info(f'Skipping s{section.section_num} Section is already stitched.')
    #     print(f'Skipping s{section.section_num} Section is already stitched.')
    #     section.clear_data()
    #     return

    # if masking:
    #     try:
    #         section.load_masks()
    #     except Exception as _:
    #         kwargs_masks = dict(
    #             roi_thresh=20,
    #             max_vert_ext=200,
    #             edge_only=True,
    #             n_lines=30,
    #             store=True,
    #             filter_size=50,
    #             range_limit=0
    #         )
    #         section.create_masks(**kwargs_masks)

    # Computes optimized coarse offset array
    # cfg = mesh.IntegrationConfig(
    #     dt=0.001,
    #     gamma=0.01,
    #     k0=0.0,  # unused
    #     k=0.1,
    #     stride=(1, 1),  # unused
    #     num_iters=1000,
    #     max_iters=100000,
    #     stop_v_max=0.001,
    #     dt_max=100
    # )
    # section.compute_coarse_mesh(cfg=cfg, overwrite=overwrite)

    # # Create margin masks for warping
    # section.build_margin_masks(grid_shape, margin, rim_size, overwrite)

    # #  Compute flows between overlaps
    section.compute_fine_flows(patch_size, stride, masking, overwrite=overwrite, ext=None)

    # # # Compute fine meshes
    get_fine_mesh(section, stride, overwrite)

    # # WARP SECTION
    margin = 0
    clahe_kwargs = dict(kernel_size=1024, clip_limit=0.201, nbins=256)
    warp_kwargs = dict(stride=stride, margin=margin, use_clahe=use_clahe,
                       clahe_kwargs=clahe_kwargs, zarr_store=zarr_store,
                       rescale_fct=rescale_fct, parallelism=parallelism,
                       rot_angle=rot_angle)

    section.warp_section(**warp_kwargs)

    # # Downscale and store stitched section
    # section.load_image()
    # if section.image is not None:
    #     section.downscale_section(fct=rescale_fct)
    #     utils.save_img(section.path_thumb, section.thumb)

    # Derotate stitched .zarr sections
    # section.rotate_and_store_stitched(rot_angle, section.path_stitched_custom.parent)

    section.clear_section()
    return


def main_patch_size(section: Section):
    stride = 10
    sizes = [20, 40, 60, 80, 100, 120]
    sizes = [60, ]
    masking = False

    section.feed_section_data()

    def iter_patch(patch_sizes: List[int]):
        for ps in patch_sizes:
            try:
                print(f'computing patch size: {ps}')
                section.compute_fine_flows(ps, stride, masking, ext=f'_{ps}')
            except ValueError as e:
                print(f'{ps}: {e}')
        return

    iter_patch(sizes)

    return


def main_coarse_align_tile_pair(sec_path: str, tid_a: int, tid_b: int, skip_xcorr=False,
                                refine=False, est_vec: Optional[Vector] = None):
    xcorr_kwargs = dict(id_a=tid_a,
                        id_b=tid_b,
                        refine=False,
                        store=True,
                        clahe=True,
                        overlaps_xy=((270, 170, 340), (270, 170, 340)),
                        min_range=((0, 20, 40), (0, 20, 40)),
                        masking=False)

    section = Section(sec_path)
    section.feed_section_data()

    if not skip_xcorr:
        shift_vec = section.compute_coarse_offset(**xcorr_kwargs)
        print(f'computed coarse shift: {shift_vec}')

    if refine:
        refine_kwargs = dict(tid_a=tid_a, tid_b=tid_b, masking=True, levels=1, max_ext=6,
                             stride=2, clahe=True, store=True, plot=True, show_plot=False,
                             est_vec=est_vec)
        best_vec = section.refine_pyramid(**refine_kwargs)
        print(f'refined vector: {best_vec}')

    return


def exp_configs() -> List[ExpConfig]:
    # Define paths and grid numbers of experiments
    paths = [
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16",
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_2\20240218",
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240416",
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702",
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-1-new-prefect\align-run-1_3",
        r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-2"
    ]
    grid_nums = [0, 1, 0, 0, 1, 1]
    sec_ranges = [[402, 4724], [1027, 5633], [450, 5486], [0, 4874], [282, 8651], [261, 8579]]
    grid_shapes = [[25, 25], [40, 30], [40, 40], [28, 23], [33, 33], [40, 40]]

    # Create ExpConfig named tuples using a loop
    experiments = []
    items = zip(paths, grid_nums, sec_ranges, grid_shapes)
    for path, grid_num, (first_sec, last_sec), grid_shape in items:
        path = utils.cross_platform_path(path)
        exp_conf = ExpConfig(path, grid_num, first_sec, last_sec, grid_shape)
        experiments.append(exp_conf)

    return experiments


def main_eval_single_ov():
    base_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702"
    dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702\_inspect\tmp"
    start, end = 300, 370
    tid_a: int = 308
    tid_b: int = 309

    res = []
    base_path = utils.cross_platform_path(base_path)
    dir_out = utils.cross_platform_path(dir_out)
    for sec_num in range(start, end):
        section_path = Path(base_path) / "sections" / f"s{sec_num}_g0"
        try:
            section = Section(section_path)
            section.feed_section_data()
            path_tile_a: str = section.tile_dicts.get(tid_a)
            path_tile_b: str = section.tile_dicts.get(tid_b)
            if path_tile_a is None or path_tile_b is None:
                print('Failed to read tile data. Check tile paths or section number.')
                return

            tile_a: Tile = Tile(path_tile_a)
            tile_b: Tile = Tile(path_tile_b)
            tile_pair: Tuple[Tile, Tile] = (tile_a, tile_b)

            shift_vec = section.get_coarse_offset(tid_a, axis=0)
            val = section.eval_ov(tile_pair, shift_vec)
            logging.info(f'Eval_OV result for tile-pair t{tid_a}-t{tid_b}: {val}')
            res.append(val)
        except NotADirectoryError as e:
            logging.warning(e)
            res.append(np.nan)
            continue

    fp_out = Path(dir_out) / 'mssim_over_ovs.png'
    plt.plot(list(range(start, end)), res, 'o-')
    plt.savefig(fp_out)
    print(f'Result saved to: {fp_out}')
    return


def main_detect_bad_seam():
    base_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702\_inspect\tmp\aligned"
    dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\20240702\_inspect\tmp"
    start, end = 300, 370
    tid_a: int = 308
    tid_b: int = 309

    base_path = utils.cross_platform_path(base_path)
    dir_out = utils.cross_platform_path(dir_out)
    tif_files = sorted(Path(base_path).glob("*.tif"))

    res = []
    for fp in tif_files:
        ov_img = skimage.io.imread(str(fp))
        val = utils.detect_bad_seam(ov_img, is_vert=False)
        res.append(val)

    fp_out = Path(dir_out) / 'bad_seam_over_ovs.png'
    plt.plot(list(range(len(res))), res, 'o-')
    plt.savefig(fp_out)
    print(f'Result saved to: {fp_out}')
    return


def main_eval_ov_opt(path_section, tid_a, tid_b):
    nr_of_blurs = 10
    delta_blur = 0.5
    init_blur = 0
    res = {}
    sigma = init_blur
    for i in range(nr_of_blurs):
        mssim = main_eval_ov(path_section, tid_a, tid_b,
                             half_width=40, plot_pair=False,
                             blur=True, sigma=sigma)
        res[sigma] = mssim
        sigma += delta_blur
    return res


def main_plot_masks(config: cfg.ExpConfig, sec_num: int, tile_id: int):
    
    path_section = Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'
    section = Section(path_section)
    section.feed_section_data()

    # Load tile-masks
    # section.load_masks()

    # Load margin masks
    if section.margin_masks is None:
        section.margin_masks = utils.load_mapped_npz(section.path_margin_masks)
        
    def plot_mask(sec: Section, tid: int):
        try:
            y, x = np.where(tid == sec.tile_id_map)
            y, x = int(y[0]), int(x[0])
            sec.margin_masks = utils.load_mapped_npz(sec.path_margin_masks)
            msk = sec.margin_masks.get((x, y))
            plt.imshow(msk, cmap='gray')
            plt.show()
        except ValueError as _:
            pass
        return


    # section.plot_masks()

    plot_mask(section, tile_id)
    return


def main_verify_tile_id_map():

    root = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\dp-2\2024_06_12"
    sec_num = 9975
    grid_num = 1
    print_missing = True

    root = Path(utils.cross_platform_path(root))
    sec_path = root / 'sections' / f's{sec_num}_g{grid_num}'

    if not sec_path.exists():
        logging.warning('Specified section number is missing in section folder')
        return

    sec = Section(sec_path)
    res = sec.verify_tile_id_map(print_ids=print_missing)
    status = 'OK' if res else 'NOK'
    print(f'Verifying section s{sec_num} finished with status {status}.')

    return


def main_check_zarr_warped():
    # Verify all chunks in .zarr file are valid
    root = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-3\2024_06_11"
    sec_nums = list(range(3978, 3982))
    grid_num = 0

    def process_zarr_section(root_dir: UniPath, sec_num, grid_num):
        root_dir = Path(root_dir)
        sec_path = root_dir / 'sections' / f's{sec_num}_g{grid_num}'

        if not sec_path.exists():
            logging.warning('Specified section number is missing in section folder')
            return

        section = Section(sec_path)
        # is_intact = utils.check_zarr_integrity(section.path_stitched)
        # print(f'Warped section s{section.section_num} file is intact: {is_intact}')

        # Downscale
        section.load_image()
        section.downscale_section(fct=1.0)
        utils.save_img(section.path_thumb, section.thumb)

        return

    root = Path(utils.cross_platform_path(root))
    for num in sec_nums:
        process_zarr_section(root, num, grid_num)

    # print(section.stitched)
    return

def eval_rot_angle():
    mat_a = [0.667831806298371, 0.744312218424671]
    mat_b = [-0.744312218424671, 0.667831806298371]

    try:
        a = np.array(mat_a)
        b = np.array(mat_b)
        rot_mat = np.array((a, b))
        rotation_angle = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        return np.degrees(rotation_angle)
    except TypeError as _:
        return None


def main_recover_coarse_mat(section_path: str):
    section_path = utils.cross_platform_path(section_path)
    path_cxcy = Path(section_path).parent.parent / '_inspect' / 'all_offsets.npz'
    sec = Section(section_path)
    cxyz = np.load(path_cxcy)
    sec_num = int(Path(section_path).name[1:].split("_")[0])
    try:
        sec.save_coarse_mat(cxyz[str(sec_num)])
    except KeyError as _:
        print(f'Coarse offset mat of section number {sec_num} is missing in offsets file!')
    return


def main_eval_ov_score(cfg, sec_num, tid_a, tid_b) -> Optional[float]:

    section = Section(str(Path(cfg.path) / 'sections' / f's{sec_num}_g{cfg.grid_num}'))
    section.feed_section_data()

    is_vert = utils.pair_is_vertical(section.tile_id_map, tid_a, tid_b)
    if is_vert is None:
        logging.info(f'Specified tile_ids are not neighbors or one of the tile-ids are missing!')
        return
    axis = 1 if is_vert else 0

    co = section.get_coarse_offset(tid_a, axis)
    tile_pair = section.load_image_pair(tid_a, tid_b, blur_fct=6.0)
    co_score = section.eval_ov(tile_pair, co)
    print(f's{sec_num} t{tid_a}-t{tid_b} seam score {co_score:.4f}')

    return co_score


def coarse_align_or_refine_tilepair(config: cfg.ExpConfig):
    start = 4000
    end = start
    tid_a = 440
    tid_b = 465
    # est_vec = (-30, -45)
    est_vec = None
    overlaps_xy = ((350, 250, 450), (350, 250, 450))
    clahe = True

    xcorr_kwargs = dict(id_a=tid_a,
                        id_b=tid_b,
                        refine=False,
                        store=False,
                        clahe=clahe,
                        overlaps_xy=overlaps_xy,
                        min_range=((0, 70, 140), (0, 70, 140)),
                        max_valid_offset=450,
                        min_overlap=2,
                        masking=True)

    refine_kwargs = dict(tid_a=tid_a, tid_b=tid_b, masking=False, levels=2, max_ext=15,
                         stride=4, clahe=clahe, store=False, plot=True, show_plot=False,
                         est_vec=est_vec)

    for sec_num in range(start, end + 1):
        my_sec = Section(str(Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'))
        print(f'Path section: {my_sec.path}')

        # my_sec.fix_coarse_offset(tid_a, tid_b,
        #                          est_vec_orig=False,
        #                          skip_xcorr=False,
        #                          refine=True,
        #                          xcorr_kwargs=xcorr_kwargs,
        #                          refine_kwargs=refine_kwargs,
        #                          plot_ov=True)
    return


def main_warp_section(config: cfg.ExpConfig, sec_num: int):
    # RENDERS SINGLE SECTION

    path_section = Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'
    section = Section(path_section)
    section.feed_section_data()

    stride = 20  # Fine resolution
    # patch_size = 40  # For fine-flows
    rim_size = 30  # Safety margin to custom overlaps
    margin = max(10, rim_size // 3)  # Elast. deformation reduction
    zarr_store = True  # Store .zarr into stitched-sections folder
    rescale_fct = 0.2  # Store .jpg thumbnail into section folder
    clahe_kwargs = dict(kernel_size=None, clip_limit=0.01, nbins=256)
    warp_kwargs = dict(stride=stride, margin=margin, use_clahe=True,
                       clahe_kwargs=clahe_kwargs, zarr_store=zarr_store,
                       rescale_fct=rescale_fct, parallelism=10)

    # Optionally, recompute margin masks
    print(f'margin: {margin}')
    overwrite = True
    # mesh_config = None
    mesh_config = mesh.IntegrationConfig(
        dt=0.001, gamma=0.01, k0=0.01, k=0.1, stride=stride,
        num_iters=1000, max_iters=20000, stop_v_max=0.001,
        dt_max=100, prefer_orig_order=True,
        start_cap=0.1, final_cap=10., remove_drift=True
    )
    section.build_margin_masks(config.grid_shape, margin, rim_size, overwrite, mesh_config)

    section.warp_section(**warp_kwargs)
    print('Warping done')
    return


def main_fix_zarr_layer():
    # Fix warped section by removing z-dimension
    cfg = mont_1
    sec_num = 1017
    sec_path = str(Path(cfg.path) / 'stitched-sections' / f's{sec_num}_g{cfg.grid_num}.zarr')
    dst_path = Path(sec_path).parent.parent / 'stitched-sections-fixed' / Path(sec_path).name
    utils.fix_zarr(sec_path, str(dst_path))
    return


def main_create_masks():
    # CREATE MASKS
    p = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\gitrepos\gfriedri_dataset-alignment\processed_data\roli-1\sections\s1522_g1"
    p = utils.cross_platform_path(p)
    section = Section(p)
    kwargs = dict(roi_thresh=20,
                  max_vert_ext=200,
                  edge_only=False,
                  n_lines=20,
                  store=True,
                  filter_size=50,
                  range_limit=0)
    section.create_masks(**kwargs)
    return


def main_eval_ov_by_ssim():
    # EVAL OV BY SSIM
    path_section = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_4\2024_06_04\sections\s1062_g0"
    tid_a = 329
    tid_b = 330
    main_eval_ov(path_section, tid_a, tid_b, half_width=30, plot_pair=True, blur=True, sigma=6)
    res_a = main_eval_ov_opt(path_section, tid_a, tid_b)
    ax, ay = [k for k in res_a.keys()], [v for v in res_a.values()]

    path_section = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\sections\s2887_g0"
    res_b = main_eval_ov_opt(path_section, tid_a, tid_b)
    bx, by = [k for k in res_b.keys()], [v for v in res_b.values()]

    def plot_res(a, b):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plt.plot(a[0], a[1], 'o')
            ax1.set_title('Image 1')
            plt.plot(b[0], b[1], 'o')
            ax2.set_title('Image 2')
            plt.show()
            return

    # plot_res((ax, ay), (bx, by))

    y_data = np.array(bx[1:]) / np.array(ax[1:])
    print(y_data)
    x_data = ax[1:]
    plt.plot(x_data, y_data, '-o')
    plt.show()
    return




def main_fine_align_section(config: cfg.ExpConfig, sec_num: int):
    # FINE-ALIGN AND WARP SECTION
    path_section = Path(config.path) / 'sections' / f's{sec_num}_g{config.grid_num}'
    section = Section(path_section)
    section.feed_section_data()
    fine_align_section(section, config.grid_shape, masking=True)
    return


def create_and_plot_masks():
    root = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\gitrepos\gfriedri_dataset-alignment\processed_data\roli-1\sections"
    # # #sec_nums = range(1000, 8000, 1000)
    sec_nums = [3032,]
    grid_num = 1

    paths_section = [Path(root) / f's{num}_g{grid_num}' for num in sec_nums]
    for path_section in paths_section:
        my_sec = Section(path_section)
        my_sec.feed_section_data()
        kwargs = dict(roi_thresh=20,
                      max_vert_ext=200,
                      edge_only=True,
                      n_lines=25,
                      store=True,
                      filter_size=50,
                      range_limit=0)

    my_sec.create_masks(**kwargs)

    rim_size = 60
    margin = max(10, rim_size // 3)
    my_sec.build_margin_masks(grid_shape=[28, 23], margin=margin, rim_size=rim_size, overwrite=True)

    my_sec.plot_masks()
    return


if __name__ == "__main__":

    # Accessing individual experiments
    configs = cfg.get_experiment_configurations()
    mont_1 = configs[cfg.ExperimentName.MONT_1]
    mont_2 = configs[cfg.ExperimentName.MONT_2]
    mont_3 = configs[cfg.ExperimentName.MONT_3]
    mont_4 = configs[cfg.ExperimentName.MONT_4]
    roli_1 = configs[cfg.ExperimentName.ROLI_1]
    roli_2 = configs[cfg.ExperimentName.ROLI_2]
    roli_3 = configs[cfg.ExperimentName.ROLI_3]
    dp2 = configs[cfg.ExperimentName.DP2]


    # COARSE ALIGNMENT
    # coarse_align_or_refine_tilepair(config=roli_3)
    # coarse_align_section()
    # main_detect_bad_seam()
    # main_refine_tilepair()
    # run_compute_coarse_offset()
    # repl_coarse_offset()


    # EVAL SEAM QUALITY ROUTINES
    # main_eval_ov_score(roli_1, sec_num=706, tid_a=866, tid_b=906)
    # main_eval_ov()
    # main_eval_single_ov()
    # main_eval_seam()
    # main_eval_ov_opt()
    # main_eval_section_ovs()
    # main_eval_ov_by_ssim()


    # PLOTTING, WARPING, FINE ALIGNMENT
    # run_plot_ov(roli_1, sec_num=2000, tid_a=664, tid_b=704, shift_vec=(50, 5))
    main_fine_align_section(roli_1, sec_num=2000)
    # visualize_fflow(roli_1, sec_num=7964, tile_id=580, xy=None)
    # main_warp_section(roli_1, sec_num=300)
    # main_patch_size(section)
    # # section.compute_coarse_mesh()


    # MASKING
    # main_create_masks()
    # main_plot_masks(roli_1, sec_num=7964, tile_id=620)
    # create_and_plot_masks()
    # main_build_margin_masks(roli_1, sec_num=300, mesh_config=None)
    # main_create_margin_overrides()


    # VALIDATION ROUTINES
    # main_verify_tile_id_map()
    # main_check_zarr_warped()


    # OTHERS
    # main_recover_coarse_mat(sec_path)  # RECOVER COARSE MAT FROM BACKUP
    # rot_angle = eval_rot_angle()
    # main_fix_zarr_layer()
