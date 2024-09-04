import collections
import gc
import logging
import math
import random
import threading
from multiprocessing import Pool
import concurrent.futures
import multiprocessing
from functools import partial, wraps
import random
from collections import OrderedDict

import jax
import jax.numpy as jnp
import zarr

# Set the logging level to suppress all log messages
logging.getLogger('jax').setLevel(logging.ERROR)

import platform
from pathlib import Path
from typing import (Tuple, List, FrozenSet, Set, Mapping,
                    Union, Dict, Any, Iterable, Optional)
import json
import matplotlib.pyplot as plt
import cv2
import glob
import skimage.io
from cv2 import phaseCorrelate
from skimage.metrics import structural_similarity
import yaml
import numpy as np
from numpy.typing import ArrayLike
from numpy import ndarray
from ruyaml import YAML
from time import time
from tqdm import tqdm
import os
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore

import inspection_utils as utils
import experiment_configs as cfg

import pyramid_levels
import s01_coarse_align_section_pairs as em_align_reg_sections
# import s01_estimate_flow_fields as em_align_est_ff
import s02_create_coarse_aligned_stack as em_align_warp_coarse_stack
#import s02_relax_meshes as em_relax
#import s03_warp_fine_aligned_sections as em_warp
from process_section_pairs import compute_sections_par
#from parameter_config import FlowFieldEstimationConfig, MeshIntegrationConfig, WarpConfig

from Section import Section, run_compute_coarse_offset, fine_align_section
from Tile import Tile

Vector2D = List[float]
Vector = Union[Tuple[int, int], Tuple[int, int, int], Union[Tuple[int], Tuple[Any, ...]]]
UniPath = Union[str, Path]

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


class Inspection:
    def __init__(self, config: cfg.ExpConfig):
        self.config = config
        self.root = Path(config.path)
        self.grid_nr = config.grid_num
        self.first_sec = config.first_sec
        self.last_sec = config.last_sec
        self.grid_shape = config.grid_shape
        self.acq_dir = utils.cross_platform_path(config.acq_dir)

        self._initialize_directories()
        self._initialize_paths()
        self._initialize_section_data()
        self._initialize_stitched_data()

    def _initialize_directories(self):
        self.dir_sections = self.root / 'sections'
        self.dir_stitched = self.root / 'stitched-sections'
        self.dir_inspect = self.root / '_inspect'
        self.dir_downscaled = self.dir_inspect / 'downscaled'
        self.dir_overlaps = self.dir_inspect / 'overlaps'
        self.dir_outliers = self.dir_inspect / 'overlaps_outliers'
        self.dir_inf_overlaps = self.dir_inspect / 'inf_overlaps'
        self.dir_coarse_stacks = self.root / 'coarse-stack'

    def _initialize_paths(self):
        self.path_exp_notes = self.root.parent / 'exp_notes.yaml'
        self.path_cxyz = self._get_inspect_path('all_offsets.npz')
        self.path_id_maps = self._get_inspect_path('all_tile_id_maps.npz')
        self.fp_eval_ov = self._get_overlaps_path('overlap_quality_smr.npz')
        self.fp_est_ff_cfg = self.root / 'fine_align_estimate_flow_fields.config'
        self.fp_missing_sections = self.root / 'missing_sections.yaml'

    def _initialize_section_data(self):
        self.section_dirs: Optional[List[Path]] = None
        self.section_names: Optional[List[str]] = None
        self.section_nums: Optional[List[int]] = None
        self.section_dicts: Optional[Dict[int, str]] = None
        self.section_nums_duplicates: Optional[List[str]] = None
        self.section_nums_skip: Optional[List[str]] = None
        self.missing_sections: Optional[List[int]] = None

    def _initialize_stitched_data(self):
        self.stitched_dirs: Optional[List[Path]] = None
        self.stitched_names: Optional[List[str]] = None
        self.stitched_nums: Optional[List[int]] = None
        self.stitched_nums_valid: Optional[List[int]] = None
        self.stitched_dicts: Optional[Dict[int, str]] = None
        self.cross_aligned_nums: Optional[List[int]] = None

    def _get_inspect_path(self, filename: str) -> Path:
        return self.dir_inspect / filename

    def _get_overlaps_path(self, filename: str) -> Path:
        return self.dir_overlaps / filename

    def get_zarr_sizes(self) -> Dict[int, int]:
        """Compute .zarr folder size in kb and return dict with sec_num as keys"""
        if self.stitched_dicts is None:
            self.list_all_section_dirs()

        subfolder_sizes = {}
        for sec_num, zarr_path in self.stitched_dicts.items():
            subfolder_sizes[sec_num] = utils.get_folder_size(zarr_path)

        # Sort dictionary by folder number
        return dict(sorted(subfolder_sizes.items()))

    @staticmethod
    def list_subdirs_w_files(folder_path: UniPath, file_name: str) -> Optional[List[str]]:
        folder_path = Path(folder_path).resolve()
        if not folder_path.exists():
            logging.warning(f'Listing files failed. Parent folder does not exist.')
            return

        subfolders_with_file = [
            str(subfolder) for subfolder in folder_path.iterdir()
            if subfolder.is_dir() and (subfolder / file_name).is_file()
        ]
        return subfolders_with_file

    @staticmethod
    def create_multiscale(zarr_path: Union[str, Path], max_layer=5, num_processes=42) -> None:

        zarr_path = utils.cross_platform_path(str(zarr_path))

        if not Path(zarr_path).exists():
            logging.warning(f'Creating multiscale .zarr failed (zarr path does not exist).')
            return

        logging.info(f'Creating multiscale volume: {zarr_path}')
        pyramid_levels.main(zarr_path=zarr_path,
                            max_layer=max_layer,
                            num_processes=num_processes
                            )
        return


    def fix_false_offsets_trace(self,
                                tid_pair: [Tuple[int, int]],
                                align_args: dict,
                                inf=False,
                                custom_sec_nums: Optional[Iterable[int]] = None
                                ) -> None:
        sec_nums = []
        # fp = self.dir_inspect / 'coarse_offset_outliers.txt'
        # if inf:
        #     fp = self.dir_inspect / 'inf_vals.txt'
        #
        # if not Path(fp).exists():
        #     logging.warning(f'Failed to load outliers/inf values from {fp}')
        #     return
        #
        # try:
        #     outliers = self.load_outliers(path_outliers=fp)
        # except ValueError as _:
        #     logging.warning(f'Empty list of outliers/inf values in {fp}')
        #     return
        #
        # sec_nums = list(outliers.keys())
        if custom_sec_nums is not None:
            sec_nums = custom_sec_nums
            # sec_nums = [num for num in sec_nums if num in custom_sec_nums]

        if not sec_nums:
            logging.warning(f'Fixing false offsets: nothing to fix in specified range of section numbers and tile-pair IDs.')
            return

        # for sec_num in sec_nums:
        #     for val in set(outliers[sec_num]):
        #         if val == tid_pair:
        #             tid_a, tid_b = tid_pair
        #             align_tile_pair(self, sec_num, tid_a, tid_b,  **align_args)

        seams_scores = []
        for sec_num in sec_nums:
            tid_a, tid_b = tid_pair
            ov_score = align_tile_pair(self, sec_num, tid_a, tid_b, **align_args)
            seams_scores.append(ov_score)

        # Store and plot results
        mssim_tuples = [(num, score) for num, score in zip(sec_nums, seams_scores)]
        dir_out = self.dir_inf_overlaps
        process_eval_ov_results(mssim_tuples, tid_pair[0], tid_pair[1], dir_out, sort=False)
        return


    def parallel_align(self, sec_nums, tid_pair, align_args):
        # TODO test parallel fix of outliers
        args = [(self, sec_num, tid_pair, align_args) for sec_num in sec_nums]
        with multiprocessing.Pool() as pool:
            pool.map(align_wrapper, args)

        return

    def compute_paddings(self):

        if self.stitched_dirs is None:
            logging.info(f'Parsing stitched .zarr folders...')
            res = utils.process_dirs(str(self.dir_stitched), utils.list_stitched)
            if res is None:
                logging.warning('Get paddings not performed. No .zarr sections found.')
                return
            self.stitched_dirs = res[0]

        dirs = [str(p) for p in self.stitched_dirs]
        shifts = em_align_reg_sections.load_shifts(dirs)
        paddings = em_align_reg_sections.get_padding_per_section(shifts)

        for i in tqdm(range(len(dirs))):
            with open(os.path.join(dirs[i], "coarse_stack_padding.json"), "w") as f:
                json.dump(dict(shift_y=int(paddings[i, 0]), shift_x=int(paddings[i, 1])), f)
        logging.info(f'Get paddings done.')
        return


    def get_cross_aligned_nums(self) -> None:
        fn = 'shift_to_previous.json'
        cross_aligned_dirs = self.list_subdirs_w_files(self.dir_stitched, fn)
        if len(cross_aligned_dirs) > 0:
            nums = [int(Path(s).name.split("_")[0][1:]) for s in cross_aligned_dirs]
            self.cross_aligned_nums = nums


    def validate_seams(self, input_dir: Optional[UniPath] = None, num_proc: int = 5) -> None:

        # Get list of overlaps folders
        source_dir = Path(input_dir) if input_dir is not None else self.dir_overlaps
        ov_dirs = self.list_dir_overlaps(source_dir)
        ov_dirs_full = [str(source_dir / name) for name in ov_dirs]

        for dir_path in ov_dirs_full:
            run_par_validate_seam(dir_path, num_proc=num_proc)


    def list_dir_overlaps(self, source_dir: Optional[UniPath] = None) -> List[str]:
        """Returns ov folder names within 'overlaps' folder"""

        matching_folders = []
        folder_path = Path(source_dir) if source_dir is not None else self.dir_overlaps
        if not folder_path.exists():
            logging.warning('No folders containing overlap images could be located.')
            return matching_folders

        # Iterate over the contents of the folder
        for item in folder_path.iterdir():
            # Check if the item is a directory and matches the specified pattern
            if (item.is_dir() and len(item.name) == 11
                    and item.name.startswith("t")
                    and item.name[1:5].isdigit()
                    and item.name[5:7] == "_t"
                    and item.name[7:].isdigit()):
                # If it matches, add it to the list of matching folders
                matching_folders.append(item.name)

        return sorted(matching_folders)

    def get_valid_sec_nums(
            self,
            start: Optional[int] = None,
            end: Optional[int] = None,
            custom_sec_nums: Optional[Iterable[int]] = None
    ) -> List[int]:

        """Get valid section numbers based on specified range or custom range.

        Args:
            start (int, optional): The first section number to consider. Defaults to None.
            end (int, optional): The last section number to consider. Defaults to None.
            custom_sec_nums (Iterable[int], optional): A custom set of section numbers to consider.
                                                       Defaults to None.

        Returns:
            List[int]: A list of valid section numbers within the specified range or custom range.
        """

        # Ensure experiment section numbers are available
        if self.section_nums is None:
            data = utils.process_dirs(
                str(self.dir_sections), utils.filter_and_sort_sections)
            if data is not None:
                _, _, section_nums, _ = data
            else:
                return []

        # Determine the set of section numbers to consider
        if custom_sec_nums is None:
            # Set default values for section numbers if not provided
            if start is None:
                start = self.section_nums[0]
            if end is None:
                end = self.section_nums[-1]
            sec_nums = set(range(start, end + 1))
        else:
            sec_nums = set(custom_sec_nums)

        valid_nums = list(sec_nums.intersection(self.section_nums))
        valid_nums.sort()

        return valid_nums

    def read_exp_notes(self):
        exp_notes: cfg.ExpNotes = cfg.load_exp_notes(str(self.path_exp_notes))
        if exp_notes is None:
            print(f'Warning: experiment notes file not found.')
            return

        self.section_nums_skip = exp_notes.Skip
        self.section_nums_duplicates = exp_notes.Duplicates

        return

    def repair_inf_offsets(self,
                           start: Optional[int] = None,
                           end: Optional[int] = None,
                           custom_sec_nums: Optional[Iterable[int]] = None,
                           masking: bool = True,
                           store: bool = True,
                           num_processes: Optional[int] = 5,
                           refine_params: Optional[Dict] = None  # TODO
                           ) -> None:
        """Repair infinite values in offset arrays across sections using refine pyramid method.

        Args:
            start (int, optional): The first section number to consider. Defaults to None.
            end (int, optional): The last section number to consider. Defaults to None.
            custom_sec_nums : Define specific set of sections to be scanned for Inf values
            masking (bool, optional): Whether to apply masking. Defaults to True.
            store (bool): Save computed offset vector to cx_cy.json.
            num_processes (int): Number of CPU cores to use for parallel processing 
        """

        # Get dict of all sections with Inf values and respective tile_id pairs
        sec_nums = self.get_valid_sec_nums(start, end, custom_sec_nums)
        sec_dirs = [self.section_dicts.get(n) for n in sec_nums]
        all_inf = utils.locate_all_inf(sec_dirs)

        if len(all_inf) == 0:
            logging.warning(f'Repair Inf offsets: no Inf values to repair were found.')
            return

        # print(len(all_inf))
        # for inf in all_inf:
        #     print(inf)

        # Use refine method to fix infinities
        arg_dicts = []
        for sec_dir, tid_list in all_inf.items():
            sec = Section(sec_dir)
            sec.feed_section_data()

            for (tid_a, tid_b) in tid_list:
                logging.info(f'fixing s{utils.get_section_num(sec_dir)} t{tid_a} t{tid_b}')

                args = dict(sec=sec,
                            tid_a=tid_a,
                            tid_b=tid_b,
                            masking=masking,
                            levels=3,
                            max_ext=24,
                            stride=6,
                            clahe=True,
                            store=store,
                            plot=True,
                            show_plot=False)

                arg_dicts.append(args)

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(refine_trace_wrapper, arg_dicts)

        return

    def plot_ov_for_section(self, sec_num: int, ov_dict: dict):
        # Plot overlaps
        sec_dict = {sec_num: ov_dict.get(sec_num)}
        self.plot_specific_ovs(sec_dict)
        return

    def plot_all_ovs_par(
            self,
            first: Optional[int] = None,
            last: Optional[int] = None,
            custom_range: Optional[Iterable[int]] = None,
            num_processes=42
    ) -> None:

        sec_nums = self.get_valid_sec_nums(first, last, custom_range)
        if not sec_nums:
            logging.warning('No valid setion numbers to plot.')
            return

        num_processes = min(len(sec_nums), num_processes)

        if len(sec_nums) == 0:
            logging.warning('plot_ovs: Nothing to plot. Invalid section range.')
            return

        # Create dict of all sections and all tile_id pairs
        ov_dict: Dict[int, List[Tuple[int, int]]] = dict()

        # Get valid tile-id neighbor pairs and add them to final dict
        for num in sec_nums:
            sec_path = self.section_dicts.get(num)
            if sec_path is not None:
                my_sec = Section(sec_path)
                my_sec.read_tile_id_map()
                ov_dict[num] = utils.get_neighbour_pairs(my_sec.tile_id_map)

        with multiprocessing.Pool(processes=num_processes) as pool:
            plot_partial = partial(self.plot_ov_for_section, ov_dict=ov_dict)
            pool.map(plot_partial, sec_nums)

        return

    def mod_time_select(self,
                        sec_nums: Set[int],
                        min_file_age: int,
                        filename: str,
                        stitched_dir: bool = False,
                        store: bool = False,
                        store_dir: Optional[UniPath] = None
                        ) -> Set[int]:
        """
              Selects section (numbers) containing specific file older than
              min_file_age (specified in hours).

              Can be used in conjunction with compute coarse offsets, when one
              aims to recompute only portion of cx_cy.jsons.

              Args:
                  sec_nums (Set[int]): Set of section numbers.
                  min_file_age (int): Minimum file age in hours.
                  filename (str): Filename to be scanned for.
                  stitched_dir (bool): look for files in stitched sections directories otherwise in sections folder
              Returns:
                  Set[int]: Selected section numbers.

              """

        def eval_thumb_fp(sec_num) -> str:
            sec_str = str(sec_num).zfill(5)
            ext = f"g{self.grid_nr}_thumb_0.2.png"
            thumb_name = "s" + sec_str + ext
            return thumb_name

        def eval_thumb_fn(sec_num) -> str:
            thumb_name = f"final_flow_s{sec_num - 1}_g0_to_s{sec_num}_g0.npy"
            return thumb_name

        now = time()
        sel_sec_nums = sec_nums.copy()
        dicts = self.stitched_dicts if stitched_dir else self.section_dicts
        for num in sec_nums:
            try:
                sec_dir = Path(dicts.get(num))
            except TypeError as _:
                logging.warning(f"Mod time select failed at s{num}")
                sel_sec_nums.remove(num)
                continue

            if sec_dir is not None:
                fp = sec_dir / filename
                # fp = sec_dir / eval_thumb_fn(num)
                # fp = sec_dir / '.zattrs'
                try:
                    mod_time = fp.stat().st_ctime
                    file_age = (now - mod_time) / 3600
                    logging.debug(f's{num} file age: {file_age:.1f} hours')
                    if file_age < min_file_age:
                        logging.debug(f'Removing {num}')
                        sel_sec_nums.remove(num)
                except FileNotFoundError:
                    logging.info(f'File {fp} not found in s{num} directory!')
                    # sel_sec_nums.remove(num)

        if store:
            if store_dir is None:
                store_dir = self.dir_inspect
            fp = Path(store_dir) / 'old_nums.json'
            with open(fp, 'w') as json_file:
                data = sorted(list(sel_sec_nums))
                for number in data:
                    json_file.write(f"{number}\n")

        logging.info(f'Number of old files: {len(sel_sec_nums)}')
        return sel_sec_nums

    def select_folders_wo_file(self, sec_nums: Set[int], filename: Optional[str] = None) -> Set[int]:

        def eval_thumb_fn(sec_num) -> str:
            sec_str = str(sec_num).zfill(5)
            if filename is not None:
                return filename

            ext = f"g{self.grid_nr}_thumb_0.2.png"
            thumb_name = "s" + sec_str + ext
            return thumb_name

        if self.section_dicts is None:
            self.init_experiment()

        sel_sec_nums = sec_nums.copy()
        for num in sec_nums:
            sec_dir = Path(self.section_dicts.get(num))
            if sec_dir is not None:
                path_cxcy = sec_dir / eval_thumb_fn(num)
                if path_cxcy.exists():
                    sel_sec_nums.remove(num)

        return sel_sec_nums

    def create_masks_exp_par(self,
                             sec_nums: Optional[Iterable[int]] = None,
                             num_processes: int = 42) -> None:
        """Parallelized version of creating tile masks

        num_processes: number of CPU cores requested for a job
        sec_nums: Section numbers to be processed. If not provided,
                  all sections within sections folder will be processed
        """
        if self.section_nums is None:
            self.init_experiment()

        if sec_nums is None:
            sec_nums = self.section_nums
        else:
            sec_nums = [num for num in sec_nums if num in self.section_nums]

        kwargs = dict(roi_thresh=20,
                      max_vert_ext=200,
                      edge_only=True,
                      n_lines=25,
                      store=True,
                      filter_size=50,
                      range_limit=170)

        part_func = partial(create_masks_for_section, **kwargs)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(part_func, sec_nums)
        return


    @staticmethod
    def verify_single_section(section: Section) -> Optional[int]:
        if not section.verify_tile_id_map(print_ids=False):
            return section.section_num
        return None

    def verify_tile_id_maps(self) -> List[int]:

        # Init experiment
        if not self.section_nums:
            self.init_experiment()

        # Construct section objects to be aligned
        section_paths = list(self.section_dirs)
        sections = [Section(p) for p in section_paths]

        # Create a pool of processes and map the verify_single_section function over section_nums
        num_proc = min(42, len(self.section_nums))
        with multiprocessing.Pool(processes=num_proc) as pool:
            results = pool.map(self.verify_single_section, sections)

        # Filter None results to collect failed section numbers
        failed_sec_nums = [num for num in results if num is not None]
        return failed_sec_nums


    @staticmethod
    def verify_stitched(zarr_path: Union[str, Path]) -> Optional[int]:
        if not utils.check_zarr_integrity(zarr_path):
            return utils.get_section_num(zarr_path)
        return None

    def verify_stitched_files(self) -> List[int]:

        if not self.stitched_dirs:
            self.init_experiment()

        # Create a pool of processes and map the verify_single_section function over section_nums
        num_proc = min(42, len(self.stitched_dirs))
        with multiprocessing.Pool(processes=num_proc) as pool:
            results = pool.map(self.verify_stitched, self.stitched_dirs)

        # Filter None results to collect failed section numbers
        failed_sec_nums = [num for num in results if num is not None]
        return failed_sec_nums


    def scan_for_missing_masks(self) -> List[int]:
        """Scans section folders of input experiment for missing tile_masks.npz files."""

        if self.section_dirs is None:
            self.init_experiment()

        missing_nums = []
        fn = 'tile_masks.npz'
        for directory in self.section_dirs:
            fp = (Path(directory) / fn)
            if not fp.exists():
                missing_nums.append(int(directory.name.split("_")[0].strip('s')))

        return missing_nums

    def backup_coarse_offsets(self):
        """Stores all coarse offset arrays into a .npz file within inspect directory
        """

        # Collect all offsets
        offsets, missing_files = utils.aggregate_coarse_offsets(self.section_dirs)
        logging.debug(f'len missing files {len(missing_files)}')
        for p in missing_files:
            logging.debug(p)

        fp_out = self.dir_inspect / "all_offsets.npz"
        np.savez(fp_out, **offsets)
        logging.info(f'Coarse offsets saved to: {fp_out}')

        fp_out2 = fp_out.with_name("all_offsets_missing_files.txt")
        with open(fp_out2, "w") as f:
            f.writelines("\n".join(missing_files))
        logging.info(f'Missing offsets saved to: {fp_out2}')
        return

    def backup_tile_id_maps(self):
        # Collect all tile ID maps
        tile_id_maps, missing_files = utils.aggregate_tile_id_maps(self.section_dirs)
        logging.debug(f'len missing files {len(missing_files)}')
        for p in missing_files:
            logging.debug(p)

        fp_out = self.dir_inspect / "all_tile_id_maps.npz"
        np.savez(fp_out, **tile_id_maps)
        logging.info(f'Tile ID maps saved to: {fp_out}')

        fp_out2 = fp_out.with_name("all_missing_tile_id_maps.txt")
        with open(fp_out2, "w") as f:
            f.writelines("\n".join(missing_files))
        logging.info(f'Missing tile ID maps saved to: {fp_out2}')
        return

    def run_compute_coarse_offset_trace(self,
                                        start: int,
                                        end: int,
                                        tid1: int,
                                        tid2: int,
                                        store: bool
                                        ):
        if self.section_dicts is None:
            self.init_experiment()

        for sec_num in range(start, end + 1):
            if sec_num in self.section_nums:
                print(f'Aligning s{sec_num}')
                section_path = self.section_dicts[sec_num]
                run_compute_coarse_offset(section_path)

        return

    def load_outliers(self, path_outliers: Optional[UniPath] = None
                      ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Load outliers data from a text file.

        :param path_outliers: Optional. Path to the file containing outliers
         data. If not provided, defaults to 'coarse_offset_outliers.txt'
         in the directory specified by self.dir_inspect.
        :return: A dictionary where keys are slice numbers and values are
        lists of tuples, each containing two integers (TileID, TileID_nn).
        """

        if path_outliers is None:
            # path_outliers = self.dir_inspect / 'coarse_offset_outliers.txt'
            path_outliers = self.dir_inspect / 'inf_vals.txt'

        outliers_data = {}

        if Path(path_outliers).exists():
            with open(path_outliers, 'r') as f:
                # Skip the header line
                next(f)

                # Read data line by line
                for line in f:
                    parts = line.strip().split('\t')
                    slice_num = int(parts[0])
                    tile_id = int(parts[-2])
                    tile_id_nn = int(parts[-1])

                    # Check if the key already exists in the dictionary
                    if slice_num in outliers_data:
                        # If the key exists, append the new outlier data to the existing list
                        outliers_data[slice_num].append((tile_id, tile_id_nn))
                    else:
                        # If the key does not exist, create a new list with the outlier data
                        outliers_data[slice_num] = [(tile_id, tile_id_nn)]

            # Check if outliers_data is empty
            if not outliers_data:
                raise ValueError("No data found in the input file.")

        return outliers_data

    def plot_specific_ovs(self,
                          ov_dict: Dict[int, List[Tuple[int, int]]],
                          dir_name_out: Optional[str] = None,
                          refine=False,
                          est_vec: Optional[Vector] = None,
                          shift_abs_dev: Optional[float] = 15.
                          ) -> None:

        for sec_num, tid_list in ov_dict.items():
            if sec_num not in self.section_nums:
                continue

            sec_path = self.section_dicts[sec_num]
            sec = Section(sec_path)
            sec.feed_section_data()

            # Specify custom shift vector here
            shift_vec = (None, 0)

            # Specify parent dir for stored images
            if dir_name_out is None:
                dir_name_out = 'overlaps'

            dir_out = self.dir_inspect / dir_name_out
            utils.create_directory(dir_out)

            # Filter duplicate tile-id pairs
            seen = set()
            unique_tid_list = [x for x in tid_list if x not in seen and not seen.add(x)]

            # skip = [(860, 900), (941, 981)]
            skip = []
            # unique_tid_list = [(651, 683),] # (624, 656)
            for (tid_a, tid_b) in unique_tid_list:
                if (tid_a, tid_b) not in skip:
                    # print(f'Plotting s{sec.section_num} t{tid_a}-t{tid_b}')
                    # logging.info(f'Plotting s{sec.section_num} t{tid_a}-t{tid_b}')

                    if refine:
                        refine_kwargs = dict(tid_a=tid_a, tid_b=tid_b, masking=False, levels=3,
                                             max_ext=24, stride=8, clahe=True, store=True,
                                             plot=False, show_plot=False, est_vec=None)
                        shift_vec = sec.refine_pyramid(**refine_kwargs)
                        # print(f'refined vector: {shift_vec}')

                    # Verify is coarse offset is within limits, otherwise skip
                    axis = 1 if utils.pair_is_vertical(sec.tile_id_map, tid_a, tid_b) else 0
                    offset = sec.get_coarse_offset(tid_a, axis)
                    offset_valid, _ = utils.vector_dist_valid(offset, est_vec, shift_abs_dev)
                    if offset_valid:
                        logging.info(f's{sec.section_num} t{tid_a}-t{tid_b} coarse offset deviation within limits.')
                        break

                    args = dict(tid_a=tid_a,
                                tid_b=tid_b,
                                shift_vec=shift_vec,
                                dir_out=dir_out,
                                show_plot=False,
                                clahe=True,
                                blur=1.0)
                    sec.plot_ov(**args)
        return

    def plot_tile_overlaps(self,
                           sec_start: Optional[int],
                           sec_end: Optional[int],
                           tid_a: int,
                           tid_b: int,
                           dir_out: Optional[UniPath]
                           ):
        """
        Stores rendered overlap regions of range of tile-pairs and specific sections.
        """

        assert tid_a != tid_b
        assert any((sec_start, sec_end))

        if sec_start is None:
            sec_start = self.section_nums[0]

        if sec_end is None:
            sec_end = self.section_nums[-1]

        if dir_out is None:
            dir_out = self.dir_inspect / 'overlaps'

        logging.info(f'plotting overlaps of tiles {tid_a, tid_b}, sections range: {sec_start, sec_end}')

        for sec_num in range(sec_start, sec_end):
            if sec_num in self.section_nums:
                my_sec = Section(self.section_dicts[sec_num])
                my_sec.feed_section_data()

                # Specify custom shift vector here
                shift_vec = (None, 0)

                args = dict(tid_a=tid_a,
                            tid_b=tid_b,
                            shift_vec=shift_vec,
                            dir_out=dir_out,
                            show_plot=False,
                            clahe=True,
                            blur=1.2)

                my_sec.plot_ov(**args)

        return

    def get_missing_sections(self) -> None:
        """Identify section numbers discontinuities in section folder"""
        if self.section_dirs is None:
            self.list_all_section_dirs()

        missing_nums = []
        if len(self.section_dirs) > 1 and self.section_dirs:
            # first: int = utils.get_section_num(str(self.section_dirs[0]))
            # last: int = utils.get_section_num(str(self.section_dirs[-1]))  # TODO: implement skipped section numbers
            first: int = self.first_sec
            last: int = self.last_sec
            section_range = set(range(first, last + 1))
            missing_nums = sorted(list(section_range - set(self.section_nums)))

        if len(missing_nums) > 0:
            utils.write_dict_to_yaml(str(self.fp_missing_sections), missing_nums)
            is_are = 'is' if len(missing_nums) == 1 else 'are'
            logging.warning(f"There {is_are} {len(missing_nums)} missing sections in 'sections' folder!")

        self.missing_sections = missing_nums
        return

    def add_missing_sections(self,
                             prefix_tile_name: str,
                             grid_shape: Tuple[int, int],
                             thickness_nm: int,
                             acquisition: str,
                             tile_height: int,
                             tile_width: int,
                             tile_overlap: int,
                             resolution_nm: float):

        def create_section_folder(section_num: int) -> Path:
            path_section = self.dir_sections / f's{section_num}_g{self.grid_nr}'
            path_section.mkdir(parents=True, exist_ok=True)
            return path_section

        def create_tile_entries(tile_paths: List[Path]):
            def_entry = {
                    'tile_id': self.grid_nr,
                    'path': '',
                    'stage_x': 0,
                    'stage_y': 0,
                    'resolution_xy': 10.0,
                    'unit': 'nm'}

            tile_entries = []
            for tp in sorted(tile_paths):
                new_entry = def_entry.copy()
                new_entry['tile_id'] = int(tp.parent.name[1:])
                new_entry['path'] = str(tp)
                new_entry['resolution_xy'] = resolution_nm
                tile_entries.append(new_entry)

            return tile_entries

        def create_section_entries():
            return {
                'file_path': str(self.dir_sections / f's{sec_num}_g{self.grid_nr}' / 'section.yaml'),
                'section_num': sec_num,
                'tile_grid_num': self.grid_nr,
                'grid_shape': self.grid_shape,
                'acquisition': acquisition,
                'thickness': thickness_nm,
                'tile_height': tile_height,
                'tile_width': tile_width,
                'tile_overlap': tile_overlap,
                'tiles': create_tile_entries(paths_tiles)
            }

        def store_tile_id_map(section_dir: Path,
                              tile_paths: List[Path],
                              grid_shape: Tuple[int, int]
                              ):

            tile_ids = [int(p.parent.name[1:]) for p in tile_paths]
            tile_id_map = utils.compute_tile_id_map(tuple(grid_shape), tile_ids)

            if tile_id_map is None:
                logging.warning('Tile id-map could not be created')
            else:
                tile_id_map_list = tile_id_map.tolist()
                with open(section_dir / 'tile_id_map.json', "w") as file:
                    json.dump(tile_id_map_list, file)
            return

        # Read missing_sections.yaml
        with open(self.fp_missing_sections, 'r') as f:
            missing_nums = yaml.safe_load(f)

        if not missing_nums:
            logging.warning('No missing numbers could be loaded form missing_sections.yaml!')
            return

        # Filter section tile-paths
        dir_tiles = Path(self.acq_dir) / 'tiles' / f'g000{self.grid_nr}'
        tile_folders = utils.scan_dirs(dir_tiles)
        if not tile_folders:
            logging.warning('Tile folders with raw EM-data not found.')
            return

        for sec_num in missing_nums:
            paths_tiles = []
            for tf in tile_folders:
                tile_path = tf / f'{prefix_tile_name}g000{self.grid_nr}_{tf.name}_s{str(sec_num).zfill(5)}.tif'
                if tile_path.is_file():
                    paths_tiles.append(tile_path)

            if not paths_tiles:
                logging.warning(f'No tiles paths found for section s{sec_num}')
                continue

            sec_dir = create_section_folder(sec_num)
            data = create_section_entries()
            utils.create_section_yaml_file(**data)
            store_tile_id_map(Path(sec_dir), paths_tiles, grid_shape)

        return

    def init_experiment(self) -> None:
        """
        Perform initial data processing for Inspection class
        """
        self.read_exp_notes()
        self.list_all_section_dirs()
        self.get_missing_sections()
        self.create_inspection_dirs()


    def list_all_section_dirs(self, skip: Optional[bool] = False) -> None:
        """Lists all section and .zarr folders stored in 'sections' and 'stitched' folders
        """

        # Sections folder
        res = utils.process_dirs(str(self.dir_sections),
                                 utils.filter_and_sort_sections)
        if res is not None:
            (self.section_dirs,
             self.section_names,
             self.section_nums,
             self.section_dicts) = res

        # Stitched sections folder
        res = utils.process_dirs(str(self.dir_stitched),
                                 utils.list_stitched)
        if res is not None:
            (self.stitched_dirs,
             self.stitched_names,
             self.stitched_nums,
             self.stitched_dicts) = res

        return


    def validate_stitched(self):
        if self.stitched_nums is None:
            self.init_experiment()

        valid_nums = []
        for num in self.stitched_nums:
            section = Section(exp.section_dicts[num])
            if section.stitched:
                valid_nums.append(num)

        self.stitched_nums_valid = valid_nums
        return


    def create_inspection_dirs(self):
        new_dirs = ('overlaps', 'traces', 'downscaled', 'inf_overlaps')
        logging.debug(f'Creating inspection infrastructure {new_dirs}')
        for leaf in new_dirs:
            create_directory(self.dir_inspect / leaf)
        return

####

# STATIC

####

def align_wrapper(args) -> None:
    self, sec_num, tid_pair, align_args = args
    tid_a, tid_b = tid_pair
    align_tile_pair(self, sec_num, tid_a, tid_b, **align_args)
    return

def read_coarse_mat(path: Union[str, Path]) -> tuple[Optional[ndarray], ndarray, ndarray]:
    """
    Return contents of a coarse data file (either npz or json).
    :param path: path to the data file (coarse.npz or cx_cy.json)
    :return:
        coarse_mesh: contents of coarse_mesh (only if .npz file is read) or None
        cx: coarse shifts between horizontal neighbors
        cy: coarse shifts between vertical neighbors
    """
    path = Path(path)
    try:
        if path.suffix == '.npz':
            arr = np.load(path)
            coarse_mesh, cx, cy = arr['coarse_mesh'], arr['cx'], arr['cy']
            # logging.debug(f'cx npz shape: {np.shape(cx)}')
            return coarse_mesh, cx, cy
        elif path.suffix == '.json':
            # Load the JSON data from the file
            with open(path, 'r') as file:
                data = json.load(file)

            # Convert the loaded data back to NumPy arrays
            cx = np.asarray(data.get('cx', {}))
            cy = np.asarray(data.get('cy', {}))
            coarse_mesh = None
            return coarse_mesh, cx, cy
        else:
            print(f"Error: Unsupported file format for '{path}'. Supported formats are '.npz' and '.json'.")

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON data in '{path}': {e}")
    except Exception as e:
        print(f"Error during read_coarse_mat: {e}")


def aggregate_coarse_shifts(paths: List[Union[str, Path]]) -> Dict[int, Tuple[ndarray, ndarray]]:
    """
    Read cx, cy matrices from 'paths' folders and aggregate them into a dict
    :param paths: section paths
    :return: agg_shifts: dictionary containing section number (int) and corresponding cx, cy matrices
    """
    # TODO: allow to select input file type (.npz or .json)
    agg_shifts = {}
    for path in paths:
        path = Path(path)
        fns = ['coarse.npz', 'cx_cy.json']
        fp_coarse_npz, fp_cx_cy = [path / f_name for f_name in fns]
        section_num = int(path.name.split('_')[0].strip('s'))

        if not fp_coarse_npz.exists():
            logging.warning(f's{section_num}: Reading cxy failed. Check if coarse-files are present.')
        else:
            logging.debug(f's{section_num}: Reading cxy from coarse.npz file...')
            _, cx, cy = read_coarse_mat(fp_coarse_npz)
            agg_shifts[section_num] = (cx, cy)

    return agg_shifts


def create_directory(dir_path: Path):
    try:
        os.makedirs(dir_path, exist_ok=True)
        logging.debug(f"Directory '{dir_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_path}' already exists.")
    except PermissionError:
        print(f"Permission denied. Unable to create directory '{dir_path}'.")
    except OSError as e:
        print(f"Error creating directory '{dir_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_traces_eval_ov(exp: Inspection):
    ids_to_plot = utils.get_tile_ids_set(str(exp.path_id_maps))
    ids_to_plot = sorted(list(ids_to_plot))
    # ids_to_plot = [867, 946]
    # ids_to_plot = [860, ]

    traces_out_dir = exp.dir_inspect / 'overlaps' / 'ov_validation'
    for tile_id in ids_to_plot:
        print(f'plotting trace of tile nr.: {tile_id}')
        plot_name = 't' + str(tile_id).zfill(4) + '_ov_ssim.png'

        args = dict(
            path_cxyz=str(exp.fp_eval_ov),
            path_id_maps=str(exp.path_id_maps),
            path_plot=str(traces_out_dir / plot_name),
            tile_id=tile_id,
            sec_range=(320, 370),  # Specify range of sections to be plotted. None for first/last slice
            show_plot=0,
        )
        utils.plot_trace_eval_ov(**args)
    return


def plot_trace_wrapper(args):
    utils.plot_trace_from_backup(**args)
    return


def postprocess_cxcy(exp: Inspection,
                     plot_traces: bool,
                     locate_inf: bool,
                     trace_ids: Optional[Iterable[int]] = None,
                     in_parallel: bool = False):
    """
      Post-processes cx_cy.json files.

      Args:
          exp (Inspection): Inspection object.
          plot_traces (bool): Whether to plot traces.
          locate_inf (bool): Whether to locate and save inf values from cx_cy.json files.
          trace_id (int): plot only specified tile_id trace
          :param in_parallel:
          :param trace_ids:
      """

    if exp.section_nums is None:
        exp.init_experiment()

    # Backup cx_cy
    if not exp.path_cxyz.exists():
        exp.backup_coarse_offsets()

    # # Override backing-up coarse offsets
    exp.backup_coarse_offsets()

    if not exp.path_id_maps.exists():
        exp.backup_tile_id_maps()
    # exp.backup_tile_id_maps()

    # Locate and save inf values from cxyz backup
    if locate_inf:
        _ = utils.locate_inf_vals(exp.path_cxyz, exp.dir_inspect, store=True)

    if plot_traces:
        # Determine traces to be plotted
        traces_out_dir = exp.dir_inspect / 'traces'
        ids_to_plot = sorted(list(utils.get_tile_ids_set(str(exp.path_id_maps))))
        if trace_ids is not None:
            ids_to_plot = trace_ids

        # Create args list for trace plotting
        args_list = []
        for tile_id in ids_to_plot:
            print(f'Plotting trace of tile nr.: {tile_id}')
            plot_name = 't' + str(tile_id).zfill(4) + '_trace.png'
            args = (str(exp.path_cxyz), str(exp.path_id_maps), str(traces_out_dir / plot_name), tile_id, (None, None), 0)
            args_list.append(args)

        if in_parallel:
            num_proc = min(40, len(ids_to_plot))
            with multiprocessing.Pool(processes=num_proc) as pool:
                pool.starmap(utils.plot_trace_from_backup, args_list)
        else:
            for i_args in args_list:
                utils.plot_trace_from_backup(*i_args)

    return


def align_tile_pair_(sec_path: str, tid_a: int, tid_b: int, refine: bool):
    my_sec = Section(sec_path)
    my_sec.feed_section_data()

    # overlaps_xy = ((200, 250, 300), (200, 270, 340))
    # min_range = ((35, 50, 65), (0, 10, 100))

    overlaps_xy = ((170, 250, 340), (170, 250, 340))
    min_range = ((40, 80, 110), (40, 80, 110))

    # Ruth 3
    overlaps_xy = ((200, 270, 340), (200, 270, 340))
    min_range = ((20, 30, 40), (20, 30, 40))

    clahe = True

    args = dict(overlaps_xy=overlaps_xy,
                min_range=min_range,
                min_overlap=2,
                filter_size=10,
                max_valid_offset=400,
                )

    offset = my_sec.compute_coarse_offset(
        tid_a, tid_b, refine, clahe=clahe, store=False, **args)

    logging.info(f's{my_sec.section_num} coarse offset: {offset}')
    print(f'computed coarse offset: {offset}')
    return


def align_tile_pair(
        exp: Inspection,
        sec_num: int,
        tid_a: int,
        tid_b: int,
        masking: bool,
        store: bool,
        refine: bool,
        plot: bool,
        refine_params: dict,
        clahe: bool = True,
) -> Optional[float]:

    sec_dir = exp.dir_sections / f's{sec_num}_g{exp.grid_nr}'

    if not sec_dir.exists():
        logging.warning(f'Section s{sec_num} not found in the sections directory!')
        return

    my_sec = Section(sec_dir)
    my_sec.feed_section_data()

    if tid_a not in my_sec.tile_id_map and tid_b not in my_sec.tile_id_map:
        logging.warning(f'Section s{sec_num} does not contain tile-pair {tid_a}-{tid_b}!')
        return

    if masking:
        my_sec.load_masks()

    logging.info(f'Aligning s{sec_num} tile_pair: {tid_a, tid_b}')

    # Optionally run only refine pyramid with desired settings
    seam_score = None
    if refine:
        offset = my_sec.refine_pyramid(tid_a, tid_b, masking, **refine_params)

        # Evaluate seam quality using current coarse offset
        tile_pair = my_sec.load_masked_pair(tid_a, tid_b, roi=True, smr=False, gauss_blur=True, sigma=6.0)
        axis = 1 if utils.pair_is_vertical(my_sec.tile_id_map, tid_a, tid_b) else 0
        current_co = my_sec.get_coarse_offset(tid_a, axis)
        seam_score = my_sec.eval_ov(tile_pair, current_co)
        if seam_score is not None:
            logging.info(f's{my_sec.section_num} t{tid_a}-t{tid_b} ov mssim: {seam_score:.2f}')
        else:
            seam_score = np.nan
            logging.warning(
                f'eval_ov (t{tid_a}-{tid_b}): estimation of overlap quality failed in section {my_sec.section_num}.')

    else:

        # overlaps_xy = ((200, 250, 300), (200, 270, 340))
        # min_range = ((35, 50, 65), (0, 10, 100))

        # # Mont2
        # overlaps_xy = ((170, 250, 340), (170, 250, 340))
        # min_range = ((0, 20, 40), (0, 20, 40))

        # # Mont3
        # overlaps_xy = ((200, 270, 340), (200, 270, 340))
        # min_range = ((20, 30, 40), (20, 30, 40))

        # # Li-3
        # overlaps_xy = ((350, 250, 450), (350, 250, 450))
        # min_range = ((50, 100, 150), (50, 100, 150))

        # # Mont4
        # overlaps_xy = ((200, 250, 350), (200, 250, 350))
        # min_range = ((0, 70, 140), (0, 70, 140))

        # # Dp2
        # overlaps_xy = ((200, 150, 350), (200, 150, 350))
        # min_range = ((0, 30, 60), (0, 30, 60))

        # Li-1
        overlaps_xy = ((200, 100, 300), (200, 100, 300))
        min_range = ((0, 50, 100), (0, 50, 100))

        args = dict(overlaps_xy=overlaps_xy,
                    min_range=min_range,
                    min_overlap=2,
                    filter_size=10,
                    masking=masking,
                    max_valid_offset=400)

        offset = my_sec.compute_coarse_offset(
            tid_a, tid_b, refine, store, clahe, **args)

        logging.info(f's{my_sec.section_num} coarse offset: {offset}')
        print(f'computed coarse offset: {offset}')

    if plot:
        dir_out = utils.cross_platform_path(str(exp.dir_inf_overlaps))
        # Plot zero overlap if Inf in coarse offset
        if offset is None or any(np.isinf(offset)):
            offset = (0, 0)
        my_sec.plot_ov(tid_a, tid_b, offset, dir_out, show_plot=False,clahe=clahe, blur=1.0)

    return seam_score


def compute_coarse_offset_wrapper(section: Section, **kwargs):
    """
    Wrapper for coarse offsets parallel computation
    """
    return Section.compute_coarse_offset(section, **kwargs)


def par_coarse_trace(
        exp: Inspection,
        kwargs: Dict,
        num_processes: int,
        nums_to_align: Optional[Iterable[int]] = None
) -> None:
    """
    Compute coarse offsets for particular tile-pair for range of
    sections in parallel
    """

    if exp.section_nums is None:
        exp.init_experiment()

    # If no section range was specified, align pairs from all sections
    if nums_to_align is None:
        start = exp.section_nums[0]
        end = exp.section_nums[-1]
        nums_to_align = set(range(start, end + 1))

    nums_to_align = set(nums_to_align).intersection(exp.section_nums)
    section_paths = [exp.section_dicts[num] for num in sorted(list(nums_to_align))]
    sections = [Section(p) for p in section_paths]

    # Filter sections without any of the requested tile_ids
    # TODO debug
    for sec in sections:
        sec.read_tile_id_map()
        tid_a, tid_b = kwargs['id_a'], kwargs['id_b']
        if tid_a not in sec.tile_id_map or tid_b not in sec.tile_id_map:
            sections.remove(sec)

    # Compute coarse offsets
    comp_off_partial = partial(compute_coarse_offset_wrapper, **kwargs)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(comp_off_partial, sections)

    return


def run_par_coarse_trace(
        exp: Inspection,
        id_a: int,
        id_b: int,
        masking: bool,
        refine: bool,
        start: Optional[int] = None,
        end: Optional[int] = None,
        custom_sec_nums: Optional[Iterable[int]] = None,
        est_vec: Optional[Vector] = None,
):
    """
    Compute coarse offsets of a specific tile-pair in parallel
    """
    jax.config.update("jax_platform_name", "cpu")

    # Select section range (optional)
    nums_to_align = None
    if custom_sec_nums is not None:
        nums_to_align = custom_sec_nums
    elif start is not None and end is not None:
        nums_to_align = set(range(start, end + 1))

    if nums_to_align is not None:
        nums_to_align = set(range(exp.first_sec, exp.last_sec)).intersection(nums_to_align)
        nums_to_align = sorted(list(nums_to_align))
    
    # Set number of CPU workers
    max_workers = 42
    num_processes = min(max_workers, len(nums_to_align)) if nums_to_align is not None else max_workers
    
    # Mont1
    # overlaps_xy = ((270, 240, 300), (265, 240, 300))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # # Ruth 2
    # overlaps_xy = ((200, 250, 350), (200, 250, 350))
    # min_range = ((40, 80, 110), (40, 80, 110))

    # # Mont 3
    # overlaps_xy = ((200, 270, 340), (200, 270, 340))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # # Mont4
    # overlaps_xy = ((200, 250, 350), (200, 250, 350))
    # min_range = ((0, 70, 140), (0, 70, 140))

    # # Li-3
    # overlaps_xy = ((350, 250, 450), (350, 250, 450))
    # min_range = ((0, 50, 100), (0, 50, 100))

    # Dp2
    overlaps_xy = ((200, 150, 350), (200, 150, 350))
    min_range = ((0, 30, 60), (0, 30, 60))

    kwargs = dict(
        id_a=id_a,
        id_b=id_b,
        refine=refine,
        store=True,
        clahe=True,
        overlaps_xy=overlaps_xy,
        min_range=min_range,
        min_overlap=2,
        filter_size=10,
        masking=masking,
        max_valid_offset=450,
        est_vec=est_vec,
        co_score_lim=None,
        co_lim_dist=15.0,
        custom_mask_params=(400, 400, 1500, 0)
    )

    par_coarse_trace(
        exp=exp,
        kwargs=kwargs,
        num_processes=num_processes,
        nums_to_align=nums_to_align)

    return


def plot_ov_wrapper(section, **kwargs):
    # print(f'Plotting s{section.section_num}')
    return Section.plot_ov(section, **kwargs)


def par_plot_ov(
        exp: Inspection,
        kwargs: Dict,
        num_processes: int,
        nums_to_align: Optional[Iterable[int]] = None):

    if exp.section_nums is None:
        exp.init_experiment()

    # If no section range was specified, align pairs from all sections
    if nums_to_align is None:
        start: int = exp.section_nums[0]
        end: int = exp.section_nums[-1]
        nums_to_align = set(range(start, end + 1))
    else:
        nums_to_align = set(nums_to_align)

    nums_to_align = nums_to_align.intersection(exp.section_nums)
    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Filter sections without any of the requested tile_ids or pair is not neighbouring
    for sec in sections:
        sec.read_tile_id_map()
        tid_a, tid_b = kwargs['tid_a'], kwargs['tid_b']
        is_vert = utils.pair_is_vertical(sec.tile_id_map, tid_a, tid_b)
        if is_vert is None or tid_a not in sec.tile_id_map or tid_b not in sec.tile_id_map:
            sections.remove(sec)

    part_func = partial(plot_ov_wrapper, **kwargs)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(part_func, sections)
    return


def eval_ov_quality_exp(exp: Inspection,
                        custom_tile_id_pairs: Optional[Iterable[Tuple[int, int]]] = None,
                        num_proc=42
                        ) -> None:

    # Create dict of all sections and all tile_id pairs
    ov_dict: Dict[int, List[Tuple[int, int]]] = dict()

    # Get valid tile-id neighbor pairs and add them to final dict
    for num in exp.section_nums[:]:

        # Omit sections out of specified range (from experiment_configs.py)
        if num not in range(exp.first_sec, exp.last_sec + 1):
            continue

        sec_path = exp.section_dicts.get(num)
        if sec_path is not None:
            my_sec = Section(sec_path)
            my_sec.read_tile_id_map()
            ov_dict[num] = utils.get_neighbour_pairs(my_sec.tile_id_map)

    inverted_ov_dict = utils.invert_(ov_dict)

    # # Select only tile pairs with first tile_id lower than threshold
    # threshold = 742
    # inverted_ov_dict = {k: v for k, v in inverted_ov_dict.items() if k[0] <= threshold}

    if not inverted_ov_dict:
        return

    # Compute only specific tile-pairs
    sel_dict = {}
    if custom_tile_id_pairs is not None:
        for pair in custom_tile_id_pairs:
            if pair in inverted_ov_dict.keys():
                sel_dict[pair] = inverted_ov_dict[pair]
        inverted_ov_dict = sel_dict

    for (tile_id_a, tile_id_b), sec_nums in inverted_ov_dict.items():
        kwargs = dict(exp=exp,
                      start=min(sec_nums),
                      end=max(sec_nums),
                      tid_a=tile_id_a,
                      tid_b=tile_id_b,
                      num_proc=num_proc)

        run_par_eval_ov(**kwargs)
    return

def process_eval_ov_results(mssim_tuples, tid_a, tid_b, dir_out: UniPath, sort: bool = True):

    all_pk_sorted_nums = OrderedDict(sorted((int(num), float(mssim)) for num, mssim in mssim_tuples if num is not None))

    # Construct paths and filenames
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    to_write = all_pk_sorted_nums
    if not sort:
        sorted_mssim = sorted(mssim_tuples, key=lambda x: x[1])
        to_write = {int(num): float(val) for num, val in sorted_mssim}

    tile_str = f't{str(tid_a).zfill(4)}_t{str(tid_b).zfill(4)}'
    fn_eval_ov = dir_out / f'{tile_str}_seams.yaml'
    logging.info(f'Writing seam scores to: {fn_eval_ov}')
    with open(str(fn_eval_ov), "w") as file:
        yaml.dump(to_write, file, default_flow_style=False)

    # Plot result
    path_plot = dir_out / f'{tile_str}_seams.png'
    utils.plot_eval_ov(all_pk_sorted_nums, tid_a, tid_b, str(path_plot))
    return



def run_par_eval_ov(
        exp: Inspection,
        start: Optional[int],
        end: Optional[int],
        tid_a: int,
        tid_b: int,
        num_proc: int,
):
    """Plots overlap regions of a specific tile-pair over sections in parallel."""
    jax.config.update("jax_platform_name", "cpu")

    if exp.section_nums is None:
        exp.init_experiment()

    # Select section range (optional)
    if start is not None and end is not None:
        assert start <= end
        nums_to_align = exp.get_valid_sec_nums(start, end)
    elif start is None and end is not None:
        nums_to_align = set(range(exp.first_sec, end))
    elif start is not None and end is None:
        nums_to_align = set(range(start, exp.last_sec))
    else:
        nums_to_align = set(range(exp.first_sec, exp.last_sec + 1))

    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]
    part_func = partial(eval_ov_wrapper, tid_a=tid_a, tid_b=tid_b)

    # Use try-finally to ensure resource cleanup
    try:
        with multiprocessing.Pool(processes=num_proc) as pool:
            # Use imap_unordered for potentially better performance with large results
            results = pool.imap_unordered(part_func, sections)
            mssim_tuples = list(results)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return

    dir_out = exp.dir_overlaps
    process_eval_ov_results(mssim_tuples, tid_a, tid_b, dir_out)
    return



def eval_ov_wrapper(section, tid_a, tid_b) -> Tuple[Optional[int], Optional[float]]:

    section.read_tile_id_map()
    if (section.tile_id_map is None
            or tid_a not in section.tile_id_map
            or tid_b not in section.tile_id_map):
        # logging.warning(f'Section {section.section_num} does
        # not contain tile_ids {tid_a} and {tid_b}.')
        return None, None

    is_vert = utils.pair_is_vertical(section.tile_id_map, tid_a, tid_b)
    if is_vert is None:
        logging.warning(f'Specified tile_ids are not neighbors in section {section.section_num}!')
        return None, None
    
    # Load tile-pair
    tile_pair = section.load_masked_pair(tid_a, tid_b, roi=True, smr=False, gauss_blur=True, sigma=6.0)
    if tile_pair is None:
        return section.section_num, np.nan
    
    axis = 1 if is_vert else 0
    shift_vec = section.get_coarse_offset(tid_a, axis)
    
    mssim = section.eval_ov(tile_pair, shift_vec)
    if mssim is not None:
        logging.info(f's{section.section_num} t{tid_a}-t{tid_b} ov mssim: {mssim:.2f}')
    else:
        mssim = np.nan
        logging.warning(f'eval_ov (t{tid_a}-{tid_b}): estimation of overlap quality failed in section {section.section_num}.')

    return section.section_num, mssim


def run_par_plot_ov(
        exp: Inspection,
        start: Optional[int],
        end: Optional[int],
        tid_a: int,
        tid_b: int,
        num_proc: int,
        dir_out: Optional[str] = None,
        sec_nums: Optional[Iterable[int]] = None
):
    """Plots overlap regions of a specific tile-pair over sections in parallel.
    """
    jax.config.update("jax_platform_name", "cpu")

    if dir_out is None:
        dir_out = exp.dir_inspect / 'overlaps'

    # Select section range (optional)
    nums_to_align = None
    if sec_nums is not None:
        nums_to_align = sec_nums
    elif None not in (start, end):
        nums_to_align = set(range(start, end+1))

    kwargs = dict(
        tid_a=tid_a,
        tid_b=tid_b,
        shift_vec=(0, None),
        dir_out=dir_out,
        show_plot=False,
        clahe=True,
        blur=1.2
    )

    par_plot_ov(
        exp=exp,
        kwargs=kwargs,
        num_processes=num_proc,
        nums_to_align=nums_to_align,
    )
    return


def roi_masks_wrapper(section, **kwargs):
    return Section.create_masks(section, **kwargs)


def roi_masks_par(exp: Inspection, num_processes: int, nums_to_align: Optional[Iterable[int]] = None):
    if exp.section_nums is None:
        exp.init_experiment()

    # If no section range was specified, align pairs from all sections
    if nums_to_align is None:
        start: int = exp.section_nums[0]
        end: int = exp.section_nums[-1]
        nums_to_align = set(range(start, end + 1))

    # Specific range
    missing_nums = []
    fn = 'tile_masks.npz'
    for directory in exp.section_dirs:
        fp = (Path(directory) / fn)
        if not fp.exists():
            missing_nums.append(int(directory.name.split("_")[0].strip('s')))

    nums_to_align = set(missing_nums)

    nums_to_align = nums_to_align.intersection(exp.section_nums)
    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Compute ROI masks
    kwargs = dict(roi_thresh=20, store=True)
    part_func = partial(roi_masks_wrapper, **kwargs)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(part_func, sections)

    return


def refine_trace_wrapper(rfn_kwargs: Dict):
    section = rfn_kwargs['sec']
    rfn_kwargs.pop('sec')
    return Section.refine_pyramid(section, **rfn_kwargs)


def refine_coo_wrapper(section, **kwargs):
    return Section.refine_coarse_offset_section(section, **kwargs)


def coo_wrapper(section, **coo_kwargs):
    return Section.compute_coarse_offset_section(section, **coo_kwargs)


def cco_par(
        exp: Inspection,
        kwargs: Dict,
        num_processes: int,
        nums_to_align: Iterable[int],
        refine=False,
) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    jax.config.update("jax_platform_name", "cpu")

    # Construct section objects to be aligned
    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Compute or refine coarse offsets
    if len(sections) <= 0:
        print('No sections were selected for coarse shift estimation.')
        return

    # Select mode of alignment
    part_func = partial(coo_wrapper, **kwargs)

    if refine:
        refine_kwargs = dict(
        masking=False,
        levels=1,
        max_ext=15,
        stride=2,
        clahe=True,
        store=True,
        plot=True,
        show_plot=False
        )
        part_func = partial(refine_coo_wrapper, **refine_kwargs)

    # Create map of alignment processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(part_func, sections)

    return


def run_coo_par(exp: Inspection,
                start: Optional[int] = None,
                end: Optional[int] = None,
                custom_sec_nums: Optional[Iterable[int]] = None,
                overwrite_cxcy=False,
                refine=False,
                masking=False,
                generate_masks=False,
                store=True,
                num_processes=42
                ):

    if exp.section_nums is None:
        exp.init_experiment()

    # Define sections to be aligned
    nums_to_align = exp.get_valid_sec_nums(start, end, custom_sec_nums)
    num_processes = min(len(nums_to_align), num_processes)

    # # Process section only if cxcy was computed before some specific time
    # nums_to_align = exp.mod_time_select(nums_to_align, min_file_age=14, stitched_dir=False)

    # Mont1
    # overlaps_xy = ((150, 200, 300), (150, 200, 300))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # # Mont2
    # overlaps_xy = ((170, 250, 340), (170, 250, 340))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # # Mont3
    # overlaps_xy = ((200, 270, 340), (200, 270, 340))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # # Mont4
    # overlaps_xy = ((200, 250, 350), (200, 250, 350))
    # min_range = ((0, 70, 140), (0, 70, 140))

    # Li-3
    # overlaps_xy = ((350, 250, 450), (350, 250, 450))
    # min_range = ((0, 50, 100), (0, 50, 100))

    # # Dp2
    # overlaps_xy = ((200, 150, 350), (200, 150, 350))
    # min_range = ((0, 30, 60), (0, 30, 60))

    # Li-1
    overlaps_xy = ((200, 100, 300), (200, 100, 300))
    min_range = ((0, 50, 100), (0, 50, 100))

    kwargs_masks = dict(
        roi_thresh=20,
        max_vert_ext=200,
        edge_only=True,
        n_lines=30,
        store=True,
        filter_size=50,
        range_limit=0
    )

    kwargs = dict(
        co=None,
        overlaps_xy=overlaps_xy,
        min_range=min_range,
        store=store,
        min_overlap=20,
        filter_size=10,
        max_valid_offset=450,
        clahe=True,
        masking=masking,
        overwrite_cxcy=overwrite_cxcy,
        rewrite_masks=generate_masks,
        kwargs_masks=kwargs_masks
    )

    if refine:
        kwargs = dict(
            masking=True,
            levels=3,
            max_ext=50,
            stride=10,
            clahe=True,
            store=True,
            plot=True,
            show_plot=False
        )

    cco_par(
        exp=exp,
        kwargs=kwargs,
        num_processes=num_processes,
        nums_to_align=nums_to_align,
        refine=refine,
    )

    return


def validate_seam(path_ov_img: UniPath) -> Tuple[int, float]:
    path_ov_img = Path(path_ov_img)
    sec_num = str.split(str(Path(path_ov_img).name), "_")[0].strip('s')
    sec_num = int(sec_num)
    try:
        logging.debug(f'processing: {path_ov_img}')
        ov_img = skimage.io.imread(str(path_ov_img))
    except FileNotFoundError as e:
        print(e)
        return sec_num, np.nan

    pair_is_vert = True if ov_img.shape[0] < ov_img.shape[1] else False
    seam_pk_ssim = utils.detect_bad_seam(ov_img, pair_is_vert)

    return sec_num, seam_pk_ssim


def run_par_validate_seam(dir_ov: str, num_proc: int = 5) -> None:
    dir_ov = utils.cross_platform_path(dir_ov)
    ov_filenames = list(Path(dir_ov).glob("*_ov.jpg"))
    if not ov_filenames:
        return

    # Process files
    num_processes = min(num_proc, len(ov_filenames))
    with Pool(num_processes) as pool:
        # results = dict(pool.map(validate_seam, ov_filenames))
        results = pool.imap_unordered(validate_seam, ov_filenames)
        mssim_tuples = list(results)

    res = OrderedDict(sorted((num, mssim) for num, mssim in mssim_tuples if num is not None))

    # # Sort sec_nums and ssim_peaks based on sec_nums
    # sec_nums = sorted(list(results.keys()))
    # ssim_peaks = [results[num] for num in sec_nums]
    #
    # # Sort worst to best quality
    # ssim_sorted = sorted(list(results.values()))[::-1]
    # nums_sorted = []
    # for ss in ssim_sorted:
    #     for k, v in results.items():
    #         if v == ss:
    #             nums_sorted.append(k)

    # worst_to_best = {k: v for k, v in zip(nums_sorted, ssim_sorted)}
    # res = worst_to_best

    # Write out results
    # path_out = Path(dir_ov)
    # dir_name = str(Path(dir_ov).name)
    # fn = f'seam_validation_' + dir_name
    # fp = str(path_out / (fn + '.json'))
    # # utils.write_dict_to_yaml(fp, worst_to_best)
    # with open(fp, 'w') as json_file:
    #     for item in res.items():
    #         json_file.write(f"{item}\n")

    tile_str = str(Path(dir_ov).name)
    fn_seams = f'{tile_str}_seams.yaml'
    fn_eval_ov = Path(dir_ov) / fn_seams
    logging.info(f'Writing seam scores to: {fn_eval_ov}')
    utils.write_dict_to_yaml(str(fn_eval_ov), res)

    # Plot data
    x, y = list(res.keys()), list(res.values())
    # x, y = sec_nums, ssim_peaks
    # plot_path = str(path_out / (fn + '.png'))

    plt.figure(figsize=(15, 9))
    plt.plot(x, y, '*-')
    plt.xlabel('Section number')
    # plt.ylabel('Seam inaccuracy (arb. units)')
    plt.ylabel('Normalized seam accuracy (arb. units)')
    # plt.title(dir_name)
    plt.title(tile_str)
    plt.grid(True)
    # plt.show()
    plot_path = str(fn_eval_ov.with_suffix('.png'))
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return


def run_validate_seams_on_fly(exp, start, end, tid_a, tid_b):
    nums_to_align = exp.get_valid_sec_nums(start, end)

    all_pk = {}
    for sec_num in nums_to_align:

        # Load image data and original coarse shift vector
        my_sec = Section(exp.section_dicts[sec_num])
        data = my_sec.get_masked_img_pair(tid_a, tid_b, masking=True)
        if data is None:
            continue

        tile_map, _, is_vert, shift_vec = data

        if tid_a in my_sec.tile_id_map and tid_b in my_sec.tile_id_map:
            res = my_sec.eval_seam(tile_map, tid_a, tid_b,
                                   shift_vec=shift_vec,
                                   is_vert=is_vert,
                                   dir_out=exp.dir_overlaps,
                                   plot=True,
                                   show_plot=False)
        else:
            res = np.nan

        if res is None:
            res = np.nan

        all_pk[sec_num] = res

    # Plot result
    x = list(all_pk.keys())
    y = list(all_pk.values())
    plt.plot(x, y, '-')
    path_plot = str(exp.dir_overlaps / f't{tid_a}_t{tid_b}_seams.png')
    plt.savefig(path_plot)
    plt.close()
    return


def run_validate_seams_on_fly_new(exp, start, end, tid_a, tid_b):
    nums_to_align = exp.get_valid_sec_nums(start, end)

    all_pk = {}
    for sec_num in nums_to_align:
        # Load image data and original coarse shift vector
        my_sec = Section(exp.section_dicts[sec_num])
        my_sec.read_tile_id_map()
        is_vert = utils.pair_is_vertical(my_sec.tile_id_map, tid_a, tid_b)
        if is_vert is None:
            logging.warning(f'Specified tile_ids are not neighbors!')
            return

        # Load tile-pair
        tile_pair = my_sec.load_masked_pair(tid_a, tid_b,
                                            roi=True, smr=False,
                                            gauss_blur=True, sigma=2.5)
        if tile_pair is None:
            return

        axis = 1 if is_vert else 0
        shift_vec = my_sec.get_coarse_offset(tid_a, axis)

        mssim = my_sec.eval_ov(tile_pair, shift_vec)
        if mssim is not None:
            logging.info(f's{my_sec.section_num} t{tid_a}-t{tid_b} ov mssim: {mssim:.2f}')
        else:
            logging.warning(f'eval_ov (t{tid_a}-{tid_b}): estimation of overlap quality failed.')

        all_pk[sec_num] = mssim

    # Plot result
    x = list(all_pk.keys())
    y = list(all_pk.values())
    plt.plot(x, y, '-')
    path_plot = str(exp.dir_overlaps / f't{tid_a}_t{tid_b}_seams.png')
    print(f'Plot saved to: {path_plot}')
    plt.savefig(path_plot)
    plt.close()
    return


def inspect_ov():
    img_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240217\_inspect\overlaps\t0700_t0701\s3574_t0700_t0701_ov.jpg"
    img_path = str(utils.cross_platform_path(img_path))
    img = skimage.io.imread(img_path)
    pair_is_vert = True if img.shape[0] < img.shape[1] else False

    _ = utils.detect_bad_seam(img, pair_is_vert, plot=True)
    return


def eval_section_ovs_wrapper(section, **kwargs) -> Dict[int, np.ndarray]:
    res = Section.eval_section_overlaps(section, **kwargs)
    ret_val = {section.section_num: res}
    return ret_val


def eval_section_ovs_par(
        exp: Inspection,
        roi_masking: bool,
        smr_masking: bool,
        nums_to_align: Iterable[int],
        num_processes=42
) -> Optional[List[Dict[int, np.ndarray]]]:
    # Construct section objects to be aligned
    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Compute or refine coarse offsets
    if len(sections) <= 0:
        print('No sections were selected for processing.')
        return

    # Select mode of alignment
    part_func = partial(eval_section_ovs_wrapper,
                        roi_masking=roi_masking,
                        smr_masking=smr_masking)

    # Create map of alignment processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        res_list = pool.map(part_func, sections)

    return res_list


def create_masks_for_section(section, **kwargs) -> None:
    """Function to create masks for a single section"""
    return Section.create_masks(section, **kwargs)


def fine_align_sections_multiproc(
        exp: Inspection,
        start: Optional[int] = None,
        end: Optional[int] = None,
        nums_to_align: Optional[Iterable[int]] = None,
        masking: bool = True,
        num_processes: int = 40
):
    # jax.config.update("jax_platform_name", "cpu")
    if exp.section_dicts is None:
        exp.init_experiment()

    if start is not None and end is not None:
        nums_to_align = list(range(start, end + 1))

    if nums_to_align is None:
        nums_to_align = list(range(exp.first_sec, exp.last_sec + 1))

    nums_to_align = [num for num in nums_to_align if num in exp.section_dicts.keys()]
    print(nums_to_align)

    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Compute or refine coarse offsets
    if not len(sections):
        print('No sections were selected for processing.')
        logging.warning('No valid sections were selected for processing. Check experiment setting and section numbers.')
        return

    # Compute without parallelization
    if num_processes == 0:
        for s in sections:
            fine_align_section(s, exp.grid_shape, masking)
        logging.info(f'Finished fine-alignment of sections {min(nums_to_align)} - {max(nums_to_align)}')
        return

    # Select mode of alignment
    part_func = partial(fine_align_section, grid_shape=exp.grid_shape, masking=masking)

    # Create map of alignment processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(part_func, sections)  # , chunksize=2
    return


# Define a Semaphore to limit the number of concurrent threads
semaphore = threading.Semaphore(5)  # Adjust the number as needed

def chunk_list(lst, chunk_size):
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def fine_align_sections_threading(
        exp: Inspection,
        start: Optional[int] = None,
        end: Optional[int] = None,
        nums_to_align: Optional[Iterable[int]] = None,
        masking: bool = True,
        chunk_size: int = 10
        ):

    if start is not None and end is not None:
        nums_to_align = list(range(start, end + 1))

    section_paths = [exp.section_dicts[num] for num in nums_to_align]
    sections = [Section(p) for p in section_paths]

    # Compute or refine coarse offsets
    if len(sections) <= 0:
        print('No sections were selected for processing.')
        return

    # Select mode of alignment
    part_func = partial(fine_align_section,
                        grid_shape=exp.grid_shape,
                        masking=masking
                        )

    semaphore = threading.Semaphore(10)

    for chunk in chunk_list(sections, chunk_size):
        # Create map of alignment threads for each section
        threads = []
        for section in chunk:
            thread = threading.Thread(
                target=process_section_with_semaphore,
                args=(semaphore, part_func, section)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads in the current chunk to complete
        for thread in threads:
            thread.join()

    return

# def process_section_with_semaphore(part_func, section):
#     with semaphore:
#         part_func(section)

def process_section_with_semaphore(semaphore, part_func, section):
    with semaphore:
        part_func(section)
#
# def compute_section_flow_fields_threading(
#         exp: Inspection,
#         grid_shape: List[int],
#         nums_to_align: Iterable[int],
#         flow_fields: bool = False,
#         masking: bool = False,  # not-used
# ):
#     # Construct path pairs to stitched sections
#     invalid = []
#     section_pairs = []
#     for num in nums_to_align:
#         if num in exp.stitched_nums:
#             num_ind = exp.stitched_nums.index(num)
#             try:
#                 nn_num = exp.stitched_nums[num_ind + 1]
#                 p1 = str(exp.stitched_dicts.get(num))
#                 p2 = str(exp.stitched_dicts.get(nn_num))
#                 section_pairs.append([p1, p2])
#             except IndexError as _:
#                 invalid.append(num)
#
#     for inv in invalid:
#         logging.warning(f'Section {inv} not cross-aligned (no neighboring section found).')
#
#     if len(section_pairs) <= 0:
#         print('No sections were selected for processing.')
#         return
#
#     # Read config file
#     path_cfg = exp.fp_est_ff_cfg
#     if not Path(path_cfg).exists():
#         logging.warning(f'Flow fields estimation failed: cfg file not found.')
#         return
#
#     with open(path_cfg) as f:
#         config = yaml.safe_load(f)
#
#     ffe_conf = FlowFieldEstimationConfig(**config["ffe_conf"])
#     stitched_section_dir = ""
#
#     part_func = partial(em_align_est_ff.estimate_flow_fields,
#                         stitched_section_dir=stitched_section_dir,
#                         ffe_conf=ffe_conf)
#
#     # Create map of alignment threads for each section
#     threads = []
#     for section_pair in section_pairs:
#         thread = threading.Thread(
#             target=process_section_pairs_with_semaphore,
#             args=(part_func, section_pair)  # Pass the function and the section pair
#         )
#         threads.append(thread)
#         thread.start()
#
#     for thread in threads:
#         thread.join()
#
#     return


def process_section_pairs_with_semaphore(part_func, section_pair):
    with semaphore:
        part_func(paths_pair=section_pair)

#
# def compute_flow_fields_par(
#         exp: Inspection,
#         nums_to_align: Iterable[int],
#         num_processes: int = 5
# ):
#     # Construct path pairs to stitched sections
#     invalid = []
#     section_pairs = []
#
#     stitched_dicts = exp.stitched_dicts
#     stitched_nums = exp.stitched_nums
#
#     for num in nums_to_align:
#         if num in stitched_nums:
#             num_ind = stitched_nums.index(num)
#             try:
#                 nn_num = stitched_nums[num_ind + 1]
#                 p1 = stitched_dicts.get(num)
#                 p2 = stitched_dicts.get(nn_num)
#                 if p1 and p2:
#                     section_pairs.append([str(p1), str(p2)])
#                 else:
#                     invalid.append(num)
#             except IndexError:
#                 invalid.append(num)
#
#     for inv in invalid:
#         logging.warning(f'Section {inv} not cross-aligned (no neighboring section found).')
#
#     if not section_pairs:
#         print('No sections were selected for processing.')
#         return
#
#     # Read config file
#     path_cfg = exp.fp_est_ff_cfg
#     if not Path(path_cfg).exists():
#         logging.warning(f'Flow fields estimation failed: cfg file not found.')
#         return
#
#     with open(path_cfg) as f:
#         config = yaml.safe_load(f)
#
#     ffe_conf = FlowFieldEstimationConfig(**config["ffe_conf"])
#     kwargs = {'stitched_section_dir': "", 'ffe_conf': ffe_conf}
#     part_func = partial(flow_fields_wrapper, **kwargs)
#
#     # chunk_size = 1
#     with (multiprocessing.Pool(processes=num_processes) as pool):
#         # pool.imap(part_func, section_pairs, chunksize=chunk_size)
#         for _ in pool.imap(part_func, section_pairs):
#             pass
#     return


# def flow_fields_wrapper(sec_pair, **ff_kwargs):
#     return em_align_est_ff.estimate_flow_fields(paths_pair=sec_pair, **ff_kwargs)


def get_size_per_tile(exp: Inspection, num: int, data_size: int) -> Optional[int]:
    # Return immediately if data_size is zero
    if data_size == 0:
        return None

    # Initialize experiment if section dictionaries are not available
    if exp.section_dicts is None:
        exp.init_experiment()

    # Build the file path to tile ID map
    fp = Path(exp.section_dicts[num]) / 'tile_id_map.json'

    try:
        # Load the tile ID map
        tile_id_map = utils.get_tile_id_map(fp)

        # Remove the placeholder value (-1) and calculate unique tiles
        unique_tiles = np.unique(tile_id_map)
        unique_tiles = unique_tiles[unique_tiles != -1]  # Remove -1 if present

        # Check if there are any tiles to process
        if unique_tiles.size == 0:
            return None

        # Calculate and return the size per tile
        return int(data_size / len(unique_tiles))

    except FileNotFoundError:
        print(f"File not found: {fp}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {fp}")
        return None


def main_get_zarr_sizes(exp: Inspection, min_size_per_tile=1000) -> List[int]:
    """Computes size of each .zarr section and store results to json file"""
    exp.init_experiment()

    zarr_sizes: Dict[int, int] = exp.get_zarr_sizes()
    zarr_sizes = dict(sorted(zarr_sizes.items(), key=lambda item: item[1]))

    output_file = str(Path(exp.root) / 'zarr_sizes.json')
    print(f'Storing .zarr section sizes to: {output_file}')
    with open(output_file, 'w') as file:
        json.dump(zarr_sizes, file, indent=4)

    size_per_tile: Dict[int, int] = {sec_num: get_size_per_tile(exp, sec_num, size) for sec_num, size in zarr_sizes.items()}
    output_file = str(Path(exp.root) / 'zarr_sizes_per_tile.json')
    print(f'Storing .zarr section per tile ratios to: {output_file}')
    with open(output_file, 'w') as file:
        json.dump(size_per_tile, file, indent=4)

    # Detect sections with size per tile smaller than limit
    sec_nums_below_limit = []
    for sec_num, size_ratio in size_per_tile.items():
        if size_ratio < min_size_per_tile:
            sec_nums_below_limit.append(sec_num)

    return sec_nums_below_limit


def cross_sec_align_par(
        exp: Inspection,
        nums_to_align: Iterable[int],
        num_processes: int,
        flow_fields: bool = False
):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    # Construct path pairs to stitched sections
    invalid = []
    section_pairs = []
    for num in nums_to_align:
        if num in exp.stitched_nums:
            num_ind = exp.stitched_nums.index(num)
            try:
                nn_num = exp.stitched_nums[num_ind + 1]
                p1 = str(exp.stitched_dicts.get(num))
                p2 = str(exp.stitched_dicts.get(nn_num))
                section_pairs.append([p1, p2])
            except IndexError as _:
                invalid.append(num)

    for inv in invalid:
        logging.warning(f'Section {inv} not cross-aligned (no neighboring section found).')

    if not flow_fields:
        part_func = partial(em_align_reg_sections.compute_shift)
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(cross_sec_wrapper, [(part_func, args) for args in section_pairs])

    if flow_fields:
        path_cfg = exp.fp_est_ff_cfg
        if not Path(path_cfg).exists():
            logging.warning(f'Flow fields estimation failed: cfg file not found.')
            return

        with open(path_cfg) as f:
            config = yaml.safe_load(f)

        ffe_conf = FlowFieldEstimationConfig(**config["ffe_conf"])
        part_func = partial(em_align_est_ff.estimate_flow_fields)
        fields_kwargs_list = [["", ffe_conf, pair] for pair in section_pairs]

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(cross_sec_wrapper, [(part_func, args) for args in fields_kwargs_list])

    return


def cross_sec_wrapper(args):
    part_func, arg_list = args
    return part_func(*arg_list)


def main_est_ff(exp: Inspection, nums_to_align: List[int], num_processes=1):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    path_cfg = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\_processing\SOFIMA\user\ganctoma\runs\Montano_1\fine_align_estimate_flow_fields.config"
    path_cfg = utils.cross_platform_path(path_cfg)

    with open(path_cfg) as f:
        config = yaml.safe_load(f)

    em_align_est_ff.estimate_flow_fields(
        stitched_section_dir=config["stitched_sections_dir"],
        ffe_conf=FlowFieldEstimationConfig(**config["ffe_conf"]),
    )
    return


def main_relax_meshes():
    path_cfg = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\_processing\SOFIMA\user\ganctoma\runs\Montano_1\relax_meshes.config"
    path_cfg = utils.cross_platform_path(path_cfg)

    with open(path_cfg) as f:
        config = yaml.safe_load(f)

    em_relax.relax_meshes(
        stitched_section_dir=config["stitched_sections_dir"],
        output_dir=config["output_dir"],
        mesh_integration=MeshIntegrationConfig(**config["mesh_integration"]),
        flow_stride=config["flow_stride"],
    )
    return


def main_warp_fine(exp: Inspection, nums_to_align: List[int], num_processes=1):
    path_cfg = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\_processing\SOFIMA\user\ganctoma\runs\Montano_1\warp_fine_aligned_sections.config"
    path_cfg = utils.cross_platform_path(path_cfg)

    with open(path_cfg) as f:
        config = yaml.safe_load(f)

    em_warp.warp_fine_aligned_sections(
        stitched_sections_dir=config["stitched_sections_dir"],
        warp_start_section=config["warp_start_section"],
        warp_end_section=config["warp_end_section"],
        output_dir=config["output_dir"],
        volume_name=config["volume_name"],
        block_size=config["block_size"],
        map_zarr_dir=config["map_zarr_dir"],
        flow_stride=config["flow_stride"],
    )
    return


def main_coarse_align_section() -> None:
    sec_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-3\2024_06_11\sections\s6174_g0"
    sec_path = utils.cross_platform_path(sec_path)
    my_sec = Section(sec_path)

    # overlaps_xy = ((200, 250, 350), (200, 250, 350))
    # min_range = ((40, 80, 110), (40, 80, 110))

    # overlaps_xy = ((200, 270, 340), (200, 270, 340))
    # min_range = ((0, 80, 120), (0, 80, 120))

    # Mont1
    overlaps_xy = ((150, 200, 300), (150, 200, 300))
    min_range = ((0, 20, 40), (0, 20, 40))

    # # Mont2
    # overlaps_xy = ((170, 250, 340), (170, 250, 340))
    # min_range = ((0, 20, 40), (0, 20, 40))
    #
    # # Mont3
    # overlaps_xy = ((200, 270, 340), (200, 270, 340))
    # min_range = ((0, 20, 40), (0, 20, 40))

    # Li-3
    overlaps_xy = ((350, 250, 450), (350, 250, 450))
    min_range = ((0, 50, 100), (0, 50, 100))

    kwargs = dict(
        co=None,
        store=True,
        overlaps_xy=overlaps_xy,
        min_range=min_range,
        min_overlap=2,
        filter_size=10,
        max_valid_offset=400,
        masking=False
    )
    my_sec.compute_coarse_offset_section(**kwargs)
    return


def main_par_process_sections() -> None:
    # PROCESS SECTIONS IN PARALLEL (CO, REFINE CO, MASKS)

    start, end = roli_1.first_sec, roli_1.last_sec
    # start, end = 2535, mont_1.last_sec
    args = {'num_jobs': 5, 'num_proc': 42, 'start': start, 'end': end}
    sec_ranges = utils.slurm_sec_ranges(**args)
    print(sec_ranges)

    # exp = Inspection(mont_1)
    # exp = Inspection(mont_2)
    # exp = Inspection(mont_3)
    # exp = Inspection(mont_4)
    exp = Inspection(roli_1)
    # exp = Inspection(roli_2)
    # exp = Inspection(roli_3)
    # exp = Inspection(dp2)

    sec_ranges = [[282, 1962], [1963, 3642], [3643, 5322], [5323, 7002], [7003, 8651]]
    start, end = sec_ranges[0]

    # data = utils.read_dict_from_yaml(exp.root / 'missing_roi_masks.yaml')
    # custom_sec_range = [k for k in data.keys()]
    # numbers = utils.extract_numbers_from_yaml(str(exp.root / 'missing_roi_masks.yaml'))
    # custom_sec_range = numbers
    # print(custom_sec_range)
    # start = 3062
    # end = 3062

    # start, end = exp.first_sec, exp.last_sec
    # sec_ranges = utils.split_list(nums=list(range(start, end + 1)), n=3)
    # custom_sec_nums = sec_ranges[2]
    # custom_sec_nums = None

    # # Section ranges based on file absence in section dirs
    # sec_nums = set(range(exp.first_sec, exp.last_sec + 1))
    # nums_wo_file = sorted(list(exp.select_folders_wo_file(sec_nums, filename='cx_cy.json')))
    # sec_ranges = utils.split_list(nums_wo_file, n=4)
    # custom_sec_nums = sec_ranges[3]
    # start, end = None, None
    # print(nums_wo_file)

    run_coo_par(
        exp=exp,
        start=start,
        end=end,  # Including
        custom_sec_nums=custom_sec_nums,
        overwrite_cxcy=True,
        masking=True,
        generate_masks=True,
        refine=False,
        store=True,
        num_processes=42
    )
    return


def main_verify_zarr_sections():
    exp = Inspection(roli_3)
    exp.init_experiment()
    failed_sec_nums = exp.verify_stitched_files()
    fp = str(exp.root / 'invalid_stitched_nums.yaml')
    utils.write_dict_to_yaml(fp, failed_sec_nums)
    return


def main_verify_cxcy_integrity():
    # VERIFY CXCY ARE READABLE
    exp = Inspection(mont_3)
    exp.init_experiment()
    sections = [Section(p) for p in exp.section_dirs]
    failed_sec_nums = exp.verify_cxcy_files(sections)
    fp = str(exp.root / 'invalid_cxcy.yaml')
    utils.write_dict_to_yaml(fp, failed_sec_nums)
    return


def main_verify_tile_id_maps():
    exp = Inspection(roli_1)
    exp.init_experiment()
    failed_sec_nums = exp.verify_tile_id_maps()
    fp = str(exp.root / 'invalid_tile_id_maps.yaml')
    utils.write_dict_to_yaml(fp, failed_sec_nums)
    return


def main_scan_missing_section_folders():
    # INIT AND SCAN FOR MISSING SECTION FOLDERS
    exp = Inspection(roli_1)
    exp.init_experiment()
    # print(f'len sections: {len(exp.section_nums)}')
    # print(f'first, last {exp.first_sec, exp.last_sec}')
    # print(f'len missing folders: {len(exp.missing_sections)}')
    utils.write_dict_to_yaml(str(exp.root / 'missing_sections.yaml'), exp.missing_sections)


def main_fix_sbem_meta():
    # FIX MISSING SECTIONS ENTRIES DUE TO CORRUPTED SBEM-IMAGE METADATA
    exp = Inspection(roli_1)
    params = {
        'prefix_tile_name': '20230523_RoLi_IV_130558_run2_',
        'grid_shape': exp.grid_shape,
        'thickness_nm': 25.0,
        'acquisition': 'run_0',
        'tile_height': 2304,
        'tile_width': 3072,
        'tile_overlap': 200,
        'resolution_nm': 10.0
    }
    exp.add_missing_sections(**params)
    return


def plot_from_seams_file(exp: Inspection,
                         fn: str,
                         nr_of_bad_performers: int = 50,
                         min_score: Optional[float] = None,
                         adapt_args: Optional[Tuple[int, float]] = (3, 2)
                         ) -> Optional[Set[int]]:

    def plot_(data_x, data_y, data_y_spline, outs_x, outs_y, dst_dir, yaml_fp):

        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(data_x, data_y, 'o', label='Data')
        plt.plot(data_x, data_y_spline, '-', label='Fitted Spline')
        plt.plot(outs_x, outs_y, 'ro', label='Outliers Below Curve')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Spline Fitting and Outlier Detection')
        new_stem = str(Path(yaml_fp).stem) + "_outliers"
        new_suffix = ".png"
        plt_fp = Path(dst_dir) / Path(new_stem).with_suffix(new_suffix)
        plt.savefig(plt_fp)
        return

    if not Path(fn).exists():
        return

    logging.info(f'Reading: {fn}')
    tile_str = str(Path(fn).stem)[:-6]
    dir_out = str(exp.dir_outliers / tile_str)
    utils.create_directory(dir_out)

    data: Dict[int, float] = utils.read_dict_from_yaml(fn)
    nan_keys = []

    if adapt_args:
        s_val, t_val = adapt_args
        nan_keys = [key for key, value in data.items() if np.isnan(value)]
        if nan_keys:
            print(f"Keys with NaN values: {nan_keys}")

        cleaned_data = {key: value for key, value in data.items() if not np.isnan(value)}
        x = np.array(list(cleaned_data.keys()))
        y = np.array(list(cleaned_data.values()))

        if len(np.unique(x)) != len(x):
            raise ValueError("x values must be unique")

        x, y = zip(*sorted(zip(x, y)))
        x, y = np.array(x), np.array(y)

        spline = UnivariateSpline(x, y, s=s_val)
        y_spline = spline(x)
        residuals = y - y_spline

        negative_residuals = residuals[residuals < 0]
        negative_residuals_indices = np.where(residuals < 0)[0]

        z_scores = np.abs(zscore(negative_residuals))
        outlier_indices = negative_residuals_indices[z_scores > t_val]

        outlier_sec_nums = x[outlier_indices]
        outlier_values = y[outlier_indices]
        res = {n: val for n, val in zip(outlier_sec_nums, outlier_values)}

        # Plot results
        plot_(x, y, y_spline, outlier_sec_nums, outlier_values, dir_out, fn)

    else:
        sorted_ov_quality = sorted(list(k for k in data.values()))
        if min_score is not None:
            res = {k: v for k, v in data.items() if v < min_score}
        else:
            res = {k: v for k, v in data.items() if v in sorted_ov_quality[:nr_of_bad_performers]}

    # Add nan seam values
    for key in nan_keys:
        res[key] = np.nan

    # Plotting overlaps
    ta, tb = str(Path(fn).stem).split("_")[:2]
    tid_a, tid_b = int(ta[1:]), int(tb[1:])

    for sec_num, qual in res.items():
        section = Section(exp.section_dicts[sec_num])
        section.feed_section_data()
        is_vert = utils.pair_is_vertical(section.tile_id_map, tid_a, tid_b)
        axis = 1 if is_vert else 0
        vec = section.get_coarse_offset(tid_a, axis)
        section.plot_ov(tid_a, tid_b, vec, dir_out,
                        rotate_vert=True, store_to_root=True)

    logging.info(f'Plotting bad performers of {str(Path(fn).stem)} finished')
    return set(res.keys())


def main_plot_ovs_with_large_dev(config: cfg.ExpConfig):
    exp = Inspection(config)
    exp.init_experiment()
    est_vec = (25, -275)
    tile_id_pairs = [(745, 785), ]
    start, end = 6800, 7050
    sec_nums = list(range(start, end))
    # sec_nums = list(range(exp.first_sec, exp.last_sec))
    ov_dict = {num: tile_id_pairs for num in sec_nums}
    dir_name_out = str(Path(exp.dir_outliers).name)
    shift_abs_dev = 15
    exp.plot_specific_ovs(ov_dict, dir_name_out, est_vec=est_vec, shift_abs_dev=shift_abs_dev)
    pass


def main_par_coarse_align_tilepair(config: cfg.ExpConfig) -> None:
    #  COMPUTE COARSE OFFSETS OF A SPECIFIC TILE-PAIR IN PARALLEL
    exp = Inspection(config)
    exp.init_experiment()

    tid_a = 781
    tid_b = 821
    est_vec = None
    # est_vec = (-40, -90)
    start, end = 0, 6000
    # masking = True
    masking = 0
    refine = 1
    num_proc = 42
    plot = 0
    postprocess = 1
    refine_args = {}

    ov_dir = exp.dir_outliers / f"t{tid_a:04d}_t{tid_b:04d}"
    custom_sec_nums = utils.get_ov_sec_nums(ov_dir)
    custom_sec_nums = None
    # EOF Input

    # Processing
    run_par_coarse_trace(exp, tid_a, tid_b, masking, refine, start, end, custom_sec_nums, est_vec)

    # Plot refined overlaps
    if plot:
        dir_out = str(exp.dir_inf_overlaps)
        run_par_plot_ov(exp, start, end, tid_a, tid_b, num_proc, dir_out, custom_sec_nums)

    # Store intp cxyz.npz and plot traces
    if postprocess:
        main_postprocess_coarse_shifts(config, trace_ids=(tid_a,))
    return


def main_postprocess_coarse_shifts(config: cfg.ExpConfig,
                                   trace_ids: Optional[Iterable[int]] = None,
                                   plot_traces: bool = True):
    exp = Inspection(config)
    if exp.section_nums is None:
        exp.init_experiment()

    postprocess_cxcy(
        exp=exp,
        plot_traces=plot_traces,
        locate_inf=False,
        trace_ids=trace_ids,
        in_parallel=True
    )
    pass


def main_par_plot_ovs_specific_tile_pair():
    # PLOT OVERLAPS OF SPECIFIC TILE-PAIR IN PARALLEL
    exp = Inspection(roli_1)
    exp.init_experiment()
    start, end = roli_1.first_sec, roli_1.last_sec
    start, end = 282, 350
    tid_a, tid_b = 820, 821
    run_par_plot_ov(exp, start, end, tid_a, tid_b, num_proc=10)
    dir_out = exp.dir_overlaps
    # run_par_plot_ov(exp, start, end, tid_a, tid_b, num_proc=10)
    return


def main_fix_outliers_and_infinities(config: cfg.ExpConfig):
    # # FIX OUTLIERS, INFINITIES AND PLOT THEM

    exp = Inspection(config)
    exp.init_experiment()

    dir_overlaps = exp.dir_inspect / 'overlaps_outliers'
    # dir_overlaps = exp.dir_inspect / 'inf_overlaps'
    tid_pairs = utils.get_ov_tid_pairs(dir_overlaps)
    # tid_pairs = [(704, 744), ]
    for tid_pair_to_align in tid_pairs:
        tid_a, tid_b = tid_pair_to_align
        str_a = str(tid_a).zfill(4)
        str_b = str(tid_b).zfill(4)
        ov_dir = dir_overlaps / ('t' + str_a + '_t' + str_b)
        custom_sec_nums = utils.get_ov_sec_nums(ov_dir)
        # custom_sec_nums = {6914}
        # # # custom_sec_nums = None
        # # # custom_sec_nums = list(range(4200, 4600))

        est_vec = (-78, -70)  # vert (+x, -y)
        # est_vec = (-196, -10)
        # est_vec = None
        masking = False
        refine = True
        store = True
        plot = True
        fix_infinities = False
        # custom_sec_nums = None
        custom_mask_params = (500, 500, 700, 700)  # top, bottom, left, right
        # custom_mask_params = (0, 0, 1, 1)  # vertical tile-pair
        # custom_mask_params = (600, 600, 0, 0)  # horizontal tile-pair

        refine_params = dict(levels=1, max_ext=60, stride=6, clahe=True, store=store,
                             plot=False, show_plot=False, est_vec=est_vec,
                             custom_mask_params=custom_mask_params)

        args = dict(masking=masking,
                    store=store,
                    refine=refine,
                    plot=plot,
                    refine_params=refine_params)

        exp.fix_false_offsets_trace(tid_pair=tid_pair_to_align,
                                    custom_sec_nums=custom_sec_nums,
                                    inf=fix_infinities,
                                    align_args=args)
    return


def main_plot_ovs_with_low_score(config: cfg.ExpConfig):
    exp = Inspection(config)
    exp.init_experiment()
    input_dir = exp.dir_overlaps
    # adapt_args = 6, 3
    adapt_args = None
    # min_score = None
    min_score = 0.79
    nr_of_bad_performers = 5
    t1, t2 = 866, 906
    # t1, t2 = None, None

    seam_yamls = utils.list_yaml_files(input_dir)
    custom_sec_nums = None
    for yaml_path in seam_yamls:
        if t1 is None:
            custom_sec_nums = plot_from_seams_file(exp, yaml_path, nr_of_bad_performers, min_score, adapt_args)
        else:
            if str(t1) in yaml_path and str(t2) in yaml_path:
                custom_sec_nums = plot_from_seams_file(exp, yaml_path, nr_of_bad_performers, min_score, adapt_args)
    return


def main_coarse_align_tilepair():
    # # Compute coarse offset for one tile-pair  # #
    exp = Inspection(roli_1)
    exp.init_experiment()
    start, end = 6219, 6700
    nums = list(range(start, end + 1))
    # nums = {4802}
    tid_a, tid_b = 502, 503
    est_vec = (-60, -30)
    # est_vec = None

    for sec_num in sorted(list(nums)):
        args = dict(sec_num=sec_num,
                    tid_a=tid_a,
                    tid_b=tid_b,
                    masking=False,
                    store=True,
                    refine=True,
                    plot=True,
                    clahe=True,
                    est_vec=est_vec)
        align_tile_pair(exp, **args)
    return


def main_plot_ovs_all_tilepairs():
    # PLOT OVERLAPS OF ALL TILE-PAIRS IN PARALLEL
    exp = Inspection(roli_1)
    exp.init_experiment()
    # mont_4_sec_ranges = [[0, 982], [983, 1964], [1965, 2946], [2947, 3928], [3929, 4874]]
    # first, last = mont_4_sec_ranges[4]
    # first, last = mont_3.first_sec, mont_3.last_sec
    first, last = 0, 10000
    exp.plot_all_ovs_par(first, last, num_processes=11)
    return


def main_plot_ovs_outs_and_infs():
    # Plot overlaps from file 'coarse_offset_outliers.txt' or 'all_inf.txt'
    exp = Inspection(roli_1)
    exp.init_experiment()
    fp = exp.dir_inspect / 'coarse_offset_outliers.txt'
    # fp = exp.dir_inspect / 'inf_vals.txt'
    outs = exp.load_outliers(fp)  # all_inf.txt if no arg is provided
    exp.plot_specific_ovs(outs, dir_name_out='overlaps_outliers', refine=False)
    return


def validate_seams_of_stored_ovs():
    # Validate seams of stored overlaps
    exp = Inspection(roli_1)
    # input_dir = exp.dir_inf_overlaps
    input_dir = exp.dir_overlaps
    num_proc = 11
    exp.validate_seams(input_dir, num_proc)  # detect_bad_seam method (line-over-line SSIM)
    return


def main_validate_seams():
    exp = Inspection(roli_1)
    exp.init_experiment()
    # start, end = 900, 1800
    start, end = None, None
    tid_a, tid_b = 866, 906
    # custom_tid_pairs = [(tid_a, tid_b),]
    # custom_tid_pairs = None
    # eval_ov_quality_exp(exp, custom_tid_pairs, num_proc=40)  # all tile-pairs, all sections using eval_ov
    run_par_eval_ov(exp, start, end, tid_a, tid_b, num_proc=42)  # one-tile pair, eval_ov method (SSIM over area)
    # # run_validate_seams_on_fly(exp, start, end, tid_a, tid_b)  #   no parallelization, uses detect_bad_seam  method
    # # run_validate_seams_on_fly_new(exp, start, end, tid_a, tid_b) # no parallelization, uses eval_ov (SSIM over common area method)
    return


def create_multiscale_volume():
    # CREATE MULTISCALE ZARR VOLUME
    path_one_level_zarr = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_2\20230603\20220426_RM0008_130hpf_fP1_f3_volume_full_fine_aligned_rewarp"
    path_one_level_zarr = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240416\fine-stacks\mont3_fs_s0450_s5486_2024-07-25"
    path_one_level_zarr = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240416\fine-stacks\mont3_fs_s0450_s5486_2024-08-04"
    utils.create_multiscale(path_one_level_zarr, max_layer=5, num_processes=12)
    exp = Inspection(mont_3)
    exp.create_multiscale(path_one_level_zarr, max_layer=4, num_processes=12)
    return


def main_cross_align_sections(config: cfg.ExpConfig):
    # CROSS-ALIGN SECTION PAIRS

    exp = Inspection(config)
    exp.init_experiment()
    exp.get_cross_aligned_nums()

    not_cross_aligned_nums = [num for num in exp.stitched_nums if num not in exp.cross_aligned_nums]
    # print(not_cross_aligned_nums)
    # nums_to_align = [sec_num for sec_num in range(mont_3.first_sec, mont_3.last_sec)]

    # Roli-3
    sec_ranges = [[1, 800], [801, 1600], [1601, 2400], [2401, 3200], [3201, 4000],
                  [4001, 4800], [4801, 5600], [5601, 6400], [6401, 7245]]

    # Roli-1
    sec_ranges = [[282, 1000], [4801, 5600], [5601, 6400], [6401, 7245]]

    # start, end = sec_ranges[3]
    start, end = 282, 1000
    nums_to_align = list(range(start, end+1))
    cross_sec_align_par(exp, nums_to_align, num_processes=2, flow_fields=False)

    # # Compute and store paddings after cross-section alignment
    # exp.compute_paddings()

    return


def check_cross_section_shifts(config: cfg.ExpConfig):
    # #  Check cross-section shifts from logs
    exp = Inspection(config)
    root_dir = exp.dir_inspect / 'logs'
    log_files = glob.glob(f"{str(root_dir)}/*.out")
    shifts_dict, errs = utils.parse_logs(log_files)
    fp_out = str(exp.dir_inspect / 'cross-section_drifts_sorted.yaml')
    utils.sort_and_write_to_yaml(shifts_dict, errs, fp_out)
    return


def main_warp_cross_aligned_stack(config: cfg.ExpConfig):
    # WARP CROSS-ALIGNED STACK
    exp = Inspection(config)
    exp.init_experiment()
    bin_fct: int = 4
    start, end = 3970, 3990

    nums_to_align = list(range(start, end + 1))

    args = dict(stitched_section_dir=str(exp.dir_stitched),
                output_dir=str(exp.dir_coarse_stacks),
                volume_name=f'cstack_{start}-{end}',
                start_section=start,
                end_section=end,
                bin=bin_fct)

    em_align_warp_coarse_stack.main(**args)
    return


def main_estimate_flow_fields():
    # ESTIMATE FLOW-FIELDS
    exp = Inspection(mont_1)
    exp.init_experiment()
    exp.get_missing_sections()

    # Select sections with old or missing .npy files
    old_nums: Set[int] = exp.mod_time_select(set(range(exp.first_sec, exp.last_sec)),
                                        204, "not-used")  # VERIFY INPUTS

    if exp.first_sec in old_nums:
        old_nums.remove(exp.first_sec)

    sec_ranges = utils.split_list(list(old_nums), n=6)
    # EOF custom selection

    sec_ranges = [[402, 1267], [1268, 2132], [2133, 2997], [2998, 3862], [3863, 4724]]  # Mont1
    start, end = sec_ranges[4]
    nums_to_align = list(range(start, end + 1))
    nums_to_align = list(range(3000, 3010))
    print(len(nums_to_align))
    main_est_ff(exp, nums_to_align=nums_to_align,  num_processes=1)
    cross_sec_align_par(exp, nums_to_align, num_processes=5, flow_fields=True)
    nums_to_align = sec_ranges[0]
    compute_section_flow_fields_threading(exp, [-1,], nums_to_align)
    compute_flow_fields_par(exp, nums_to_align, num_processes=10)
    return


def main_fix_zarr_structure():
    # FIX warped sections and move them to different folder
    # Removes first dimension from zarr section
    exp = Inspection(mont_1)
    exp.init_experiment()

    def fix_stitched(stitched_dirs: List[Path], dst_dir: str) -> None:
        if len(stitched_dirs) == 0:
            logging.warning(f'No zarr sections to fix were found at {stitched_dirs}')
            return

        for sdir in stitched_dirs:
            dst_path = str(Path(dst_dir) / Path(sdir).name)
            utils.fix_zarr(str(sdir), dst_path)
        return

    destination_dir = str(exp.root / "stitched-sections-fixed")
    if not Path(destination_dir).exists():
        logging.warning(f'Destination folder does not exist.')

    fix_stitched(exp.stitched_dirs, destination_dir)
    return


def main_repair_inf_values():
    # REPAIR ALL INF VALUES IN ALL SECTIONS
    exp = Inspection(roli_1)
    exp.init_experiment()
    mont_4_sec_ranges = [[0, 982], [983, 1964], [1965, 2946], [2947, 3928], [3929, 4874]]
    start, end = mont_4.first_sec, mont_4.last_sec
    start, end = mont_4_sec_ranges[4]
    start, end = 1011, 1011
    exp.repair_inf_offsets(start=start,
                           end=end,
                           custom_sec_nums=None,
                           masking=True,
                           store=True,
                           num_processes=8
                           )
    return


def plot_ovs_from_out_or_inf_file(config: cfg.ExpConfig):
    # Plot overlaps from file 'coarse_offset_outliers.txt' or 'all_inf.txt'
    exp = Inspection(config)
    exp.init_experiment()
    outs = exp.load_outliers()
    exp.plot_specific_ovs(outs)
    return


def main_par_eval_seams(config: cfg.ExpConfig):
    # Compute overlap quality in parallel
    exp = Inspection(config)
    exp.init_experiment()
    nums = list(range(2886, 2890))

    kwargs = dict(roi_masking=True,
                  smr_masking=False,
                  exp=exp,
                  num_processes=12,
                  nums_to_align=nums
                  )
    result = eval_section_ovs_par(**kwargs)

    # Export results:
    res_dict = {}
    for res in result:
        for k, v in res.items():
            res_dict[str(k)] = v

    fp_out = exp.dir_overlaps / 'overlap_quality_smr.npz'
    np.savez(fp_out, **res_dict)

    plot_traces_eval_ov(exp)
    return


def main_par_multiproc(config: cfg.ExpConfig):

    exp = Inspection(config)
    exp.init_experiment()

    # fp_tbs = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\to_be_stitched_nums.json"
    # fp_tbs = utils.cross_platform_path(fp_tbs)
    # to_be_stitched: List[int] = sorted(list(utils.load_set_from_json(fp_tbs)))
    #
    # fp_st = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\new_stitched.json"
    # fp_st = utils.cross_platform_path(fp_st)
    # stitched_list: List[int] = sorted(list(utils.load_set_from_json(fp_st)))
    #
    # old_nums = [n for n in to_be_stitched if n not in stitched_list]
    # sec_ranges = utils.split_list(list(old_nums), n=10)
    # sec_ranges = utils.slurm_sec_ranges(num_jobs=10,
    #                                     num_proc=42,
    #                                     start=exp.first_sec,
    #                                     end=exp.last_sec
    #                                     )
    # print(sec_ranges)

    # print(old_nums)
    # # nums_to_align = sec_ranges[1]

    # start, end = sec_ranges[6]
    # start, end = 6201, 6400
    # nums_to_align = list(range(start, end + 1))
    # nums_to_align = [5281, 5285, 5286, 5287, 5289, 5290]
    # nums_to_align = old_nums

    # # # Section ranges based on file non-presence in section dirs
    # sec_nums = set(range(exp.first_sec, exp.last_sec))
    # nums_wo_file: List[int] = list(exp.select_folders_wo_file(sec_nums, filename='cx_cy.json'))
    # sec_ranges = utils.split_list(nums_wo_file, n=6)
    # nums_to_align = sec_ranges[0]

    # Nums from .json file
    # fp = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\old_margin_masks.json"
    # fp = utils.cross_platform_path(fp)
    # old_nums: Set[int] = utils.load_set_from_json(fp)
    # old_nums: List[int] = sorted(list(old_nums))
    # sec_ranges = utils.split_list(old_nums, n=6)
    # nums_to_align = sec_ranges[5]

    # Nums based on not stitched sections
    # not_stitched = set(exp.section_nums) - set(exp.stitched_nums)
    # print(sorted(list(not_stitched)))
    # sec_ranges = utils.split_list(sorted(list(not_stitched)), n=7)
    # print(sec_ranges)
    # nums_to_align = sec_ranges[6]
    # start, end = None, None
    # print(sec_ranges)

    start = 282
    end = None
    # nums_to_align = list(range(2485, 2490))
    nums_to_align = None
    nums_to_align = [1033, 1434, 1638, 1788,
                     1934, 2382, 2683, 2693,
                     2740, 3089, 3232, 3233,
                     3234, 3235, 3243, 3437,
                     4582, 5185, 5333, 5940,
                     6185, 6438, 7335, 7984,
                     8091, 8155, 8335, 8509]

    fp = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-1\2024_05_17\sections\missing_meshes.txt"
    nums_to_align = nums_from_missing_meshes(fp)
    print(len(nums_to_align))

    # sec_ranges = [[282, 1500], [1501, 2500], [2501, 3200],
    #               [3201, 3900], [3901, 4601], [4601, 5300],
    #               [5301, 6000], [6001, 6701], [6701, 7400],
    #               [7401, 8000], [8001, 8651]]

    fine_align_sections_multiproc(exp, start=start, end=end,
                                  nums_to_align=nums_to_align[25:],
                                  masking=True, num_processes=26)

    # fine_align_sections_threading(exp, nums_to_align=nums_to_align, masking=True)
    return


def nums_from_missing_meshes(path_missing_meshes: UniPath) -> List[int]:
    fp = utils.cross_platform_path(path_missing_meshes)
    nums_to_align = []
    if not Path(fp).exists():
        print(f'Missing meshes file not loaded at following path: {fp}')
        return nums_to_align

    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                sec_num = int(line.split("_g")[0][3:])
                nums_to_align.append(sec_num)
            except ValueError as _:
                print(f'error at line: {line}')
                continue

    return nums_to_align

def main_nums_from_missing_meshes():
    path_missing_meshes = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-1\2024_05_17\sections\missing_meshes.txt"
    nums = nums_from_missing_meshes(path_missing_meshes)
    print(nums)
    print(len(nums))
    return



if __name__ == "__main__":

    # Accessing individual alignment experiments
    configs = cfg.get_experiment_configurations()
    mont_1 = configs[cfg.ExperimentName.MONT_1]
    mont_2 = configs[cfg.ExperimentName.MONT_2]
    mont_3 = configs[cfg.ExperimentName.MONT_3]
    mont_4 = configs[cfg.ExperimentName.MONT_4]
    roli_1 = configs[cfg.ExperimentName.ROLI_1]
    roli_2 = configs[cfg.ExperimentName.ROLI_2]
    roli_3 = configs[cfg.ExperimentName.ROLI_3]
    dp2 = configs[cfg.ExperimentName.DP2]


    # # PRE- and POST-PROCESS ROUTINES
    # main_postprocess_coarse_shifts(config=roli_1, plot_traces=True, trace_ids=None)
    # main_scan_missing_section_folders()
    # main_fix_sbem_meta()
    # main_verify_zarr_sections()
    # main_verify_cxcy_integrity()
    # main_verify_tile_id_maps()


    # # # COARSE ALIGNMENT
    # main_coarse_align_section()
    # main_par_process_sections()
    # main_fix_outliers_and_infinities(config=roli_1)
    # main_coarse_align_tilepair()
    # main_par_coarse_align_tilepair(config=roli_1)
    # main_par_compute_coarse_offsets_tile_pair(config=roli_1)
    # main_repair_inf_values()



    # # PLOT OVERLAPS
    # main_plot_ovs_with_large_dev(config=roli_1)
    # main_par_plot_ovs_specific_tile_pair()
    # main_plot_ovs_all_tilepairs()
    # main_plot_ovs_outs_and_infs()
    # main_plot_ovs_with_low_score(roli_1)
    # plot_ovs_from_out_or_inf_file(roli_1)



    # SEAM VALIDATION
    # main_validate_seams()
    # validate_seams_of_stored_ovs()
    # main_par_eval_seams(roli_1)


    # # MULTIPROCESSING, RENDERING & FINE ALIGNMENT
    main_par_multiproc(roli_1)


    # FINE-ALIGNMENT
    # main_warp_fine(exp, nums_to_align=nums_to_align, num_processes=1)
    # main_estimate_flow_fields()
    # main_relax_meshes()


    # CROSS-SECTION ALIGNMENT
    # main_cross_align_sections(roli_1)
    # check_cross_section_shifts(roli_1)
    # main_warp_cross_aligned_stack(roli_1)


    # OTHERS
    # create_multiscale_volume()
    # failed_nums = exp.verify_stitched_files()
    # failed_sec_nums = main_get_zarr_sizes(Inspection(mont_3), min_size_per_tile=1000)
    # utils.rename_folders(str(exp.dir_stitched), start, end)
    # main_fix_zarr_structure()
    # main_nums_from_missing_meshes()





