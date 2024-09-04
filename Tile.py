import glob
import multiprocessing

import cv2
import logging
import numpy as np
from pathlib import Path

import scipy.ndimage
import skimage
from scipy import ndimage
from typing import Tuple, List, Set, Union, Optional, Dict, Mapping, Any
from matplotlib import pyplot as plt
import os

import inspection_utils as utils
from inspection_utils import Num
import mask_utils as mutils
# from SOFIMA.eval_smearing import detect_smearing

# Set up logging
DEBUG = False
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.WARNING)

TileXY = Tuple[int, int]

class Tile:
    def __init__(self, path: str):
        self.tile_path = Path(utils.cross_platform_path(path))
        self.tile_id: Optional[int] = utils.get_tile_num(self.tile_path)
        self.tile_xy: Optional[TileXY] = None
        self.img_data: Optional[np.ndarray] = None
        self.roi_mask: Optional[np.ndarray[bool]] = None
        self.smr_mask: Optional[np.ndarray[bool]] = None
        self.tile_mask: Optional[np.ndarray[bool]] = None
        self._ssim: Optional[float] = None
        self._shift_to_prev_x: Optional[float] = None
        self._shift_to_prev_y: Optional[float] = None
        self._mean_brightness: Optional[float] = None
        self._std: Optional[float] = None
        self._clip_under: Optional[float] = None
        self._clip_over: Optional[float] = None
        self._hist: Optional[np.ndarray] = None
        self._sharpness: Optional[float] = None
        self._smearing: Optional[int] = None

    def __str__(self):
        return (
            f'Tile ID: {self.tile_id}\n'
            f'Sharpness: {self._sharpness}\n'
            f'Mean brightness: {self._mean_brightness}\n'
            f'Standard deviation: {self._std}\n'
            f'Clipping: {self._clip_under, self._clip_over}\n'
        )

    def validate_smr_mask(self):
        """Check whether smr mask meets quality criteria"""
        pass
        return


    def create_smearing_mask(self,
                             max_vert_ext: int = 200,
                             mask_top_edge: int = 20,
                             edge_only=False
                             ) -> Optional[np.ndarray[bool]]:
        """
        Computes smearing distortion mask for one tile-image.

        Args:
            max_vert_ext: Number of top lines for which to compute smearing mask. Defaults to 200.
            mask_top_edge: Number of top lines to be masked entirely. Defaults to 20.
                           Mask will not be computed if n_lines == 0.
            edge_only: If True, do not compute smearing mask, only mask top edge
        Returns:
            Boolean mask indicating regions affected by smearing.

        """

        # Optionally, plot results TODO refactor
        path_plot = None
        # path_plot = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\scripts\pythonProject\inspection\tmp\masks.png"
        # path_plot = utils.cross_platform_path(path_plot)

        if max_vert_ext == 0:
            self.smr_mask = None
            return

        # Load image data if not already loaded
        if self.img_data is None:
            self.load_image()

        h, w = self.img_data.shape
        mask_tile = np.full(shape=(max_vert_ext, w), fill_value=False, dtype=bool)

        # Compute mask only for the top N lines and optionally skip smr mask computation
        mask_tile[:min(mask_top_edge, max_vert_ext)] = True
        if edge_only:
            self.smr_mask = mask_tile
            return mask_tile

        # Compute smearing mask
        smr_input = self.img_data[:np.shape(mask_tile)[0]]
        print(f'smr input shape: {np.shape(smr_input)}')
        print(f'mask tile shape: {np.shape(mask_tile)}')
        smr_mask = mutils.get_smearing_mask(smr_input,mask_top_edge,path_plot, plot=False)

        mask_tile |= smr_mask
        self.smr_mask = mask_tile
        return mask_tile


    def create_roi_mask(
            self,
            thresh: int = 20,
            filter_size: int = 10,
            range_limit: int = 20,
    ) -> Optional[np.ndarray[bool]]:

        if self.img_data is None:
            self.load_image()

        def detect_resin(a: np.ndarray) -> np.ndarray[bool]:
            # Identify silver particles
            _, binary_image = cv2.threshold(a, thresh, 255, cv2.THRESH_BINARY)
            binary_image = cv2.bitwise_not(binary_image)

            # Blur particles and perform thresholding again
            # binary_image = cv2.GaussianBlur(binary_image, (755, 755), 0)
            binary_image = ndimage.gaussian_filter(binary_image, sigma=125)
            _, binary_image = cv2.threshold(binary_image, thresh + 1, 255, cv2.THRESH_BINARY)

            # Fill holes
            _mask = utils.fill_holes(binary_image)
            return _mask

        def mask_low_dynamic_range(a: np.ndarray):
            # Masks areas with insufficient dynamic range.
            a_mask = (ndimage.maximum_filter(a, filter_size) -
                      ndimage.minimum_filter(a, filter_size)
                      ) < range_limit
            a_mask = utils.fill_holes(a_mask)
            return a_mask

        # Compute mask
        mask = detect_resin(self.img_data)
        # mask_resin = mutils.detect_resin(self.img_data, thresh)
        if range_limit != 0:
            mask_void = mask_low_dynamic_range(self.img_data)
            mask = mask | mask_void

        return None if not mask.any() else mask


    def merge_masks(self) -> None:
        """ Merges the smearing mask and ROI mask into a single mask."""

        # Load image data if not already loaded
        if self.img_data is None:
            self.load_image()

        tile_mask = np.full_like(self.img_data, fill_value=False)

        if self.smr_mask is not None:
            tile_mask |= self.smr_mask

        if self.roi_mask is not None:
            tile_mask |= self.roi_mask

        self.tile_mask = tile_mask
        return


    def load_image(self, clahe: bool = False) -> None:
        self.img_data = None
        try:
            self.img_data = skimage.io.imread(str(self.tile_path))
        except (FileNotFoundError, OSError):
            print(f"Error loading image from file: {self.tile_path}")

        if clahe and self.img_data is not None:
            self.img_data = utils.apply_clahe(self.img_data)
        return

    def _compute_ssim(self, ref_img: np.ndarray) -> Optional[float]:
        ssim_val = None
        if self.img_data is not None and self.img_data.shape == ref_img.shape:
            ssim_val = utils.compute_ssim(self.img_data, ref_img)
            self._ssim = ssim_val
        return ssim_val

    def _compute_sharpness(self):
        if self.img_data is not None:
            self._sharpness = utils.compute_sharpness(self.img_data)
        else:
            print('Sharpness not computed. Image data not initialized.')

    def compute_mean_brightness(self):
        if self.img_data is not None:
            mb = np.round(np.mean(self.img_data), 2)
            self._mean_brightness = mb

    def compute_tile_std(self):
        if self.img_data is not None:
            con = np.round(np.std(self.img_data), 2)
            self._std = con


    def compute_clipping(self) -> None:
        if self.img_data is not None:
            histogram = np.histogram(self.img_data, bins=256, range=(0, 256))[0]
            self._hist = histogram

            first_non_zero_bin = np.argmax(histogram > 0)
            last_non_zero_bin = 255 - np.argmax(histogram[::-1] > 0)

            first_non_zero_bin = max(1, first_non_zero_bin)
            last_non_zero_bin = min(254, last_non_zero_bin)

            hist_values = histogram.astype(int)
            underexp = np.sum(hist_values[:first_non_zero_bin])
            overexp = np.sum(hist_values[last_non_zero_bin + 1:])
            img_size = np.prod(self.img_data.shape)
            c_u, c_o = (round(val / img_size, 4) for val in (underexp, overexp))
            self._clip_under = c_u
            self._clip_over = c_o

    def compute_tile_stats(self):
        if self.img_data is not None:
            self.compute_mean_brightness()
            self.compute_tile_std()
            self.compute_clipping()
            self._compute_sharpness()
        else:
            print('Load tile image data first!')


    def get_tile_stats(self) -> Dict[str, Optional[Union[float, tuple]]]:
        tiles_entries = {
            'ssim': None,
            'shift_to_prev_tile_x': None,
            'shift_to_prev_tile_y': None,
            'mean_brightness': None,
            'contrast': None,
            'clip_under': None,
            'clip_over': None,
            'sharpness': None,
            'smearing': None,
        }

        vals = [
            self._ssim,
            self._shift_to_prev_x,
            self._shift_to_prev_y,
            self._mean_brightness,
            self._std,
            self._clip_under,
            self._clip_over,
            self._sharpness,
            self._smearing
        ]

        def fix_types(values: List[Optional[Num]]) -> List[Any[float, Tuple[float]]]:
            for i, v in enumerate(values):
                if isinstance(v, np.float64):
                    values[i] = float(v)
                elif isinstance(v, tuple):
                    new = []
                    for t_val in v:
                        if isinstance(t_val, np.float64):
                            t_val = float(t_val)
                        new.append(t_val)
                    values[i] = tuple(new)
            return values

        vals = fix_types(vals)
        tiles_entries = {k: v for k, v in zip(tiles_entries, vals)}
        return tiles_entries


    def plot_histogram(self):
        plt.figure()
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(self._hist)
        plt.xlim([0, 256])  # Set the x-axis limits to cover the entire range of pixel values
        plt.show()


def process_tile(tile_path: str) -> dict:
    # Used for writing results in Section parallel processing
    tile = Tile(tile_path)
    tile.load_image()
    tile.compute_tile_stats()
    res_dict = {tile.tile_id: tile.get_tile_stats()}
    return res_dict


def tst_get_smr_mask():
    # tile_path = r"/tungstenfs/scratch/gmicro_sem/gfriedri/montruth/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2/tiles/g0000/t0705/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2_ROI_02_g0000_t0705_s05146.tif"
    tile_path = r"/tungstenfs/scratch/gmicro_sem/gfriedri/montruth/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2/tiles/g0000/t0860/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2_ROI_02_g0000_t0860_s00974.tif"
    tile_path = r"/tungstenfs/scratch/gmicro_sem/gfriedri/montruth/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2/tiles/g0000/t0980/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2_ROI_02_g0000_t0980_s00980.tif"
    tile_path = utils.cross_platform_path(tile_path)
    my_tile = Tile(tile_path)
    _ = my_tile.create_smearing_mask(max_vert_ext=2000, edge_only=True, mask_top_edge=20)
    return




def main():
    tile_path = r"/tungstenfs/landing/gmicro_sem/gemini_data/20220429_scr_Ruth_Run0001/tiles/g0000/t0237/20220429_scr_Ruth_Run0001_g0000_t0237_s02119.tif"
    my_tile = Tile(tile_path)
    my_tile.load_image()
    my_tile.compute_tile_stats()
    print(my_tile)
    # my_tile.plot_histogram()


def tst_detect_bad_seam():

    root = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240217\_inspect\overlaps_25022024\t0700_t0701_orig"
    sec_num: int = 3574
    tile_id_a: int = 700
    tile_id_b: int = 701

    root = utils.cross_platform_path(root)
    ov_img_name = f's{sec_num}_t{str(tile_id_a).zfill(4)}_t{str(tile_id_b).zfill(4)}_ov.jpg'
    ov_img_fp = Path(root) / ov_img_name
    ov_img = skimage.io.imread(ov_img_fp)
    return


def main_get_roi_mask(roi_thresh):
    def apply_mask(image, mask):
        modified = image.copy()
        modified = modified.astype(np.float32)
        modified[mask] = np.nan
        return modified

    def norm_img(data) -> np.ndarray:
        # Change to 8-bit depth
        norm_gray = (data - np.min(data)) / (np.max(data) - np.min(data))
        return (norm_gray * 255).astype(np.uint8)

    def replace_nan_with_max(arr):
        # Get the maximum value of the data type of the array
        max_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else np.finfo(arr.dtype).max

        # Replace nan values with the maximum value
        arr[np.isnan(arr)] = max_val
        return arr


    tile_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-1\2024_05_17\_inspect\masks\fs_range\20230523_RoLi_IV_130558_run2_g0001_t0701_s08000.tif"
    dir_out = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\roli-1\2024_05_17\_inspect\masks"

    tile_path = utils.cross_platform_path(tile_path)
    dir_out = utils.cross_platform_path(dir_out)
    my_tile = Tile(tile_path)
    my_tile.load_image()

    for range_limit in (0, 25, 50, 75):
        for fs in (50, ):
            print(f'Filter size, range limit: {fs, range_limit} ')
            # plt.imshow(my_tile.img_data, cmap='gray')
            # plt.show()

            msk = my_tile.create_roi_mask(roi_thresh, fs, range_limit)
            # msk = my_tile.create_roi_mask(roi_thresh)
            # plt.imshow(msk*255, cmap='gray')
            # plt.show()

            masked_img = apply_mask(my_tile.img_data, msk)
            # plt.imshow(masked_img, cmap='gray')
            # plt.show()

            # masked_img = replace_nan_with_max(masked_img)
            # masked_img = norm_img(masked_img)

            path_out = str(Path(dir_out) / f'fs_{fs}_rng_{range_limit}.png')
            plt.imshow(masked_img, cmap='gray', vmin=0, vmax=255)
            # # plt.show()
            plt.imsave(path_out, masked_img)
            plt.close()
    return


def tst_get_smr_mask_debug():
    # tile_path = r"/tungstenfs/scratch/gmicro_sem/gfriedri/montruth/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2/tiles/g0000/t0980/20230315_Ruth_20220511_RM0008_126hpf_fP10_f2_ROI_02_g0000_t0980_s00980.tif"
    # tile_path = r"/tungstenfs/scratch/gmicro_sem/gfriedri/_EM_acquisitions/20230523_RoLi_IV_130558_run2/tiles/g0001/t0906/20230523_RoLi_IV_130558_run2_g0001_t0906_s01522.tif"
    tile_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_2\masks_analysis\roi_masks\s03030\20220831_Ruth_20220426_RM0008_130hpf_fP1_f3_run001_g0001_t0643_s03030.tif"
    tile_path = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_2\masks_analysis\roi_masks\s03030\20220831_Ruth_20220426_RM0008_130hpf_fP1_f3_run001_g0001_t0524_s03030.tif"

    tile_path = utils.cross_platform_path(tile_path)
    my_tile = Tile(tile_path)
    _ = my_tile.create_smearing_mask(max_vert_ext=0, edge_only=False, mask_top_edge=30)

    plt.imshow(my_tile.smr_mask, cmap='gray')
    plt.show()
    return


def compute_roi_masks(dir_path: str, thresh: int) -> None:
    """Compute ROI masks for all .tif files in specified folder"""
    
    def prep_files() -> Optional[Tuple[List[str], str]]:
        tile_paths = list_tiff_files(dir_path)
        if len(tile_paths) == 0:
            print('No files to process were found. Check input directory.')
            return

        dir_out = Path(tile_paths[0]).parent / 'masks'
        if not dir_out.exists():
            dir_out.mkdir(parents=True)
        return tile_paths, str(dir_out)
    
    def list_tiff_files(directory: str) -> List[str]:
        directory_path = Path(directory)
        tiff_files = directory_path.glob("*.tif")
        tiff_file_paths = [str(file_path) for file_path in tiff_files]
        return tiff_file_paths

    def downscale_image(img: np.ndarray, sf: float) -> np.ndarray:
        d_img = skimage.transform.resize(img, output_shape=(
            int(img.shape[0] * sf), int(img.shape[1] * sf)),
                                         anti_aliasing=False, preserve_range=True)
        return d_img

    def save_masks_as_png(bin_masks: List[np.ndarray], file_names: List[str], dir_out: str,
                          scale_factor: float = 0.5) -> None:
        for i, mask in enumerate(bin_masks):
            file_path = Path(file_names[i])
            file_name = file_path.stem + "_mask.jpg"
            output_path = Path(dir_out) / file_name

            try:
                img_orig = skimage.io.imread(str(file_path), as_gray=True)
                img_scaled = downscale_image(img_orig, scale_factor)
                bin_mask = downscale_image(mask, scale_factor)
                bin_mask = np.where(bin_mask, 255, 0).astype(np.uint8)
                img_stack = np.hstack((img_scaled, bin_mask))
                plt.imsave(output_path, img_stack, cmap='gray')
            except (FileNotFoundError, AttributeError) as _:
                pass
        return

    res = prep_files()
    if res is None:
        return

    paths, out_dir = res
    masks = [Tile(p).create_roi_mask(thresh) for p in paths]
    save_masks_as_png(masks, paths, out_dir)
    return

def roi_mask_analysis(parent_dir: str) -> None:
    # Compute masks for all tif files within specified folder and store them into masks folder
    def list_folders_in(directory: str) -> List[str]:
        directory = utils.cross_platform_path(directory)
        directory_path = Path(directory)
        folders = [item.name for item in directory_path.iterdir() if item.is_dir()]
        return folders

    dirs = list_folders_in(parent_dir)
    if not dirs:
        print(f'No folders in {parent_dir}')
        return

    # dirs = ['s01012',]
    for d in dirs:
        dir_path = utils.cross_platform_path(str(Path(parent_dir) / d))
        print(f'Processing dir: {dir_path}')
        compute_roi_masks(dir_path, thresh=20)
    return

def roi_mask_analysis_wrapper(dir_path: str) -> None:
    compute_roi_masks(dir_path, thresh=20)
    return


def mask_analysis(parent_dir: str):
    def list_folders_in(directory: str) -> List[str]:
        directory = utils.cross_platform_path(directory)
        directory_path = Path(directory)
        folders = [item.name for item in directory_path.iterdir() if item.is_dir()]
        return folders

    parent_dir = utils.cross_platform_path(parent_dir)
    dirs = list_folders_in(parent_dir)
    dir_paths = [Path(parent_dir) / d for d in dirs]
    for d in dir_paths:
        print(d)

    with multiprocessing.Pool(processes=len(dir_paths)) as pool:
        pool.map(roi_mask_analysis_wrapper, dir_paths)
    return


if __name__ == "__main__":
    # main()
    # tst_get_smr_mask()
    # tst_get_smr_mask_debug()
    # tst_detect_bad_seam()
    main_get_roi_mask(roi_thresh=20)


    # Compute masks for all .tif files within specified subdirectories in input dir.
    # Store them into masks folder in every subdirectory.
    # par_dir = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_1\2024_04_16\masks_analysis"
    # mask_analysis(par_dir)

    # tile_path = r"/tungstenfs/landing/gmicro_sem/gemini_data/20220429_scr_Ruth_Run0001/tiles/g0000/t0240/20220429_scr_Ruth_Run0001_g0000_t0240_s04402.tif"
    # tile_path = utils.cross_platform_path(tile_path)
    # tile = Tile(tile_path)
    # tile.load_image()
    # plt.imshow(tile.img_data, cmap='gray')
    # plt.show()


