import concurrent
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, OrderedDict
from zipfile import BadZipFile

import cv2
import csv
import glob
import json
import logging
import os
import shutil
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from numcodecs import Blosc
import numpy as np
from numpy import ndarray
import platform
from pathlib import Path
import re

from numpy._typing import ArrayLike
from ruyaml import YAML
import skimage
from scipy import ndimage
from scipy.ndimage import convolve
from scipy.interpolate import CloughTocher2DInterpolator
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import subprocess
from tqdm import tqdm
from typing import Tuple, List, Set, Union, Optional, Dict, Iterable, Any, Sequence
import yaml

from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
import zarr

import pyramid_levels

Num = Union[int, float]
TileXY = tuple[int, int]
Vector = Union[tuple[int, int], tuple[int, int, int]]  # [z]yx order
UniPath = Union[str, Path]
TileCoord = Union[Tuple[int, int, int, int], Tuple[int, int]]  # (c, z, y, x)
GridXY = Tuple[Any, Any, Any]
MaskMap = Dict[TileXY, Optional[np.ndarray]]

# # #  Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.WARNING)


def fix_zarr(path_orig: str, path_new: str):
    zarr_volume = zarr.open(path_orig, mode='r')
    img_data = zarr_volume['0']

    if img_data.ndim == 2:
        logging.info(f'{Path(path_orig).name} already fixed. Moving to destination folder.')
        shutil.move(path_orig, path_new)
        return

    sec_name = str(Path(path_new).name)
    sec_dir = str(Path(path_new).parent)
    logging.info(f'Storing section {sec_name}')
    store_section_zarr(img_data[0, ...], sec_name, Path(sec_dir))
    return

def store_section_zarr(
        img_data: np.ndarray,
        section_name: str,
        out_dir: UniPath
) -> None:

    zarr_path = str(Path(out_dir) / section_name)
    store = parse_url(zarr_path, mode="w").store

    write_image(
        image=img_data,
        group=zarr.group(store=store),
        scaler=Scaler(max_layer=0),
        axes="yx",
        storage_options=dict(
            chunks=(2744, 2744),
            compressor=Blosc(
                cname="zstd",
                clevel=3,
                shuffle=Blosc.SHUFFLE,
            ),
            overwrite=True,
            write_empty_chunks=False,
        ),
    )

    return


def create_zarr(
        output_dir: str,
        volume_name: str,
        yx_size: tuple[int, int],
) -> str:
    target_dir = Path(output_dir) / volume_name
    store = parse_url(str(target_dir), mode="w").store
    zarr_root = zarr.group(store=store)

    chunks = (1, 2744, 2744)
    write_image(
        image=np.zeros(
            (1, yx_size[0], yx_size[1]),
            dtype=np.uint8,
        ),
        group=zarr_root,
        axes="zyx",
        scaler=Scaler(max_layer=0),
        storage_options=dict(
            chunks=chunks,
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE),
            overwrite=True,
            write_empty_chunks=True,
        ),
    )

    return str(target_dir)

def scan_dirs(dir_path) -> List[Path]:
    p = Path(dir_path)
    if not p.is_dir():
        print(f"{dir_path} is not a valid directory.")
        return []

    # List only top-level directories
    dirs = [Path(d) for d in p.glob('*') if d.is_dir()]
    return dirs


def find_files_with_substring(dir_path, substring):
    p = Path(dir_path)
    if not p.is_dir():
        print(f"{dir_path} is not a valid directory.")
        return []

    matching_files = [str(file) for file in p.rglob('*') if file.is_file() and substring in file.name]
    return matching_files


def create_multiscale(zarr_path: Union[str, Path], max_layer=5, num_processes=42) -> None:

    zarr_path = cross_platform_path(str(zarr_path))

    if not Path(zarr_path).exists():
        logging.warning(f'Creating multiscale .zarr failed (zarr path does not exist).')
        return

    logging.info(f'Creating multiscale volume: {zarr_path}')
    pyramid_levels.main(zarr_path=zarr_path, max_layer=max_layer, num_processes=num_processes)
    return



def create_section_yaml_file(file_path,
                             section_num,
                             tile_grid_num,
                             grid_shape,
                             acquisition,
                             thickness,
                             tile_height,
                             tile_width,
                             tile_overlap,
                             tiles):
    data = {
        'section_num': section_num,
        'tile_grid_num': tile_grid_num,
        'grid_shape': grid_shape,
        'acquisition': acquisition,
        'thickness': thickness,
        'tile_height': tile_height,
        'tile_width': tile_width,
        'tile_overlap': tile_overlap,
        'tiles': tiles
    }

    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    logging.info(f"YAML file created at {file_path}")
    return


def norm_img(data) -> np.ndarray:
    logging.debug(f'norm_img shape:{data.shape}')
    norm_gray = (data - np.min(data)) / (np.max(data) - np.min(data))
    return (norm_gray * 255).astype(np.uint8)


# def plot_images_with_overlay(image1, image2, alpha=0.5):
#     fig, ax = plt.subplots(figsize=(10, 5))
#
#     # Plot the first image
#     ax.imshow(image1, cmap='gray')
#
#     # Plot the second image with transparency
#     ax.imshow(image2, cmap='jet', alpha=alpha)
#
#     # Set axis labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#
#     # Show the plot
#     plt.show()
#     return


def build_tiles_coords(tile_id_map: np.ndarray) -> Optional[Tuple[TileXY]]:
    """Builds tile coordinates map from the given tile ID map.

    Args:
        tile_id_map (np.ndarray): The tile ID map.

    Returns:
        Optional[Tuple[TileXY]]: A tuple of tile coordinates (x, y).
    """
    if not isinstance(tile_id_map, np.ndarray):
        logging.error('Build tile coords failed: wrong input data.')
        return None

    tc = []
    y, x = tile_id_map.shape
    for i in range(y):
        for j in range(x):
            if int(tile_id_map[i, j]) != -1:
                tc.append((j, i))
    return tuple(tc)


def crop_bbox(img: np.ndarray,
              pad: Optional[int] = 700,
              offset: Optional[Vector] = None
              ) -> np.ndarray:
    """
    Crop the bounding box around the non-zero elements in the image.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        pad:
        offset:

    Returns:
        np.ndarray: Cropped image with the maximum extent of black canvas removed.

    """

    is_vert = True if img.shape[0] < img.shape[1] else False
    axis = 1 if is_vert else 0

    if is_vert:
        left = pad + offset[1 - axis]
        right = left + img.shape[axis]
        top = 0
        bottom = 0
    else:
        left = 0
        right = 0
        top = pad + offset[1 - axis]
        bottom = top + img.shape[axis]

    # Crop the image
    print(f'cropping params: {top, bottom, left, right}')
    cropped_img = img[top:bottom, left:right]
    print(f'orig shape: {img.shape}')
    print(f'new shape: {cropped_img.shape}')

    return cropped_img


def crop_nan(img: np.ndarray[float]) -> np.ndarray:
    """Remove all columns in the input image that contain nan values"""
    nans = np.argwhere(np.isnan(img))
    nan_col_indices = set(nans[:, 1])
    mask = np.ones(img.shape[1], dtype=bool)
    mask[list(nan_col_indices)] = False

    # Crop out columns based on the mask
    cropped_arr = img[:, mask]
    return cropped_arr


def fill_holes(mask):
    num_mask = np.asarray(mask, dtype=int)
    filled_num_mask = ndimage.binary_fill_holes(num_mask)
    filled_mask = filled_num_mask.astype(bool)
    return filled_mask

def get_smearing_mask(
  img: np.ndarray,
  mask_top_edge: int = 0,
  path_plot: Optional[str] = None,
  plot=False
) -> Optional[np.ndarray]:
  """Commutes mask of a distortion appearing at the top of the EM-images.

  Estimate the presence and extent of a smearing distortion at the top of the
  input image and return it as a boolean mask.

  Args:
    img: input image for detection of distortion at its top border
    mask_top_edge: number of lines at the top of the image to be masked entirely
    path_plot: filepath where to save the mask image (if plot=True)
    plot: switch to execute creation of various mask graphs

  Returns:
    Mask of smearing distortion with the shape same as the input image
  """

  det_args = dict(
    img=img,
    segment_width=1000,
    sigma=1.5,
    dx=50,
    dy=4
  )

  # Run smearing detection
  smr_map = detect_smearing2d(**det_args)
  smr_map_interp = interpolate_nan_2d(smr_map)
  smr_map_interp = gaussian(smr_map_interp, sigma=2)
  mask = create_mask(smr_map_interp, threshold=0.1)

  clean_args = dict(
    mask=mask,
    min_size=800,
    portion=1.0,
    max_vert_extent=150,
    top=mask_top_edge
  )

  def clean_mask(mask, top, min_size, portion, max_vert_extent):

    # Mask entire top lines
    if top > 0:
      mask[:top] = True

    # Mask top right border # TODO investigate if needed
    mask[:, -1] = True

    # Fill binary holes
    mask = fill_holes(mask)

    # Unmask all lines below line nr. 'max_vert_extent'
    if 0 < max_vert_extent < mask.shape[0]:
      mask[max_vert_extent:] = False

    # Mask small masking irregularities
    if portion > 0:
      mask = flood_smearing(mask, portion)

    # Remove True islands with small area
    if min_size > 0:
      mask = remove_isolated(mask, min_size)

    return mask

  mask_final2 = clean_mask(**clean_args)


  # mask_filled = fill_holes(mask)
  # # mask_flooded = flood_pixels(mask_filled, ratio=0.4)
  # # mask_flood = flood_smearing(mask_flooded, portion=1.0)
  # mask_flood = flood_smearing(mask_filled, portion=1.0)
  # # mask_flood2 = flood_smearing(mask_flood, portion=.5)
  # # mask_final = fill_holes(mask_flood2)
  # mask_final = fill_holes(mask_flood)
  # mask_final2 = remove_isolated(mask_final, min_size=800)

  # if plot:
  #   to_plot = [smr_map, smr_map_interp, mask_flood,
  #              mask_flood, mask_final, mask_final2
  #              ]
  #   plot_smearing(plots=to_plot, path_plot=path_plot)

  return mask_final2


def crop_ov(ov_img: np.ndarray, pad: int, offset: Vector,
            is_vert: bool, tile_shape: Tuple[int, int]) -> np.ndarray:

    # plt.imshow(ov_img, cmap='gray')
    # plt.show()

    dy, dx = 100, 100  # Half-widths of seam region
    axis = 1 if is_vert else 0

    h, w = np.shape(ov_img)
    h_mid, w_mid = int(h / 2), int(w / 2)
    c0 = int(pad / 2) + offset[1 - axis]

    if is_vert:
        left = max(int(pad / 2), c0)
        right = min(int(pad / 2), c0) + tile_shape[axis]
        top = h_mid - dy
        bottom = h_mid + dy
    else:
        left = w_mid - dx
        right = w_mid + dx
        top = max(int(pad / 2), c0)
        bottom = min(int(pad / 2), c0) + tile_shape[axis]

    cropped_ov = ov_img[top:bottom, left:right]

    # plt.imshow(ov_img, cmap='gray')
    # plt.show()
    # plt.imshow(cropped_ov, cmap='gray')
    # plt.show()

    return cropped_ov


def detect_bad_seam(
    ov_img: np.ndarray,
    is_vert: bool,
    stride: int = 1,
    plot: bool = False,
    path_plot: Optional[str] = None,
    show_plot: bool = False
) -> Optional[float]:

    # Rotate overlap plot
    if not is_vert:
        ov_img = np.rot90(ov_img, k=1)

    # Remove all columns in the image that contain nan values
    ov_crop = crop_nan(ov_img)

    # Apply one-directional Gaussian blurring
    sigma = 2
    ov_crop = ndimage.gaussian_filter(ov_crop, sigma=(0, sigma))

    y, x = ov_crop.shape
    if y < stride:
        logging.warning(f'Cropped area is too small in (y) {ov_crop.shape} \
        to be used with specified stride.')
        return
    if x < stride:
        logging.warning(f'Cropped area is too small in (x) {ov_crop.shape} \
        to be used with specified stride.')
        return

    # Find discontinuity using SSIM
    peak_to_ssim = bad_seam_ssim(ov_crop, plot, path_plot, show_plot)
    return peak_to_ssim


def bad_seam_ssim(
        img: np.ndarray,
        plot: bool = False,
        path_plot: Optional[str] = None,
        show_plot: bool = False
) -> float:

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def pk_detect(a, n=7):
        pk_arr = []
        for i in range(len(a) - n):
            pk_arr.append(np.max(a[i:i + n]) + abs(np.min(a[i:i + n])))
        return np.array(pk_arr)

    # Change to 8-bit depth
    def norm_img(data) -> np.ndarray:
        norm_gray = (data - np.min(data)) / (np.max(data) - np.min(data))
        return (norm_gray * 255).astype(np.uint8)

    res = []
    dy = 10  # Width of an image stripe to be used for SSIM
    line_range = np.arange(img.shape[0] - 1 - dy, step=dy // 2)

    for i in line_range:
        a = img[i: i + dy]
        b = img[i + 1: i + dy + 1]
        a, b = [skimage.filters.gaussian(i_img, sigma=1.5) for i_img in (a, b)]
        a = norm_img(a)
        b = norm_img(b)
        res.append(ssim(a, b, win_size=5, full=False))

    ssim_vals = np.array(res)
    min_peak = np.min(ssim_vals)
    min_pk_vs_mean = 1/(min_peak / np.mean(ssim_vals))

    # diff_res = ssim_vals[1:] - ssim_vals[:-1]
    # diff_res = moving_average(np.array(diff_res))
    #
    # pk_spectrum = pk_detect(diff_res, n=2)
    # # plt.plot(pk_spectrum, 'o-')
    # # # plt.plot(diff_res, 'o-')
    # # plt.show()
    # # return
    #
    # diff_res = pk_spectrum
    # diff_mean, diff_std = np.mean(diff_res), np.std(diff_res)
    # max_peak = np.max(diff_res)
    #
    # pk_vs_mean: float = max_peak / diff_mean
    # pk_vs_std: float = max_peak / diff_std
    #
    # # mean = np.mean(ssim_vals)
    # # std_dev = np.std(ssim_vals)
    # #
    # # # Calculate upper and lower bounds
    # # upper_bound = mean + 2 * std_dev
    # # lower_bound = mean - 2 * std_dev
    # #
    # # # Filter outliers
    # # filtered_ssim_vals = ssim_vals[(ssim_vals >= lower_bound) & (ssim_vals <= upper_bound)]
    # # mean = np.mean(filtered_ssim_vals)
    # # std_dev = np.std(filtered_ssim_vals)
    # # pk = min(ssim_vals)
    # # print(pk)
    # # print(mean, std_dev)
    # # print((mean-pk)/std_dev)
    # # pk_vs_mean = (mean-pk)/std_dev
    #
    # # plt.subplot(1, 2, 1)
    # # plt.plot(res, line_range[::-1], 'o-')
    # # plt.xlabel('Mean SSIM')
    # # plt.ylabel('Line Index [pix]')
    # #
    # # plt.subplot(1, 2, 2)
    # # _label = f'max pk vs. meanSSIM: {pk_vs_mean}, max pk vs. stddev: {pk_vs_std}'
    # # plt.plot(diff_res, '-o', label=f'{_label}')
    # # plt.axhline(diff_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    # #
    # # # Plot one standard deviation band
    # # plt.fill_between(range(len(diff_res)), diff_mean - diff_std, diff_mean + diff_std,
    # #                  color='orange', alpha=0.3, label='1 Std Dev Band')
    # #
    # # # Plot two standard deviation band
    # # plt.fill_between(range(len(diff_res)), diff_mean - 2 * diff_std, diff_mean + 2 * diff_std,
    # #                  color='yellow', alpha=0.3, label='2 Std Dev Band')
    # #
    # # plt.xlabel('Mean SSIM (difference)')
    # # plt.ylabel('Diff. SSIM value')
    # # plt.legend()
    # # plt.show()
    #
    # if plot:
    #     plt.figure(figsize=(12, 8), dpi=100)
    #     plt.subplot(311)
    #     plt.imshow(img, cmap='gray')
    #     plt.axis('off')
    #
    #     plt.subplot(312)
    #     _label = f'inv. min pk vs. meanSSIM: {min_pk_vs_mean:.2f}'
    #     plt.plot(res, line_range[::-1], 'o-', label=_label)
    #     plt.xlabel('Mean SSIM')
    #     plt.ylabel('Line Index [pix]')
    #     plt.title('Mean SSIM')
    #     plt.legend()
    #     plt.grid(True)
    #
    #     plt.subplot(313)
    #     _label = f'max pk vs. meanSSIM: {pk_vs_mean:.3f}, max pk vs. stddev: {pk_vs_std:.2f}'
    #     plt.plot(diff_res, '-o', color='orange', label=_label)
    #     plt.axhline(diff_mean, color='red', linestyle='dashed', linewidth=2, label='Mean')
    #     plt.fill_between(range(len(diff_res)), diff_mean - diff_std, diff_mean + diff_std,
    #                      color='orange', alpha=0.3, label='1 Std Dev Band')
    #     plt.fill_between(range(len(diff_res)), diff_mean - 2 * diff_std, diff_mean + 2 * diff_std,
    #                      color='yellow', alpha=0.3, label='2 Std Dev Band')
    #     plt.xlabel('Mean SSIM (difference)')
    #     plt.ylabel('Diff. SSIM value')
    #     plt.title('Difference SSIM')
    #     plt.grid(False)
    #     plt.legend()
    #     plt.tight_layout()
    #
    #     if path_plot is not None:
    #         print(f'export path: {path_plot}')
    #         plt.savefig(path_plot)
    #
    #     if show_plot:
    #         plt.show()
    #
    #     plt.close()
    return 1 / min_pk_vs_mean


def get_tid_idx(tile_id_map, tile_id) -> Optional[Tuple[int, int]]:
    # Extract y, x coordinate of tile_id in tile_id_map
    # logging.debug(tile_id_map)
    if tile_id not in tile_id_map:
        # logging.info(f'Tile_ID: {tile_id} not present in section!')
        return None

    if tile_id == -1:
        logging.info(f'Tile_ID: {tile_id} has undefined mask!')
        return None

    coords = np.where(tile_id == tile_id_map)
    y, x = int(coords[0][0]), int(coords[1][0])
    return y, x


def plot_trace_from_backup(
        path_cxyz: str,
        path_id_maps: str,
        path_plot: str,
        tile_id: int,
        sec_range: Tuple[Optional[int], Optional[int]],
        show_plot: bool,
):
    """Plots traces from input cxyz tensor

    Args:
        path_cxyz: str -  path to aggregated file containing all coarse offsets
        path_id_map: str - path to aggregated file containing all tile_id_maps
        path_plot: str - path here to store resulting graph
        tile_id: int - ID of the tile trace to be plotted
        sec_range: Optional[tuple(int, int)] - range of section numbers to be plotted

    :return:
    """

    def plot_traces(x_axis: np.ndarray, traces: np.ndarray, _path_plot: str,
                    _tile_id: int, _vert_nn_tile_id: Optional[int], _show_plot: bool) -> None:

        """Plots array of both coarse offset vectors' values """

        fig, ax = plt.subplots(figsize=(15, 9))
        labels = ('c0x', 'c0y', 'c1x', 'c1y')
        for j in range(traces.shape[0]):
            ax.plot(x_axis, traces[j, :], '-', label=f'{labels[j]}')

        # Add labels, title, and legend
        ax.set_xlabel('Section number')
        ax.set_ylabel('Shift [pix]')
        nn_tile_id = '' if _vert_nn_tile_id is None else f" ({str(_vert_nn_tile_id)})"
        ax.set_title(f'Coarse Offsets for Tile ID {_tile_id} {nn_tile_id}')
        ax.legend(loc='upper right')
        ax.grid(True)

        # Adjust x-axis and y-axis ticks density
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=30))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))

        plt.savefig(_path_plot)
        if _show_plot:
            plt.show()

        plt.close(fig)
        return

    path_cxyz = Path(cross_platform_path(path_cxyz))
    path_id_maps = Path(cross_platform_path(path_id_maps))
    path_plot = cross_platform_path(path_plot)

    if path_cxyz.exists() and path_id_maps.exists():
        # Load coarse offsets
        cxyz_obj = np.load(path_cxyz)
        cxyz_keys = list(cxyz_obj.files)
        sec_nums = set(map(int, cxyz_keys))

        # Load tile_id_maps
        tile_id_maps = np.load(path_id_maps)
        tile_id_maps_keys = list(tile_id_maps.files)
        sec_nums_id_maps = set(map(int, tile_id_maps_keys))

        # Compatible section numbers
        sec_nums = sec_nums.intersection(sec_nums_id_maps)

        if len(sec_nums) == 0:
            # Not possible to map tile_id_map files to the coarse offset files
            print(f'Available offsets maps and tile_id_maps do not match.')
            return

        # Select range of sections to be processed
        first, last = sec_range
        if first is None:
            first = min(sec_nums)

        if last is None:
            last = max(sec_nums)

        if first > last:
            logging.warning('Plot traces: wrong section range definition.')
            return

        sec_nums_plot = set(np.arange(first, last, step=1))
        sec_nums_plot = sec_nums.intersection(sec_nums_plot)
        logging.info(f'Trace plotting: {len(sec_nums_plot)} sections will be processed.')

        if len(sec_nums_plot) > 1:

            arr = np.full(shape=(4, last - first), fill_value=np.nan)
            x_axis_sec_nums = np.arange(first, last)
            vert_nn_tile_id = None
            for i, num in enumerate(x_axis_sec_nums):
                if num in sec_nums_plot:
                    tile_id_map = tile_id_maps[str(num)]
                    if vert_nn_tile_id is None:
                        vert_nn_tile_id = get_vert_tile_id(tile_id_map, tile_id)
                    coord = get_tid_idx(tile_id_map, tile_id)
                    if coord is not None:
                        y, x = coord
                        try:
                            shifts = cxyz_obj[str(num)][:, :, y, x]
                            arr[:, i] = shifts.flatten().transpose()
                        except IndexError as _:
                            logging.warning(f'Trace plotting: unable to determine shifts of s{num} t{tile_id}')
                    else:
                        continue

            if not np.all(np.isnan(arr)):
                plot_traces(x_axis_sec_nums, arr, path_plot, tile_id, vert_nn_tile_id, show_plot)
        else:
            print(f'Nothing to plot. Sections {first} : {last} not in available'
                  f'range: [{min(sec_nums)} : {max(sec_nums)}].')
            return
    else:
        print(f'Input files are missing. Check path_cxyz: {path_cxyz}')
    return


def plot_trace_eval_ov(
        path_cxyz: str,
        path_id_maps: str,
        path_plot: str,
        tile_id: int,
        sec_range: Tuple[Optional[int], Optional[int]],
        show_plot: bool,
):
    """Plots trace from input tensor

    Args:
        path_cxyz: str -  path to aggregated file containing all coarse offsets
        path_id_map: str - path to aggregated file containing all tile_id_maps
        path_plot: str - path here to store resulting graph
        tile_id: int - ID of the tile trace to be plotted
        sec_range: Optional[tuple(int, int)] - range of section numbers to be plotted

    :return:
    """

    def plot_traces(x_axis: np.ndarray, traces: np.ndarray, _path_plot: str, _tile_id: int, _show_plot: bool) -> None:
        # Plot four graphs of coarse offsets
        fig, ax = plt.subplots(figsize=(15, 9))
        labels = ('horizontal pairs', 'vertical pairs')
        for j in range(len(labels)):
            ax.plot(x_axis, traces[j, :], '-', label=f'{labels[j]}')

        # Add labels, title, and legend
        ax.set_xlabel('Section number')
        ax.set_ylabel('Overlaps similarity (SSIM)')
        ax.set_title(f'Similarity plots of tile-pair overlaps {_tile_id}')
        ax.legend(loc='upper right')

        # Set grid
        ax.grid(True)

        # Adjust x-axis ticks density
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the plot and show if needed
        plt.savefig(_path_plot)
        if _show_plot:
            plt.show()

        # Close the plot
        plt.close()

        return

    print(f'path_cxyz: {path_cxyz}')
    logging.warning(f'path_cxyz: {path_cxyz}')
    path_cxyz = Path(cross_platform_path(path_cxyz))
    path_id_maps = Path(cross_platform_path(path_id_maps))
    path_plot = cross_platform_path(path_plot)

    if not path_cxyz.exists() or not path_id_maps.exists:
        print(f'Input files are missing. Check path_cxyz: {path_cxyz}')
    else:
        # Load coarse offsets
        cxyz_obj = np.load(path_cxyz)
        cxyz_keys = list(cxyz_obj.files)
        sec_nums = set(map(int, cxyz_keys))

        # Load tile_id_maps
        tile_id_maps = np.load(path_id_maps)
        tile_id_maps_keys = list(tile_id_maps.files)
        sec_nums_id_maps = set(map(int, tile_id_maps_keys))

        # Compatible section numbers
        sec_nums = sec_nums.intersection(sec_nums_id_maps)

        if len(sec_nums) == 0:
            # Not possible to map tile_id_map files to the coarse offset files
            print(f'Available offsets maps and tile_id_maps do not match.')
            return

        # Select range of sections to be processed
        first, last = sec_range
        if first is None:
            first = min(sec_nums)

        if last is None:
            last = max(sec_nums)

        if first > last:
            print('Nothing to plot: wrong section range definition.')
            return

        sec_nums_plot = set(np.arange(first, last, step=1))
        sec_nums_plot = sec_nums.intersection(sec_nums_plot)
        logging.info(f'{len(sec_nums_plot)} sections will be processed.')

        if len(sec_nums_plot) <= 1:
            print(f'Nothing to plot. Sections {first} : {last} not in available'
                  f'range: [{min(sec_nums)} : {max(sec_nums)}].')
            return
        else:
            arr = np.full(shape=(2, last - first), fill_value=np.nan)
            x_axis_sec_nums = np.arange(first, last)

            for i, num in enumerate(x_axis_sec_nums):
                if num in sec_nums_plot:
                    tile_id_map = tile_id_maps[str(num)]
                    coord = get_tid_idx(tile_id_map, tile_id)
                    if coord is not None:
                        y, x = coord
                        shifts = cxyz_obj[str(num)][:, y, x]
                        arr[:, i] = shifts.flatten().transpose()
                    else:
                        continue

            if not np.all(np.isnan(arr)):
                plot_traces(x_axis_sec_nums, arr, path_plot, tile_id, show_plot)
    return


def aggregate_tile_id_maps(
        section_dirs: List[UniPath]
) -> Tuple[Dict[str, ndarray], List[str]]:
    """
    Load tile_id_maps arrays from input folder and return them
    as a dictionary with section numbers as keys

    :param section_dirs:
    :return: dictionary mapping tile_id array to section number
    """
    maps: dict = {}
    failed_paths = []
    fn_map = 'tile_id_map.json'
    for section_path in tqdm(section_dirs):
        fp_map = Path(section_path) / fn_map
        if not fp_map.exists():
            logging.debug(f's{get_section_num(section_path)} tile_id_map.json file does not exist')
            failed_paths.append(f's{get_section_num(section_path)}\n')
            continue

        key = str(get_section_num(section_path))
        maps[key] = get_tile_id_map(fp_map)

    return maps, failed_paths


def aggregate_coarse_offsets(
        section_dirs: List[UniPath]
) -> Tuple[Dict[str, ndarray], List[str]]:
    """
    Load coarse offset arrays from input folder and return them
    as a dictionary with section numbers as keys

    Coarse offsets can be read either from coarse.npz or from cx_cy.json file.
    :param section_dirs:
    :return:
    """
    offsets: dict = {}
    failed_paths = []
    for section_path in tqdm(section_dirs):
        cxy_path = None
        cxy_names = ('cx_cy.json',)
        for name in cxy_names:
            path_to_check = Path(section_path) / name
            if path_to_check.exists():
                cxy_path = path_to_check
                break

        if cxy_path is None:
            logging.debug(f's{get_section_num(section_path)} coarse-offsets file does not exist')
            failed_paths.append(f's{get_section_num(section_path)}\n')
            continue

        _, cx, cy = read_coarse_mat(cxy_path)
        key = str(get_section_num(section_path))
        offsets[key] = np.array((cx, cy))

    return offsets, failed_paths



def backup_tile_id_maps(dir_sections: UniPath, dir_out: Optional[UniPath]) -> None:
    """
    Store all tile ID maps from sections inside the input folder into a .npz file.

    :param dir_sections: Path to the parent directory containing section directories.
    :param dir_out: Optional. Path to the output directory. If not provided, the result
                    is stored in the input directory.
    :return: None
    """

    dir_sections = Path(dir_sections)
    all_sections = filter_and_sort_sections(str(dir_sections))

    # Collect all tile ID maps
    tile_id_maps, missing_files = aggregate_tile_id_maps(all_sections)
    logging.debug(f'len missing files {len(missing_files)}')
    for p in missing_files:
        logging.debug(p)

    fp_out = (dir_out or dir_sections) / "all_tile_id_maps.npz"
    np.savez(fp_out, **tile_id_maps)
    logging.info(f'Tile ID maps saved to: {fp_out}')

    fp_out2 = fp_out.with_name("all_missing_tile_id_maps.txt")
    with open(fp_out2, "w") as f:
        f.writelines("\n".join(missing_files))
    logging.info(f'Missing tile ID maps saved to: {fp_out2}')

    return


def apply_clahe(image, clip_limit=2., grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def backup_coarse_offsets(dir_sections: UniPath, dir_out: Optional[UniPath]) -> None:
    """
    Store all coarse offset arrays from section inside input folder into a .npz file.

    :param dir_sections: Path to the parent directory containing section directories.
    :param dir_out: Optional. Path to the output directory. If not provided, the result
                is stored in the input directory
    :return: None
    """

    dir_sections = Path(dir_sections)
    all_sections = filter_and_sort_sections(str(dir_sections))

    # Collect all offsets
    offsets, missing_files = aggregate_coarse_offsets(all_sections)
    logging.debug(f'len missing files {len(missing_files)}')
    for p in missing_files:
        logging.debug(p)

    fp_out = (dir_out or dir_sections) / "all_offsets.npz"
    np.savez(fp_out, **offsets)
    logging.info(f'Coarse offsets saved to: {fp_out}')

    fp_out2 = fp_out.with_name("all_offsets_missing_files.txt")
    with open(fp_out2, "w") as f:
        f.writelines("\n".join(missing_files))
    logging.info(f'Missing offsets saved to: {fp_out2}')

    return


def copy_stats_yaml(src_folder, dest_folder):

    # Iterate over folders in the source directory
    for folder_name in tqdm(os.listdir(src_folder)):
        folder_path = os.path.join(src_folder, folder_name)

        # Check if it's a directory and its name matches the expected pattern
        if os.path.isdir(folder_path) and folder_name.startswith('s') and folder_name.endswith('_g1'):
            # Construct the source and destination paths for stats.yaml
            src_stats_yaml = os.path.join(folder_path, 'stats.yaml')
            dest_section_folder = os.path.join(dest_folder, folder_name)
            if Path(dest_section_folder).exists():
                dest_stats_yaml = Path(dest_section_folder) / 'stats.yaml'

                # Copy the file
                shutil.copy(src_stats_yaml, dest_stats_yaml)
                # print(f"Copied {src_stats_yaml} to {dest_stats_yaml}")


def create_directory(dir_path: UniPath):
    dir_path = Path(cross_platform_path(str(dir_path)))
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print(f"Directory '{dir_path}' already exists.")
    except PermissionError:
        print(f"Permission denied. Unable to create directory '{dir_path}'.")
    except OSError as e:
        print(f"Error creating directory '{dir_path}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def cross_platform_path(path: str) -> str:
    """
       Normalize the given filepath to use forward slashes on all platforms.
       Additionally, remove Windows UNC prefix "\\tungsten-nas.fmi.ch\" if present

       Parameters:
           :rtype: object
           :param path: The input filepath.

       Returns:
           str: The normalized filepath.
       """

    def win_to_ux_path(win_path: str, remove_substring=None) -> str:
        if remove_substring:
            win_path = win_path.replace(remove_substring, '/tungstenfs')
        linux_path = win_path.replace('\\', '/')
        linux_path = linux_path.replace('//', '', 1)
        return linux_path

    def ux_to_win_path(ux_path: str, remove_substring=None) -> str:
        if remove_substring:
            ux_path = ux_path.replace(remove_substring, r'\\tungsten-nas.fmi.ch\tungsten')
        win_path = ux_path.replace('/', '\\')
        return win_path

    if path is None:
        return ''

    # Get the operating system name
    os_name = platform.system()

    if os_name == "Windows" and "/" in path:
        # Running on Windows but path in UX style
        path = ux_to_win_path(path, remove_substring="/tungstenfs")

    elif os_name == "Windows" and r"\\tungsten-nas.fmi.ch\tungsten" in path:
        path = path.replace(r"\\tungsten-nas.fmi.ch\tungsten", "W:")
        path = path.replace('\\', '/')

    elif os_name == "Linux" and "\\" in path:
        # Running on UX but path in Win style
        rs = r"\\tungsten-nas.fmi.ch\tungsten"
        path = win_to_ux_path(path, remove_substring=rs)

    return path


def aggregate_coarse_shifts(paths: List[Union[str, Path]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Read cx, cy matrices from 'paths' folders and aggregate them into a dict

    :param paths: section paths
    :return: agg_shifts: dictionary containing section number (int)
                         and corresponding cx, cy matrices
    """
    agg_shifts = {}
    for path in paths:
        path = Path(path)
        fns = ['coarse.npz', 'cx_cy.json']
        fp_coarse_npz, fp_cx_cy = [path / f_name for f_name in fns]
        section_num = int(path.name.split('_')[0].strip('s'))

        if fp_coarse_npz.exists():
            logging.debug(f's{section_num}: Reading cxy from coarse.npz file...')
            _, cx, cy = read_coarse_mat(fp_coarse_npz)
        elif fp_cx_cy.exists():
            logging.debug(f's{section_num}: Reading cxy from cx_cy.json ...')
            _, cx, cy = read_coarse_mat(fp_cx_cy)
        else:
            logging.warning(f's{section_num}: Reading cxy failed. Check if coarse-files are present.')
            cx, cy = [], []
        agg_shifts[section_num] = (cx, cy)
    return agg_shifts


def cxyz_stats(
        section_paths: Optional[List[str]],
        ref_sec_num: int
) -> Dict:
    """
    Perform statistical analysis of cxyz tensor
    :param section_paths:
    :param ref_sec_num: section number, which is being investigated
    :return: average cxy matrix, stddev matrix
    """

    stats_dict = {'mean': ndarray,
                  'std': ndarray,
                  'shift_tensor': ndarray,
                  'cxyz_tf': Dict,
                  'valid': bool,
                  'inv_coords': List}

    # Read cxyz shifts {(section number + cxy) ...}
    cxyz: Dict[int, Tuple[ndarray, ndarray]] = aggregate_coarse_shifts(section_paths)

    # Get section numbers
    sec_nums = list(cxyz.keys())

    # Get tile_id_maps {section_num: tile_id_map}
    tile_id_maps = {}
    for num, path in zip(sec_nums, section_paths):
        path_tile_id_map = Path(path) / 'tile_id_map.json'
        tile_id_maps[num] = get_tile_id_map(path_tile_id_map)

    # Add tile_id_maps and section paths to cxyz tensor
    cxyz_tf = {}
    for num, i_path in zip(sec_nums, section_paths):
        cxy_tf = {
            'cxy': cxyz.get(num),
            'tile_id_map': tile_id_maps.get(num),
            'section_path': i_path
        }
        cxyz_tf[num] = cxy_tf

    # Compute mean_cxy, std_cxy
    mean_mat, stddev_mat, shift_tensor, cxyz_valid, invalid_coords = get_mean_mat(
        cxyz_tf, ref_sec_num)

    # Add cxyz_tf to result and return output
    stats = [mean_mat, stddev_mat, shift_tensor, cxyz_tf, cxyz_valid, invalid_coords]
    for k, v in zip(stats_dict.keys(), stats):
        stats_dict[k] = v

    return stats_dict


def get_aggregation_folders(
        fp_failed_section: Union[str, Path],
        n_before: int,
        n_after: int
) -> Optional[List[Path]]:
    """
    Get list of folders to look for cx, cy interpolation

    Get list of folder neighboring folder names with respect to 'failed_section_nr'
    Example: Corrupted section num: 1137 with n_before=2 and n_after=3 will look for
             section numbers: 1135, 1136, 1137, 1138, 1139, 1140
    :param fp_failed_section: folder name of a section with corrupted cx, cy matrices
    :param n_before: number of neighboring section folders to be looked for (earlier sections)
    :param n_after: number of neighboring section folders to be looked for (newer sections)
    :return: paths of neighbors including the reference failed folder name
    """

    def coarse_files_present(dir_path: Union[str, Path]) -> bool:
        # Scan for coarse shift files and return False if not present in the specified folder
        fp_cx_cy = Path(dir_path) / 'cx_cy.json'
        if fp_cx_cy.exists():
            return True
        else:
            return False

    dir_name = Path(cross_platform_path(str(fp_failed_section)))

    if not dir_name.exists():
        # TODO: broken path to failed section or path doesn't exist
        print(f'Dir name {dir_name} does not exist')
        return None
    else:
        section_num = get_section_num(str(dir_name))
        grid_id = dir_name.name.split('_')[1]
        num_range = list(range(section_num - n_before, section_num + n_after + 1))
        paths = []
        for n in num_range:
            section_name = 's' + str(n) + '_' + grid_id
            dir_name_n = dir_name.parent / section_name
            if not dir_name_n.exists():
                logging.debug(f"Section folder not found: {dir_name_n}")
            else:
                if coarse_files_present(dir_name_n):
                    paths.append(dir_name_n)
                else:
                    logging.warning(f"No coarse shifts present in {dir_name_n.name}.")
        return paths


def filter_dirs_with_file(section_paths: List[str], file_name: str) -> List[str]:
    """
    Filter directories that contain a specific file.

    :param section_paths: The root directory to start searching.
    :param file_name: The name of the file to check for.
    :return: A list of directories containing the file.
    """
    matching_dirs = []
    for section_dir in section_paths:
        for dirpath, dirnames, filenames in os.walk(section_dir):
            if file_name in filenames:
                matching_dirs.append(dirpath)

    return matching_dirs


def get_mean_mat(
        coarse_tensor: dict,
        ref_sec_num: int
) -> Tuple[ndarray, ndarray, ndarray, bool, List]:
    """
    Compute average coarse shifts array and corresponding stddev array
    :param coarse_tensor: dictionary of
                {section_num: {'cxy': coarse mat, 'tile_id_map': map, 'section_path': dir_path}}
    :param ref_sec_num: number of slice for which the mean arrays are computed
    :return: two arrays  of shape (2, 2, x, y) - mean and stddev coarse shift matrices
    """

    ref_cxy = np.array(coarse_tensor[ref_sec_num]['cxy'])  # tuple(cx, cy)

    # Filter out -1 values from tile_id_map first
    tile_id_map = coarse_tensor[ref_sec_num]['tile_id_map']
    valid_tile_ids = tile_id_map[tile_id_map != -1]

    # Initialize aggregation array
    cxyz = np.full(np.shape(ref_cxy), np.nan)[..., np.newaxis]
    sec_nums = sorted(list(coarse_tensor.keys()))
    cxyz = np.repeat(cxyz, len(sec_nums), axis=-1)

    # Get x,y coordinates mapped to tile_id numbers
    valid_tid_mapping = {tile_id: np.where(tile_id == tile_id_map) for tile_id in valid_tile_ids}

    # Aggregate all coarse shift values
    for tile_id, (tid_x_coord, tid_y_coord) in valid_tid_mapping.items():
        for z, num in enumerate(sec_nums):

            if num == ref_sec_num:
                continue

            cxy_tf: dict = coarse_tensor[num]
            tile_map = cxy_tf['tile_id_map']

            if tile_id in tile_map:
                coord = np.where(tile_id == tile_map)
                x, y = coord[0][0], coord[1][0]
                for i, cc in enumerate(cxy_tf['cxy']):
                    for j, mat in enumerate(cc):
                        cxyz[i][j][tid_x_coord[0]][tid_y_coord[0]][z] = mat[x, y]
            else:
                logging.debug(f'TileID {tile_id} not in section {num} tile map.')
                pass

    # Clean-up the cxyz tensor from NaN and Inf values
    ma_sliced_cxyz = np.ma.masked_invalid(cxyz)

    # Compute mean cxy and stddev_cxy for cleaning purposes
    mean_cxy = np.mean(ma_sliced_cxyz, axis=4)
    std_cxy = np.std(ma_sliced_cxyz, axis=4)

    # Remove outliers from cxyz
    final_cxyz = rm_outliers_from_tensor(ma_sliced_cxyz, mean_cxy, std_cxy)
    final_cxyz = np.ma.masked_invalid(final_cxyz)

    # Evaluate if all traces have enough numeric values for mean and stddev computation
    final_cxy_valid, inv_vals = validate_traces(final_cxyz, tile_id_map)

    # Compute mean cxy and stddev_cxy
    mean_cxy = np.mean(final_cxyz, axis=4)
    std_cxy = np.std(final_cxyz, axis=4)

    return mean_cxy, std_cxy, final_cxyz, final_cxy_valid, inv_vals


def get_ov_tid_pairs(directory: UniPath) -> List[Tuple[int, int]]:
    """
    Extracts pairs of tile IDs from folder names in the specified directory.

    Folders should follow the format 'tXXXX_tYYYY', where 'XXXX' and 'YYYY' are integers.

    Parameters:
    - directory (UniPath): The path to the directory containing folders.

    Returns:
    - List[Tuple[int, int]]: A list of tuples, each containing a pair of tile IDs.
    """
    pattern = re.compile(r'^t(\d{4})_t(\d{4})$')  # Exact match for 'tXXXX_tYYYY'
    matches: List[Tuple[int, int]] = []

    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    for entry in dir_path.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                matches.append((num1, num2))

    return sorted(matches)


def get_ov_sec_nums(directory: UniPath) -> List[int]:
    # Regular expression pattern to match "s0510_t0754_t0786_ov.jpg"
    pattern = re.compile(r's(\d+)_t\d+_t\d+_ov\.jpg')
    numbers = []

    # Create a Path object for the directory
    dir_path = Path(cross_platform_path(str(directory)))
    if not dir_path.exists():
        return numbers

    # Iterate over all entries in the directory
    for entry in dir_path.iterdir():
        if entry.is_file():  # Check if it's a file
            match = pattern.match(entry.name)
            if match:
                s_number = int(match.group(1))  # Integer following 's'
                numbers.append(s_number)

    return sorted(numbers)


def get_folder_size(folder_path: str) -> Optional[int]:

    if not Path(folder_path).exists():
        return None

    # Run the du command to get the size of the folder
    command = ['du', '-s', folder_path]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        # Extract the size from the command output
        size_in_kb = int(result.stdout.split()[0])  # Size is in kilobytes
        return size_in_kb
    else:
        print("Error: Unable to get folder size.")
        return None


def get_traces_length(cxyz: ndarray) -> ndarray:
    """
    Compute number of non-Nan and non-Inf values
    in given cxyz array along z-dimension
    """
    valid_mask = np.logical_and(np.isfinite(cxyz), ~np.isnan(cxyz))
    valid_cxyz_count = np.sum(valid_mask, axis=-1, keepdims=False)
    logging.debug(f'traces valid z-dim {valid_cxyz_count}')
    return valid_cxyz_count


def validate_traces(
        coarse_shift_tensor: ndarray,
        tile_id_map: ndarray,
) -> Tuple[bool, List]:
    """
    Compute length of each tile-id trace in (z-dimension)

    :param coarse_shift_tensor: array of coarse shift matrices ((cx, cy), N-sections)
    :param tile_id_map:
    :return:
        - True if all traces have more numerical items than 'min_trace_len'
        - List of coordinates of tile_ids with less numerical items than 'min_trace_len'
    """

    # Minimum amount of numeric values in a given trace
    # global min_trace_len
    logging.debug(f'cxyz shape: {np.shape(coarse_shift_tensor)}')
    traces_len_mat = get_traces_length(coarse_shift_tensor)
    mask = tile_id_map != -1  # mask non-existing section tile-ids

    logging.debug(f'tile-id map: {tile_id_map}')
    logging.debug(f'mask: {mask}')

    invalid_mask = ~np.logical_or(~mask, traces_len_mat > min_trace_len)
    invalids = np.argwhere(invalid_mask)
    traces_are_valid = True if len(invalids) == 0 else False
    return traces_are_valid, invalids



def rm_outliers_from_tensor(
        tensor: np.ndarray,
        mean_mat: np.ndarray,
        std_mat: np.ndarray
) -> np.ndarray:
    """
    Mask outliers in 3D tensor of coarse offsets arrays
    using average and stddev 2D matrix

    :param tensor: input tensor of shape (2,2,x,y,z)
    :param mean_mat: array of shape (2,2,x,y)
    :param std_mat: array of shape (2,2,x,y)
    :return: filtered tensor
    """

    # Broadcast mean_mat and std_mat to match the shape of tensor
    mod_mean_mat = mean_mat[..., np.newaxis]
    mod_std_mat = std_mat[..., np.newaxis]

    # Compute the lower and upper bounds for each element
    # global std_band
    low_bound = mod_mean_mat - std_band * mod_std_mat
    up_bound = mod_mean_mat + std_band * mod_std_mat

    # Create a mask and identify outliers
    mask = ((tensor >= low_bound) & (tensor < up_bound))
    filtered_tensor = np.where(mask, tensor, np.nan)

    return filtered_tensor



def define_specific_sections_names(start: int, end: int, grid_num: int) -> Optional[List[str]]:
    # Generate desired list of section names

    nums = range(start, end+1)

    def names_generator(sec_numbers: Iterable[int]) -> List[str]:
        names = []
        pre = 's'
        for i in sec_numbers:
            post = f'_g{grid_num}'
            name = pre + str(i) + post
            names.append(name)
        return None if len(names) == 0 else names

    sec_names = names_generator(nums)
    return sec_names


def define_specific_section_paths(parent_path: str, section_names: Optional[List[str]]) -> Optional[List[str]]:
    paths = []
    if section_names is not None:
        paths = [str(Path(parent_path) / name) for name in section_names]
    return None if len(paths) == 0 else paths


def list_arrays(group, prefix=''):
    """ Recursively list all array names in the given Zarr group. """
    keys = []
    for key in group.keys():
        item = group[key]
        if isinstance(item, zarr.core.Array):
            keys.append(prefix + key)
        elif isinstance(item, zarr.hierarchy.Group):
            keys.extend(list_arrays(item, prefix=prefix + key + '/'))
    return keys


def read_zarr_volume(path_volume: Union[Path, str]) -> Optional[zarr.core.Array]:
    path_volume = cross_platform_path(str(path_volume))
    try:
        vol = zarr.open(store=path_volume, mode='r')
        logging.debug(vol.info)  # Prints detailed info about the volume

        # Check if there are any arrays or groups within the volume
        if not hasattr(vol, 'info'):  # If the .info property does not exist
            raise AttributeError("Zarr object does not have info attribute, possibly due to corruption.")

        # Checking content directly
        if not vol.keys():
            logging.error("The .zarr volume is empty or corrupted with no members.")
            return None

        keys = list_arrays(vol)  # Assuming list_arrays function can handle recursive listing
        if not keys:
            logging.error("No data arrays found within the .zarr volume.")
            return None

        return vol

    except (zarr.errors.PathNotFoundError, AttributeError) as e:
        logging.error(f"Failed to read .zarr volume due to: {str(e)}")
        return None

def verify_zarr_volume(path_volume: Union[Path, str]) -> bool:
    """
    Check if the Zarr volume at the specified path can be successfully read.

    Args:
    path_volume (Union[Path, str]): The path to the Zarr volume.

    Returns:
    bool: True if the Zarr volume is successfully read, False otherwise.
    """
    volume = read_zarr_volume(path_volume)
    if volume is not None:
        logging.info("Zarr file read successfully.")
        return True
    else:
        logging.warning(f"Failed to read .zarr file: {path_volume}")
        return False



def vector_dist_valid(co: Optional[Vector] = None,
                      est_vec: Optional[Vector] = None,
                      co_lim_dist: Optional[float] = 15.
                      ) -> Tuple[bool, Optional[float]]:
    # Verifies that the absolute pixel distance of current coarse offset is within
    # limits from estimated vector
    abs_dist = None
    vec_valid = False
    if est_vec is not None and co is not None:
        abs_diff = np.abs(np.asarray(co) - np.asarray(est_vec))
        abs_dist = np.linalg.norm(abs_diff)
        vec_valid = abs_dist < co_lim_dist
    return vec_valid, abs_dist



def check_zarr_integrity(zarr_path: UniPath):
    is_intact = True  # Assume the file is intact unless proven otherwise
    logging.info(f'reading: {zarr_path}')
    try:
        # Load the Zarr file
        dataset = zarr.open(store=zarr_path, mode='r')
        dataset = dataset['0']

        # Get the total shape of the dataset and the chunk shape
        total_shape = dataset.shape
        chunk_shape = dataset.chunks

        logging.debug(f'total shape: {total_shape}')

        # Calculate the number of chunks along each dimension
        num_chunks = [int(np.ceil(s / c)) for s, c in zip(total_shape, chunk_shape)]

        # Iterate through every possible chunk index
        for chunk_index in np.ndindex(*num_chunks):
            try:
                # Compute the slice for each chunk based on its index
                slices = tuple(slice(i*c, min((i+1)*c, s)) for i, c, s in zip(chunk_index, chunk_shape, total_shape))
                # Attempt to access the chunk
                data = dataset[slices]
                logging.info(f'{chunk_index} {np.shape(data)}')
                # if np.all(data == 0):  # Assuming '0' is the fill value for missing chunks
                #     logging.info(f"Warning: Chunk {chunk_index} seems to be filled with default values.")
                #     is_intact = False
            except KeyError:
                logging.warning(f"Error: Missing or inaccessible chunk at index {chunk_index} in {zarr_path}")
                is_intact = False
            except Exception as e:
                logging.warning(f"Error reading chunk at index {chunk_index} in {zarr_path}: {e}")
                is_intact = False

    except Exception as e:  # Use a broad Exception if specific Zarr errors are not available
        logging.warning(f"Failed to load Zarr file {zarr_path}: {e}")
        is_intact = False

    if is_intact:
        logging.info(f'zarr file at following path is valid: {zarr_path}')
    return is_intact


def find_black_stripe(image: np.ndarray, kern_len=100) -> Union[Set[int], np.ndarray]:
    # Define the kernel for detecting the stripe of black pixels of len 'ker_len'
    # Perform in both x and y direction
    kernel = np.array(np.ones(kern_len), dtype=bool)
    kernels = (kernel.reshape(1, -1), kernel.reshape(-1, 1))

    conv_images = {}
    zero_locations = {}
    for i, ker in enumerate(kernels):
        # Apply convolution to detect the black stripe
        conv_images[i] = convolve(image, ker, mode='constant')

        # Find the location of the black stripe
        y, x = np.where(conv_images[i] == 0)
        # Return the coordinates (x, y) of the black stripe location
        zero_locations[i] = list(zip(y, x))

    conv_img = np.logical_and(conv_images[0], conv_images[1])
    return zero_locations, conv_img


def filter_low_pass(image: np.ndarray, brightness_limit: float) -> np.ndarray:
    if not 0 <= brightness_limit <= 255:
        raise ValueError("Threshold intensity value must be between 0 and 255!")

    # Convert RGB image to grayscale
    if image.ndim == 3:
        filtered_img = rgb2gray(image[..., :3]) * 255
    else:
        filtered_img = np.copy(image)
    assert filtered_img.ndim == 2, "Warning, image is RGB but greyscale is required!"
    # Perform low-pass thresholding
    filtered_img[filtered_img > brightness_limit] = 255
    filtered_img[filtered_img != 255] = 0
    return filtered_img


def plot_img(data: np.ndarray):
    plt.imshow(data, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_overlay(ref_img, mov_img, down_fct=0.2):

    # Downscale both images
    img1_downscaled = ndimage.zoom(ref_img, down_fct)
    img2_downscaled = ndimage.zoom(mov_img, down_fct)

    plt.figure()
    cmap1 = 'viridis'
    cmap2 = 'inferno'
    plt.imshow(img1_downscaled, cmap=cmap1)
    plt.imshow(img2_downscaled, cmap=cmap1, alpha=0.5)  # Adjust alpha to control transparency
    plt.show()

def compute_histogram(data):
    hist, centers = skimage.exposure.histogram(data)
    return hist, centers


def compute_tile_pair_shift(ref_img: np.ndarray, tst_img: np.ndarray) -> List[float]:
    shift, error = cv2.phaseCorrelate(ref_img.astype(np.float32), tst_img.astype(np.float32))
    shift = [-shift[1], -shift[0]]  # Negate and swap dx and dy
    return shift

def compute_tile_id_map(grid_shape: Tuple[int, int], tile_ids: List[int]) -> Optional[ArrayLike]:

    def trim_neg_ones(arr):
        """
        Removes borders filled entirely with -1 from a 2D numpy array.
        """
        if arr.size == 0:
            return arr

        rows = np.any(arr != -1, axis=1)
        cols = np.any(arr != -1, axis=0)
        return arr[np.ix_(rows, cols)]

    def create_sbem_grid():
        grid = np.full(grid_shape, fill_value=-1)
        rows, cols = grid.shape

        for row_pos in range(rows):
            for col_pos in range(cols):
                tile_index = row_pos * cols + col_pos
                if tile_index in tile_ids:
                    grid[row_pos, col_pos] = tile_index
        return trim_neg_ones(grid)

    if len(tile_ids) == 0:
        logging.warning(f"len tiles: {len(tile_ids)}")
        return None

    return create_sbem_grid()
    
    
def compute_ssim(a: ndarray, b: ndarray) -> float:
    SSIM = skimage.metrics.structural_similarity(a, b)
    normalized_similarity = (SSIM + 1) / 2
    return float(round(normalized_similarity, 3))


def register_tile_pair(
        ref_img: ndarray,
        moving_img: ndarray,
        shift_vec: Optional[Tuple[int, int]] = None,
) -> Tuple[ndarray, ndarray, List[Num]]:
    """
    Register and crop tile-pair. If shift vector is provided,
    do not register, but return shifted and cropped images.

    :param ref_img: raw .tif reference image data
    :param moving_img: raw. tif test image-data
    :param shift_vec: optional shift vector (y, x) used to shift and crop resulting images
    :return: Tuple of:
                - cropped reference image
                - cropped shifted testing image
                - computed shift vector between tile-images
    """
    if not shift_vec:
        shift = compute_tile_pair_shift(ref_img, moving_img)
        shift = list(map(int, shift))
    else:
        shift = list(map(int, shift_vec))

    cx, cy = abs(shift[0]), abs(shift[1])
    crop_vals = ((cy, cy), (cx, cx))

    ref_img = skimage.util.crop(ref_img, crop_vals)
    shifted_img = ndimage.shift(moving_img, shift)
    shifted_img = skimage.util.crop(shifted_img, crop_vals)

    return ref_img, shifted_img, shift


def shift_tile_pair(
        ref_img: ndarray,
        moving_img: ndarray,
        shift: List[Num]
) -> Tuple[ndarray, ndarray]:
    """
    Register and crop tile-pair
    :param shift:
    :param ref_img:
    :param moving_img:
    :return:
    """

    cx, cy = abs(shift[0]), abs(shift[1])
    crop_vals = ((cy, cy), (cx, cx))

    ref_img = skimage.util.crop(ref_img, crop_vals)
    shifted_img = ndimage.shift(moving_img, shift)
    shifted_img = skimage.util.crop(shifted_img, crop_vals)

    return ref_img, shifted_img


def parse_inf_log(file_path: str) -> Optional[Dict[int, Tuple[int, ...]]]:
    """
    Parse a tab-delimited text file containing slice numbers and coarse offsets.

    Args:
        file_path (str): The path to the input text file.

    Returns:
        Optional[Dict[int, np.ndarray]]: A dictionary mapping slice numbers to
            coarse offset coordinates.
        Returns None if the file is not found or an error occurs during parsing.
    """
    data_dict = {}

    try:
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)  # Skip the header

            for row in reader:
                key = int(row[0])
                values = tuple(map(int, row[1:]))
                data_dict[key] = values
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None

    return data_dict


def parse_logs(log_files: List[str]) -> Tuple[Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]], Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Parse log files and construct a dictionary with section numbers as keys,
    and lists of (xy shift, xy error) tuples as values. Also record xy errors separately.
    """
    section_data = defaultdict(list)
    xy_errors = {}

    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                if 'Sections' in line:
                    # Extract section numbers and xy shifts/errors using regex
                    match = re.search(r'Sections \((\d+), (\d+)\) cross-section xy shift \(([-\d]+), ([-\d]+)\) xy err \(([-\d]+), ([-\d]+)\)', line)
                    if match:
                        sec_start = int(match.group(1))
                        sec_end = int(match.group(2))
                        xy_shift = (int(match.group(3)), int(match.group(4)))
                        xy_err = (int(match.group(5)), int(match.group(6)))

                        section_data[sec_start].append((xy_shift, xy_err))
                        xy_errors[(sec_start, sec_end)] = xy_err

    return section_data, xy_errors


def compute_absolute_error(xy_err: Tuple[int, int]) -> float:
    """
    Compute the absolute error magnitude from xy error tuple.
    """
    return sqrt(xy_err[0]**2 + xy_err[1]**2)



def rename_folders(root_dir: str, start: int, end: int):
    """
    Rename .zarr subfolders with numbers from start to end to .zarr_skip

    Args:
    - root_dir (str): Root directory containing subfolders to rename.
    """
    for folder_name in os.listdir(root_dir):
        if folder_name.endswith('.zarr'):
            try:
                folder_num = int(folder_name.split('_')[0][1:])  # Extract number from folder name
                if start <= folder_num <= end:
                    new_name = folder_name.replace('.zarr', '.zarr_skip')
                    os.rename(os.path.join(root_dir, folder_name), os.path.join(root_dir, new_name))
                    print(f'Renamed {folder_name} to {new_name}')
            except ValueError:
                print(f'Ignoring folder {folder_name} with invalid format')


def represent_tuple(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', list(data), flow_style=True)


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by a specified angle and center it in a larger canvas using Pillow.

    Args:
    - image (np.ndarray): Input image as a NumPy array.
    - angle (float): Angle in degrees by which to rotate the image.

    Returns:
    - np.ndarray: Rotated image centered in a larger canvas.
    """

    pil_image = Image.fromarray(image)  # Ensure image is in uint8 format
    rotated_image = pil_image.rotate(angle, expand=True, resample=Image.BICUBIC)
    return np.array(rotated_image)



def sort_and_write_to_yaml(section_data: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]], xy_errors: Dict[Tuple[int, int], Tuple[int, int]], output_file: str):
    """
    Sort section data by absolute error magnitude and write to YAML file.
    """
    yaml.add_representer(tuple, represent_tuple)

    sorted_entries = []

    for sec_num, shifts_errors in section_data.items():
        for shift, err in shifts_errors:
            section_pair = (sec_num, sec_num + 1)
            absolute_error = compute_absolute_error(xy_errors.get(section_pair, (0, 0)))
            sorted_entries.append({
                'section_numbers': section_pair,
                'xy_shift': shift,
                'xy_error': err,
                'absolute_error': absolute_error
            })

    sorted_entries.sort(key=lambda x: x['absolute_error'])  # Sort by absolute error

    with open(output_file, 'w') as yaml_file:
        yaml.dump(sorted_entries, yaml_file, default_flow_style=False)

def plot_histogram(image, hist, hist_centers):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].axis('off')
    axes[1].plot(hist_centers, hist, lw=2)
    axes[1].set_title('histogram of gray values')
    plt.show()


def find_contours(image: np.ndarray, level: float):
    if image is not None:
        contours = skimage.measure.find_contours(image / 255.0, level)
    else:
        contours = []
    return contours


def display_contours(filtered_image, contours):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(filtered_image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def save_img(path: str, data: np.ndarray):

    if not isinstance(data, np.ndarray):
        logging.warning(f'Image {Path(path).stem} could not be resized.')
        return

    print(f'saving mini to: {path}')
    # cv2.imwrite(path, cv2.convertScaleAbs(data))
    skimage.io.imsave(path, data)
    return


def downscale_image(img: np.ndarray, fct: float) -> np.ndarray:
    """
    Downscale an image using OpenCV.

    Args:
        img (np.ndarray): The input image.
        fct (float): The scaling factor.

    Returns:
        np.ndarray: The downscaled image.
    """
    return cv2.resize(img, None, fx=fct, fy=fct, interpolation=cv2.INTER_AREA)


def downscale_section(image_array: np.ndarray,
                      downscale_factor: float
                      ) -> np.ndarray:
    # Calculate the new dimensions after downscaling
    new_width = int(image_array.shape[1] * downscale_factor)
    new_height = int(image_array.shape[0] * downscale_factor)
    thumb = cv2.resize(image_array,
                       (new_width, new_height),
                       interpolation=cv2.INTER_AREA)
    return thumb


def eval_ov_static(img_a, img_b, offset, is_vert: bool, half_width=50, **kwargs) -> Tuple[float, List[str]]:

    def check_zero_dimension(image_array):
        return any(dim == 0 for dim in image_array.shape)

    log_messages: List[str] = []
    MIN_OV_WIDTH = 15
    axis = 1 if is_vert else 0
    if is_vert:
        img_a = np.rot90(img_a, k=1)
        img_b = np.rot90(img_b, k=1)
        offset = (-offset[0], offset[1])

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
        return np.nan, log_messages
    if check_zero_dimension(ov_b):
        logging.warning('Cropping second overlap overlap resulted in error')
        return np.nan, log_messages

    # Remove masked regions
    ov_stacked = np.rot90(np.hstack((ov_a, ov_b)), k=-1)
    ov_stacked = crop_nan(ov_stacked)
    logging.debug(f'eval_ov: cropped stack ov shape {ov_stacked.shape}')

    if any((size < MIN_OV_WIDTH for size in ov_stacked.shape)):
        msg = f'eval_ov: cropped ov-shape is under limit ({MIN_OV_WIDTH} pixels)'
        log_messages.append(msg)
        return np.nan, log_messages

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
    ov_ac, ov_bc = map(norm_img, (ov_ac, ov_bc))
    mssim = ssim(ov_ac, ov_bc)
    mssim = (mssim + 1) / 2
    logging.debug(f'mssim: {mssim:.3f}')

    return mssim, log_messages


def get_collection_mask(coll_xy_shape: Tuple[int, ...]) -> np.ndarray:
    h, w = coll_xy_shape
    center = (int(h / 2), int(w / 2))
    radius = int(h / 3)
    rr, cc = skimage.draw.disk(center, radius)
    mask = np.ones((h, w), dtype=bool)
    mask[rr, cc] = False
    return mask


def compute_sharpness(img: ndarray, *args) -> float:
    # metric 'edges' computes sharpness as a mean value of image processed by Sobel operator
    mask = get_collection_mask(np.shape(img)) if not args else args[0]
    if np.shape(mask) != np.shape(img):
        mask = get_collection_mask(np.shape(img))

    masked_grad_img = np.ma.array(get_grad_img(img.astype(np.float32)), mask=mask, dtype=np.float32)
    sh = np.round(np.mean(masked_grad_img), 2)
    return sh


def get_section_num(section_path: UniPath) -> Optional[int]:
    try:
        num = int(Path(section_path).name.split('_')[0].strip('s'))
        return num
    except (ValueError, IndexError):
        return None

def get_section_path_by_number(paths: List[str], target_number: int) -> Optional[str]:
    # Filter paths based on whether the target_number is present in the folder name
    matching_paths = [path for path in paths if str(target_number) in os.path.basename(path)]

    if matching_paths:
        # Return the first matching path
        return str(cross_platform_path(str(matching_paths[0])))
    else:
        # Return None if no match is found
        return None


def get_section_nums_from_log(log_file_path: str) -> Optional[List[int]]:
    # Get all unique section numbers from a repair_coarse_shifts log file.
    # Example : 's00274_t0609 mean:[-437.    2.] stddev:[1.8 1.6]' -> [274: int, ...]

    if not Path(log_file_path).is_file():
        print(f"Specified log file doesn't exist.")
        return []

    numbers = []
    with (open(log_file_path, 'r') as file):
        for line in file:
            numbers.append(int(line.split("_")[0].strip("s")))

        unique_numbers = set(numbers)
    return sorted(list(unique_numbers))


def get_tile_id_from_tile_path(path: str) -> Optional[int]:
    if Path(path).suffix == '.tif':
        tile_str = Path(path).stem.split("t")[-1].split("_")[0]
        return int(tile_str)
    else:
        return None


def get_vert_tile_id(tile_id_map: np.ndarray, tile_id: int) -> Optional[int]:
    """
    Retrieves the tile ID located directly below the specified tile ID in the given tile ID map.

    Example:
        tile_id_map = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])
        get_vert_tile_id(tile_id_map, 5) returns:  8
        get_vert_tile_id(tile_id_map, 9) returns: None
    """
    tile_id = int(tile_id)
    if not isinstance(tile_id, int):
        raise ValueError(f"Invalid tile_id '{tile_id}' specification: must be an integer.")

    if tile_id < 0:
        logging.warning(f"Invalid tile_id specification (must be non-negative)!")
        return None

    if tile_id not in tile_id_map:
        # logging.warning(f"Invalid tile_id specification ({tile_id} not in tile_id_map)!")
        return None

    y, x = np.where(tile_id == tile_id_map)
    y, x = y[0], x[0]
    try:
        return int(tile_id_map[y + 1][x])
    except IndexError as _:
        # logging.warning(f"tile_id {tile_id} not found in tile_id_map.")
        return None


def get_tile_num(section_path: UniPath) -> Optional[int]:
    try:
        num = int(Path(section_path).name.split('_')[-2].strip('t'))
        return num
    except (ValueError, IndexError):
        return None


def get_tile_id_map(path_tid_map: UniPath) -> np.ndarray:
    """
    Load a JSON file containing a tile ID map and return it as a NumPy array.

    Args:
        path_tid_map (UniPath): Path to the JSON file.

    Returns:
        np.ndarray: Tile ID map as a NumPy array.
    """
    try:
        with open(path_tid_map, "r") as file:
            mp = np.array(json.load(file))
        return mp
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Handle file not found or JSON decoding errors gracefully.
        raise ValueError(f"Error loading tile ID map from {path_tid_map}: {str(e)}")


def get_grad_img(data: np.ndarray) -> np.ndarray:
    scale = 1
    delta = 0
    ksize = 3
    ddepth = cv2.CV_32F
    # grad_x = cv2.Sobel(data, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # grad_y = cv2.Sobel(data, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x = cv2.Scharr(data, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Scharr(data, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    return cv2.magnitude(grad_x, grad_y)

def get_path_by_tile_id(
        fp_yaml: UniPath,
        tile_id: Union[str, int]
) -> Optional[str]:
    try:
        # Load the YAML data
        with open(fp_yaml, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

        # Find the 'path' associated with the given 'tile_id'
        for tile in data.get('tiles', []):
            if tile.get('tile_id') == int(tile_id):
                return tile.get('path')

        # If the 'tile_id' is not found, return None or raise an exception as needed.
        return None

    except yaml.YAMLError as e:
        # Handle YAML parsing errors gracefully.
        print(f"Error parsing YAML data: {e}")
        return None


def get_tile_shape(fp_yaml: UniPath) -> Optional[TileXY]:
    try:
        # Load the YAML data
        with open(fp_yaml, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
            h = data['tile_height']
            w = data['tile_width']
        return h, w
    except Exception as e:
        print(f"Error in get_tile_shape occurred: {e}")
        return None


def locate_inf_(
        cxy: Optional[ndarray],
        tile_id_map: ndarray,
        section_num: int
) -> Tuple[List[Tuple[int, Any, Any]], List[Any]]:
    """
    Locate Inf in corrupted section coarse matrices
    :param cxy: cx_cy array
    :param tile_id_map:
    :param section_num:
    :return: tuple of :
            Inf positions within cxy matrix [example: (0, 6, 5) cx matrix, col_id=6, row_id=5]
            tile_id(s) corresponding to Inf values
    """
    inf_coords_out = []
    inf_tile_ids = []

    for i, cc in enumerate(cxy):
        logging.debug(f'loc inf cc shape: {np.shape(cxy)}')
        inf_coords = np.argwhere(np.isinf(cc[0]))
        logging.debug(f'loc inf coord {inf_coords}')

        for i_coord in inf_coords:
            if len(i_coord) == 3:
                _, col, row = i_coord
            else:
                col, row = i_coord

            inf_loc = (i, col, row)
            inf_coords_out.append(inf_loc)
            inf_tile_ids.append(tile_id_map[col, row])

    logging.info(f"Section num: {section_num} Inf values at tile_id(s): {inf_tile_ids}")
    return inf_coords_out, inf_tile_ids


def locate_all_inf(section_dirs: Iterable[UniPath]) -> Dict[str, Set]:
    
    all_inf = {}

    for sec_dir in section_dirs:
        cxy_path = Path(sec_dir) / 'cx_cy.json'
        sec_num = get_section_num(sec_dir)
        if cxy_path.exists():
            _, cx, cy = read_coarse_mat(cxy_path)
            cxy = np.array((cx, cy))
            tile_id_map = read_tile_id_map(sec_dir)
            res = locate_inf(cxy, tile_id_map, sec_num)
            if res:
                _, tids = res
                all_inf[str(sec_dir)] = set(tids)
                
    return all_inf


def locate_inf(
        cxy: Optional[ndarray],
        tile_id_map: ndarray,
        section_num: int
) -> Optional[Tuple[List[TileCoord], List[int]]]:
    """
    Locate Inf in corrupted section coarse matrices
    :param cxy: cx_cy array
    :param tile_id_map:
    :param section_num:
    :return: tuple of :
            Inf positions within cxy matrix [example: (0, 6, 5) cx matrix, col_id=6, row_id=5]
            tile_id(s) corresponding to Inf values
    """

    inf_coords = np.where(np.isinf(cxy))
    coords_list = list(zip(*inf_coords))
    tids_list = []

    for crd in coords_list:
        tile_id_a = tile_id_from_coord(crd, tile_id_map)

        # Determine the direction of the tile neighbor
        dx = 1 if crd[0] == 0 else 0
        dy = 1 if crd[0] != 0 else 0

        # Compute the neighbor's coordinates
        nn_dxdy = np.zeros_like(crd)
        nn_dxdy[-2:] = (dy, dx)  # Assign dy and dx directly
        nn_crd = crd + nn_dxdy

        # Get the tile ID of the neighboring coordinate
        tile_id_b = tile_id_from_coord(nn_crd, tile_id_map)

        tids_list.append((tile_id_a, tile_id_b))

    if len(coords_list) == 0:
        return None

    # if len(tids_list) > 0:
    #     coords_w_key = [(int(section_num),) + c + tids
    #                     for c, tids in zip(coords_list, tids_list)]
    # else:
    #     coords_w_key = [(int(section_num),) + c for c in coords_list]


    logging.info(f"Section num: {section_num} Inf values at tile_id pairs: {tids_list}")
    return coords_list, tids_list



def get_shift_grid(max_ext: int, stride: int, shift_vec: Vector, is_vert: bool) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Generate a grid of 2D shift vectors with specific stride and maximum extent.

    The function creates a meshgrid of x and y displacements and combines them into a list
    of shift vectors. Each shift vector represents a 2D displacement.

    Parameters:
    - max_ext (int): Maximum extent in pixels for both x and y displacements.
    - stride (int): Stride for mesh points, determining the spacing between displacements.

    Returns:
    - List[Tuple[int, int]]: A list of tuples where each tuple represents a 2D shift vector.
    """

    if not is_vert:
        x_lim = min(-6, max_ext + shift_vec[0])
        x_disp, y_disp = np.meshgrid(
            np.arange(-max_ext + shift_vec[0], x_lim, stride),
            np.arange(-max_ext + shift_vec[1], max_ext + shift_vec[1] + 1, stride), indexing='ij')
    else:
        y_lim = min(-6, max_ext + shift_vec[1])
        x_disp, y_disp = np.meshgrid(
            np.arange(-max_ext + shift_vec[0], max_ext + shift_vec[0], stride),
            np.arange(-max_ext + shift_vec[1], y_lim, stride), indexing='ij')

    shifts = list(zip(x_disp.flatten().astype(int), y_disp.flatten().astype(int)))

    return shifts, x_disp, y_disp


def get_tile_ids_set(path_all_tid_maps: str) -> Set[int]:
    try:
        path_tid_maps = Path(path_all_tid_maps).parent / 'all_tile_id_maps.npz'
        tid_maps = np.load(str(path_tid_maps), allow_pickle=True)
    except FileNotFoundError as _:
        tid_maps = None
        logging.warning('Error reading all_tile_id_maps.npz')

    tile_ids = set()
    for tid in tid_maps.values():
        tile_ids |= set(tid.flatten())

    tile_ids.remove(-1)

    return tile_ids


def locate_inf_vals(
        path_cxyz: Union[str, Path],
        dir_out: Union[str, Path],
        store: bool) -> Optional[List[Tuple[int]]]:
    """
    Find all Inf values in a backed-up coarse shift tensor.

    :param path_cxyz: Path to the backed-up coarse shift .npz file
    :param dir_out: Directory where to store results
    :param store: Activates storing the results to a text file
    :return: List containing tuples of section numbers, all Inf
            coordinates and corresponding tile IDs
    """

    try:
        cxyz = np.load(str(path_cxyz), allow_pickle=True)
    except FileNotFoundError as e:
        print('Error reading coarse tensor file.')
        return

    try:
        path_tid_maps = Path(path_cxyz).parent / 'all_tile_id_maps.npz'
        tid_maps = np.load(str(path_tid_maps), allow_pickle=True)
    except FileNotFoundError as _:
        tid_maps = None
        logging.warning('Error reading all_tile_id_maps.npz')

    all_coords = []
    all_tids = []
    tids_list = ()

    for section_num in cxyz:
        inf_coords = np.where(np.isinf(cxyz[section_num]))
        coords_list = list(zip(*inf_coords))

        # Get TileIDs of a corrupted tile-pair
        if tid_maps is not None:
            tile_id_map = tid_maps[section_num]
            tids_list = []
            for crd in coords_list:
                tile_id_a = tile_id_from_coord(crd, tile_id_map)

                # Determine the direction of the tile neighbor
                dx = 1 if crd[0] == 0 else 0
                dy = 1 if crd[0] != 0 else 0

                # Compute the neighbor's coordinates
                nn_dxdy = np.zeros_like(crd)
                nn_dxdy[-2:] = (dy, dx)  # Assign dy and dx directly
                nn_crd = crd + nn_dxdy

                # Get the tile ID of the neighboring coordinate
                tile_id_b = tile_id_from_coord(nn_crd, tile_id_map)

                tids_list.append((tile_id_a, tile_id_b))

        if len(tids_list) > 0:
            coords_w_key = [(int(section_num),) + c + tids
                            for c, tids in zip(coords_list, tids_list)]
        else:
            coords_w_key = [(int(section_num),) + c for c in coords_list]

        all_coords.extend(coords_w_key)
        all_tids.append(tids_list)

    if store:
        path_out = str(Path(dir_out) / 'inf_vals.txt')
        logging.info(f"Storing Inf values to: {path_out}")
        np.savetxt(fname=path_out, X=all_coords, fmt='%s', delimiter='\t',
                   header=f'Slice\tC\tZ\tY\tX\tTileID\tTileID_nn')

    return all_coords


def tst_locate_inf_vals():
    path_cxyz = r"\\tungsten-nas.fmi.ch\tungsten\scratch\gmicro_sem\gfriedri\tgan\Stack_alignments\Montano_3\20240217\_inspect\all_offsets.npz"
    path_cxyz = cross_platform_path(path_cxyz)
    dir_out = str(Path(path_cxyz).parent)
    store = True

    res = locate_inf_vals(path_cxyz, dir_out, store)

    return



def tile_id_from_coord(coord: TileCoord, tile_id_map: np.ndarray) -> Optional[int]:
    """
    Get the tile ID from tile coordinates.

    Args:
        coord (TileCoord): Tile coordinates, either (c, z, y, x) or (y, x).
        tile_id_map (np.ndarray): Array containing tile IDs.

    Returns:
        Optional[int]: The tile ID if found, None otherwise.
    """
    if len(coord) in (2, 4):
        # Unpack the last two elements of the tuple as y, x
        y, x = coord[-2:]
    else:
        return None

    try:
        return int(tile_id_map[int(y), int(x)])
    except (ValueError, IndexError):
        return None


def read_coarse_mat_new(path: UniPath) -> tuple[Optional[ndarray], ndarray, ndarray]:
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
            arr = np.load(str(path))
            coarse_mesh, cx, cy = arr['coarse_mesh'], arr['cx'], arr['cy']
            # logging.debug(f"cx npz shape: {np.shape(cx)}")
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
        print(f"Error during read_coarse_mat_new: {e}")


def process_dirs(directory_path: str, filter_function)\
        -> Optional[Tuple[List[Path], List[str], List[int], Dict[int, str]]]:
    """
    Process directories and return lists and dictionaries based on section number.

    :param directory_path: Path to the directory to process.
    :param filter_function: Function to filter and sort the directories.
    :return: Lists and dictionaries of directories, names, numbers, and dicts based on section number.
    """

    if not Path(directory_path).exists():
        return None
    else:
        dirs = [Path(p) for p in filter_function(directory_path)]
        if not dirs:
            logging.warning(f'No stitched directories were loaded!')
            return None
        else:
            names = [d.name for d in dirs]
            nums = [get_section_num(d) for d in dirs]
            dirs_dict = {num: str(p) for num, p in zip(nums, dirs)}

        return dirs, names, nums, dirs_dict


def read_coarse_mat(path: UniPath) -> Tuple[Optional[ndarray], ndarray, ndarray]:
    # TODO: fix dimensionality of saved cx_cy.json !
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
            arr = np.load(str(path))
            coarse_mesh, cx, cy = arr['coarse_mesh'], arr['cx'], arr['cy']
            logging.debug(f'cx npz shape: {np.shape(cx)}')
            return coarse_mesh, cx, cy
        elif path.suffix == '.json':
            # Load the JSON data from the file
            with open(path, 'r') as file:
                data = json.load(file)

            # Convert the loaded data back to NumPy arrays
            cx = np.asarray(data.get('cx', {}))
            cy = np.asarray(data.get('cy', {}))
            if cx.ndim == 4:
                cx = cx[:, 0, ...]
                cy = cy[:, 0, ...]
            if cx.ndim == 5:
                cx = cx[:, 0, 0, ...]
                cy = cy[:, 0, 0, ...]

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



def get_tile_paths(path: UniPath) -> Optional[List[str]]:
    section_yaml = Path(path) / "section.yaml"
    try:
        with open(section_yaml, 'r') as file:
            contents = yaml.safe_load(file)
            if contents:
                return [s["path"] for s in contents["tiles"]]
            else:
                return None
    except FileNotFoundError as e:
        print(f"{e} \n {section_yaml} does not exists!")
        return None


def get_tile_ids(path: UniPath) -> Optional[List[int]]:
    section_yaml = Path(path) / "section.yaml"
    try:
        with open(section_yaml, 'r') as file:
            contents = yaml.safe_load(file)
            if contents:
                return [int(s["tile_id"]) for s in contents["tiles"]]
            else:
                return None
    except FileNotFoundError as e:
        print(f"{e} \n {section_yaml} does not exists!")
        return None


def get_tile_dicts(path: UniPath) -> Optional[Dict[int, str]]:
    section_yaml = Path(path) / "section.yaml"
    try:
        with open(section_yaml, 'r') as file:
            contents = yaml.safe_load(file)
            if contents:
                return {int(s["tile_id"]): str(cross_platform_path(s["path"]))
                        for s in contents["tiles"]}
            else:
                return None
    except FileNotFoundError as e:
        print(f"{e} \n {section_yaml} does not exists!")
        return None



def get_tile_map(path: UniPath) -> Optional[Dict[int, str]]:
    section_yaml = Path(path) / "section.yaml"
    try:
        with open(section_yaml, 'r') as file:
            contents = yaml.safe_load(file)
            if contents:
                return {tile["tile_id"]: tile["path"] for tile in contents["tiles"]}
            else:
                return None
    except FileNotFoundError as e:
        print(f"{e} \n {section_yaml} does not exists!")
        return None


def pair_is_vertical(tile_id_map: np.ndarray, tile_id_a: int, tile_id_b: int
                     ) -> Optional[bool]:

    assert isinstance(tile_id_map, np.ndarray), f"Incorrect tile_id_map type {type(tile_id_map)}. Check if it was loaded correctly."

    if tile_id_a < 0 or tile_id_b < 0:
        logging.warning('f(pair_is_vertical): tile_id must be greater than -1!')
        return None

    if tile_id_a == tile_id_b:
        logging.warning('f(pair_is_vertical): tile_ids must differ!')
        return None

    if tile_id_a not in tile_id_map:
        # logging.warning(f'Tile ID {tile_id_a} not present in TileID map')
        return None
    elif tile_id_b not in tile_id_map:
        # logging.warning(f'Tile ID {tile_id_b} not present in TileID map')
        return None
    else:
        y, x = np.where(tile_id_a == tile_id_map)
        y, x = y[0], x[0]
        try:
            tile_id_up = tile_id_map[y - 1, x]
            if tile_id_b == tile_id_up:
                return True
        except IndexError as _:
            pass
            # logging.warning(f"{tile_id_a} doesn't have a vertical neighbor.")

        try:
            tile_id_down = tile_id_map[y + 1, x]
            if tile_id_b == tile_id_down:
                return True
        except IndexError as _:
            pass
            # logging.warning(f"{tile_id_a} doesn't have a vertical neighbor.")

        if tile_id_b in tile_id_map:
            return False
        else:
            return None


def pair_is_horizontal(tile_id_map: np.ndarray, tile_id_a: int, tile_id_b: int
                       ) -> Optional[bool]:
    if tile_id_a < 0 or tile_id_b < 0:
        logging.warning('pair_is_horizontal: tile_id must be greater than -1!')
        return None

    if tile_id_a == tile_id_b:
        logging.warning('pair_is_horizontal: tile_id must differ!')
        return None

    if tile_id_a not in tile_id_map:
        logging.warning(f'Tile ID {tile_id_a} not present in TileID map')
        return None
    elif tile_id_b not in tile_id_map:
        logging.warning(f'Tile ID {tile_id_b} not present in TileID map')
        return None
    else:
        y, x = np.where(tile_id_a == tile_id_map)
        y, x = y[0], x[0]
        try:
            tile_id_hor = tile_id_map[y, x + 1]
            if tile_id_b == tile_id_hor:
                return True
        except IndexError as _:
            logging.warning(f"{tile_id_a} doesn't have a horizontal neighbor.")

        try:
            tile_id_hor = tile_id_map[y, x - 1]
            if tile_id_b == tile_id_hor:
                return True
        except IndexError as _:
            logging.warning(f"{tile_id_a} doesn't have a horizontal neighbor.")

        if tile_id_b in tile_id_map:
            return False
        else:
            return None


def get_shift(cx_cy: np.ndarray[float],
              tile_id_map: np.ndarray,
              tile_id: int,
              axis: int
              ) -> Optional[Vector]:
    """Returns a shift vector to tile 'tile_id' in 'tile_id_map'.

    Params:
        cx_cy: coarse shift matrix
        tile_id_map: numpy array containing tile IDs
        tile_id: tile number for which to extract shift vector
        axis: 0 for horizontal pair, 1 for vertical pair
    Returns:
        Vector or None if Inf
    """

    coord = np.where(tile_id_map == tile_id)
    y, x = coord[0][0], coord[1][0]
    try:
        vec = cx_cy[axis, :, y, x]
        if np.inf in vec:
            # logging.warning(f"t{tile_id} nothing to plot: shift vector contains Inf value")  # TODO plot?
            return None
        else:
            vec = cx_cy[axis, :, y, x].astype(np.int64)
            return tuple(vec)

    except TypeError as _:
        logging.warning(f"t{tile_id} nothing to plot: shift vector not defined")
        return None

def imread_cv2(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

def read_section_yaml(path: UniPath) -> Optional[List]:
    section_yaml = Path(path) / "section.yaml"
    try:
        with open(section_yaml, 'r') as file:
            contents = yaml.safe_load(file)
            if contents:
                return contents
            else:
                return None
    except FileNotFoundError as e:
        print(f"{e} \n {section_yaml} does not exists!")
        return None


def read_tile_id_map(path: UniPath) -> Optional[ndarray]:
    """
    Return contents of a tile_id_map.json within specified directory path
    :param path: Path to directory containing tile_id_map.json
    :return: tile id map as a numpy array
    """
    fp_json = Path(path) / 'tile_id_map.json'
    if not fp_json.exists():
        print(f'tile_id_map file is missing: {fp_json}')
        return None
    else:
        return get_tile_id_map(fp_json)


def create_coarse_mat(tile_id_map: np.ndarray) -> np.ndarray:
    """
    Creates a dummy coarse offset array based on the section tile_id_map array.
    :param tile_id_map: np.ndarray tile_id_map
    :return: Coarse offset array full of NaN values
    """
    yx_shape = np.shape(tile_id_map)
    conn_x = np.full((2, yx_shape[0], yx_shape[1]), np.nan)
    conn_y = np.full((2, yx_shape[0], yx_shape[1]), np.nan)
    return np.array((conn_x, conn_y))


def scan_dir(dir_path):
    p = Path(dir_path)
    if not p.is_dir():
        print(f"{dir_path} is not a valid directory.")
        return [], []

    dirs = []
    files = []

    for item in p.rglob('*'):
        if item.is_dir():
            dirs.append(item)
        elif item.is_file():
            files.append(item)

    return dirs, files


def save_coarse_mat(
        cxy_mat: np.ndarray,
        path: UniPath,
        file_format: str
) -> None:
    """
    Save coarse offsets array to a file in the specified format ('json' or 'npz').
    :param cxy_mat: cxy array to save
    :param path: directory path where to store the coarse-shift array
    :param file_format: format for saving ('json' or 'npz')
    :return: None
    """
    path = Path(path)

    if not path.is_dir():
        print(f'Error: The directory {path} does not exist.')
        return

    try:
        if file_format == 'json':
            cx_0, cx_1 = cxy_mat[0][0], cxy_mat[0][1]
            cy_0, cy_1 = cxy_mat[1][0], cxy_mat[1][1]
            cx = [cx_0.tolist()], [cx_1.tolist()]
            cy = [cy_0.tolist()], [cy_1.tolist()]
            data = {
                "cx": list(cx),
                "cy": list(cy)
            }
            with open(path / 'cx_cy.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)
        elif file_format == 'npz':
            fn_fix = path / 'coarse_fixed.npz'
            np.savez(fn_fix, cxy_mat)
        else:
            print(f'Error: Unsupported file format "{file_format}". Supported formats are "json" and "npz".')

    except Exception as e:
        print(f'Error during save_coarse_mat: {e}')


def save_coarse_mesh(
        c_mesh: np.ndarray,
        path: UniPath,
        file_format: str
) -> None:
    """
    Save coarse mesh to a file in the specified format ('json' or 'npz').
    :param c_mesh: to save
    :param path: folder name where to store mesh
    :param file_format: format for saving ('json' or 'npz')
    :return: None
    """

    def float_encoder(obj):
        if isinstance(obj, np.float32):
            return round(obj, 2)
        return obj

    path = Path(path)

    if not path.is_dir():
        print(f'Error: The directory {path} does not exist.')
        return

    try:
        if file_format == 'json':
            cx = [c_mesh[0].tolist()]
            cy = [c_mesh[1].tolist()]
            data = {
                "cx": list(cx),
                "cy": list(cy)
            }
            with open(path / 'coarse_mesh.json', 'w') as json_file:
                json.dump(data, json_file, indent=4, default=float_encoder)
        elif file_format == 'npz':
            fn_fix = path / 'coarse_mesh.npz'
            np.savez(fn_fix, c_mesh)
        else:
            print(f'Error: Unsupported file format "{file_format}". Supported formats are "json" and "npz".')

    except Exception as e:
        print(f'Error during save_coarse_mesh: {e}')


def filter_and_sort_sections(sections_dir: str) -> Optional[List[str]]:
    """
    Filter and sort section directories within a parent directory.

    Only section names in form 's0xxxx_gy' where x and y are numeric will be returned.
    Sorting according to the section number from smallest to largest.
    :param sections_dir: Path to the parent directory containing section directories.
    :return: List of sorted and filtered section directory names.
    """

    # Define a regex pattern to filter section directory names
    pattern = r's\d+_g\d+'
    regex_pattern = re.compile(pattern)

    # Use glob to filter the section directory names
    dirs = glob.glob(str(Path(sections_dir) / "*"))

    # Filter and sort the matching section directory names
    sorted_dirs = sorted([dir_name for dir_name in dirs
                          if os.path.isdir(dir_name) and regex_pattern.match(Path(dir_name).name)],
                         key=lambda name: int(Path(name).name.split('_')[0].strip('s')))

    return sorted_dirs if sorted_dirs else None


def plot_flow_components(fine_flow, xy, title_prefix, transpose=False):
    """
    Plot horizontal and vertical components of a fine flow.

    Args:
        fine_flow (dict): Dictionary containing flow components.
        xy (str): Key in fine_flow to access the data.
        title_prefix (str): Prefix for the plot titles.
        transpose (bool): Whether to transpose the data before plotting.
    """
    if xy in fine_flow.keys():
        # Create a new figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Plot horizontal component in the first subplot
        data = fine_flow[xy][0, ...].T if transpose else fine_flow[xy][0, ...]
        cax1 = ax1.matshow(data, cmap='viridis')
        ax1.set_title(f'{title_prefix} - Horizontal Component')
        fig.colorbar(cax1, ax=ax1)

        # Plot vertical component in the second subplot
        data = fine_flow[xy][1, ...].T if transpose else fine_flow[xy][1, ...]
        cax2 = ax2.matshow(data, cmap='viridis')
        ax2.set_title(f'{title_prefix} - Vertical Component')
        fig.colorbar(cax2, ax=ax2)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Display the figure
        plt.show()


def plot_all_flow_components(fine_x, fine_y, xy, transpose=False):
    """
    Plot horizontal and vertical components of fine flows for both X and Y with a shared colorbar.

    Args:
        fine_x (dict): Dictionary containing X flow components.
        fine_y (dict): Dictionary containing Y flow components.
        xy (str): Key to access the data.
        transpose (bool): Whether to transpose the data before plotting.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(axes[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    im1 = None
    im3 = None

    # Plot fine X horizontal component
    if xy in fine_x.keys():
        data = fine_x[xy][0, ...].T if not transpose else fine_x[xy][0, ...]
        im1 = axes[0, 0].matshow(data, cmap='viridis')
        axes[0, 0].set_title('Fine Flow X - Horizontal Component')

    # Plot fine X vertical component
    if xy in fine_x.keys():
        data = fine_x[xy][1, ...].T if not transpose else fine_x[xy][1, ...]
        im2 = axes[0, 1].matshow(data, cmap='viridis')
        axes[0, 1].set_title('Fine Flow X - Vertical Component')

    # Plot fine Y horizontal component
    if xy in fine_y.keys():
        data = fine_y[xy][0, ...] if not transpose else fine_y[xy][0, ...].T
        im3 = axes[1, 0].matshow(data, cmap='viridis')
        axes[1, 0].set_title('Fine Flow Y - Horizontal Component')

    # Plot fine Y vertical component
    if xy in fine_y.keys():
        data = fine_y[xy][1, ...] if not transpose else fine_y[xy][1, ...].T
        im4 = axes[1, 1].matshow(data, cmap='viridis')
        axes[1, 1].set_title('Fine Flow Y - Vertical Component')

    # Add a shared colorbar
    if im1 is not None:
        fig.colorbar(im1, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.show()
    elif im3 is not None:
        fig.colorbar(im3, cax=cax, orientation='vertical')
        plt.tight_layout()
        plt.show()
    else:
        print(f'Nothing to plot')
        return


def load_set_from_json(file_path) -> Set[int]:
    try:
        # Read each line from the file and convert to a set of integers
        with open(file_path, 'r') as json_file:
            int_list = [int(line.strip()) for line in json_file]
        return set(int_list)
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return set()
    except IOError as e:
        print(f"Error: An I/O error occurred. {e}")
        return set()
    except ValueError as e:
        print(f"Error: A value error occurred. {e}")
        return set()


def load_mapped_npz(fp: str) -> Optional[MaskMap]:
    file_path = Path(fp)
    if not file_path.exists():
        logging.info(f"File '{file_path}' not found.")
        return None

    fmt_data = {}
    try:
        data = np.load(fp, allow_pickle=True)
        for key in data.keys():
            fmt_data[eval(key)] = data[key]
        return fmt_data

    except BadZipFile:
        logging.warning(f"File '{file_path}' is not a valid zip file.")
        return None

    except Exception as e:
        logging.warning(f"An unknown error occurred while loading '{file_path}': {e}")
        return None


def list_yaml_files(directory_path):
    search_pattern = os.path.join(directory_path, '*.yaml')
    yaml_files = glob.glob(search_pattern)
    return yaml_files


def list_stitched(dir_stitched: str) -> List[str]:

    dirs = glob.glob(str(Path(dir_stitched) / "*"))
    stitched_dirs = sorted([str(d) for d in dirs if Path(d).suffix == '.zarr'],
                           key=lambda name: int(Path(name).name.split('_')[0].strip('s')))
    return stitched_dirs


def plot_eval_ov(all_pk, tid_a, tid_b, path_plot: str):
    """Plots the results of the overlap evaluation."""
    x = list(all_pk.keys())
    y = list(all_pk.values())

    plt.figure(figsize=(10, 6))
    marker = '-' if len(x) > 200 else 'o-'
    plt.plot(x, y, marker)
    plt.xlabel('Section Number')
    plt.ylabel('MSSIM')
    plt.title(f'Overlap Regions for Tile Pair t{tid_a} and t{tid_b}')

    # Enable the grid
    plt.grid(True)

    # Set more x-tick marks with meaningful intervals
    min_x, max_x = min(x), max(x)
    interval = (max_x - min_x) // 10 or 1
    xticks = np.arange(min_x - min_x % interval, max_x + interval, interval)
    plt.xticks(xticks)

    print(f'Plot saved to: {path_plot}')
    plt.savefig(path_plot)
    plt.close()




def read_dict_from_yaml(file_path: UniPath) -> Dict[int, float]:
    """Reads a dictionary with integer keys and float values from a YAML file."""
    with open(str(file_path), 'r') as file:
        return yaml.safe_load(file)


def write_dict_to_yaml(file_path: str, data: Union[Dict[int, float], Iterable[int]]):
    """
    Write a dictionary with integer keys and float values, or an iterable of integers, to a YAML file.

    :param file_path: Path to the YAML file.
    :param data: Dictionary or iterable to be written to the file.
    """

    if isinstance(data, dict):
        # Convert NumPy types to native Python types, if any
        converted_data = {int(k): float(v) for k, v in data.items()}
    elif isinstance(data, Iterable):
        converted_data = [int(item) for item in data]
    else:
        raise ValueError(
            "The 'data' parameter must be a dictionary with integer keys and float values, or an iterable of integers."
        )
    try:
        with open(file_path, "w") as file:
            yaml.dump(converted_data, file, default_flow_style=False)
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")



def update_section_value(yaml_path, section_num, key_to_update, new_value):
    yaml = YAML(typ="rt")

    with open(yaml_path, 'r') as file:
        data = yaml.load(file)

    for section in data['sections']:
        if section['section_num'] == section_num:
            section[key_to_update] = new_value

    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)


def visualize_nns(cx_mat: ndarray, cy_mat: ndarray, path_fig: str) -> None:
    # Visualize the relative positions of horizontal and vertical nearest neighbors.
    # Every vector represents a tile *pair*.
    f, ax = plt.subplots(1, 2, figsize=(10, 5))

    sy, sx = np.shape(cx_mat[0, 0, ...])
    range_x = tuple(np.arange(sx))
    range_y = tuple(np.arange(sy) * -1)
    delta_lim = 0.5

    ax[0].quiver(range_x, range_y, cx_mat[0, 0, ...], cx_mat[1, 0, ...])
    ax[0].set_ylim(range_y[2]-delta_lim, range_y[0]+delta_lim)
    ax[0].set_xlim(range_x[0]-delta_lim, range_x[2]-delta_lim)
    ax[0].set_title('horizontal NNs')

    ax[1].quiver(range_x, range_y, cy_mat[0, 0, ...], cy_mat[1, 0, ...])
    ax[1].set_ylim(range_y[2]+delta_lim, range_y[0]+delta_lim)  # -1.5, 0.5
    ax[1].set_xlim(range_x[0]-delta_lim, range_x[2]+delta_lim)  # -0.5, 2.5
    ax[1].set_title('vertical NNs')

    plt.savefig(path_fig)
    plt.close()
    return


def slurm_sec_ranges(
        num_jobs: int,
        num_proc: int,
        start: int,
        end: int
) -> List[List[int]]:
    """
    Get optimal section ranges for SLURM job submission.

    Args:
        num_jobs (int): Maximum desired number of submitted jobs to SLURM.
        num_proc (int): Number of processors reserved for each job.
        start (int): Starting section number to be processed.
        end (int): Ending section number to be processed.

    Returns:
        List of pairs of section ranges to be processed.
    """

    sections_per_job = np.ceil((end - start) / num_proc)
    end_ext = start + int(sections_per_job) * num_proc

    batch_size = (end_ext - start) // num_proc
    step_size = int((num_proc * batch_size) / num_jobs)

    logging.debug(f'step size: {step_size}')

    a = np.arange(start, end, step_size)
    if a[-1] != end and a[-1] < end:
        a = np.append(a, end)

    sec_ranges = [[a, b] for a, b in zip(a[:-1], a[1:])]
    for i, pair in enumerate(sec_ranges[1:]):
        pair[0] += 1

    logging.debug(f"sec: {sec_ranges}")
    return sec_ranges


def split_list(nums: List[int], n: int) -> List[List[int]]:
    len_sub = (len(nums) + n - 1) // n
    return [nums[i * len_sub: (i + 1) * len_sub] for i in range(n)]


def get_neighbour_pairs(tid_map: np.ndarray) -> List[Tuple[int, int]]:
    """
    Generates pairs of neighboring elements in a 2D numpy array
     where each element is not equal to -1.
    
    Usually used to get tile-neighbors from tile_id_map.
    Args:
        tid_map (np.ndarray): A 2D numpy array.

    Returns:
        list: A list of tuples containing pairs of neighboring elements.
    """
    pairs = []
    for i in range(tid_map.shape[0]):
        for j in range(tid_map.shape[1]):
            if tid_map[i, j] != -1:
                if i + 1 < tid_map.shape[0] and tid_map[i + 1, j] != -1:
                    pairs.append((tid_map[i, j], tid_map[i + 1, j]))
                if j + 1 < tid_map.shape[1] and tid_map[i, j + 1] != -1:
                    pairs.append((tid_map[i, j], tid_map[i, j + 1]))
    return pairs


def invert_(ov_dict: Dict[int, List[Tuple[int, int]]]) -> Dict[Tuple[int, int], List[int]]:
    """
    Inverts a dictionary where the keys are integers and the values are lists of integer pairs.
    The resulting dictionary has the pairs as keys and lists of original keys as values.

    Args:
        ov_dict (Dict[int, List[Tuple[int, int]]]): The original dictionary to invert.
            Keys are integers and values are lists of integer pairs (tuples).

    Returns:
        Dict[Tuple[int, int], List[int]]: A new dictionary where each key is a pair from the
            original lists, and the corresponding value is a list of integers that were the keys
            in the original dictionary.

    Example:
        ov_dict = {
            1: [(2, 3), (4, 5)],
            2: [(2, 3), (6, 7)],
            3: [(8, 9)]
        }

        new_dict = invert_ov_dict(ov_dict)
        # new_dict will be:
        # {
        #     (2, 3): [1, 2],
        #     (4, 5): [1],
        #     (6, 7): [2],
        #     (8, 9): [3]
        # }
    """
    new_dict = {}
    for key, value_list in ov_dict.items():
        for pair in value_list:
            if pair not in new_dict:
                new_dict[pair] = []
            new_dict[pair].append(key)
    return new_dict



def plot_ssim_grid(ssim_values: Sequence[Tuple[float]],
                   shifts: List[Tuple[int, int]]):
    """Creates a scatter plot with color-mapped SSIM values

    :param ssim_values: tuple of mean SSIM values
    :param shifts: list of shift vectors used to offset the moving image
    """

    plt.scatter(*zip(*shifts), c=ssim_values, cmap='viridis', marker='o')
    plt.colorbar(label='SSIM Value')
    plt.xlabel('X Displacement')
    plt.ylabel('Y Displacement')
    plt.title('Scatter Plot of SSIM Values on Shift Grid')
    plt.show()
    return


def plot_refined_field(
        data: Dict[Tuple[int, int], float],
        path_plot: Optional[str] = None,
        show_plot: bool = False
) -> None:

    # Extract x and y coordinates from keys
    x_values = [coord[0] for coord in data.keys()]
    y_values = [coord[1] for coord in data.keys()]

    # Extract color values from dictionary values
    colors = list(data.values())

    # Plot the scatter plot
    plt.scatter(x_values, y_values, c=colors, cmap='viridis')
    plt.colorbar(label='Values')
    plt.grid(True)  # Show grid based on input coordinates
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot with Grid')

    if path_plot is not None:
        if Path(path_plot).parent.exists():
            plt.savefig(path_plot)

    if show_plot:
        plt.show()
    return


def get_pyramid(levels=3, max_ext=50, stride=10) -> List[Tuple[int, int]]:
    # Define pyramid of search parameters
    params = [(max_ext // N, stride // N) for N in range(1, levels + 1)]
    params = [tup for tup in params if 0 not in tup]
    return params


def plot_refined_grid(
        interp_data: Tuple[GridXY, GridXY],
        path_plot: Optional[str],
        show_plot=False
):

    # Plot the filled contour plot
    (x, y, z), (gx, gy, gz) = interp_data

    plt.contourf(gx, gy, gz, cmap='viridis')
    plt.colorbar(label='Inaccuracy [fct. of SSIM]')
    plt.scatter(x, y, c=z, cmap='viridis', edgecolors='k', linewidth=0.5)
    plt.xlabel('Coarse offset X [pix]')
    plt.ylabel('Coarse offset Y [pix]')
    plt.title('Seam inaccuracy in coarse offset space')

    if show_plot:
        plt.show()

    if path_plot is not None:
        if Path(path_plot).parent.exists():
            plt.savefig(path_plot)

    plt.close()
    return


def interp_coarse_grid(
        coarse_grid_xy: Tuple[np.ndarray, np.ndarray],
        coarse_grid_z: List[float],
) -> Tuple[Vector, Tuple[GridXY, GridXY]]:

    cgx, cgy = coarse_grid_xy
    cgz = np.array(coarse_grid_z)
    cgx, cgy, cgz = [a.flatten() for a in (cgx, cgy, cgz)]

    # Fine coarse offset grid
    fgx, fgy = np.meshgrid(
        np.linspace(min(cgx), max(cgx), 100),
        np.linspace(min(cgy), max(cgy), 100)
    )

    # Create CloughTocher2DInterpolator instance
    interp = CloughTocher2DInterpolator((cgx, cgy), cgz)

    # Perform interpolation on the finer grid
    fgz = interp(fgx, fgy)

    # Find the index of the smallest value in fine grid data
    min_index = np.argmin(fgz)

    # Convert the index to 2D coordinates
    min_index_2d = np.unravel_index(min_index, fgz.shape)
    min_coord = (int(np.round(fgx[min_index_2d])),
                 int(np.round(fgy[min_index_2d])))

    # For plotting purposes
    coarse_grid_xyz = (cgx, cgy, cgz)
    fine_grid_xyz = (fgx, fgy, fgz)
    refined_data = coarse_grid_xyz, fine_grid_xyz

    logging.info(f'estimated offset: {min_coord}')
    return min_coord, refined_data



if __name__ == "__main__":
    pass
    # tst_locate_inf_vals()
    # tst_reg_pair()
    # tst_section_registered()
    # num_of_reg_sections()
    # tst_register_tile_pair()
