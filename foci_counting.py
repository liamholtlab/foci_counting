from PySide2.QtWidgets import QApplication
from pathlib import Path
from magicgui import magicgui
from PySide2.QtWidgets import QMessageBox
from helpers import (
    IJ_threshold_minimum,
    foci_thresh,
    draw_label_on_image,
    measure_nuclei_from_mask,
    segment_nuclei_th,
    segment_nuclei_stardist,
    threshold_methods,
    vol_threshold_methods,
    segmentation_methods,
    save_foci_props,
    save_nuclei_props,
    load_settings,
    save_settings,
    measure_volume
)
from skimage import (
    io,
    filters,
    measure,
    segmentation,
    morphology,
    exposure
)
import matplotlib.pyplot as plt
import numpy as np
from nd2reader import ND2Reader
from scipy import ndimage
import pandas as pd
import glob
import os
import traceback
from tifffile import imread, imwrite, TiffFile

settings_file = "foci_counting.pkl"


def process_files(input_dir,
                  input_mask_dir,
                  output_dir,
                  nucleus_ch=1,
                  foci_ch=3,
                  intensity_ch=2,
                  bit_depth=12,
                  file_type='nd2',
                  segmentation_method='StarDist',
                  rescale_factor=0.5,
                  saturate_perc=6,
                  sm_radius=4,
                  cl_radius=2,
                  seed_distance=35,
                  th_method="FoCo",
                  foco_sm_r=0,
                  foco_bk_r=0,
                  int_cutoff=0.5,
                  volume_th_method='yen',
                  z_microns_per_voxel=0.5,
                  xy_microns_per_voxel=0.1342,
                  home_z_level=6,
                  expand_roi_px=5,
                  min_z_level=1,
                  max_z_level=11,
                  max_px_valid=10):

    # label all foci with area greater than 100px, for helping estimate size cutoff
    foci_min_area_for_label = 100

    # Threshold used for FoCo method
    custom_thresh = 2 ** bit_depth * int_cutoff

    # Type conversion map
    type_map = {8: 'uint8', 12: 'uint16', 16: 'uint16', 32: 'uint32'}

    # StarDist Model, if using...
    if segmentation_method == 'StarDist':
        from stardist.models import StarDist2D
        sd_model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # create the directories in the output for saving the check images
    for dir_ in ["segmentation", "foci", "volume"]:
        dir_ = os.path.join(output_dir, dir_)
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # Data structures for the output data frames: all foci data, and summary nucleus data
    # Note: summary nucleus data will include zero counts for nuclei without any foci (not included in foci data)
    foci_data = pd.DataFrame()
    nucleus_data = pd.DataFrame()

    # Read in nd2 or tif files from input directory
    img_files = glob.glob(os.path.join(input_dir, f"*.{file_type}"))

    if file_type == 'tif':
        # remove any files that end in _mask.tif from the list, case in-sensitive
        img_files = [f for f in img_files if not f.lower().endswith("_mask.tif")]

    if len(img_files) == 0:
        # Often, this happens when the user enters a mistake in the path
        return pd.DataFrame(), pd.DataFrame(), f"No {file_type} files found in the path: '{input_dir}'"
    else:
        file_count = len(img_files)

    # Process each file, each FOV
    for img_file in img_files:
        file_root = os.path.splitext(os.path.split(img_file)[1])[0]
        print(f"Processing {file_root}")

        if file_type == 'nd2':
            images = ND2Reader(img_file)
            print(images.sizes)

            if 'z' in images.sizes:
                z_levels = True
                bundle = 'czyx'
            else:
                z_levels = False
                bundle = 'cyx'

            if 'v' in images.sizes:
                n_fov = images.sizes['v']
                bundle = 'v' + bundle
            else:
                n_fov = 1

            if not ('c' in images.sizes and 'y' in images.sizes and 'x' in images.sizes):
                print(f"Error: unexpected shape of input image: {images.sizes}")
                continue

            images.bundle_axes = bundle
            images = images[0]  # ND2Reader adds an extra dimension to the beginning with bundle_axes

        else:
            images = imread(img_file)  # io.imread(img_file)
            n_fov = 1

            # Sort out if the z or the color is first, we want color first
            tif_ = TiffFile(img_file)
            num_color_ch = tif_.imagej_metadata['channels']

            if len(images.shape) == 4:
                z_levels = True
                num_z_levels = tif_.imagej_metadata['slices']

                if images.shape[0] == num_z_levels and images.shape[1] == num_color_ch:
                    images = images.transpose(1, 0, 2, 3)

            elif len(images.shape) == 3:
                z_levels = False

                if images.shape[0] > num_color_ch and images.shape[2] == num_color_ch:
                    images = images.transpose(2, 0, 1)
            else:
                print(f"Error: unexpected shape of input image: {images.shape}")
                continue

        print("n_fov:", n_fov)

        # Check if there is a mask file for the current image: this is only allowed for single FOV images
        mask_file = ""
        if segmentation_method == 'LabeledMask':
            if n_fov > 1:
                print(f"Error: LabeledMask method only works with single FOV images, using 'StarDist' for segmentation.")
                segmentation_method = 'StarDist'
            else:
                # select the mask file that matches the current image file: <filename>_mask.tif or <filename>_MASK.tif
                # If the mask directory is different from the input directory, also can be <filename>.tif
                mask_files = (os.path.join(input_mask_dir, f"{file_root}_mask.tif"),
                              os.path.join(input_mask_dir, f"{file_root}_MASK.tif"))
                if file_type != 'tif' or not os.path.samefile(input_dir, input_mask_dir):
                    mask_files = mask_files + (os.path.join(input_mask_dir, f"{file_root}.tif"), )

                for file_ in mask_files:
                    if os.path.exists(file_):
                        mask_file = file_
                        break
                if mask_file == "":
                    print(f"Error: No mask file found for {file_root}, using 'StarDist' for segmentation.")
                    segmentation_method = 'StarDist'
                else:
                    print(f"Using mask file: {mask_file}")

        for i in range(n_fov):
            if n_fov > 1:
                fov = images[i]
            else:
                fov = images

            # z-project the dapi channel
            if z_levels:
                dapi_img_stack = fov[nucleus_ch]
                dapi_img = np.max(dapi_img_stack, axis=0)
            else:
                dapi_img = fov[nucleus_ch]

            if segmentation_method == 'Thresholding':
                # segment nuclei using thresholding and watershed
                df, labeled_nuclei, nucleus_img_overlay = segment_nuclei_th(dapi_img,
                                                                            saturate_perc,
                                                                            sm_radius,
                                                                            cl_radius,
                                                                            seed_distance)
            elif segmentation_method == 'StarDist':
                # segment with stardist
                df, labeled_nuclei, nucleus_img_overlay = segment_nuclei_stardist(dapi_img,
                                                                                  sd_model,
                                                                                  rescale_factor)
            elif segmentation_method == 'LabeledMask':
                labeled_nuclei = imread(mask_file)
                df, nucleus_img_overlay = measure_nuclei_from_mask(dapi_img, labeled_nuclei)
            else:
                print(f"Error: unknown threshold method {th_method}, using 'StarDist'")
                df, labeled_nuclei, nucleus_img_overlay = segment_nuclei_stardist(dapi_img,
                                                                                  sd_model,
                                                                                  rescale_factor)
            # save the image overlay for checking segmentation results
            plt.imsave(os.path.join(output_dir, "segmentation", f"{file_root}_v{i}.png"), nucleus_img_overlay)

            if z_levels:
                # Measure the volume using nuclei channel and the MASK from segmentation
                df_volume, volume_overlay_stack = measure_volume(dapi_img_stack, labeled_nuclei, volume_th_method,
                                                                 z_microns_per_voxel, xy_microns_per_voxel,
                                                                 home_z_level, expand_roi_px,
                                                                 min_z_level, max_z_level, max_px_valid)
                df = df.join(df_volume.set_index('label'), on='label', how='left')

                # save the image overlay stack for checking segmentation results at each z level
                io.imsave(os.path.join(output_dir, "volume", f"{file_root}_v{i}.tif"), volume_overlay_stack)

            # z-project the intensity channel
            if intensity_ch >= 0:
                if z_levels:
                    int_img_stack = fov[intensity_ch]
                    int_img = np.mean(int_img_stack, axis=0)
                else:
                    int_img = fov[intensity_ch]

                # Calculate mean intensity of the background (for calculating CTCF)
                th = filters.threshold_otsu(int_img)
                mean_bk = np.mean(int_img[int_img < th])

                # Get mean intensity in each nucleus, join with original df
                df_int = pd.DataFrame(measure.regionprops_table(labeled_nuclei,
                                                                int_img,
                                                                properties=['label', 'mean_intensity']))
                df = df.join(df_int.set_index('label'), on='label', how='left')
                df.rename({'mean_intensity': f'mean_intensity_ch{intensity_ch + 1}'}, axis=1, inplace=True)

                # CTCF
                df[f'background_intensity_ch{intensity_ch + 1}'] = mean_bk
                df[f'CTCF_ch{intensity_ch + 1}'] = df['area'] * (
                        df[f'mean_intensity_ch{intensity_ch + 1}'] - df[f'background_intensity_ch{intensity_ch + 1}'])
            else:
                int_img = None

            # z-project the foci channel (max project and avg project)
            if foci_ch >= 0:
                if z_levels:
                    h2ax_img_stack = fov[foci_ch]
                    h2ax_img = np.max(h2ax_img_stack, axis=0)
                    h2ax_img_avgpr = np.mean(h2ax_img_stack, axis=0)
                else:
                    h2ax_img = fov[foci_ch]
                    h2ax_img_avgpr = fov[foci_ch]

                # Calculate mean intensity of the background (for calculating CTCF) using average projection
                th = filters.threshold_otsu(h2ax_img_avgpr)
                mean_bk = np.mean(h2ax_img_avgpr[h2ax_img_avgpr < th])

                # Get mean intensity in each nucleus, join with original df
                df_int = pd.DataFrame(measure.regionprops_table(labeled_nuclei,
                                                                h2ax_img_avgpr,
                                                                properties=['label', 'mean_intensity']))
                df = df.join(df_int.set_index('label'), on='label', how='left')
                df.rename({'mean_intensity': f'mean_intensity_ch{foci_ch+1} (foci)'}, axis=1, inplace=True)

                # CTCF
                df[f'background_intensity_ch{foci_ch+1} (foci)'] = mean_bk
                df[f'CTCF_ch{foci_ch+1} (foci)'] = df['area'] * (
                        df[f'mean_intensity_ch{foci_ch+1} (foci)'] - df[f'background_intensity_ch{foci_ch+1} (foci)'])

                # identify H2AX foci
                # Max of the image - check bit depth
                img_max = np.max(h2ax_img)
                if img_max > 2 ** bit_depth - 1:
                    print(f"Warning: image max is {img_max}, using {bit_depth} bit depth")

                h2ax_img = exposure.rescale_intensity(h2ax_img,
                                                      out_range=(0, 2 ** bit_depth - 1)).astype(type_map[bit_depth])
                h2ax_img_uint8 = exposure.rescale_intensity(h2ax_img, out_range=(0, 255)).astype('uint8')

                if th_method == 'minimum':
                    # use to 8-bit, otherwise the minimum threshold algorithm has trouble finding 2 peaks
                    h2ax_th = IJ_threshold_minimum(h2ax_img_uint8)
                    h2ax_mask = h2ax_img_uint8 > h2ax_th
                elif th_method == 'yen':
                    res = morphology.white_tophat(h2ax_img, morphology.disk(1))
                    h2ax_th = filters.threshold_yen(h2ax_img - res)
                    h2ax_mask = (h2ax_img - res) > h2ax_th
                elif th_method == 'FoCo':
                    h2ax_mask = foci_thresh(h2ax_img, foco_sm_r, foco_bk_r, custom_thresh)
                else:
                    print(f"Error: unknown threshold method {th_method}, using 'FoCo'")
                    h2ax_mask = foci_thresh(h2ax_img, foco_sm_r, foco_bk_r, custom_thresh)

                h2ax_mask = ndimage.binary_fill_holes(h2ax_mask)
                labeled_h2ax = measure.label(h2ax_mask)

                h2ax_image_label_overlay = segmentation.mark_boundaries(h2ax_img_uint8,
                                                                        labeled_h2ax,
                                                                        color=[1, 0, 0],
                                                                        mode='inner')
                h2ax_image_label_overlay = segmentation.mark_boundaries(h2ax_image_label_overlay,
                                                                        labeled_nuclei,
                                                                        color=[0, 1, 0],
                                                                        mode='inner')

                # Count foci for each nucleus
                foci_counts = []
                for row in df.iterrows():
                    coords = row[1]['coords']
                    rr, cc = list(zip(*coords))

                    # mask out any foci outside current nucleus, then threshold and count
                    mult_img = np.zeros_like(h2ax_mask)
                    mult_img[rr, cc] = True
                    nucleus_h2ax_mask = h2ax_mask * mult_img

                    # label and count
                    nucl_labeled_h2ax = measure.label(nucleus_h2ax_mask)
                    foci_df = pd.DataFrame(measure.regionprops_table(nucl_labeled_h2ax,
                                                                     h2ax_img,
                                                                     properties=['label',
                                                                                 'area',
                                                                                 'mean_intensity',
                                                                                 'centroid'
                                                                                 ]))
                    for foci_row in foci_df[foci_df['area'] > foci_min_area_for_label].iterrows():
                        # label all large foci with their area on the check image
                        draw_label_on_image(h2ax_image_label_overlay,
                                            int(foci_row[1]['centroid-0']),
                                            int(foci_row[1]['centroid-1']),
                                            f"{round(foci_row[1]['area'], 0)}",
                                            text_color=[0, 0, 1])
                    # Save the foci areas + count for each nucleus, and add to main data frame
                    if len(foci_df) > 0:
                        foci_df.rename({'area': 'foci_area',
                                        'mean_intensity': 'foci_mean_intensity'}, axis=1, inplace=True)
                        foci_df.drop(['label', 'centroid-0', 'centroid-1'], axis=1, inplace=True)
                        foci_df['file'] = file_root
                        foci_df['fov'] = i
                        foci_df['nucleus_label'] = row[1]['label']
                        foci_df['foci_count'] = len(foci_df)
                        foci_data = pd.concat([foci_data, foci_df], axis=0, ignore_index=True)

                    # Save the foci count separately to join with the nucleus df
                    foci_counts.append([row[1]['label'], len(foci_df)])

                # SAVE FOCI IMAGE FOR CHECKING
                plt.imsave(os.path.join(output_dir, "foci", f"{file_root}_v{i}.png"), h2ax_image_label_overlay)

                # add foci counts to the nucleus df
                foci_counts_df = pd.DataFrame(foci_counts, columns=['label', 'foci_count'])
                df = df.join(foci_counts_df.set_index('label'), on='label', how='left')

            # add file and fov to the nucleus df, drop unwanted columns and concat to full nucleus df
            df['file'] = file_root
            df['fov'] = i
            df.rename({'label': 'nucleus_label',
                       'area': 'nucleus_area',
                       'solidity': 'nucleus_solidity'}, axis=1, inplace=True)
            df.drop(['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'coords', 'centroid-0', 'centroid-1'], axis=1, inplace=True)
            nucleus_data = pd.concat([nucleus_data, df], axis=0, ignore_index=True)

    # Fix up the return data
    if len(nucleus_data) > 0:
        # Place file, fov and label columns first
        cols = nucleus_data.columns.tolist()
        cols.insert(0, cols.pop(cols.index('file')))
        cols.insert(1, cols.pop(cols.index('fov')))
        cols.insert(2, cols.pop(cols.index('nucleus_label')))
        nucleus_data = nucleus_data[cols]

    if len(foci_data) > 0:
        # Re-order columns
        foci_data = foci_data[['file',
                               'fov',
                               'nucleus_label',
                               'foci_count',
                               'foci_area',
                               'foci_mean_intensity']]
        msg = f"Done!  {file_count} files processed."
    else:
        msg = f"{file_count} files processed, but no foci found.  Please check input parameters."

    print("Done...")
    return nucleus_data, foci_data, msg


@magicgui(
    Input_Directory={"label": "Input directory (nd2/tif)", "mode": "d"},
    Input_Mask_Directory={"label": "Nuclei mask directory (tif)", "mode": "d"},
    Output_Directory={"label": "Output directory", "mode": "d"},
    Bit_depth={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [8, 12, 16, 32],
    },
    File_type={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": ['nd2', 'tif'],
    },
    # Stack_order={
    #     "widget_type": "RadioButtons",
    #     "orientation": "horizontal",
    #     "choices": ['vczyx', 'vcyx', 'czyx', 'cyx'],
    # },
    Segmentation_method={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": segmentation_methods,
    },
    StarDist_Rescale_factor={"min": 0, "max": 1.0},
    #CE_pixel_saturation={"label": "CE px saturation (%)", "min": 0, "max": 100},
    #Smoothing_radius={"min": 1, "max": 10},
    #Closing_radius={"min": 1, "max": 10},
    #WS_seed_distance={"label": "WS seed distance", "min": 1, "max": 1000},
    Volume_Threshold_Method={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": vol_threshold_methods,
    },
    Z_microns_per_voxel={"min": 0.01, "step": 0.01},
    XY_microns_per_voxel={"min": 0.0001, "step": 0.0001},
    Home_z_level={"min": 1, "step": 1},
    z_level_range={"label": "Z levels min/max"},
    Expand_ROI_px={"min": 0, "step": 1},
    Max_px_valid={"min": 0, "step": 1},
    Foci_Threshold_Method={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": threshold_methods,
    },
    Foco_sm_size={"label": "AdpM filter size (FoCo)", "min": 0, "max": 100},
    Foco_bk_radius={"label": "Opening radius (FoCo)", "min": 0, "max": 100},
    Intensity_cutoff={"label": "Intensity cutoff (FoCo)", "min": 0, "max": 1.0},
    Nucleus_area_range={"label": "Nucleus area min/max (px)"},
    Nucleus_min_solidity={"min": 0, "max": 1.0},
    Foci_max_area={"label": "Foci max area (px)", "min": 0, "max": 1000000},
    Count_Foci={"widget_type": "PushButton"},
    call_button="Re-filter data"
)
def count_foci_widget(
    Input_Directory=Path("."),
    Mask_directory_same_as_Input=True,
    Input_Mask_Directory=Path("."),
    Output_Directory=Path("."),
    Bit_depth=12,
    File_type='nd2',
    #Stack_order='vczyx',
    Nucleus_Channel=2,
    Segmentation_method="StarDist",
    StarDist_Rescale_factor=0.5,
    #CE_pixel_saturation=6,
    #Smoothing_radius=4,
    #Closing_radius=2,
    #WS_seed_distance=35,
    Volume_Threshold_Method='otsu',
    Z_microns_per_voxel=0.5,
    XY_microns_per_voxel=0.1342,
    Home_z_level=6,
    z_level_range=(1, 11),
    Expand_ROI_px=5,
    Max_px_valid=10,
    Intensity_channel=3,
    Foci_Channel=4,
    Foci_Threshold_Method="FoCo",
    Foco_sm_size=0,
    Foco_bk_radius=0,
    Intensity_cutoff=0.5,
    Foci_max_area=250,
    Nucleus_area_range=(1500, 10000),
    Nucleus_min_solidity=0.92,
    Count_Foci=True
    # *NOTE* channel numbers are 1-based, and Intensity_channel set to 0 means no intensity channel measurement
):
    try:
        print(f'Filtering data with new cutoffs.  Old (filtered) file will be overwritten.')

        if Mask_directory_same_as_Input:
            Input_Mask_Directory = Input_Directory

        # Save parameter settings to file
        save_settings(settings_file, count_foci_widget)

        # Re-load full data frames, including all nuclei and all foci data (no filtering)
        if os.path.exists(os.path.join(Output_Directory, "foci_data.txt")):
            foci_df = pd.read_csv(os.path.join(Output_Directory, "foci_data.txt"), sep='\t')
        else:
            foci_df = pd.DataFrame()

        if os.path.exists(os.path.join(Output_Directory, "nucleus_data.txt")):
            nucleus_df = pd.read_csv(os.path.join(Output_Directory, "nucleus_data.txt"), sep='\t')
        else:
            nucleus_df = pd.DataFrame()

        # Apply filters, save final results
        apply_filters(foci_df,
                      nucleus_df,
                      os.path.join(Output_Directory, "final_results.txt"),
                      Foci_max_area,
                      Nucleus_area_range[0],
                      Nucleus_area_range[1],
                      Nucleus_min_solidity)

        # Save foci and nuclei props as histograms and scatter plots
        if len(foci_df) > 0:
            save_foci_props(foci_df,
                            os.path.join(Output_Directory, "Foci_properties.png"),
                            Foci_max_area)
        if len(nucleus_df) > 0:
            save_nuclei_props(nucleus_df,
                              os.path.join(Output_Directory, "Nucleus_properties.png"),
                              Nucleus_area_range[0],
                              Nucleus_area_range[1],
                              Nucleus_min_solidity)

    except Exception:
        error_message = traceback.format_exc()
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("An error occurred")
        msg_box.setInformativeText(error_message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()


def apply_filters(foci_df,
                  nucleus_df,
                  filename,
                  foci_max_area,
                  nucleus_min_area,
                  nucleus_max_area,
                  nucleus_min_solidity):

    # Drop foci by area cutoff, and re-count
    if len(foci_df) > 0:
        filtered_foci_df = foci_df[foci_df['foci_area'] < foci_max_area].copy()
        count_df = pd.DataFrame(filtered_foci_df.groupby(['file', 'fov', 'nucleus_label']).size(),
                                columns=['foci_count_r'])

        # Update foci counts in the nucleus table
        filtered_nucleus_df = nucleus_df.join(count_df, on=['file', 'fov', 'nucleus_label'], how='left')
        filtered_nucleus_df['foci_count'] = filtered_nucleus_df['foci_count_r'].fillna(0)
        filtered_nucleus_df.drop(['foci_count_r'], axis=1, inplace=True)
    else:
        filtered_nucleus_df = nucleus_df.copy()

    if len(filtered_nucleus_df) > 0:
        # Apply min/max area and solidity filters to the nucleus data and save final results
        filtered_nucleus_df = filtered_nucleus_df[(filtered_nucleus_df['nucleus_area'] > nucleus_min_area) &
                                                  (filtered_nucleus_df['nucleus_area'] < nucleus_max_area) &
                                                  (filtered_nucleus_df['nucleus_solidity'] > nucleus_min_solidity)]
        filtered_nucleus_df.to_csv(filename, sep='\t', index=False)


def count_foci():

    try:
        # Load params
        Input_Directory = count_foci_widget.Input_Directory.value
        Mask_directory_same_as_Input = count_foci_widget.Mask_directory_same_as_Input.value
        Input_Mask_Directory = count_foci_widget.Input_Mask_Directory.value
        Output_Directory = count_foci_widget.Output_Directory.value
        Nucleus_Channel = count_foci_widget.Nucleus_Channel.value
        Foci_Channel = count_foci_widget.Foci_Channel.value
        Intensity_channel = count_foci_widget.Intensity_channel.value
        Bit_depth = count_foci_widget.Bit_depth.value
        Threshold_Method = count_foci_widget.Foci_Threshold_Method.value
        Foco_sm_size = count_foci_widget.Foco_sm_size.value
        Foco_bk_radius = count_foci_widget.Foco_bk_radius.value
        Intensity_cutoff = count_foci_widget.Intensity_cutoff.value
        Nucleus_min_area = count_foci_widget.Nucleus_area_range.value[0]
        Nucleus_max_area = count_foci_widget.Nucleus_area_range.value[1]
        Nucleus_min_solidity = count_foci_widget.Nucleus_min_solidity.value
        Foci_max_area = count_foci_widget.Foci_max_area.value
        #CE_pixel_saturation = count_foci_widget.CE_pixel_saturation.value
        #Smoothing_radius = count_foci_widget.Smoothing_radius.value
        #Closing_radius = count_foci_widget.Closing_radius.value
        #WS_seed_distance = count_foci_widget.WS_seed_distance.value
        File_type = count_foci_widget.File_type.value
        Segmentation_method = count_foci_widget.Segmentation_method.value
        StarDist_Rescale_factor = count_foci_widget.StarDist_Rescale_factor.value
        Volume_Threshold_Method = count_foci_widget.Volume_Threshold_Method.value
        Z_microns_per_voxel = count_foci_widget.Z_microns_per_voxel.value
        XY_microns_per_voxel = count_foci_widget.XY_microns_per_voxel.value
        Home_z_level = count_foci_widget.Home_z_level.value
        Expand_ROI_px = count_foci_widget.Expand_ROI_px.value
        Min_z_level = count_foci_widget.z_level_range.value[0]
        Max_z_level = count_foci_widget.z_level_range.value[1]
        Max_px_valid = count_foci_widget.Max_px_valid.value

        if Mask_directory_same_as_Input:
            Input_Mask_Directory = Input_Directory

        # Save parameter settings to file
        save_settings(settings_file, count_foci_widget)

        # Execute foci finding

        # parameters for thresholding method of segmentation (placeholder, not used)
        CE_pixel_saturation = 6
        Smoothing_radius = 4
        Closing_radius = 2
        WS_seed_distance = 35

        # channel numbers are 1-based, fix for proper indexing
        Intensity_channel = Intensity_channel-1
        Nucleus_Channel = Nucleus_Channel-1
        Foci_Channel = Foci_Channel-1
        Home_z_level = Home_z_level-1
        Min_z_level = Min_z_level-1
        Max_z_level = Max_z_level-1

        nucleus_df, foci_df, msg = process_files(input_dir=Input_Directory,
                                                 input_mask_dir=Input_Mask_Directory,
                                                 output_dir=Output_Directory,
                                                 nucleus_ch=Nucleus_Channel,
                                                 foci_ch=Foci_Channel,
                                                 intensity_ch=Intensity_channel,
                                                 bit_depth=Bit_depth,
                                                 file_type=File_type,
                                                 segmentation_method=Segmentation_method,
                                                 rescale_factor=StarDist_Rescale_factor,
                                                 saturate_perc=CE_pixel_saturation,
                                                 sm_radius=Smoothing_radius,
                                                 cl_radius=Closing_radius,
                                                 seed_distance=WS_seed_distance,
                                                 th_method=Threshold_Method,
                                                 foco_sm_r=Foco_sm_size,
                                                 foco_bk_r=Foco_bk_radius,
                                                 int_cutoff=Intensity_cutoff,
                                                 volume_th_method=Volume_Threshold_Method,
                                                 z_microns_per_voxel=Z_microns_per_voxel,
                                                 xy_microns_per_voxel=XY_microns_per_voxel,
                                                 home_z_level=Home_z_level,
                                                 expand_roi_px=Expand_ROI_px,
                                                 min_z_level=Min_z_level,
                                                 max_z_level=Max_z_level,
                                                 max_px_valid=Max_px_valid)

        # Save full data frames, including all nuclei and all foci data (no filtering)
        if len(foci_df) > 0:
            foci_df.to_csv(os.path.join(Output_Directory, "foci_data.txt"),
                           sep='\t',
                           index=False)
            save_foci_props(foci_df,
                            os.path.join(Output_Directory, "Foci_properties.png"),
                            Foci_max_area)

        if len(nucleus_df) > 0:
            nucleus_df.to_csv(os.path.join(Output_Directory, "nucleus_data.txt"),
                              sep='\t',
                              index=False)
            save_nuclei_props(nucleus_df,
                              os.path.join(Output_Directory, "Nucleus_properties.png"),
                              Nucleus_min_area,
                              Nucleus_max_area,
                              Nucleus_min_solidity)

        # Apply filters, save final results
        if len(foci_df) > 0 or len(nucleus_df) > 0:
            apply_filters(foci_df,
                          nucleus_df,
                          os.path.join(Output_Directory, "final_results.txt"),
                          Foci_max_area,
                          Nucleus_min_area,
                          Nucleus_max_area,
                          Nucleus_min_solidity)

        # Inform user of results
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Finished")
        msg_box.setText(msg)
        msg_box.exec_()

    except Exception:
        error_message = traceback.format_exc()
        print(error_message)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("An error occurred")
        msg_box.setInformativeText(error_message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()


count_foci_widget.Count_Foci.clicked.connect(count_foci)

if __name__ == "__main__":

    load_settings(settings_file, count_foci_widget)

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    w = count_foci_widget.show()

    # Adjust the widths of some entry boxes
    w.Nucleus_Channel.width = 100
    w.Intensity_channel.width = 100
    w.Foci_Channel.width = 100
    w.Foci_Threshold_Method.width = 100
    w.Intensity_cutoff.width = 100
    w.Foco_sm_size.width = 100
    w.Foco_bk_radius.width = 100
    w.Nucleus_min_solidity.width = 100
    w.Foci_max_area.width = 100
    #w.CE_pixel_saturation.width = 100
    #w.Smoothing_radius.width = 100
    #w.Closing_radius.width = 100
    #w.WS_seed_distance.width = 100
    w.StarDist_Rescale_factor.width = 100
    w.Z_microns_per_voxel.width = 100
    w.XY_microns_per_voxel.width = 100
    w.Home_z_level.width = 100
    w.Expand_ROI_px.width = 100
    w.Max_px_valid.width = 100

    w.z_level_range.value_0.width = 100
    w.z_level_range.value_1.width = 100
    w.Nucleus_area_range.value_0.width = 100
    w.Nucleus_area_range.value_1.width = 100

    w.Nucleus_area_range.value_0.max = 1000000
    w.Nucleus_area_range.value_1.max = 1000000
    w.Nucleus_area_range.value_0.step = 10
    w.Nucleus_area_range.value_1.step = 10

    app.exec_()
