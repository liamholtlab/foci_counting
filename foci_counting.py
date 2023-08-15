from PyQt5.QtWidgets import QApplication
from pathlib import Path
from magicgui import magicgui
from qtpy.QtWidgets import QMessageBox
from helpers import (
    IJ_threshold_minimum,
    foci_thresh,
    draw_label_on_image,
    segment_nuclei,
    threshold_methods,
    save_foci_props,
    save_nuclei_props
)
from skimage import (
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


def process_files(input_dir,
                  output_dir,
                  nucleus_ch=1,
                  foci_ch=3,
                  bit_depth=12,
                  th_method="FoCo",
                  int_cutoff=0.5):

    # label all foci with area greater than 100px, for helping estimate size cutoff
    foci_min_area_for_label = 100

    # Threshold used for FoCo method
    custom_thresh = 2 ** bit_depth * int_cutoff

    # Type conversion map
    type_map = {8: 'uint8', 12: 'uint16', 16: 'uint16', 32: 'uint32'}

    # create the directories in the output for saving the check images
    for dir_ in ["segmentation", "foci"]:
        dir_ = os.path.join(output_dir, dir_)
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    # Data structures for the output data frames: all foci data, and summary nucleus data
    # Note: summary nucleus data will include zero counts for nuclei without any foci (not included in foci data)
    foci_data = pd.DataFrame()
    nucleus_data = []

    # Read in nd2 files from input directory
    nd2_files = glob.glob(os.path.join(input_dir, "*.nd2"))

    if len(nd2_files) == 0:
        # Commonly, this happens when the user enters a mistake in the path
        return pd.DataFrame(), f"No nd2 files found in the path: '{input_dir}'"
    else:
        file_count = len(nd2_files)

    # Process each file, each FOV
    for nd2_file in nd2_files:
        file_root = os.path.splitext(os.path.split(nd2_file)[1])[0]
        print(f"Processing {file_root}")

        images = ND2Reader(nd2_file)
        images.bundle_axes = 'vczyx'

        for i, fov in enumerate(images[0]):

            # z-project the dapi channel
            dapi_img_stack = fov[nucleus_ch]
            dapi_img = np.max(dapi_img_stack, axis=0)

            # segment nuclei using thresholding and watershed
            df, labeled_nuclei, nucleus_img_overlay = segment_nuclei(dapi_img)

            # save the image overlay for checking segmentation results
            plt.imsave(os.path.join(output_dir, "segmentation", f"{file_root}_v{i}.png"), nucleus_img_overlay)

            # z-project the foci channel
            h2ax_img_stack = fov[foci_ch]
            h2ax_img = np.max(h2ax_img_stack, axis=0)

            # identify H2AX foci by thresholding
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
                h2ax_mask = foci_thresh(h2ax_img, 0, 0, custom_thresh)
            else:
                print(f"Error: unknown threshold method {th_method}, using 'FoCo'")
                h2ax_mask = foci_thresh(h2ax_img, 0, 0, custom_thresh)

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
                    foci_df.rename({'area': 'foci_area', 'mean_intensity': 'foci_mean_intensity'}, axis=1, inplace=True)
                    foci_df.drop(['label', 'centroid-0', 'centroid-1'], axis=1, inplace=True)
                    foci_df['file'] = file_root
                    foci_df['fov'] = i
                    foci_df['nucleus_label'] = row[1]['label']
                    foci_df['foci_count'] = len(foci_df)
                    foci_data = pd.concat([foci_data, foci_df], axis=0, ignore_index=True)

                # Save the nucleus data as a single row, included foci count
                nucleus_data.append([file_root,
                                     i,
                                     row[1]['label'],
                                     row[1]['area'],
                                     row[1]['solidity'],
                                     len(foci_df)])

            # SAVE FOCI IMAGE FOR CHECKING
            plt.imsave(os.path.join(output_dir, "foci", f"{file_root}_v{i}.png"), h2ax_image_label_overlay)

    # Fix up the return data
    if len(nucleus_data) > 0:
        nucleus_data = pd.DataFrame(nucleus_data, columns=['file',
                                                           'fov',
                                                           'nucleus_label',
                                                           'nucleus_area',
                                                           'nucleus_solidity',
                                                           'foci_count'
                                                           ])
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
    Input_Directory={"label": "Input directory (nd2)", "mode": "d"},
    Output_Directory={"label": "Output directory", "mode": "d"},
    Bit_depth={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": [8, 12, 16, 32],
    },
    Threshold_Method={
        "widget_type": "RadioButtons",
        "orientation": "horizontal",
        "choices": threshold_methods,
    },
    Intensity_cutoff={"label": "Intensity cutoff (FoCo)", "min": 0, "max": 1.0},
    Nucleus_min_area={"label": "Nucleus min area (px)"},
    Nucleus_max_area={"label": "Nucleus max area (px)"},
    Nucleus_min_solidity={"min": 0, "max": 1.0},
    Foci_max_area={"label": "Foci max area (px)"},
    Count_Foci={"widget_type": "PushButton"},
    call_button="Re-filter data"
)
def count_foci_widget(
    Input_Directory=Path("."),
    Output_Directory=Path("."),
    Nucleus_Channel=1,
    Foci_Channel=3,
    Bit_depth=12,
    Threshold_Method="FoCo",
    Intensity_cutoff=0.5,
    Nucleus_min_area=1600,
    Nucleus_max_area=8000,
    Nucleus_min_solidity=0.92,
    Foci_max_area=250,
    Count_Foci=True
):
    print(f'Filtering data with new cutoffs.  Old (filtered) file will be overwritten.')

    # Re-load full data frames, including all nuclei and all foci data (no filtering)
    foci_df = pd.read_csv(os.path.join(Output_Directory, "foci_data.txt"), sep='\t')
    nucleus_df = pd.read_csv(os.path.join(Output_Directory, "nucleus_data.txt"), sep='\t')

    # Apply filters, save final results
    apply_filters(foci_df,
                  nucleus_df,
                  os.path.join(Output_Directory, "final_results.txt"),
                  Foci_max_area,
                  Nucleus_min_area,
                  Nucleus_max_area,
                  Nucleus_min_solidity)

    # Save foci and nuclei props as histograms and scatter plots
    save_foci_props(foci_df,
                    os.path.join(Output_Directory, "Foci_properties.png"),
                    Foci_max_area)
    save_nuclei_props(nucleus_df,
                      os.path.join(Output_Directory, "Nucleus_properties.png"),
                      Nucleus_min_area,
                      Nucleus_max_area,
                      Nucleus_min_solidity)


def apply_filters(foci_df,
                  nucleus_df,
                  filename,
                  foci_max_area,
                  nucleus_min_area,
                  nucleus_max_area,
                  nucleus_min_solidity):

    # Drop foci by area cutoff, and re-count
    filtered_foci_df = foci_df[foci_df['foci_area'] < foci_max_area].copy()
    count_df = pd.DataFrame(filtered_foci_df.groupby(['file', 'fov', 'nucleus_label']).size(),
                            columns=['foci_count_r'])

    # Update foci counts in the nucleus table
    filtered_nucleus_df = nucleus_df.join(count_df, on=['file', 'fov', 'nucleus_label'], how='left')
    filtered_nucleus_df['foci_count'] = filtered_nucleus_df['foci_count_r'].fillna(0)
    filtered_nucleus_df.drop(['foci_count_r'], axis=1, inplace=True)

    # Apply min/max area and solidity filters to the nucleus data and save final results
    filtered_nucleus_df = filtered_nucleus_df[(filtered_nucleus_df['nucleus_area'] > nucleus_min_area) &
                                              (filtered_nucleus_df['nucleus_area'] < nucleus_max_area) &
                                              (filtered_nucleus_df['nucleus_solidity'] > nucleus_min_solidity)]
    filtered_nucleus_df.to_csv(filename, sep='\t', index=False)


def count_foci():

    Input_Directory = count_foci_widget.Input_Directory.value
    Output_Directory = count_foci_widget.Output_Directory.value
    Nucleus_Channel = count_foci_widget.Nucleus_Channel.value
    Foci_Channel = count_foci_widget.Foci_Channel.value
    Bit_depth = count_foci_widget.Bit_depth.value
    Threshold_Method = count_foci_widget.Threshold_Method.value
    Intensity_cutoff = count_foci_widget.Intensity_cutoff.value
    Nucleus_min_area = count_foci_widget.Nucleus_min_area.value
    Nucleus_max_area = count_foci_widget.Nucleus_max_area.value
    Nucleus_min_solidity = count_foci_widget.Nucleus_min_solidity.value
    Foci_max_area = count_foci_widget.Foci_max_area.value

    try:
        nucleus_df, foci_df, msg = process_files(Input_Directory,
                                                 Output_Directory,
                                                 Nucleus_Channel,
                                                 Foci_Channel,
                                                 Bit_depth,
                                                 Threshold_Method,
                                                 Intensity_cutoff)

        # Save full data frames, including all nuclei and all foci data (no filtering)
        foci_df.to_csv(os.path.join(Output_Directory, "foci_data.txt"),
                       sep='\t',
                       index=False)
        nucleus_df.to_csv(os.path.join(Output_Directory, "nucleus_data.txt"),
                          sep='\t',
                          index=False)

        # Apply filters, save final results
        apply_filters(foci_df,
                      nucleus_df,
                      os.path.join(Output_Directory, "final_results.txt"),
                      Foci_max_area,
                      Nucleus_min_area,
                      Nucleus_max_area,
                      Nucleus_min_solidity)

        # Save foci and nuclei props as histograms and scatter plots
        save_foci_props(foci_df,
                        os.path.join(Output_Directory, "Foci_properties.png"),
                        Foci_max_area)
        save_nuclei_props(nucleus_df,
                          os.path.join(Output_Directory, "Nucleus_properties.png"),
                          Nucleus_min_area,
                          Nucleus_max_area,
                          Nucleus_min_solidity)

        # Inform user of results
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Finished")
        msg_box.setText(msg)
        msg_box.exec_()

    except Exception as err:
        error_message = traceback.format_exc()
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("An error occurred")
        msg_box.setInformativeText(error_message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()


count_foci_widget.Count_Foci.clicked.connect(count_foci)


if __name__ == "__main__":
    app = QApplication([])
    w = count_foci_widget.show()

    # Adjust the widths of some entry boxes
    w.Nucleus_Channel.width = 100
    w.Foci_Channel.width = 100
    w.Threshold_Method.width = 100
    w.Intensity_cutoff.width = 100
    w.Nucleus_min_area.width = 100
    w.Nucleus_max_area.width = 100
    w.Nucleus_min_solidity.width = 100
    w.Foci_max_area.width = 100

    app.exec_()