import numpy as np
from skimage import (
    filters,
    measure,
    feature,
    segmentation,
    morphology,
    exposure,
    transform
)
from scipy import ndimage
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from csbdeep.utils import normalize

threshold_methods = ["FoCo", "minimum", "yen"]
segmentation_methods = ['StarDist', 'Threshold']


def load_settings(filename, main_widget):
    try:
        fp = open(filename, "rb")
        settings = pickle.load(fp)
        fp.close()
        for widget in main_widget:
            if widget.name in settings:
                widget.value = settings[widget.name]
        return True
    except FileNotFoundError:
        return False
    except pickle.UnpicklingError:
        return False
    except EOFError:
        return False


def save_settings(filename, main_widget):
    try:
        fp = open(filename, "wb")
        settings = {}
        for widget in main_widget:
            settings[widget.name] = widget.value
        pickle.dump(settings, fp)
        fp.close()
        return True
    except FileNotFoundError:
        return False
    except pickle.PicklingError:
        return False


def segment_nuclei_stardist(img,
                            model,
                            scale_factor=0.5):

    labels, _ = model.predict_instances(normalize(transform.rescale(img, scale_factor)))
    labels = transform.resize(labels, img.shape, order=0, preserve_range=True)

    # clear those touching border
    labels = segmentation.clear_border(labels)

    # df is a data frame of identified nuclei
    props = ['label', 'area', 'solidity', 'bbox', 'coords', 'centroid']
    df = pd.DataFrame(measure.regionprops_table(labels, properties=props))

    # SAVE IMAGE FOR CHECKING - overlap nucleus area/solidity for each
    img_uint8 = exposure.rescale_intensity(img, out_range=(0, 255)).astype('uint8')
    image_label_overlay = segmentation.mark_boundaries(img_uint8,
                                                       labels,
                                                       color=[0, 1, 0],
                                                       mode='inner')
    for row in df.iterrows():
        label_ = f"{row[1]['label']}: {round(row[1]['area'], 0)}|{round(row[1]['solidity'], 2)}"
        draw_label_on_image(image_label_overlay,
                            int(row[1]['centroid-0']),
                            int(row[1]['centroid-1']),
                            label_,
                            text_color=[1, 1, 1])

    return df, labels, image_label_overlay


def segment_nuclei_th(img,
                      saturate_perc=6,
                      sm_radius=4,
                      closing_radius=2,
                      seed_distance=35):

    # saturate 6% of pixels (3% on top and 3% on bottom)
    p1, p2 = np.percentile(img, (saturate_perc/2, 100 - (saturate_perc/2)))
    dapi_img_proc = exposure.rescale_intensity(img, in_range=(p1, p2))

    # smoothing with median filter
    dapi_img_proc = filters.median(dapi_img_proc, morphology.disk(sm_radius))

    # threshold
    th = filters.threshold_otsu(dapi_img_proc)
    dapi_mask = dapi_img_proc > th

    # fill holes
    dapi_mask = ndimage.binary_fill_holes(dapi_mask)

    # close shapes: helps with nuclei not fully segmented
    dapi_mask = morphology.binary_closing(dapi_mask, footprint=morphology.disk(closing_radius))

    # watershed - separates touching:
    labeled_image = measure.label(dapi_mask)

    # (1) distance map
    distance = ndimage.distance_transform_edt(dapi_mask)
    distance = ndimage.median_filter(distance, 4)

    # (2) local maxima (seeds of watershed)
    peak_idx = feature.peak_local_max(distance,
                                      exclude_border=False,
                                      labels=labeled_image,
                                      min_distance=seed_distance)
    # (peak_local_max returns indices - create a mask)
    peak_mask = np.zeros_like(distance, dtype=bool)
    peak_mask[tuple(peak_idx.T)] = True
    markers, n = ndimage.label(peak_mask)

    # (3) watershed: fill from the local maxima markers
    labels = segmentation.watershed(-distance, markers, mask=dapi_mask)

    # clear those touching border
    labels = segmentation.clear_border(labels)

    # remove small objects
    labels = morphology.remove_small_objects(labels)

    # df is a data frame of identified nuclei
    props = ['label', 'area', 'solidity', 'bbox', 'coords', 'centroid']
    df = pd.DataFrame(measure.regionprops_table(labels, properties=props))

    # SAVE IMAGE FOR CHECKING - overlap nucleus area/solidity for each
    img_uint8 = exposure.rescale_intensity(img, out_range=(0, 255)).astype('uint8')
    image_label_overlay = segmentation.mark_boundaries(img_uint8,
                                                       labels,
                                                       color=[0, 1, 0],
                                                       mode='inner')
    for row in df.iterrows():
        label_ = f"{row[1]['label']}: {round(row[1]['area'], 0)}|{round(row[1]['solidity'], 2)}"
        draw_label_on_image(image_label_overlay,
                            int(row[1]['centroid-0']),
                            int(row[1]['centroid-1']),
                            label_,
                            text_color=[1, 1, 1])

    return df, labels, image_label_overlay


# Thresholding method, copied from ImageJ code
def IJ_threshold_minimum(img, nbins=256):
    def bimodal_test(y):
        modes = 0
        n = len(y)
        for k in range(1, n - 1):
            if y[k - 1] < y[k] and y[k + 1] < y[k]:
                modes += 1
                if modes > 2:
                    return False
        if modes == 2:
            return True
        return False

    ihisto, bins = np.histogram(img, bins=nbins)

    i = 0
    while not bimodal_test(ihisto):
        if i > 10000:
            print("Minimum: threshold not found after 10000 iterations.")
            return -1

        # smooth with 3 point running mean filter until histogram is bimodal
        ihisto = np.convolve(ihisto, [1 / 3, 1 / 3, 1 / 3], mode='same')
        i += 1

    # The threshold is the minimum between the two peaks
    for i in range(1, nbins - 1):
        if ihisto[i - 1] > ihisto[i] and ihisto[i + 1] >= ihisto[i]:
            return i

    print("Minimum: threshold not found after 10000 iterations.")
    return -1


def adpmedian(g, Smax):
    # SMAX must be an odd, positive integer greater than 1.
    Smax = int(Smax)
    if (Smax <= 1) or (Smax / 2 == round(Smax / 2)):
        print("SMAX must be an odd integer > 1.")
        return g

    # ADPMEDIAN Perform adaptive median filtering. The function was taken
    # from Gonzalez RC, Woods RE, Eddins SLU. Digital Image Processing Using
    # MATLAB. Pearson Prentice Hall; 2004, and converted to python.

    # F = ADPMEDIAN(G, SMAX) performs adaptive median filtering of
    # image G.  The median filter starts at size 3-by-3 and iterates up
    # to size SMAX-by-SMAX. SMAX must be an odd integer greater than 1.

    # Initial setup.
    f = np.zeros_like(g)
    alreadyProcessed = np.zeros_like(g).astype('bool')

    # Begin filtering.
    for k in range(3, Smax + 2, 2):
        footprint = morphology.rectangle(k, k)
        zmin = filters.rank.minimum(g, footprint)
        zmax = filters.rank.maximum(g, footprint)
        zmed = filters.rank.median(g, footprint)
        processUsingLevelB = (zmed > zmin) & (zmax > zmed) & ~alreadyProcessed

        zB = (g > zmin) & (zmax > g)
        outputZxy = processUsingLevelB & zB
        outputZmed = processUsingLevelB & ~zB

        f[outputZxy] = g[outputZxy]
        f[outputZmed] = zmed[outputZmed]

        alreadyProcessed = alreadyProcessed | processUsingLevelB

        if alreadyProcessed.all():
            break

    # Output zmed for any remaining unprocessed pixels. Note that this
    # zmed was computed using a window of size Smax-by-Smax, which is
    # the final value of k in the loop.
    f[~alreadyProcessed] = zmed[~alreadyProcessed]
    return f


def foci_thresh(distnuclcrop, sm_r, bk_r, thresh):
    # Applying adaptive median filter of size 3
    if sm_r > 0:
        distnuclcrop = adpmedian(distnuclcrop, sm_r)
        # distnuclcrop = filters.rank.median(distnuclcrop, morphology.disk(1))

    # Quantifying the foci image background by morphological opening. For the
    # morphological opening we used a structuring element disk. The radius of
    # the disk is a user-defined parameter, which designates a minimal foci radius in pixels
    if bk_r > 0:
        background = morphology.opening(distnuclcrop, morphology.disk(bk_r)) # radius = 4
        #background = morphology.white_tophat(distnuclcrop, morphology.disk(bk_r))  # radius = 1
    else:
        background = np.zeros_like(distnuclcrop)

    # Subtracting the image background
    distnuclcrop = distnuclcrop - background

    # Quantifying the Otsu's threshold
    level = filters.threshold_otsu(distnuclcrop)

    # Applying H-maxima transform with the Otsu's threshold as a parameter
    # BW = imextendedmax(distnuclcrop, level)
    BW = morphology.h_maxima(distnuclcrop, level)

    # Applying obtained foci mask to the foci image with subtracted background
    locmax = distnuclcrop * BW

    # Thresholding the obtained image with the user-defined parameter 'Minimal intensity of foci'
    locmaxthresh = locmax > thresh

    return locmaxthresh


def draw_label_on_image(img, r, c, label, text_color=[1, 1, 1], text_size=0.3, text_lw=1, label_w=130, label_h=10):
    if (r + label_h) >= len(img):
        r_pos = r - ((r + label_h) - len(img))
    else:
        r_pos = r
    if (c + label_w) >= len(img[0]):
        c_pos = c - ((c + label_w) - len(img[0]))
    else:
        c_pos = c
    cv2.putText(img,
                label,
                (int(c_pos), int(r_pos)),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                text_color,
                text_lw,
                cv2.LINE_AA)


def save_foci_props(df, filename, max_area):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    bin_max = np.percentile(df['foci_area'], 99.90)
    sns.histplot(data=df, x='foci_area', bins=np.linspace(0, bin_max, 50), stat='percent', ax=axs[0])
    axs[0].axvline(max_area, color='red')

    axs[1].scatter(df['foci_area'], df['foci_count'], s=2, alpha=0.5)
    axs[1].axvline(max_area, color='red')
    axs[1].set_xlabel('foci_area')
    axs[1].set_ylabel('(unfiltered) foci_count')

    fig.tight_layout()
    fig.savefig(filename)


def save_nuclei_props(df, filename, min_area, max_area, min_solidity):
    # Save histograms of nucleus Area, Solidity, and scatter of Area vs Solidity
    # Draw lines to indicate cutoff values
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    bin_max = np.percentile(df['nucleus_area'], 99.90)
    if max_area > bin_max:
        bin_max = np.max(df['nucleus_area'])
    sns.histplot(data=df, x='nucleus_area', bins=np.linspace(0, bin_max, 50), stat='percent', ax=axs[0])
    axs[0].axvline(min_area, color='red')
    axs[0].axvline(max_area, color='red')

    bin_min = np.percentile(df['nucleus_solidity'], 0.10)
    if min_solidity < bin_min:
        bin_min = np.min(df['nucleus_solidity'])
    sns.histplot(data=df, x='nucleus_solidity', bins=np.linspace(bin_min, 1, 50), stat='percent', ax=axs[1])
    axs[1].axvline(min_solidity, color='red')

    axs[2].scatter(df['nucleus_area'], df['nucleus_solidity'], s=2, alpha=0.5)
    axs[2].set_xlim(0, bin_max)
    axs[2].set_ylim(bin_min, 1)
    axs[2].axvline(min_area, color='red')
    axs[2].axvline(max_area, color='red')
    axs[2].axhline(min_solidity, color='red')
    axs[2].set_xlabel('nucleus_area')
    axs[2].set_ylabel('nucleus_solidity')

    fig.tight_layout()
    fig.savefig(filename)