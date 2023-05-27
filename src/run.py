import sys
from intersect_lines import intersectLines
from load_images import load_images
from generate_templates import generate_templates
from generate_weight_matrix import generate_matrix
from stitches_processing import do_stitching, process_st_candidates

from findmaxima2d import find_local_maxima, find_maxima

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
import skimage.io
import skimage.transform
from skimage.feature import match_template

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from math import floor, ceil, dist

from scipy.spatial.distance import cdist

import json

q = 0.992
template_b_w = 0.8
ntol = 1.8
min_st_th = 1.6
clustering_th_c = 1/6
stitching_th_c = 1/5

cfg_fn = "base.txt"

with open(cfg_fn, "r") as file:
    base = file.readline().strip()

def declination(endpoints):
    return np.degrees(np.arctan((endpoints[0][0] - endpoints[1][0])/(endpoints[0][1] - endpoints[1][1])))

# Get the output file path
output_fn = sys.argv[1]

# Check if the -v option is provided
verbose = False
if len(sys.argv) > 2 and sys.argv[2] == "-v":
    verbose = True
    # Get the image filenames
    image_filenames = sys.argv[3:]
else:
    image_filenames = sys.argv[2:]

out = []

# for image in load_images(filenames = images):
for p, image in enumerate(load_images(filenames=image_filenames, base = base)):
    w = image.shape[1]
    h = image.shape[0]

    clustering_th = h * clustering_th_c
    stitching_th = h * stitching_th_c

    # preparation of templates for an image with given size
    inc_template_shape = (round(h/10)*2+1, round(h/10)*3+1)
    st_template_shape = (round(h / 10) * 2 + 1, round(h / 10) * 2 + 1)
    inc_result_shape = (h - inc_template_shape[0] + 1, w - inc_template_shape[1] + 1)
    st_result_shape = (h - st_template_shape[0] + 1, w - st_template_shape[1] + 1)
    inc_templates = (np.ones(inc_template_shape),
            np.ones(inc_template_shape))
    inc_templates[0][int(floor(inc_template_shape[0]/4))+1:-int(ceil(inc_template_shape[0]/4)),int(floor(inc_template_shape[1]/4))+1:] = 0
    inc_templates[1][int(floor(inc_template_shape[0]/4))+1:-int(ceil(inc_template_shape[0]/4)),:-int(ceil(inc_template_shape[1]/4))] = 0
    inc_templates_center = ((inc_template_shape[0]//2, int(floor(inc_template_shape[1]/4))+1), (inc_template_shape[0]//2, inc_template_shape[1] - int(floor(inc_template_shape[1]/4))))

    template_weights = [np.linspace(1, 0, inc_result_shape[1]).reshape(1, inc_result_shape[1]).repeat(inc_result_shape[0], axis=0), 
                        np.linspace(0, 1, inc_result_shape[1]).reshape(1, inc_result_shape[1]).repeat(inc_result_shape[0], axis=0)]

    inc_endpoints = []

    if verbose:
        fig, ax = plt.subplots(ncols=3,nrows=4, figsize=(20, 10))
        ax[0,0].imshow(image, cmap="gray")
        ax[1,0].imshow(image, cmap="gray")

        ax[0,0].set_title("Original image (red channel)")
        ax[0,1].set_title("Left incision endpoint corelation")
        ax[0,2].set_title("Left incision endpoint corelation (thresholded)")
        ax[1,0].set_title("Detected incision and stitches overlayed")
        ax[1,1].set_title("Right incision endpoint corelation")
        ax[1,2].set_title("Right incision endpoint corelation (thresholded)")
        ax[2,0].set_title("Upper stitch endpoint corelation (weighed) histogram")
        ax[2,1].set_title("Upper stitch endpoint corelation (weighed)")
        ax[2,2].set_title("Upper stitch endpoint corelation (weighed, thresholded)")
        ax[3,0].set_title("Lower stitch endpoint corelation (weighed) histogram")
        ax[3,1].set_title("Lower stitch endpoint corelation (weighed)")
        ax[3,2].set_title("Lower stitch endpoint corelation (weighed, thresholded)")

    # find incision endpoints
    for i, template in enumerate(inc_templates):
        result = match_template(image, template)*template_weights[i]    # match with endpoint template and weigh the result so that points on the sides are more important

        # find maximum and compute incision endpoint
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        endpoint = (int(y + inc_templates_center[i][0]), int(x + inc_templates_center[i][1]))
        inc_endpoints.append(endpoint)

        if verbose:
            ax[i,1].imshow(result)
            ax[i,2].imshow(result>(result.max()*0.95), cmap="gray")

    if verbose:
        line = LineCollection([np.fliplr(np.array(inc_endpoints))], colors='r', linewidths=2)
        ax[1,0].add_collection(line)

    gamma = -declination(inc_endpoints)
    k = (inc_endpoints[1][0] - inc_endpoints[0][0])/h                       # compute k in k*x + q of incision line
    wtha = inc_endpoints[0][0]/h - k*inc_endpoints[0][1]/w                  # compute points on the sides of the image so that line through them is the same as incision line
    wthb = inc_endpoints[1][0]/h + k * (w - inc_endpoints[1][1])/w

    st_templates = generate_templates(h, gamma)
    st_centers = st_templates[-1]           # get centers of templates
    st_templates = st_templates[0:-1]       # remove coordinates of centers from templates
    template_weights.append(generate_matrix(st_result_shape[0], st_result_shape[1], wtha, wthb, 4, 0.4, -0.065, 20)) # upper side stitches weights
    template_weights.append(generate_matrix(st_result_shape[0], st_result_shape[1], wtha, wthb, 0.4, 4, 0.065, 20))  # lower side

    # coordinates and masks for upper and lower side
    maxima_xs = []
    maxima_ys = []
    st_masks = []

    # find stitch endpoint candidates for both sides
    for i, (template, template_b) in enumerate(st_templates, len(inc_templates)):
        mtb = match_template(image, template_b)*template_weights[i] # template matching - only circles
        result = match_template(image, template)*template_weights[i] # template matching - stitches endpoints
        if verbose: ax[i,1].imshow(result)

        # quantile thresholding
        th = np.quantile(result, q)
        if verbose:
            ax[i,0].hist(result.ravel(), bins=50) # histogram of stitches endpoints corelation
            ax[i,0].axvline(th, color='r', linestyle='--', label=f'Quantile: {q}')
        st_mask_i = np.logical_and(result>max(th, min_st_th), result>mtb) # replace low values of quantile (images without stitches) with fixed value min_st_th, use only points with endpoint corelation higher than circle corelation
        st_masks.append(st_mask_i)
        if verbose: ax[i,2].imshow(st_mask_i, cmap="gray")

        # maxima detection - stitch candidates
        local_max = find_local_maxima(result)

        y, x = np.where(local_max)
        ax[i,1].plot(x,y,"o")

        y, x, _ = find_maxima(result,local_max,ntol)

        if verbose:
            ax[i,1].plot(x,y,"rx")
            ax[i,2].plot(x,y,"rx")

        maxima_xs.append(x)
        maxima_ys.append(y)
    
    x, y = process_st_candidates(maxima_xs, maxima_ys, st_masks, clustering_th)
    if verbose:
        ax[2,2].plot(x[0],y[0],"go")
        ax[3,2].plot(x[1],y[1],"go")

    stitches = do_stitching(x, y, st_centers, stitching_th)
    intersections = []
    alphas = []

    ptAB = np.fliplr(np.array(inc_endpoints))
    ptA = tuple(ptAB[0,:])
    ptB = tuple(ptAB[1,:])
    p1 = ptAB[0,:]
    p2 = ptAB[1,:]
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    if dy == 0:
        inc_alpha = 90.0
    elif dx == 0:
        inc_alpha = 0.0
    else:
        inc_alpha = 90 + 180.*np.arctan(dy/dx)/np.pi
    inc_len = dist(ptA, ptB)

    for stitch in stitches:
        if verbose:
            line = LineCollection([stitch], colors='g', linewidths=2)
            ax[1,0].add_collection(line)
        xi, yi, valid, r, s = intersectLines(stitch[0], stitch[1], ptA, ptB)
        if valid:
            p1 = stitch[0]
            p2 = stitch[1]
            dx = p2[0]-p1[0]
            dy = p2[1]-p1[1]
            if dy == 0:
                st_alpha = 90.0
            elif dx == 0:
                st_alpha = 0.0
            else:
                st_alpha = 90 + 180.*np.arctan(dy/dx)/np.pi
                
            intersections.append(s*inc_len)
            alphas.append((st_alpha - inc_alpha)%180)
    img_dict = {"filename": image_filenames[p], "incision_polyline": inc_endpoints,
      "crossing_positions": intersections,
      "crossing_angles": alphas}
    out.append(img_dict)
    if verbose:
        plt.tight_layout()
        fig.show()
        plt.show()
with open(output_fn,'w') as fw:
   json.dump(out, fw)