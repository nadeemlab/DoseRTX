#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:40:31 2023

@author: ndahiya
"""

import SimpleITK as sitk
import os
import numpy as np
from skimage.transform import resize
from data import create_dataset
from models import create_model

def get_model_dataset(opt):
  """
  Get the trained model and dataset based on options.

  Args:
    opt (TYPE): test options.

  Returns:
    model (TYPE): dose prediction model.
    dataset (TYPE): dose prediction dataset.

  """
  opt.num_threads = 0   # test code only supports num_threads = 1
  opt.batch_size = 1    # test code only supports batch_size = 1
  opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
  opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
  opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
  dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
  model = create_model(opt)      # create a model given opt.model and other options
  model.setup(opt)               # regular setup: load and print networks; create schedulers
  
  if opt.eval:
    model.eval()
  
  return model, dataset

def match_arr_to_dicom(dicom_arr, ref_itk):
  dicom_itk = sitk.GetImageFromArray(dicom_arr)
  dicom_itk.SetOrigin(ref_itk.GetOrigin())
  dicom_itk.SetSpacing(ref_itk.GetSpacing())
  dicom_itk.SetDirection(ref_itk.GetDirection())
  
  return dicom_itk

def get_crop_settings(oar):
    # Use to get crop settings
    # Don't use cord or eso as they spread through more slices
    # If total number of slices is less than 128 then don't crop at all
    # Use start and end index from presence of any anatomy or ptv
    # If that totals more than 128 slices then leave as is.
    # If that totals less than 128 slices then add slices before and after to make total slices to 128

    oar1 = oar.copy()
    oar1[np.where(oar == 1)] = 0
    oar1[np.where(oar == 2)] = 0

    # For 2D cropping just do center cropping 256x256
    center = [0, oar.shape[1] // 2, oar1.shape[2] // 2]
    start = [0, center[1] - 150, center[2] - 150]
    end = [0, center[1] + 150, center[2] + 150]

    depth = oar1.shape[0]
    if depth < 128:
        start[0] = 0
        end[0] = depth

        return start, end

    first_slice = -1
    last_slice = -1
    for i in range(depth):
        frame = oar1[i]
        if np.any(frame):
            first_slice = i
            break
    for i in range(depth - 1, -1, -1):
        frame = oar1[i]
        if np.any(frame):
            last_slice = i
            break

    expanse = last_slice - first_slice + 1
    if expanse >= 128:
        start[0] = first_slice
        end[0] = last_slice

        return start, end

    # print('Get\'s here')
    slices_needed = 128 - expanse
    end_slices = slices_needed // 2
    beg_slices = slices_needed - end_slices

    room_available = depth - expanse
    end_room_available = depth - last_slice - 1
    beg_room_available = first_slice

    leftover_beg = beg_room_available - beg_slices
    if leftover_beg < 0:
        end_slices += np.abs(leftover_beg)
        first_slice = 0
    else:
        first_slice = first_slice - beg_slices

    leftover_end = end_room_available - end_slices
    if leftover_end < 0:
        first_slice -= np.abs(leftover_end)
        last_slice = depth - 1
    else:
        last_slice = last_slice + end_slices

    if first_slice < 0:
        first_slice = 0

    start[0] = first_slice
    end[0] = last_slice

    return start, end
  
def get_caselist(txt_file):
    datasets = []
    with open(txt_file, 'r') as f:
        for dset in f:
            datasets.append(dset.strip())
    return datasets
  
def resample(img, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetReferenceImage(ref_image)
    img = resampler.Execute(img)

    return img


def get_dataset(in_dir, case, suffix, itk=False):
    filename = os.path.join(in_dir, case + suffix)
    img = sitk.ReadImage(filename)
    if itk is False:
        img = sitk.GetArrayFromImage(img)

    return img
  
def crop_resize_img(img, start, end, is_mask=False):
    # Crop to setting given by start/end coordinates list, assuming depth,height,width

    img_cropped = img[start[0]:end[0]+1, start[1]:end[1], start[2]:end[2]]
    img_cropped = np.moveaxis(img_cropped, 0, -1)  # Slices last

    order = 0
    if is_mask is False:
        order = 1
    img_resized = resize(img_cropped, (128,128,128), order=order, preserve_range=True, anti_aliasing=False).astype('float32')
    if is_mask is True:
        img_resized = img_resized.astype('uint8')

    img_resized = np.moveaxis(img_resized, -1, 0)  # Slices first again

    return img_resized