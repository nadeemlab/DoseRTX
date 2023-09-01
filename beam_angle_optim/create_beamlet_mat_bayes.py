#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:43:09 2023

@author: ndahiya
"""
from numpy import loadtxt
import os
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from .utils import get_dataset, get_crop_settings, match_arr_to_dicom, resample, crop_resize_img
import SimpleITK as sitk

class CreateBeamletMat():
  def __init__(self):
    """
    Class to create a beamlet array given beams.

    Returns:
      None.

    """
    self.ct_itk = None
    self.beam_maps = None
    self.points = None
    self.infMatrix = None
    self. normalize = True
    
  def setup_dataset(self, in_dir, case_name):
    """
    In bayesian optimization we will create different beamlet matrices for the
    same dataset based on selected beams. We can store some necessary data just
    once, such as orig CT, DOSE etc.

    Args:
      in_dir (TYPE): base in directory for data.
      case_name (TYPE): case name of data.

    Returns:
      None.

    """
    self.beam_maps = loadtxt(os.path.join(in_dir, 'beams_ind_' + case_name + '.txt'), dtype=int)
    self.points = loadtxt(os.path.join(in_dir, 'Points_' + case_name + '.txt'))
    
    infMatrix = pd.read_csv(os.path.join(in_dir, 'inf_matrix_' + case_name), sep='\t', header=None)
    self.infMatrix = csr_matrix((infMatrix[2], (infMatrix[0] - int(1), infMatrix[1] - int(1))))
    
    self.dose = get_dataset(in_dir, case_name, '_dose_ECHO.nrrd', itk=True)  # echo dose
    dose_arr = sitk.GetArrayFromImage(self.dose)
    self.dose_shape = dose_arr.shape
    self.tmaxm = dose_arr.max()
    self.tminm = dose_arr.min()
    
    self.ct_itk = get_dataset(in_dir, case_name, '_CT.nrrd', itk=True)
    
    self.ptv = get_dataset(in_dir, case_name, '_PTV.nrrd')
    
    # OAR + PTV but without Eso/Cord used to get crop Z boundaries
    oar_arr = get_dataset(in_dir, case_name, '_RTSTRUCTS.nrrd')
    oar_arr[np.where(self.ptv == 1)] = 6
    
    self.crop_start, self.crop_end = get_crop_settings(oar_arr)
    self.ptv = crop_resize_img(self.ptv, self.crop_start, self.crop_end, is_mask=True)
    
  def get_beamlet(self, beams):
    """
    Given a list of beams get a new beamlet matrix.

    Args:
      beams (TYPE): list of beams.

    Returns:
      None.

    """
    beamlets_to_consider = self.get_beamlets_to_consider(beams)
    infMatrix = self.infMatrix[:, beamlets_to_consider]
    
    inf_sum = infMatrix.sum(axis=1)
    beamlet_info = np.column_stack((self.points[:, 0:3], inf_sum.A1))
    
    beamlet_arr = np.zeros(self.dose_shape, dtype=float)
    for row in beamlet_info:
      curr_pt = (row[0], row[1], row[2])
      curr_val = row[3]
      curr_indx = self.dose.TransformPhysicalPointToIndex(curr_pt)  # X,Y,Z
      beamlet_arr[curr_indx[2], curr_indx[1], curr_indx[0]] = curr_val
    
    beamlet_arr = self.normalize_beamlet_arr(beamlet_arr)
    beamlet_itk = match_arr_to_dicom(beamlet_arr, self.dose)
    beamlet_itk = resample(beamlet_itk, self.ct_itk)
    beamlet_arr = sitk.GetArrayFromImage(beamlet_itk)
    
    beamlet_arr = crop_resize_img(beamlet_arr, self.crop_start, self.crop_end, is_mask=False)
    beamlet_arr[np.where(self.ptv == 1)] = 60  # PTV volume set to prescribed dose
    
    return beamlet_arr
    
  def get_beamlets_to_consider(self, beams):
    """
    Given a list of beams, get an array of beamlets to consider.

    Args:
      beams (TYPE): list of beams.

    Returns:
      None.

    """
    # Only consider unique beams. Although calling code makes sure that beams are unique
    # before calling this function
    beamlets_to_consider = []
    
    for b in list(set(beams)):
      startB = self.beam_maps[b, 0]
      endB = self.beam_maps[b, 1]
      beamlets_to_consider.append(np.arange(startB, endB))
      
    return np.hstack(beamlets_to_consider)
  
  def normalize_beamlet_arr(self, beamlet_arr):
    """
    Optionally normalize beamlet array to have same range as dose array.

    Args:
      beamlet_arr (TYPE): current beamlet array.

    Returns:
      None.

    """
    if self.normalize is True:
      rmin = beamlet_arr.min()
      rmax = beamlet_arr.max()
      rrange = rmax - rmin
      if rrange > 0:
        beamlet_arr = ((beamlet_arr - rmin) / rrange) * (self.tmaxm - self.tminm) + self.tminm  # Beamlet array normalized to have same range as dose array
      else:
        print('\tBeamlet all zeros.')
    
    return beamlet_arr
        