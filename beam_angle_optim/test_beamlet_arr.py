#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:41:46 2023

@author: ndahiya
"""

import numpy as np
import SimpleITK as sitk
# fname = 'LUNG1-002.npz'
# full_case = dict(np.load(fname))
# Beam1 = full_case['BEAM']

# fname = 'beamlet.npz'
# case = np.load(fname)
# Beam2 = case['BEAM']

# print(Beam1.shape, Beam2.shape)
# print(Beam1.dtype, Beam2.dtype)
# print(Beam1.min(), Beam2.min())
# print(Beam1.max(), Beam2.max())

fname = 'LUNG1-002_CT2DOSE.nrrd'
dose1 = sitk.ReadImage(fname)
dose1 = sitk.GetArrayFromImage(dose1)

fname = 'pred.nrrd'
dose2 = sitk.ReadImage(fname)
dose2 = sitk.GetArrayFromImage(dose2)

print(dose1.min(), dose1.max())
print(dose2.min(), dose2.max())
print(dose1.dtype, dose2.dtype)