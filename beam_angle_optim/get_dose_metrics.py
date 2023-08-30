#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:07:38 2023

@author: ndahiya
"""

import numpy as np

class DoseMetrics():
  def __init__(self):
    """
    Class to compute dose metrics.

    Args:

    Returns:
      None.

    """
    pass
    
  def compute_dose_metrics(self, dose, oar, ptv):
    """
    Compute dose metrics given, dose, oars and ptv masks.

    Args:
      dose (TYPE): dose array.
      oar (TYPE): organs at risk masks.
      ptv (TYPE): ptv tumor mask.

    Returns:
      None.

    """
        
    oar_metrics = np.zeros(5 + 3, dtype='float32')  # 5 OAR metrics (mean) and 3 PTV metrics (D1, D95, D99)
    oar_metrics_max = np.zeros(5, dtype='float32')  # 5 OAR metrics (max)
    
    # OAR metrics (For now mean dose received by OAR)
    for i in range(5):
      oar_dose = dose[np.where(oar[i] == 1)]
      oar_metrics[i] = oar_dose.mean()
      oar_metrics_max[i] = oar_dose.max()
    
    ptv_dose = dose[np.where(ptv == 1)]
    # D1, dose received by 1% percent voxels (99th percentile)
    D1 = np.percentile(ptv_dose, 99)
    
    # D95, dose received by 95% percent voxels (5th percentile)
    D95 = np.percentile(ptv_dose, 5)

    # D99, dose received by 99% percent voxels (1st percentile)
    D99 = np.percentile(ptv_dose, 1)
    
    oar_metrics[5] = D1
    oar_metrics[6] = D95
    oar_metrics[7] = D99
    
    return oar_metrics, oar_metrics_max
