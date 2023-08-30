#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:56:56 2023

@author: ndahiya
"""

from options.test_options import TestOptions
import argparse
from .utils import get_model_dataset
from .create_beamlet_mat_bayes import CreateBeamletMat
from .get_dose_metrics import DoseMetrics
from .bayesian_optimization  import BayesianBeamOptimizer
import os
import numpy as np

def get_casename(filepath):
  return filepath.split('/')[-1].strip('.npz')

def check_duplicate_beams(beams):
  
  return True if len(beams) != len(set(beams)) else False
    
def setup_main_options():
  """
  Set up options for main script.

  Returns:
    None.

  """
  parser = argparse.ArgumentParser(description='Bayesion Dose Beamlet Optimization.')
  parser.add_argument(
    "--plan_name", 
    help="Name of the model.",
    type=str,
    default='echoBeamMomLossW0_1'
    )
  parser.add_argument(
    "--bayes_iter", 
    help="Number of Bayesian Optimization iterations.",
    type=int,
    default=40
    )
  
  return parser
  
if __name__ == '__main__':
    
  parser = setup_main_options()
  main_opts = parser.parse_known_args()[0] # Tuple of main options and other options to be parsed by TestOptions
  
  opt = TestOptions().parse()  # get test options
  opt.name = main_opts.plan_name
  
  num_bayes_iter = main_opts.bayes_iter
  dset_path = os.path.join(opt.dataroot, opt.phase)
  model, dataset = get_model_dataset(opt)
  
  for i, data in enumerate(dataset):
    case_name = get_casename(data['A_paths'][0])
    print('Setting up dataset: ', case_name)
    
    model.set_input(data) # Cloned to GPU
    gt_dose = data['B'][0,0].numpy() # 128x128x128
    oar = data['A'][0,2:7].numpy().astype(np.uint8)   # 5x128x128x128
    ptv = data['A'][0,7].numpy().astype(np.uint8)     # 128x128x128
    
    comp_metrics = DoseMetrics()
    gt_oar_metrics, gt_oar_metrics_max = comp_metrics.compute_dose_metrics(dose=gt_dose, oar=oar, ptv=ptv)
    
    optim = BayesianBeamOptimizer(verbose_level=2, random_state=1, num_iter=num_bayes_iter)
    beamlet = CreateBeamletMat()
    beamlet.setup_dataset(in_dir=dset_path, case_name=case_name)
    
    print('Starting optimization:')
    best_so_far = -10
    for i in range(num_bayes_iter):
      print('  Iteration num: ', i+1)
      
      next_point = optim.get_next_point()
      beams = [int(beam) for beam in next_point.values()]
      
      if check_duplicate_beams is True:
        target = -10
      else:
        curr_beamlet = beamlet.get_beamlet(beams)
        model.update_input_beam(curr_beamlet)
        
        model.test()           # run inference
        pred_dose = model.get_current_visuals()['fake_Dose'].detach().cpu().numpy()[0,0]
        
        pred_oar_metrics, pred_oar_metrics_max = comp_metrics.compute_dose_metrics(dose=pred_dose, oar=oar, ptv=ptv)
        
        dvh_errors = np.abs(gt_oar_metrics - pred_oar_metrics)
        dvh_errors_max = np.abs(gt_oar_metrics_max - pred_oar_metrics_max)
        dvh_errors_all = np.append(dvh_errors, dvh_errors_max, 0)
        
        dvh_score  = dvh_errors_all.mean()
        dose_score = np.mean(np.abs(gt_dose - pred_dose))
        target = -(0.4 * dvh_score + 0.6 * dose_score)
      
      optim.update_optimizer(target=target)
      
      best_so_far = max(target, best_so_far)
      print('    Best so far {}'.format(best_so_far))
      