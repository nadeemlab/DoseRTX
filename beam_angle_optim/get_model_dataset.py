#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:58:17 2023

@author: ndahiya
"""

from data import create_dataset
from models import create_model

def get_model_dataset(opt):
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

  
  