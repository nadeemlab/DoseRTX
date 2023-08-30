#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 19:44:15 2023

@author: ndahiya
"""
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

class BayesianBeamOptimizer():
  def __init__(self, verbose_level=2, random_state=1, num_iter=40):
    self.num_iter=num_iter
    self.verbose_level = verbose_level
    self.random_state = random_state
    self.curr_iter = 0
    self.next_point = None
    self.optimizer = self.create_beam_angle_optim()
  
  def get_next_point(self):
    xi = self.get_utility_xi()
    utility = UtilityFunction(kind="ei", xi=xi)
    self.next_point = self.optimizer.suggest(utility)
    self.curr_iter += 1
    
    return self.next_point
  
  def update_optimizer(self, target):
    self.optimizer.register(params=self.next_point, target=target)
    
  def create_beam_angle_optim(self):
    optimization_params = {}
    for i in range(7):
      optimization_params['beam_{}'.format(i)] = (0, 71)
    
    optimizer = BayesianOptimization(
      f=None,
      pbounds=optimization_params,
      verbose=self.verbose_level,
      random_state=self.random_state,
    )
    
    return optimizer
  
  def get_utility_xi(self):
    if self.curr_iter <= 15:
      xi = 0.1
    elif self.curr_iter > 15 and self.curr_iter <= 30:
      xi = 0.01
    else:
      xi = 0.001
    
    return xi