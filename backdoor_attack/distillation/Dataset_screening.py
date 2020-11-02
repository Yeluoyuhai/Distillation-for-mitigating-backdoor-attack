import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics

import torch
import torch.nn.functional as F
import ignite
from ignite.engine import create_supervised_evaluator

import backdoor_attack as bd
import net

class Dataset_screening:
  def __init__(self, dl, model_a, model_b, device):
    self.dl = dl
    self.model_a = model_a
    self.model_b = model_b
    self.device = device
    self.util = bd.device_util(device)
  
  def _eval(self, engine, batch):
    self.model_a.eval()
    self.model_b.eval()

    with torch.no_grad():
      x, t = batch
      x = self.util.to_device(x)
      self.y_a.append( 
        self.util.from_device(
          F.softmax(self.model_a(x), dim=1)
        ).numpy()
      )
      self.y_b.append(
        self.util.from_device(
          F.softmax(self.model_b(x), dim=1)
        ).numpy()
      )


  def run(self):
    self.model_a.to(self.device)
    self.model_b.to(self.device)
    evaluator = ignite.engine.engine.Engine(self._eval)

    self.y_a = []
    self.y_b = []
    evaluator.run(self.dl)

    self.y_a = np.concatenate(self.y_a, axis=0)
    self.y_a = np.argmax(self.y_a, axis=1)
    self.y_b = np.concatenate(self.y_b, axis=0)
    self.y_b = np.argmax(self.y_b, axis=1)

    self.negative_idx = self.y_a == self.y_b
    self.negative_dataset = {
      'x': self.dl.dataset.x[self.negative_idx],
      't': self.dl.dataset.t[self.negative_idx]
    }

    self.positive_idx = self.y_a != self.y_b
    self.positive_dataset = {
      'x': self.dl.dataset.x[self.positive_idx],
      't': self.dl.dataset.t[self.positive_idx]
    }
  
  def eval(self, t_correct):
    t = self.dl.dataset.t
    confmat = sklearn.metrics.confusion_matrix(self.positive_idx, t!=t_correct)
    report = sklearn.metrics.classification_report(self.positive_idx, t!=t_correct)

    return confmat, report