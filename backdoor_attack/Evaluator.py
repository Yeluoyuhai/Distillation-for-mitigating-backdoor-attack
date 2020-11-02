import os
import time
import numpy as np

# import torch
# import torch.nn.functional as F

import ignite
from ignite.engine import create_supervised_evaluator

import backdoor_attack as bd

class Evaluator:
  def __init__(self, model, n_class, device):
    self.model = model
    self.device = device

    self.evaluator = create_supervised_evaluator(
      self.model,
      metrics={
        'confmat':ignite.metrics.ConfusionMatrix(n_class),
        'accuracy': ignite.metrics.Accuracy()
        },
      device=self.device
    )
  
  def run(self, test_dl, plot=False, out=None, **kwargs):
    self.evaluator.run(test_dl)
    confmat = self.evaluator.state.metrics['confmat']
    accuracy = self.evaluator.state.metrics['accuracy']
    
    print('Accuracy:', accuracy)
    
    if not out is None:
      np.savez(os.path.join(out+'.npz'), confmat=confmat, accuracy=accuracy)
    
      bd.plot_util.plot_confusion_matrix(
          confmat,
          out=os.path.join(out+'.png'),
          **kwargs
      )
    else:
      bd.plot_util.plot_confusion_matrix(
          confmat,
          **kwargs
      )
    
    return confmat, accuracy