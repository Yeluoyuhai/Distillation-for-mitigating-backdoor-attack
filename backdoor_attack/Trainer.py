import os
import time

import torch
import torch.nn.functional as F

import ignite
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.engine import Events
from ignite.handlers.early_stopping import EarlyStopping

import backdoor_attack as bd

class Trainer:
  def __init__(self, train_ds, test_ds, model, device):
    self.train_ds = train_ds
    self.test_ds = test_ds
    self.model = model
    self.device = device
    self.util = bd.device_util(self.device)

  def run(self, optimizer, epoch, out):
    self.model.to(self.device)
    self.optimizer = optimizer(self.model.parameters())
    self.optimizer.step()

    trainer = create_supervised_trainer(
      self.model,
      self.optimizer,
      F.cross_entropy,
      device=self.device
    )

    evaluator = create_supervised_evaluator(
      self.model,
      metrics={
        'acc' : ignite.metrics.Accuracy(),
        'loss' : ignite.metrics.Loss(F.cross_entropy)
      },
      device=self.device
    )

    self.log = {
      'train/loss': [],
      'test/loss': [],
      'train/accuracy': [],
      'test/accuracy': []
    }

    self.max_acc = 0
    @trainer.on(Events.EPOCH_COMPLETED)
    def logger(engine):
      # Evaluation against training dataset
      evaluator.run(self.train_ds)
      tr_loss = evaluator.state.metrics['loss']
      tr_acc = evaluator.state.metrics['acc']

      # Evaluation against test dataset
      evaluator.run(self.test_ds)
      te_loss = evaluator.state.metrics['loss']
      te_acc = evaluator.state.metrics['acc']

      # Save the best model
      if te_acc > self.max_acc:
        self.max_acc = te_acc
        self.model.to('cpu')
        # Save result
        torch.save(self.model.state_dict(), os.path.join(out, 'best_model.pt'))
        self.model.to(self.device)

      # Report
      time_elp = time.time()-self.time_st
      print(
        "{}/{} Epoch, Train/Test Loss: {:.4f}/{:.4f}, \
        Train/Test Accuracy: {:.4f}/{:.4f}, \
        Elapsed/Remaining time: {:.2f}/{:.2f} [min]"
        .format(
          engine.state.epoch,
          engine.state.max_epochs,
          tr_loss,
          te_loss,
          tr_acc,
          te_acc,
          time_elp/60,
          ( (time_elp/engine.state.epoch) * (engine.state.max_epochs-engine.state.epoch) ) / 60
        )
      )

      # logs
      self.log['train/loss'].append(tr_loss)
      self.log['test/loss'].append(te_loss)
      self.log['train/accuracy'].append(tr_acc)
      self.log['test/accuracy'].append(te_acc)

    self.time_st = time.time()
    trainer.run(self.train_ds, max_epochs=epoch)
    self.time_elp = self.time_st - time.time()

    self.model.to('cpu')
