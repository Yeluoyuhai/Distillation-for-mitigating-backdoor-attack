import os
import time

import torch
import torch.nn.functional as F

import ignite
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.engine import Events

import backdoor_attack as bd
from backdoor_attack import device_util

class Distillator:
  def __init__(self, train_ds, test_ds, teacher_model, student_model, device):
    self.train_ds = train_ds
    self.test_ds = test_ds
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.device = device
    self.util = bd.device_util(self.device)

  def _update(self, engine, batch):
    self.teacher_model.eval()
    self.student_model.train()
    self.optimizer.zero_grad()
    x, _ = batch
    x = self.util.to_device(x)
    t = F.softmax(self.teacher_model(x)/self.t, dim=1)
    y = F.log_softmax(self.student_model(x)/self.t, dim=1)
    loss = F.kl_div(y, t)*(self.t*self.t)
    loss.backward()
    self.optimizer.step()

  def run(self, optimizer, epoch, temp, out):
    self.temp = temp
    self.t = self.util.to_device(torch.tensor(temp['t'][temp['epoch']==0]))

    self.teacher_model.to(self.device)
    self.student_model.to(self.device)
    self.optimizer = optimizer(self.student_model.parameters())
    self.optimizer.step()

    trainer = ignite.engine.engine.Engine(self._update)

    evaluator = create_supervised_evaluator(
      self.student_model,
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
        self.student_model.to('cpu')
        # Save result
        torch.save(self.student_model.state_dict(), os.path.join(out, 'best_model.pt'))
        self.student_model.to(self.device)

      # Report
      time_elp = time.time()-self.time_st
      print(
        "{}/{} Epoch, Train/Test Loss: {:.4f}/{:.4f}, \
        Train/Test Accuracy: {:.4f}/{:.4f}, \
        Temperature: {}, \
        Elapsed/Remaining time: {:.2f}/{:.2f} [min]"
        .format(
          engine.state.epoch,
          engine.state.max_epochs,
          tr_loss,
          te_loss,
          tr_acc,
          te_acc,
          self.t,
          time_elp/60,
          ( (time_elp/engine.state.epoch) * (engine.state.max_epochs-engine.state.epoch) ) / 60
        )
      )

      # logs
      self.log['train/loss'].append(tr_loss)
      self.log['test/loss'].append(te_loss)
      self.log['train/accuracy'].append(tr_acc)
      self.log['test/accuracy'].append(te_acc)

      # temperature scheduling
      if engine.state.epoch in self.temp['epoch']:
        self.t = self.util.to_device(torch.tensor(temp['t'][temp['epoch']==engine.state.epoch]))

    self.time_st = time.time()
    trainer.run(self.train_ds, max_epochs=epoch)
    self.time_elp = self.time_st - time.time()

    self.teacher_model.to('cpu')
    self.student_model.to('cpu')
