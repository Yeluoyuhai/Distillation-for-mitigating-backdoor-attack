import matplotlib.pyplot as plt
import seaborn
import os

def plot_training_logs(log, out=None):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(log['train/loss'], label='train')
  ax.plot(log['test/loss'], label='test')
  ax.set_title('Loss')
  ax.legend()

  if not out is None:
    fig.savefig(os.path.join(out, 'loss.png'), dpi=300)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(log['train/accuracy'], label='train')
  ax.plot(log['test/accuracy'], label='test')
  ax.set_title('Accuracy')
  ax.legend()

  if not out is None:
    fig.savefig(os.path.join(out, 'accuracy.png'), dpi=300)

  plt.show()

def plot_confusion_matrix(confmat, out=None, **kwargs):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  seaborn.heatmap(confmat, ax=ax, **kwargs)

  if not out is None:
    fig.savefig(out, dpi=300)

  plt.show()