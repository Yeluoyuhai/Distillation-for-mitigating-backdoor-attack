import numpy as np

def one_dot_mnist(x, mask=None, val=255):
  if mask is None:
    mask = np.zeros((28, 28)).astype(np.bool)
    mask[25, 25] = True

  x[mask] = val
  return x


def gtsrb(x, mask=None, val=np.array([255, 201, 14])):
  x[26:28, 15:17, :] = val[np.newaxis, np.newaxis, :]
  return x