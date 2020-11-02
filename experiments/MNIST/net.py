import torch
import torch.nn.functional as F

import pytorch_pfn_extras as ppe

class CNV(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = ppe.nn.LazyConv2d(None, 32, 3, 1)
    self.conv2 = ppe.nn.LazyConv2d(None, 32, 3, 1)
    self.fc1 = ppe.nn.LazyLinear(None, 256)
    self.fc2 = ppe.nn.LazyLinear(None, 10)

  def forward(self, x):
    h1 = F.relu(self.conv1(x))
    h2 = F.relu(self.conv2(h1))
    h3 = h2.flatten(start_dim=1)
    h4 = F.relu(self.fc1(h3))
    h5 = self.fc2(h4)
    return h5
