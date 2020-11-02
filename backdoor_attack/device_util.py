import torch

class device_util():
  def __init__(self, device):
    self.device = device
  
  def to_device(self, torch_ary):
    self.check_type(torch_ary)
    if self.device.type == 'cuda':
      return torch_ary.to(self.device)
    else:
      return torch_ary
    
  def check_type(self, ary):
    assert type(ary) == torch.Tensor, 'Input type must be torch.Tensor: Input type is {}'.format(str(type(ary)))
  
  def from_device(self, torch_ary):
    self.check_type(torch_ary)
    if self.device.type == 'cuda':
      return torch_ary.cpu()
    else:
      return torch_ary
