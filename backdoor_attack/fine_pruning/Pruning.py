from torch.nn.utils.prune import BasePruningMethod

class ActivationStructed(BasePruningMethod):
  def __init__(self, amount, n, dim=-1):
    self.amount = amount
    self.n = n
    self.dim = dim
  
  def compute_mask(self, t, default_mask):
    tensor_size = t.shape[self.dim]

    nparams_to_prune = 