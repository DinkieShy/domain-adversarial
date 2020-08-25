import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models.detection.roi_heads import RoIHeads

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class Classifier(nn.Module):
  def __init__(self, in_features, domains):
    super(Classifier, self).__init__()
    
    self.classifier = nn.Sequential(
        nn.Linear(in_features, 50),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(50, domains)
    )

  def forward(self, x):
    if x.dim() == 4:
        assert list(x.shape[2:]) == [1, 1]
    x = x.flatten(start_dim=1)
    y = self.classifier(x)
    return y

class DomainAdversarialHead(nn.Module):
  def __init__(self, in_features, domains):
    super(DomainAdversarialHead, self).__init__()
    self.gradientReversal = GradientReversalLayer()
    self.classifier = Classifier(in_features, domains)
    self.domain_count = domains

  def forward(self, features, proposals, labels=None):
    y = self.gradientReversal.apply(features)
    predictions = self.classifier(y)

    if self.training:
      labels = torch.cat(labels, dim=0)
      domainLoss = {"domainLoss": F.cross_entropy(predictions, labels)}
    else:
      domainLoss = {}

    return predictions, domainLoss