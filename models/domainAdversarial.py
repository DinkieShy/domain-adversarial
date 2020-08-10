import torch.nn as nn
import torch.nn.functional as F

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()
    
    def forward(self, x):
        return x

    def backward(self, x):
        return -x

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.Dropout2d(),
    )
    
    self.classifier = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(50, 10),
    )

  def forward(self, x):
    features = self.feature_extractor(x)
    features = features.view(x.shape[0], -1)
    logits = self.classifier(features)
    return logits

class DomainAdversarialHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.gradientReversal = GradientReversalLayer()
    self.classifier = Classifier()

  def forward(self, x):
    y = self.gradientReversal.forward(x)
    y = self.classifier(y)
    return y