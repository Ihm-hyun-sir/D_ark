import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import Dataset
import timm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.io import read_image

from utils import *

from matplotlib import pyplot as plt


model_name = 'resnet50'
net = timm.create_model(model_name,num_classes=7, pretrained=True)
print(net)