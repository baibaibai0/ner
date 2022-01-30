import torch
from torch.autograd import Variable
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel
from model.config import hyper
import torch.nn.functional as F
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda")


xx = torch.linspace(-10, 10, 10000)
print(xx.shape)
yy = torch.sigmoid(xx)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(xx, yy)
plt.show()