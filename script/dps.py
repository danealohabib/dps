# chaospy generates pseudorandom sequence, e.g. Hammersley seq.
import chaospy

# GPU-enabled vector library and optimization
import torch
import torch.optim as optim

# Libraries for plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Other numerical/data libraries
import pandas as pd
import numpy as np

dtype = torch.double
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pd.options.mode.chained_assignment = None

np.random.seed(123456789)
torch.manual_seed(123456789)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
output_folder = './output/params/notef/'

# Read and label Data
data = pd.read_csv('../DATA/dpsdata.csv')
data["zeros"] = np.zeros_like(data.p.values)
data["ones"] = np.ones_like(data.p.values)
