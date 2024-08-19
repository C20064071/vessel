import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold



#シード値の固定
def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "/root/work/data/2021"
result_path = "/root/work/result/MIM/decoder/1dconv"

transform = transforms.Resize((48,512))

x_data=[]

for i in range(1,302):
    img=np.load(file=data_path+"/deck_depth_map/data/hull"+str(i)+".npy")
    img=torch.tensor(img).reshape(1,48,521)
    img=transform(img)
    img=np.array(img[0])
    x_data.append(img)

x_data=np.array(x_data)
x_data = np.delete(x_data, range(117, 119), axis=0)
x_data=np.flip(x_data,axis=2)
x_data=np.array(x_data).reshape(299,1,48,512)
x_data=torch.tensor(x_data)

X_MAX = x_data.max()
x_data /= X_MAX

y =np.genfromtxt(data_path + "/hist_naked.dat", dtype=float, skip_header=1)
y = np.delete(y, range(117, 119), axis=0)
y_data=y[:,4:5]
#性能指数K

class HullDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    

batch_size=16

data=torch.tensor(x_data).float().clone().detach()
targets=torch.tensor(y_data).float().clone().detach()

dataset=HullDataset(data,targets)

kfold=KFold(n_splits=5,shuffle=True,random_state=0)

train_loaders=[]
val_loaders=[]

for train_index,val_index in kfold.split(dataset):
    train_dataset=Subset(dataset,train_index)
    val_dataset=Subset(dataset,val_index)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)

    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
