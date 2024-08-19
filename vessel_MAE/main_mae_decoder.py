import argparse
import json

import torch
import random
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold


# from model.vessel_MIM_fusion_concat import vessel_MIM
from model.vessel_MIM_fusion_add import vessel_MIM
from args import get_args

#PARAMETERS
args = get_args()

random_seed = args.random_seed

batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
patch_size = args.patch_size
mask_ratio = args.mask_ratio
alpha = args.loss_ratio

data_path = args.data_path
result_path = args.result_path



#ランダムシードの固定
def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fit(net, optimizer, num_epochs, train_loader, test_loader, device, history,fold,alpha):

    # tqdmライブラリのインポート
    from tqdm.notebook import tqdm

    base_epochs = len(history)
  
    for epoch in range(base_epochs, num_epochs+base_epochs):

        # 1エポックあたりの累積損失(平均化前)
        train_fom_loss, train_mae_loss=0, 0
        val_fom_loss, val_mae_loss = 0, 0

        best_val_loss=1.0

        # 1エポックあたりのデータ累積件数
        n_train, n_test = 0, 0

        #訓練フェーズ
        net.train()
        # for inputs, labels in tqdm(train_loader):
        for inputs, labels in train_loader:
            # 1バッチあたりのデータ件数
            train_batch_size = len(labels)
            # 1エポックあたりのデータ累積件数
            n_train += train_batch_size
    
            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 予測計算
            fom_loss, mae_loss = net(inputs,labels)

            # 損失計算
            # 割合とかは後で考える
            # alpha=0
            # alpha = 1 / (1 + np.exp(-10 * ((epoch+1)/num_epochs - 0.5)))
            # alpha = 1 / (1 + np.exp(-20 * ((epoch+1)/num_epochs - 0.5)))
            loss = alpha * fom_loss + (1-alpha) * mae_loss

            # 勾配計算
            loss.backward()

            # パラメータ修正
            optimizer.step()

            # lossは平均計算が行われているので平均前の損失に戻して加算
            train_fom_loss += fom_loss.item() * train_batch_size
            train_mae_loss += mae_loss.item() * train_batch_size 
            

        #予測フェーズ
        net.eval()

        for inputs_test, labels_test in test_loader:
            # 1バッチあたりのデータ件数
            test_batch_size = len(labels_test)
            # 1エポックあたりのデータ累積件数
            n_test += test_batch_size

            # GPUヘ転送
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

            # 予測計算
            fom_loss_test, mae_loss_test = net(inputs_test,labels_test)

            # 損失計算
            # fom_loss_test, mae_loss_test = criterion(outputs_test, labels_test)
            # loss_test = fom_loss_test + mae_loss_test 


            # lossは平均計算が行われているので平均前の損失に戻して加算
            val_fom_loss += fom_loss_test.item() * test_batch_size
            val_mae_loss += mae_loss_test.item() * test_batch_size

        # 損失計算
        rmse_train_fom_loss=(train_fom_loss / n_train)**0.5
        rmse_train_mae_loss=(train_mae_loss / n_train)**0.5
        rmse_val_fom_loss=(val_fom_loss / n_test)**0.5     
        rmse_val_mae_loss=(val_mae_loss / n_test)**0.5     
    

        # 結果表示
        if (epoch+1)%10==0:
            print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}],train_fom_loss: {rmse_train_fom_loss:.5f},val_fom_loss: {rmse_val_fom_loss:.5f},train_mae_loss: {rmse_train_mae_loss:.5f},val_mae_loss: {rmse_val_mae_loss:.5f}')
        # 記録
        item = np.array([epoch+1,
                        rmse_train_fom_loss,rmse_val_fom_loss,
                        rmse_train_mae_loss,rmse_val_mae_loss])

        history = np.vstack((history, item))

        save_path = result_path+'/model'+str(fold)+'.pth'

        if rmse_val_fom_loss<best_val_loss:
            best_val_loss=rmse_val_fom_loss
            # モデルの状態辞書を取得
            model_state = net.state_dict()
            # モデルの状態辞書を保存
            torch.save(model_state, save_path)
        

    return history



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

historys=[]


for fold in range(5):
    model=vessel_MIM(patch_size=patch_size, mask_ratio=mask_ratio)
    net=model.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = np.zeros((0,5))

    train_loader = train_loaders[fold]
    val_loader = val_loaders[fold]

    history = fit(net, optimizer, num_epochs, train_loader, val_loader, device, history,fold,alpha=alpha)
    historys=np.append(historys,history)


historys=historys.reshape(5,num_epochs,5)

np.save(result_path+'/log', historys)

print("End.")