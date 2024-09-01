import numpy as np
import pandas as pd
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Normalization,TimeSeriesDataset,train_model, plot_result, test, check_accuracy
from Model import CCPerceptron
import argparse

parser = argparse.ArgumentParser(description='parameters')

parser.add_argument('--n_layer', type=int, help='layers',default=2, required=False)
parser.add_argument('--device', type=int, help='gpu',default=1, required=False)
parser.add_argument('--dataset', type=str, help='dataset',default='PSM', required=True)

args = parser.parse_args()

torch.cuda.set_device(args.device)
print(torch.cuda.current_device())

def get_data(dataset):
    if dataset == 'SWaT':
        TRAIN_PATH = r'SWaT_normal.csv'
        train_data = pd.read_csv(TRAIN_PATH)
        train_data.drop('Timestamp', axis=1, inplace=True)
    if dataset == 'PSM':
        TRAIN_PATH = 'PSM/data/train.csv'
        train_data = pd.read_csv(TRAIN_PATH)
        train_data.drop('timestamp_(min)', axis=1, inplace=True)
        train_data.fillna(0, inplace=True)
    return train_data


train_data = get_data(dataset)



# hyperparameter
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
NUM_EPOCHS = 10  # epoch
window_size=100
step_size=1
n_feature=train_data.shape[-1]


train_data,_=Normalization(train_data)
train_data=train_data.values[np.arange(window_size)[None, :] + np.arange(0,train_data.shape[0]-window_size+1,step_size)[:, None]]

train_data1 = train_data[:int(np.floor(.8 * train_data.shape[0]))]
val_data = train_data[int(np.floor(.8 * train_data.shape[0])):int(np.floor(train_data.shape[0]))]
train_data=train_data1
print(train_data.shape)

model=CCPerceptron(window_size,train_data.shape[-1],args.n_layer).to(DEVICE)



# dataloader
train_set=TimeSeriesDataset(train_data)
val_set=TimeSeriesDataset(val_data)
train_loader=DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader=DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler_lr=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[4,9,13],gamma=0.25)
loss_fn=nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()  

model,train_loss_all,val_loss_all=train_model(train_loader, val_loader, model, optimizer, loss_fn, scaler, DEVICE,NUM_EPOCHS,scheduler_lr)
print(' training over!!!! ')

print('-------------start testing-------------')
def get_testdata(dataset):
    if dataset == 'SWaT':
        TEST_PATH = r'SWaT_attack.csv'
        test_data = pd.read_csv(TEST_PATH)
        test_label = test_data['lable']
        test_data.drop(['lable', 'Timestamp'], axis=1, inplace=True)
    if dataset == 'PSM':
        TEST_PATH = r'PSM/data/test.csv'
        test_data = pd.read_csv(TEST_PATH)
        test_data.drop('timestamp_(min)', axis=1, inplace=True)
        TEST_label_PATH = r'PSM/data/test_label.csv'
        test_label = pd.read_csv(TEST_label_PATH)
        test_label.drop('timestamp_(min)', axis=1, inplace=True)
        test_label=test_label['label']
    return test_data, test_label


test_data, test_label = get_testdata(dataset)


TRAIN_PATH = r'SWaT_normal.csv'
train_data = pd.read_csv(TRAIN_PATH)
train_data.drop('Timestamp', axis=1, inplace=True)
train_data,min_max_scaler=Normalization(train_data)

TRAIN_PATH = 'PSM/data/train.csv'
train_data = pd.read_csv(TRAIN_PATH)
train_data.drop('timestamp_(min)', axis=1, inplace=True)
train_data.fillna(0, inplace=True)
train_data,min_max_scaler=Normalization(train_data)


##parameter
step_size=100

#data preprocessing
x_scaled = min_max_scaler.transform(test_data)
test_data = pd.DataFrame(x_scaled)

test_data=test_data.values[np.arange(window_size)[None, :] + np.arange(0,test_data.shape[0]-window_size+1,step_size)[:, None]]
print(test_data.shape)

# dataloader
test_set = TimeSeriesDataset(test_data)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

loss_fn_each=nn.MSELoss(reduction='none')
windows_labels = []
for i in range(0,len(test_label) - window_size+1,step_size):
    windows_labels.append(list(np.int_(test_label[i:i + window_size])))
y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]


results_test,y_pred = test(test_loader, DEVICE, loss_fn_each, model,'LSTM')
# plot_result(results_test)



def get_bestF1(lab, scores, PA=False):
    scores = np.array(scores)
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones

    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    new_scores = scores[sortid]

    if PA:
        lab_diff = np.insert(lab, len(lab), 0) - np.insert(lab, 0, 0)
        a_st = np.arange(len(lab) + 1)[lab_diff == 1]
        a_ed = np.arange(len(lab) + 1)[lab_diff == -1]

        thres_a = np.array([np.max(scores[a_st[i]:a_ed[i]]) for i in range(len(a_st))])
        sort_a_id = np.flip(np.argsort(thres_a))  # big to small
        cum_a = np.cumsum(a_ed[sort_a_id] - a_st[sort_a_id])

        last_thres = np.inf
        TPs = np.zeros_like(new_lab)
        for i, a_id in enumerate(sort_a_id):
            TPs[(thres_a[a_id] <= new_scores) & (new_scores < last_thres)] = cum_a[i - 1] if i > 0 else 0
            last_thres = thres_a[a_id]
        TPs[new_scores < last_thres] = cum_a[-1]
    else:
        TPs = np.cumsum(-new_lab) + ones

    FPs = np.cumsum(new_lab - 1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs

    N = len(lab) - np.flip(TPs > 0).argmax()  
    TPRs = TPs[:N] / ones  ###让TPR很大
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)


    anomaly_ratio = ones / len(lab)
    FPR_bestF1_TPR1 = anomaly_ratio / (1 - anomaly_ratio) * (2 / F1s[maxid] - 2)
    TPR_bestF1_FPR0 = F1s[maxid] / (2 - F1s[maxid])
    return {'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid],
            'FPR': FPRs[maxid], 'maxid': maxid,
            'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0}

test_label=test_label[:len(results_test)]
test_label=np.array(test_label)
best_score=get_bestF1(test_label,results_test)
print(best_score)

