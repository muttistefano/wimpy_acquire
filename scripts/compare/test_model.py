import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
from math import sin,cos
from natsort import natsorted
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch import dropout, float32, from_numpy, flatten, no_grad
from torch.autograd import Variable
import copy

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
torch.set_printoptions(profile="full")

writer = SummaryWriter('ppew')

main_path = sys.argv[2]

laser_array = np.load(main_path + "laser_fine.npy")
tf_array    = np.load(main_path + "tf_fine.npy")

print(laser_array.shape)
print("GPU avail : ",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print("DEvice: ",device)



class CustomDataset(Dataset):
    def __init__(self, laser_in,tf_label_in, transform_in=None, target_transform_in=None):
        self.laser              = laser_in
        self.tf_label           = tf_label_in
        self.transform          = transform_in
        self.target_transform   = target_transform_in
        self.outputs            = []
        print(len(laser_in),len(tf_label_in))
        # print(tf_label_in)

    def __len__(self):
        return len(self.tf_label) - 1

    def __getitem__(self, idx):
        return self.laser[idx], self.tf_label[idx]



set_complete = CustomDataset(laser_array.astype(np.float32),tf_array)


train_size = int(len(set_complete) * 0.65)
valid_size = int(len(set_complete) * 0.20)
test_size  = len(set_complete)  - train_size - valid_size
train_set, valid_set, test_set = random_split(set_complete, [train_size,valid_size,test_size ])


# batch_size_train = 1024

# train_loader = DataLoader(train_set, batch_size=batch_size_train ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
# valid_loader = DataLoader(valid_set, batch_size=batch_size_train ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)
test_loader  = DataLoader(test_set , batch_size=1 ,shuffle=True, num_workers=0,pin_memory=False,persistent_workers=False)


class RNN(nn.Module):
    def __init__(self, hidden_size, num_layers,fd_n,fd_e):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn   = nn.GRU(input_size=510,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=0.0)
        self.fcx   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcx1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcx2  = nn.Linear(in_features=fd_e,out_features=1)
        self.fcy   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcy1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcy2  = nn.Linear(in_features=fd_e,out_features=1)
        self.fcw   = nn.Linear(in_features=hidden_size,out_features=fd_n)
        self.fcw1  = nn.Linear(in_features=fd_n,out_features=fd_e)
        self.fcw2  = nn.Linear(in_features=fd_e,out_features=1)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out_rnn, _ = self.rnn(x,h0)
        outx       = self.fcx(F.relu(out_rnn[:, -1, :]))
        outx       = self.fcx1(F.relu(outx))
        outx       = self.fcx2(F.relu(outx))
        outy       = self.fcy(F.relu(out_rnn[:, -1, :]))
        outy       = self.fcy1(F.relu(outy))
        outy       = self.fcy2(F.relu(outy))
        outw       = self.fcw(F.relu(out_rnn[:, -1, :]))
        outw       = self.fcw1(F.relu(outw))
        outw       = self.fcw2(F.relu(outw))

        return outx,outy,outw


model = RNN(1500,2,150,25)

model.load_state_dict(torch.load(sys.argv[1]))
# model.load_state_dict(torch.load("/home/blanker/wimpy/src/wimpy_acquire/scripts/fine_tune/model_xyw_fine.net"))
model.eval()

model.float()
model.to(device)



data_std_mean = np.load("/home/blanker/wimpy/src/wimpy_acquire/scripts/pre_process/data_std_mean.npy")


# ls_mean = data_std_mean[0]
# ls_std  = data_std_mean[1]
tf_mean = data_std_mean[2]
tf_std  = data_std_mean[3]


model.eval()

with no_grad():
    for i, data in enumerate(test_loader, 0):
        start_time = time.time()
        inputs, labels = torch.tensor(data[0],dtype=torch.float32).to(device),torch.tensor(data[1],dtype=torch.float32).to(device)
        outputs_x, outputs_y, outputs_w  = model(inputs)
        print("--- %s seconds ---" % (time.time() - start_time))

        writer.add_scalars("x_test", {
            'test_label_x': (labels[0,0].item() * tf_std)+ tf_mean,
            'test_out_x': (outputs_x[0][0].item() * tf_std)+ tf_mean,
        }, i)
        # print((labels[0,0].item() * tf_std)+ tf_mean)
        # print((outputs_x[0][0].item() * tf_std)+ tf_mean)
        # print("\n\n")
        writer.add_scalars("y_test", {
            'test_label_y': (labels[0,1].item() * tf_std)+ tf_mean,
            'test_out_y': (outputs_y[0][0].item() * tf_std)+ tf_mean,
        }, i)
        writer.add_scalars("w_test", {
            'test_label_w': (labels[0,2].item() * tf_std)+ tf_mean,
            'test_out_w': (outputs_w[0][0].item() * tf_std)+ tf_mean,
        }, i)
        writer.flush()

# torch.save(model.state_dict(), "model_xyw_fine.net")
# loss_train = np.asarray(loss_train)
# loss_valid = np.asarray(loss_valid)
# test_eval  = np.asarray(test_eval)

# np.save("loss_train_fine_xyw",loss_train)
# np.save("loss_valid_fine_xyw",loss_valid)
# np.save("test_eval_fine_xyw",test_eval)
# # np.save("test_eval_fine_old_xyw",test_eval_old)



