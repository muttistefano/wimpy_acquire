import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

train_loss_1 = []
valid_loss_1 = []

train_loss_2 = []
valid_loss_2 = []

with open(sys.argv[1], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    train_loss_1.append(float(row[2]))

with open(sys.argv[2], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    valid_loss_1.append(float(row[2]))

with open(sys.argv[3], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    train_loss_2.append(float(row[2]))

with open(sys.argv[4], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    valid_loss_2.append(float(row[2]))



train_loss_1 = np.asarray(train_loss_1)
valid_loss_1 = np.asarray(valid_loss_1)
train_loss_2 = np.asarray(train_loss_2)
valid_loss_2 = np.asarray(valid_loss_2)

print(len(train_loss_1),len(valid_loss_1))
print(len(train_loss_2),len(valid_loss_2))

fig1, ax1 = plt.subplots(2,1)
ax1[0].grid(True, which="both", ls="-")
ax1[1].grid(True, which="both", ls="-")
# ax1[0].set_xlabel("epochs",fontsize=32)
ax1[1].set_xlabel("epochs",fontsize=32)
ax1[0].set_ylabel("log-loss",fontsize=32)
ax1[1].set_ylabel("log-loss",fontsize=32)
ax1[0].set_yscale('log')
ax1[1].set_yscale('log')
# ax1[0].set_title("test_0")
# ax1[1].set_title("test_3")
ax1[0].set_yticks([5,0.5,0.1,0.04,0.01,0.004])
ax1[1].set_yticks([5,0.5,0.1,0.04,0.01,0.004])
ax1[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1[0].plot(np.arange(0,len(train_loss_1)*5,5),train_loss_1,label="train loss")
ax1[0].plot(np.arange(0,len(valid_loss_1)*5,5),valid_loss_1,label="validation loss")
ax1[1].plot(np.arange(0,len(train_loss_2)*5,5),train_loss_2,label="train loss")
ax1[1].plot(np.arange(0,len(valid_loss_2)*5,5),valid_loss_2,label="validation loss")
ax1[0].annotate("       {:.4f}".format(train_loss_1[-1]),(np.arange(0,len(train_loss_1)*5,5)[-1],train_loss_1[-1]))
ax1[0].annotate("       {:.4f}".format(valid_loss_1[-1]),(np.arange(0,len(valid_loss_1)*5,5)[-1],valid_loss_1[-1]))
ax1[1].annotate("       {:.4f}".format(train_loss_2[-1]),(np.arange(0,len(train_loss_2)*5,5)[-1],train_loss_2[-1]-0.001))
ax1[1].annotate("       {:.4f}".format(valid_loss_2[-1]),(np.arange(0,len(valid_loss_2)*5,5)[-1],valid_loss_2[-1]))
# plt.legend()
fig1.legend(*ax1[1].get_legend_handles_labels(),loc='upper center', ncol=4)
plt.show()