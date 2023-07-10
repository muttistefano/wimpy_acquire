import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

x_label = []
x_out   = []

y_label = []
y_out   = []

w_label = []
w_out   = []

with open(sys.argv[1], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_label.append(float(row[2]))

with open(sys.argv[2], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_out.append(float(row[2]))

with open(sys.argv[3], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    y_label.append(float(row[2]))

with open(sys.argv[4], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    y_out.append(float(row[2]))

with open(sys.argv[5], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_label.append(float(row[2]))

with open(sys.argv[6], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_out.append(float(row[2]))


x_label = np.asarray(x_label)[0:30]
x_out   = np.asarray(x_out)[0:30]
y_label = np.asarray(y_label)[0:30]
y_out   = np.asarray(y_out)[0:30]
w_label = np.asarray(w_label)[0:30]
w_out   = np.asarray(w_out)[0:30]

# print(len(train_loss_1),len(valid_loss_1))
# print(len(train_loss_2),len(valid_loss_2))

fig1, ax1 = plt.subplots(3,1)
# ax1[0].grid(True, which="both", ls="-")
# ax1[1].grid(True, which="both", ls="-")
ax1[0].set_xlabel("test sample",fontsize=32)
ax1[1].set_xlabel("test sample",fontsize=32)
ax1[2].set_xlabel("test sample",fontsize=32)
ax1[0].set_ylabel("x[m]",fontsize=32)
ax1[1].set_ylabel("y[m]",fontsize=32)
ax1[2].set_ylabel("w[rad]",fontsize=32)
# ax1[0].set_title("test_0")
# ax1[1].set_title("test_3")
# ax1[0].set_yticks([5,0.5,0.1,0.04,0.01,0.004])
# ax1[1].set_yticks([5,0.5,0.1,0.04,0.01,0.004])
# ax1[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1[0].plot(np.arange(0,len(x_label)),x_label,label="train loss")
ax1[0].plot(np.arange(0,len(x_out)),x_out,label="validation loss")
ax1[1].plot(np.arange(0,len(y_label)),y_label,label="train loss")
ax1[1].plot(np.arange(0,len(y_out)),y_out,label="validation loss")
ax1[2].plot(np.arange(0,len(w_label)),w_label,label="train loss")
ax1[2].plot(np.arange(0,len(w_out)),w_out,label="validation loss")
# ax1[0].annotate("       {:.4f}".format(train_loss_1[-1]),(np.arange(0,len(train_loss_1))[-1],train_loss_1[-1]))
# ax1[0].annotate("       {:.4f}".formatd(valid_loss_1[-1]),(np.arange(0,len(valid_loss_1))[-1],valid_loss_1[-1]))
# ax1[1].annotate("       {:.4f}".format(train_loss_2[-1]),(np.arange(0,len(train_loss_2))[-1],train_loss_2[-1]))
# ax1[1].annotate("       {:.4f}".format(valid_loss_2[-1]),(np.arange(0,len(valid_loss_2))[-1],valid_loss_2[-1]))
# plt.legend()

fig1.legend(*ax1[2].get_legend_handles_labels(),loc='upper center', ncol=4)

plt.show()