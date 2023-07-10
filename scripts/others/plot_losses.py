import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

train_loss = []
valid_loss = []

with open(sys.argv[1], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    train_loss.append(float(row[2]))

with open(sys.argv[2], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    valid_loss.append(float(row[2]))



train_loss = np.asarray(train_loss)
valid_loss = np.asarray(valid_loss)
print(len(train_loss),len(valid_loss))

fig1, ax1 = plt.subplots()
plt.grid(True, which="both", ls="-")
plt.xlabel("epochs",fontsize=32)
plt.ylabel("log-loss",fontsize=32)
plt.yscale('log')
plt.yticks([1,0.5,0.1,0.04,0.01,0.004])
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.plot(np.arange(0,len(train_loss)*5,5),train_loss,label="train loss")
plt.plot(np.arange(0,len(valid_loss)*5,5),valid_loss,label="validation loss")

plt.annotate("       {:.4f}".format(train_loss[-1]),(np.arange(0,len(train_loss)*5,5)[-1],train_loss[-1]))
plt.annotate("       {:.4f}".format(valid_loss[-1]),(np.arange(0,len(valid_loss)*5,5)[-1],valid_loss[-1]))

plt.legend()

plt.show()