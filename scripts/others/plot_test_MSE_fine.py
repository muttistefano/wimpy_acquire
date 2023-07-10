import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

w_label = []
w_out = []
x_label = []
x_out = []
y_label = []
y_out = []


with open(sys.argv[1], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_label.append(float(row[2]))

with open(sys.argv[2], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_out.append(float(row[2]))

with open(sys.argv[3], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_label.append(float(row[2]))

with open(sys.argv[4], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_out.append(float(row[2]))

with open(sys.argv[5], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    y_label.append(float(row[2]))

with open(sys.argv[6], 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    y_out.append(float(row[2]))




w_label = np.asarray(w_label)
w_out   = np.asarray(w_out)
x_label = np.asarray(x_label)
x_out   = np.asarray(x_out)
y_label = np.asarray(y_label)
y_out   = np.asarray(y_out)


print(len(x_label),len(x_out))
print(len(y_label),len(y_out))
print(len(w_label),len(w_out))



def mse(A,B):
  mse = ((A - B)**2).mean(axis=0)
  return mse

print(mse(w_label,w_out))
print(mse(y_label,y_out))
print(mse(x_label,x_out))
