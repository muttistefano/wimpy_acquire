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


with open("W_test_label_w.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_label.append(float(row[2]))

with open("W_test_out_w.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    w_out.append(float(row[2]))

with open("x_test_label_x.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_label.append(float(row[2]))

with open("x_test_out_x.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    x_out.append(float(row[2]))

with open("y_test_label_y.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    # print(row)
    y_label.append(float(row[2]))

with open("y_test_out_y.csv", 'r') as file:
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
