# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import random
import numpy as np

xiter = 5000
leng = 15
plt.figure(figsize=(10,5)) 
listnum = ['1','10','11']
plt.xlim(0,xiter)
plt.ylim(0, 0.015)
colors = ['r', 'g', 'b', 'y', '#0072BC', np.array((72,118,255))/255.0]


file3 = open('./loss/out_train_cbd_true_divide.out', 'r')
file2 = open('./loss/divide_loss.txt')
x = []
y1 = []
y2 = []
y3 = []
y4 = []
for i in range(xiter):
    x.append(i)

    s1 = file2.readline()
    # print(s1.split(' '))
    y1.append(float(s1.split(' ')[1][:-1])/10)
    y2.append(float(s1.split(' ')[2][:-2]))

    s1 = file3.readline()
    s2 = file3.readline()
    y3.append(float(s2))
    # print(s1.split(' '))
    y4.append(float(s1.split(' ')[-2]))

x = [x[ii] for ii in range(0, xiter, leng)]
y1 = [sum(y1[ii:ii+leng])/leng for ii in range(0, xiter, leng)]
y2 = [sum(y2[ii:ii+leng])/leng for ii in range(0, xiter, leng)]
y3 = [sum(y3[ii:ii+leng])/leng for ii in range(0, xiter, leng)]
y4 = [sum(y4[ii:ii+leng])/leng for ii in range(0, xiter, leng)]
# plt.plot(x, y3, colors[2])
# plt.plot(x, y4, colors[3])
plt.plot(x, y1, colors[0])
plt.plot(x, y2, colors[2])

# plt.savefig('./pltpic/'+str(i)+'jpg')
# plt.close('all')

plt.xlabel('Training Iterations', fontsize=17)
plt.ylabel('Training Loss', fontsize=17)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# print(plt.figure.figsize)
plt.legend(('Guidance Loss', 'Propagation Loss'),fontsize=17)
# plt.savefig('./TrainingLoss.eps', format='eps')
plt.show()
