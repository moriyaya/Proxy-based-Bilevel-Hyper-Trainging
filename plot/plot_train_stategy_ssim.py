import numpy as np
import matplotlib.pyplot as plt


lowjt = np.array([0.965, 0.958, 0.939, 0.921, 0.906, 0.891, 0.874, 0.858, 0.840, 0.820])
highjt = np.array([0.965, 0.958, 0.937, 0.919, 0.903, 0.887, 0.873, 0.860, 0.849, 0.836])
ours = np.array([0.966, 0.960, 0.941, 0.926, 0.912, 0.892, 0.876, 0.862, 0.843, 0.821])
xlabel = ['0.6%', '1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%']

num = 8
x = np.arange(num)
width = 0.3
labels_sta = ['Naive', 'Ours']

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lowjt[:num], width, label=labels_sta[0], color=np.array((72,118,255))/255.0)
rects2 = ax.bar(x + width/2, ours[:num], width, label=labels_sta[1], color='r')
ax.set_ylim([0.855, 0.969])
ax.set_ylabel("SSIM", fontsize=20)
ax.set_xlabel("$\sigma$", fontsize=27)
ax.set_xticks(x)
ax.set_xticklabels(xlabel[:num],  fontsize=10)
ax.legend(fontsize=18)
ax.tick_params(labelsize=20)
fig.tight_layout()

plt.show()


# xlabel = [xlabel[i]+'('+str(ours[i])+')'  for i in range(len(xlabel))]
# 
# x = np.arange(len(xlabel))
# width = 0.4
# labels_hyper = ['Low-JT', 'High-JT', 'ours']
# 
# fig, ax = plt.subplots()
# ax.set_ylim([-0.01, 0.02])
# 
# rects1 = ax.bar(x - width, lowjt - ours, width, label=labels_hyper[0])
# rects2 = ax.bar(x, highjt - ours, width, label=labels_hyper[1])
# # rects3 = ax.bar(x + width, ours - ours, width, label=labels_hyper[2])
# 
# # ax.set_ylabel("Subtracting With Our Method", fontsize=14)
# ax.set_xlabel("Sigma and Ours PSNR", fontsize=20)
# ax.set_xticks(x)
# ax.set_xticklabels(xlabel, rotation=-20, fontsize=5)
# 
# ax.legend(fontsize=15)
# ax.tick_params(labelsize=10)
# 
# 
# # ax.legend(loc='lower right')
# # plt.show()
# plt.savefig('fuck.pdf')
