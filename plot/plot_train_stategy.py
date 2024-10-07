import numpy as np
import matplotlib.pyplot as plt


lowjt = np.array([33.51, 32.72, 30.92, 29.65, 28.70, 27.89, 27.21, 26.52, 25.95, 25.31])
highjt = np.array([33.43, 32.56, 30.76, 29.55, 28.63, 27.84, 27.24, 26.70, 26.30, 25.87])
ours = np.array([33.74, 32.95, 31.08, 29.83, 28.83, 28.05, 27.33, 26.71,  26.06, 25.45])
xlabel = ['0.6%', '1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%']

num = 8
x = np.arange(num)
width = 0.3
labels_sta = ['Naive', 'Ours']

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, lowjt[:num], width, label=labels_sta[0], color=np.array((72,118,255))/255.0)
rects2 = ax.bar(x + width/2, ours[:num], width, label=labels_sta[1], color='r')
ax.set_ylim([26, 34])
ax.set_ylabel("PSNR", fontsize=20)
ax.set_xlabel("$\sigma$", fontsize=27)
ax.set_xticks(x)
ax.set_xticklabels(xlabel[:num],  fontsize=10)
ax.legend(fontsize=18)
ax.tick_params(labelsize=20)
fig.tight_layout()

plt.show()




# x = np.arange(len(xlabel))
# width = 0.4
# labels_hyper = ['Low-JT', 'High-JT', 'ours']
# 
# fig, ax = plt.subplots()
# ax.set_ylim([-0.5, 0.5])
# 
# rects1 = ax.bar(x - width, lowjt - ours, width, label=labels_hyper[0])
# rects2 = ax.bar(x, highjt - ours, width, label=labels_hyper[1])
# # rects3 = ax.bar(x + width, ours - ours, width, label=labels_hyper[2])

# ax.set_ylabel("Subtracting With Our Method", fontsize=18)
# # ax.set_xlabel("Sigma and Ours PSNR", fontsize=20)
# ax.set_xticks(x)
# ax.set_xticklabels(xlabel, rotation=-20, fontsize=5)
# 
# ax.legend(fontsize=15)
# ax.tick_params(labelsize=10)
# fig.tight_layout()
# 
# 
# # ax.legend(loc='lower right')
# # plt.show()
# plt.savefig('fuck.pdf')
