import numpy as np
from matplotlib import pyplot as plt  
from skimage.io import imread


fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

ax1.set_xticks([], [])
ax1.set_yticks([], [])
ax2.set_xticks([], [])
ax2.set_yticks([], [])
ax3.set_xticks([], [])
ax3.set_yticks([], [])
ax4.set_xticks([], [])
ax4.set_yticks([], [])
ax5.set_xticks([], [])
ax5.set_yticks([], [])
ax6.set_xticks([], [])
ax6.set_yticks([], [])

blur1 = imread('../blur1.png', True)/255.0
ax1.imshow(blur1, 'gray')

weight = np.load('../weight1.npy')
weight = np.squeeze(weight)
im = ax2.imshow(weight, cmap=plt.cm.spring)
ax2.set_xlabel('Uniform noise', fontsize=20)

weight = np.ones(blur1.shape) / 0.01
ax3.imshow(weight, cmap=plt.cm.spring)


blur1 = imread('../blur.png', True)/255.0
ax4.imshow(blur1, 'gray')

weight = np.load('../weight.npy')
weight = np.squeeze(weight)
im = ax5.imshow(weight, cmap=plt.cm.spring)

gt = blur1
sigma = np.ones(blur1.shape)
sigma[0 : gt.shape[0]/2, 0 : gt.shape[1]/2] = 0.01
sigma[0 : gt.shape[0]/2, gt.shape[1]/2 : gt.shape[1]] = 0.02
sigma[gt.shape[0]/2 : gt.shape[0], 0 : gt.shape[1]/2] = 0.03
sigma[gt.shape[0]/2 : gt.shape[0], gt.shape[1]/2 : gt.shape[1]] = 0.04

im = ax6.imshow(sigma, cmap=plt.cm.spring)


# fig.colorbar(im)
# cbar = plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

# plt.savefig('weight.pdf')
plt.show()
