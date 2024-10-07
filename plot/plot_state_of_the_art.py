import numpy as np
import matplotlib.pyplot as plt

def pro_data(data):
    return list(map(float, data.split('&')))

def plot_sub(ax, psnr, ssim, ylim1, ylim2, colors, labelsize=8, xlabel='%1'):
    x = np.array(range(len(psnr)))
    width = 0.3
    legend = np.array(('FD', 'CSF', 'MLP', 'IRCNN', 'FDN', 'Ours'))

    for i in range(len(x)):
        ax.bar(x[i]-width/2, psnr[i], width, color=colors[i], label=legend[i])
    # ax.bar(x-width/2, psnr, width, color=colors)
    ax.set_ylim(ylim1)
    ax.set_xticks([],[])
    ax.tick_params(labelsize=labelsize)
    ax.set_xlabel(xlabel, fontsize=10)

    ax1 = ax.twinx()
    ax1.bar(x+width/2, ssim, width, color=colors)
    ax1.set_ylim(ylim2)
    ax1.tick_params(labelsize=labelsize)
    # ax1.legend()

def get_data(q,w,e,r,t, y,u, n):
    return [q[n], w[n], e[n], r[n], t[n], y[n], u[n]]


# SOTA
sigma1_psnr = pro_data('30.87 & 29.14 & 30.37 & 29.90 & 31.77 & 31.26')
sigma1_ssim = pro_data('0.882 & 0.835 & 0.834 & 0.916 & 0.935 & 0.930')
sigma2_psnr = pro_data('29.33 & 28.70 & 29.06 & 29.39 & 29.90 & 29.79')
sigma2_ssim = pro_data('0.848 & 0.823 & 0.816 & 0.903 & 0.899 & 0.901')
sigma3_psnr = pro_data('28.56 & 28.09 & 27.32 & 28.70 & 28.86 & 28.87')
sigma3_ssim = pro_data('0.817 & 0.811 & 0.801 & 0.884 & 0.875 & 0.879')
sigma4_psnr = pro_data('27.62 & 27.40 & 27.09 & 28.11 & 28.15 & 28.24')
sigma4_ssim = pro_data('0.779 & 0.792 & 0.752 & 0.866 & 0.857 & 0.867')
sigma5_psnr = pro_data('26.69 & 26.68 & 26.39 & 27.62 & 27.60 & 27.71')
sigma5_ssim = pro_data('0.737 & 0.770 & 0.736 & 0.851 & 0.841 & 0.853')

sigma6_psnr = pro_data('25.77 & 25.96 & 25.67 & 26.70 & 26.72 & 27.07')
sigma6_ssim = pro_data('0.692 & 0.746 & 0.712 & 0.828 & 0.821 & 0.830')
sigma7_psnr = pro_data('24.88 & 25.26 & 25.14 & 23.53 & 24.51 & 26.60')
sigma7_ssim = pro_data('0.645 & 0.720 & 0.698 & 0.708 & 0.733 & 0.815')
colors = np.array([(152,245,255), (255,246,143), (127,255,212), (127,255,0),(255,185,15), (255,0,0)])/255.0
x_label = ['1%', '2%', '3%', '4%', '5%', '6%', '7%']
line = 1.3
ourline = 1.7

fig = plt.figure(figsize=(6,5))
ax1 = fig.add_subplot(221)
ours = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr,  5)
fdn = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 4)
ircnn =get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 3) 
mlp = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 2)
csf = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 1)
fd = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 0)

x = range(len(ours))
ax1.plot(x, fd, '--', linewidth=line)
ax1.plot(x, csf, '--', linewidth=line)
ax1.plot(x, mlp, '--', linewidth=line)
ax1.plot(x, ircnn, '--', linewidth=line)
ax1.plot(x, fdn, '--', linewidth=line)
ax1.plot(x, ours, color='r', linewidth=ourline)
ax1.set_xticks(x)
ax1.set_xticklabels(x_label)
ax1.set_ylabel('PSNR', fontsize=11)
ax1.set_xlabel('$\sigma$', fontsize=13)
ax1.grid(linestyle='-.')

ax2 = fig.add_subplot(222)
ours = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim, sigma6_ssim, sigma7_ssim, 5)
fdn = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 4)
ircnn =get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 3) 
mlp = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 2)
csf = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 1)
fd = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 0)

x = range(len(ours))
ax2.plot(x, fd, '--', linewidth=line)
ax2.plot(x, csf, '--', linewidth=line)
ax2.plot(x, mlp, '--', linewidth=line)
ax2.plot(x, ircnn, '--', linewidth=line)
ax2.plot(x, fdn, '--', linewidth=line)
ax2.plot(x, ours, color='r', linewidth=ourline)
ax2.set_xticks(x)
ax2.set_xticklabels(x_label)
ax2.set_ylabel('SSIM', fontsize=11)
ax2.set_xlabel('$\sigma$', fontsize=13)
ax2.grid(linestyle='-.')


# samelevel_difflevel
# fig = plt.figure(figsize=(6,2.5))
ax3 = fig.add_subplot(223)
ours = [28.87, 28.87, 28.87, 28.87 , 28.87]
fdn = [17.92, 25.28 , 28.86, 28.57 , 28.19]
ircnn = [10.14, 22.06 , 28.86, 28.58 , 28.20]
csf = [15.54, 23.26, 28.09, 28.25, 27.96]
mlp = [14.54, 21.06, 27.32, 27.25, 27.16]
fd = [28.56, 28.56, 28.56, 28.56, 28.56]


x = range(len(ours))
ax3.set_ylim((10, 31))
ax3.plot(x, fd, '--', linewidth=line)
ax3.plot(x, csf, '--', linewidth=line)
ax3.plot(x, mlp, '--', linewidth=line)
ax3.plot(x, ircnn, '--', linewidth=line)
ax3.plot(x, fdn, '--', linewidth=line)
ax3.plot(x, ours, color='r', linewidth=ourline)
ax3.set_xticks(x)
ax3.set_xticklabels(x_label)
ax3.set_ylabel('PSNR', fontsize=11)
ax3.set_xlabel('$\sigma$', fontsize=13)
ax3.grid(linestyle='-.')

ax4 = fig.add_subplot(224)
ours = [0.879, 0.879, 0.879, 0.879, 0.879]
fdn = [0.347, 0.756 , 0.875, 0.862, 0.851]
ircnn = [0.106, 0.624 , 0.875, 0.874, 0.861]
mlp = [0.1405,0.387,0.811,0.7277,0.684]
fd = [0.817, 0.817, 0.817, 0.817, 0.817]
csf = [0.1605, 0.587,0.811,0.796,0.777]

x = range(len(ours))
ax4.set_ylim((0.1, 0.97))
ax4.plot(x, fd, '--', linewidth=line)
ax4.plot(x, csf, '--', linewidth=line)
ax4.plot(x, mlp, '--',  linewidth=line)
ax4.plot(x, ircnn, '--', linewidth=line)
ax4.plot(x, fdn, '--', linewidth=line)
ax4.plot(x, ours, color='r', linewidth=ourline)
ax4.set_xticks(x)
ax4.set_xticklabels(x_label)
ax4.set_ylabel('SSIM', fontsize=11)
ax4.set_xlabel('$\sigma$', fontsize=13)
ax4.grid(linestyle='-.')


legend = np.array(('FD', 'CSF', 'MLP', 'IRCNN', 'FDN', 'Ours'))
lgd = fig.legend([ax1, ax2, ax3, ax4], labels=legend, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6, fontsize=10)
# 
fig.tight_layout()
fig.savefig('compare_SOTA.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

