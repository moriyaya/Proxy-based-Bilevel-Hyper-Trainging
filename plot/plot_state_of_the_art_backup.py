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

fig = plt.figure(figsize=(6,2.5))
ax1 = fig.add_subplot(121)
ours = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr,  5)
fdn = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 4)
ircnn =get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 3) 
mlp = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 2)
csf = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 1)
fd = get_data(sigma1_psnr, sigma2_psnr, sigma3_psnr, sigma4_psnr, sigma5_psnr, sigma6_psnr, sigma7_psnr, 0)

x = range(len(ours))
ax1.plot(x, fd, '--')
ax1.plot(x, csf, '--')
ax1.plot(x, mlp, '--')
ax1.plot(x, ircnn, '--')
ax1.plot(x, fdn, '--')
ax1.plot(x, ours, color='r', lw=2)
ax1.set_xticks(x)
ax1.set_xticklabels(x_label)
ax1.set_ylabel('PSNR', fontsize=11)
ax1.set_xlabel('$\sigma$', fontsize=13)
ax1.grid(linestyle='-.')

ax = fig.add_subplot(122)
ours = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim, sigma6_ssim, sigma7_ssim, 5)
fdn = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 4)
ircnn =get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 3) 
mlp = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 2)
csf = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 1)
fd = get_data(sigma1_ssim, sigma2_ssim, sigma3_ssim, sigma4_ssim, sigma5_ssim,  sigma6_ssim, sigma7_ssim, 0)

x = range(len(ours))
ax.plot(x, fd, '--')
ax.plot(x, csf, '--')
ax.plot(x, mlp, '--')
ax.plot(x, ircnn, '--')
ax.plot(x, fdn, '--')
ax.plot(x, ours, color='r')
ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_ylabel('SSIM', fontsize=11)
ax.set_xlabel('$\sigma$', fontsize=13)
ax.grid(linestyle='-.')

legend = np.array(('FD', 'CSF', 'MLP', 'IRCNN', 'FDN', 'Ours'))
lgd = fig.legend([ax1, ax], labels=legend, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=10)
# 
fig.tight_layout()
fig.savefig('compare_SOTA.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

# fig = plt.figure()
# ax1 = plt.subplot2grid((5,6), (0,0), colspan=2, rowspan=2)
# ax2 = plt.subplot2grid((5,6), (0,2), colspan=2, rowspan=2)
# ax3 = plt.subplot2grid((5,6), (0,4), colspan=2, rowspan=2)
# ax4 = plt.subplot2grid((5,6), (2,0), colspan=3, rowspan=3)
# ax5 = plt.subplot2grid((5,6), (2,3), colspan=3, rowspan=3)
# 
# plot_sub(ax1, sigma1_psnr, sigma1_ssim, (29, 32), (0.830, 0.950), colors, xlabel='1%')
# plot_sub(ax2, sigma2_psnr, sigma2_ssim, (27.5, 30), (0.800, 0.910), colors, xlabel='2%')
# plot_sub(ax3, sigma3_psnr, sigma3_ssim, (28, 28.9), (0.800, 0.890), colors, xlabel='3%')
# plot_sub(ax4, sigma4_psnr, sigma4_ssim, (27, 28.2), (0.750, 0.870), colors, xlabel='4%')
# plot_sub(ax5, sigma5_psnr, sigma5_ssim, (26, 27.8), (0.700, 0.860), colors, xlabel='5%')
# 
# lgd = fig.legend([ax1,ax2,ax3,ax4,ax5], labels=legend, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6)
# 
# fig.subplots_adjust(wspace =-10, hspace =0)
# fig.tight_layout()
# fig.savefig('state_PSNR1.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()
# 