import numpy as np
import matplotlib.pyplot as plt

def pro_data(data):
    return list(map(float, data.split('&')))

def get_data(q,w,e,r,t, y,u, n):
    return [q[n], w[n], e[n], r[n], t[n], y[n], u[n]]

colors = np.array([(152,245,255), (255,246,143), (127,255,212), (127,255,0),(255,185,15), (255,0,0)])/255.0
x_label = ['1%', '2%', '3%', '4%', '5%']

fig = plt.figure(figsize=(6,2.5))
ax1 = fig.add_subplot(121)
ours = [28.87, 28.87, 28.87, 28.87 , 28.87]
fdn = [17.92, 25.28 , 28.86, 28.57 , 28.19]
ircnn = [10.14, 22.06 , 28.86, 28.58 , 28.20]
csf = [15.54, 23.26, 28.09, 28.25, 27.96]
mlp = [14.54, 21.06, 27.32, 27.25, 27.16]
fd = [28.56, 28.56, 28.56, 28.56, 28.56]

line = 1.3
ourline = 1.7

x = range(len(ours))
ax1.set_ylim((10, 31))
ax1.plot(x, fd, '--', linewidth=line)
ax1.plot(x, csf, '-.', linewidth=line)
ax1.plot(x, mlp, '-.', linewidth=line)
ax1.plot(x, ircnn, '-.', linewidth=line)
ax1.plot(x, fdn, '-.', linewidth=line)
ax1.plot(x, ours, color='r', linewidth=ourline)
ax1.set_xticks(x)
ax1.set_xticklabels(x_label)
ax1.set_ylabel('PSNR', fontsize=11)
ax1.set_xlabel('$\sigma$', fontsize=13)
ax1.grid(linestyle='-.')

ax = fig.add_subplot(122)
ours = [0.879, 0.879, 0.879, 0.879, 0.879]
fdn = [0.347, 0.756 , 0.875, 0.862, 0.851]
ircnn = [0.106, 0.624 , 0.875, 0.874, 0.861]
mlp = [0.1405,0.387,0.811,0.7277,0.684]
fd = [0.817, 0.817, 0.817, 0.817, 0.817]
csf = [0.1605, 0.587,0.811,0.796,0.777]

x = range(len(ours))
ax.set_ylim((0.1, 0.97))
ax.plot(x, fd, '--', linewidth=line)
ax.plot(x, csf, '-.', linewidth=line)
ax.plot(x, mlp, '-.',  linewidth=line)
ax.plot(x, ircnn, '-.', linewidth=line)
ax.plot(x, fdn, '-.', linewidth=line)
ax.plot(x, ours, color='r', linewidth=ourline)
ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_ylabel('SSIM', fontsize=11)
ax.set_xlabel('$\sigma$', fontsize=13)
ax.grid(linestyle='-.')

legend = np.array(('FD', 'CSF', 'MLP', 'IRCNN', 'FDN', 'Ours'))
lgd = fig.legend([ax1, ax], labels=legend, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=6, fontsize=10)
# 
fig.tight_layout()
fig.savefig('compare_difflevel.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

