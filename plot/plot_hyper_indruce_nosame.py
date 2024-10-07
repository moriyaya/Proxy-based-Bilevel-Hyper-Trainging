import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# labels_hyper = ['No Hyper', 'Manually Designed Hyper', 'Ours']
labels_hyper = ['No Guidance', 'With Guidance']
labels = ['0.6%', '1%', '2%', '3%', '4%', '5%']
no_hyper = [32.97, 32.20, 30.59, 29.37, 28.39, 27.48]
manual_hyper = [33.60, 32.76, 30.95, 29.66, 28.70, 27.87]
ours = [33.74, 32.95, 31.08, 29.83, 28.83, 28.05]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

colors = np.array([(152,245,255), (255,246,143), (127,255,212), (127,255,0),(255,185,15), (255,0,0)])/255.0

fig, ax = plt.subplots()
ax.set_ylim([27, 34.5])
rects1 = ax.bar(x - width/2, no_hyper, width, label=labels_hyper[0], color=colors[4])
rects2 = ax.bar(x + width/2, manual_hyper, width, label=labels_hyper[1], color='r')
# rects3 = ax.bar(x + width, ours, width, label=labels_hyper[2], color='r')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('PSNR', fontsize=20)
ax.set_xlabel('$\sigma$', fontsize=30)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend(fontsize=13)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-2, 2),  # 3 points vertical offset
                    rotation=-30,
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
# autolabel(rects3)
ax.tick_params(labelsize=20)

fig.tight_layout()

plt.show()
