import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# labels_hyper = ['No Hyper', 'Manually Designed Hyper', 'Ours']
labels_hyper = ['No Guidance (tiny)', 'With Guidance (tiny)', 'No Guidance (general)', 'With Guidance (general)']
labels = ['0.6%', '1%', '2%', '3%', '4%', '5%']
no_hyper = [0.961, 0.954, 0.936, 0.916, 0.897, 0.875]
manual_hyper = [0.966, 0.959, 0.939, 0.920, 0.903, 0.887]
ours = [0.966, 0.960, 0.941, 0.926, 0.912, 0.892]
no_hyper_tiny = [0.961, 0.955, 0.936, 0.916, 0.895, 0.868]
manual_hyper_tiny = [0.962, 0.956, 0.937, 0.918, 0.899, 0.879]

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

colors = np.array([(152,245,255), (255,246,143), (127,255,212), (127,255,0),(255,185,15), (255,0,0), (107,142,35), (122,103,238)])/255.0

fig, ax = plt.subplots()
ax.set_ylim([0.865, 0.985])
# rects1 = ax.bar(x - width/2, no_hyper, width, label=labels_hyper[0], color=colors[4])
# rects2 = ax.bar(x + width/2, manual_hyper, width, label=labels_hyper[1], color='r')
# rects3 = ax.bar(x + width, ours, width, label=labels_hyper[2], color='r')
rects1 = ax.bar(x - width*1.5, no_hyper_tiny, width, label=labels_hyper[0], color=colors[4])
rects2 = ax.bar(x - width*0.5, manual_hyper_tiny, width, label=labels_hyper[1], color=colors[0])
# rects3 = ax.bar(x + width, ours, width, label=labels_hyper[2], color='r')
rects4 = ax.bar(x + width*0.5, no_hyper, width, label=labels_hyper[2], color=colors[7])
rects5 = ax.bar(x + width*1.5, manual_hyper, width, label=labels_hyper[3], color='r')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('SSIM', fontsize=20)
ax.set_xlabel('$\sigma$', fontsize=27)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)

# ylabel = [0.88,0.90,0.92,0.94,0.96,0.98]
# ax.set_yticklabels(ylabel, fontsize=20)
# plt.rc('ytick', labelsize=20)
ax.legend(fontsize=13)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(-3, 2),  # 3 points vertical offset
                    rotation=-30,
                    textcoords="offset points",
                    ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
ax.tick_params(labelsize=20)

fig.tight_layout()

plt.show()
