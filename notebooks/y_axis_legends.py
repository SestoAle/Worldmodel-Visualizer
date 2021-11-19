import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
cmap = cm.get_cmap('tab10')
sns.set_theme(style="dark")
# from matplotlib.backends.qt_compat import QtWidgets
# qApp = QtWidgets.QApplication(sys.argv)
# print(qApp.desktop().physicalDpiX())
# plt.matplotlib.rcParams['figure.dpi'] = qApp.desktop().physicalDpiX()

y = np.random.randint(0, 10, 100)
x = np.arange(100)
print(y)


fig = plt.figure()
# axes = plt.subplot()
# # plt.axis('off')
gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
ax1, ax2 = gs.subplots(sharex='col', sharey='row')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


# axes.clear()
colors = []
for p in y:
    colors.append(cmap(p))
ax2.scatter(x, y, c=colors, s=10, marker='s')
# plt.set_xlim(0, 100)

# ax1.scatter(x, y, c=colors, s=10, marker='s')
# y_ticks = ['Action 1', 'Action 2', 'Action 2', 'Action 2', 'Action 2', 'Action 2' ,'Action 2' ,'Action 2', 'Action 2',
#            'Action 2']
# y_ticks = ['' for i in range(10)]
# ax1.set_yticks(np.arange(10))
# ax1.set_yticklabels(y_ticks)
tick_colors = [cmap(p)[:3] for p in range(10)]
ax1.tick_params(axis='y', direction='in', labelsize=5, pad=-30)
for ticklabel, tickcolor in zip(ax1.get_yticklabels(), tick_colors):
    print(ticklabel)
    ticklabel.set_color(tickcolor)
ax1.figure.set_size_inches(2, 10)

ax1.label_outer()
ax2.label_outer()

plt.show()
