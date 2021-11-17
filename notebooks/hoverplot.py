#%%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#%%

def plot_v_line(x, line, axes):
    global line1
    global line2
    global axis1
    global axis2
    if x is not None:
        if axes == axes1:
            line2.remove()
            line2 = plt.axvline(x)
        else:
            line1.remove()
            line1 = plt.axvline(x)
        plt.draw()


#%%
rgb = cm.get_cmap('tab10')
fig1 = plt.figure()
axes1 = plt.subplot()
points = np.random.randn(500, 2)
axes1.plot(points[:, 0], points[:, 1])
line1 = axes1.axvline(0)
axes1.set_xlim(-3, 3)
# fig = plt.gcf()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fig2 = plt.figure()
axes2 = plt.subplot()
points = np.random.randn(500, 2)
axes2.plot(points[:, 0], points[:, 1])
line2 = axes2.axvline(0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

#%%
fig1.canvas.mpl_connect("motion_notify_event", lambda x: plot_v_line(x.xdata, line1, axes1))
fig2.canvas.mpl_connect("motion_notify_event", lambda x: plot_v_line(x.xdata, line2, axes2))


#%%

plt.show()