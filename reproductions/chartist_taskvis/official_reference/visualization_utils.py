# -*- coding: utf-8 -*-
"""
@author: Sruthi and Yao
"""

"""usage
fig = plot_img(im, DPI)
plot_visualization(x,y,duration)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def plot_img(im, DPI=100):
    width, height = im.size # original image size
    fig = plt.figure(figsize=(width/DPI,height/DPI))
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.imshow(im)
    return fig

def plot_scanpath(x, y, duration):
    plt.xticks([])
    plt.yticks([])
    #colors=plt.cm.rainbow(np.linspace(0,1,40))
    #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    plt.plot(x, y, '-o', color='blue',lw=3,alpha=0.4)

    for i in range(len(x)):
        if i == 0:
            li = plt.plot(x[i],y[i],'-o', color='blue', ms=25,alpha=0.99,zorder=10)
            plt.setp(li, 'markersize', 15)
        # elif i == len(x)-1:
        #     dur = duration[i] / 10 if duration[i] / 10>10 else 10
        #     dur = dur if dur<15 else 15
        #     li = plt.plot(x[i],y[i],'-o', color='red', ms=dur,alpha=0.99,zorder=5)
        #     plt.setp(li, 'markersize', 15)
        else:
            dur = duration[i] / 10 if duration[i] / 10>10 else 10
            dur = dur if dur<15 else 15
            li = plt.plot(x[i],y[i],'-o', color='yellow', ms=dur,alpha=0.6,zorder=5)
            plt.setp(li, 'markersize', 15)

        # li = plt.plot(x[i],y[i],'-o', color='blue', ms=25,alpha=0.99,zorder=10)
        # plt.setp(li, 'markersize', 15)

        # plt.annotate(i, (x[i], y[i]), color='yellow',weight='bold',fontsize=10)

def plot_heatmap(x, y, duration):
    sns.kdeplot(
        x=x, 
        y=y, 
        # cut = 0,
        bw_method = 0.4,
        bw_adjust = 0.6,
        cbar = False,
        levels = 110,
        # shade = True,
        cmap = cm.jet,
        alpha = 0.2
    )
    plt.gca().set_axis_off()  # Hide the axis
    plt.margins(0, 0)  # Set margins to zero
    plt.tight_layout(pad=0)  # Minimize the padding

def plot_scanpath_colormap(x, y, duration):
    plt.xticks([])
    plt.yticks([])
    colors=plt.cm.rainbow(np.linspace(0,1,40))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    plt.plot(x, y, '-ro', color='blue',lw=3,alpha=0.4)
    for i in range(len(x)):
        dur = duration[i] / 10 if duration[i] / 10>10 else 10
        dur = dur if dur<20 else 20
        li = plt.plot(x[i],y[i],'-o',ms=dur,alpha=0.6,zorder=5)
        plt.setp(li, 'markersize', 20)

        #plt.annotate(i, (x[i], y[i]), color='yellow',weight='bold',fontsize=10)
