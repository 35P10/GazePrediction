import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0.5, 0.8, N+4)
    #mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4) # full transparent
    return mycmap

def plot_heatmap(image_path, heat_map_path, alpha=0.5, display=False, save=None, axis='on'):
    image = Image.open(image_path)
    heat_map = Image.open(heat_map_path)

    mycmap = transparent_cmap(plt.cm.jet)

    assert image.size[0] == heat_map.size[0] and image.size[1] == heat_map.size[1]

    # display
    plt.imshow(image)
    plt.imshow(heat_map, alpha=alpha, cmap=mycmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.close()


if __name__ == "__main__":
    # read image
    image_path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/targets/economist_daily_chart_4.png'

    # read heatmap
    heat_map_path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_3000_5000_10000/salmaps/economist_daily_chart_4_10000.png'
    #heat_map_path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/MDSEM_bucket_500_2000_5000/salmaps/economist_daily_chart_4_3000.png'
    #heat_map_path = '/netpool/homes/wangyo/Dataset/Massvis/targets393/GT_INDIV_SIGMA10/heatmaps/10000/economist_daily_chart_4/abl.png'

    plot_heatmap(image_path, heat_map_path, alpha=0.7, save='test.png', axis='off')
