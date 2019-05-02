#
# Plot Predicted Rainfall Data given single dataset
#
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from colormap_JMA import Colormap_JMA

# plot rainfall map for single data
# This routine expects "data" to be a numpy array in [Tsize,Xsize,Ysize] dimendion
def plot_rainfall_map(data,fname,pic_path):
    pic = data
    # print
    print('Plotting: ',fname)
    print('min and max values:',np.min(pic),np.max(pic))
    # plot
    cm = Colormap_JMA()
    # output as stationary image
    fig, ax = plt.subplots(1,6,figsize=(18, 3.5))
    fig.suptitle("Precip output : "+fname, fontsize=20)
    for nt in range(6):
        id = nt*2+1
        pos = nt+1
        dtstr = str((id+1)*5)
        # plotting
        plt.subplot(1,6,pos)
        im = plt.imshow(pic[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
        plt.title(dtstr+"min")
        plt.grid()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # save as png
    plt.savefig(pic_path+'/'+fname+'.png')
    plt.close()
        
if __name__ == '__main__':
    # test plotting capability by random value
    data = np.random.rand(12,200,200)*50.0
    fname = 'tset_output'
    pic_path = './'

    scaling = 'log'

    plot_rainfall_map(data,fname,pic_path)


