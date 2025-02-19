import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class conv_block(nn.Module):

    def __init__(self, in_c, out_c, size = 3, padding = 1, pool = False):
        super().__init__()
        if pool:
            self.convo = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_c, out_c, kernel_size = size, padding = padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
                )

        else :
            self.convo = nn.Sequential(
                            nn.Conv2d(in_c, out_c, kernel_size = size ,padding = padding),
                            nn.BatchNorm2d(out_c),
                            nn.ReLU()
                            )

    def forward(self, x):
        return self.convo(x)


class CNN(nn.Module):
    def __init__(self, input_channels= 3, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(input_channels, 32, size = 3),
            conv_block(32,32, size = 3),
            nn.Dropout2d(0.2),
            conv_block(32,64, size=3, pool= True),
            conv_block(64,64,size=3),
            nn.Dropout2d(0.2),
            conv_block(64,128,size=3, pool= True),
            conv_block(128,128,size=3),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):

        return self.classification(self.conv(x))
    

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs):
    
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar