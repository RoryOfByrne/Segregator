import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import util.log
from util import constants

logger = util.log.logger

# TODO: Change functions to take 'filename' cos these parameters are getting out of hand

def plot_cluster(model, X, n, feature_model):
    fig = plt.figure(1, figsize=(4, 3), dpi=constants.DPI)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    labels = model.model.labels_

    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.dist = 12

    logger.info("Showing graph...")
    # plt.show()
    ax.azim = 200
    ax.elev = 10
    plt.savefig(''.join((constants.GRAPH_DIR, constants.CLUSTER_DIR, "%s-%s-%s.png" % (model.__str__(), feature_model, n))), dpi='figure')

def plot_truth(X, real_labels, model_name, n, feature_model):
    fig = plt.figure(2, figsize=(4, 3), dpi=constants.DPI)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    for name, label in [('realdonaldtrump', 0),
                        ('markhumphrys', 1),
                        ('barackobama', 2)]:
        ax.text3D(X[real_labels == label, 0].mean(),
                  X[real_labels == label, 1].mean(),
                  X[real_labels == label, 2].mean(),
                  name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(real_labels, [0, 1, 2]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    ax.azim = 200
    ax.elev = 10
    plt.show()
    plt.savefig(''.join((constants.GRAPH_DIR, constants.CLUSTER_DIR, "%s-%s-%s-truth.png" % (model_name, feature_model, n))), dpi='figure')