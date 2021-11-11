import matplotlib.pyplot as plt
import numpy as np
from src.DirichletMixtureModel import DirichletMixtureModel
import torch


def train(data, n_clusters, n_transpositions=12, n_dimensions=12, n_iter=1001, lr=.1, plot=True):
    """
    Helper function, trains a DMM on the data.
    :param data: np.ndarray, data of shape (num_data_points, dimension)
    :param n_clusters: int, number of cluster
    :param n_transpositions: int, number of transpositions, usually 12
    :param n_dimensions: int, dimension of the data, usually 12
    :param n_iter: int, number of gradient descent iterations
    :param lr: float, learning rate
    :param plot: bool, whether to plot the evolution of the loss
    :return: DirichletMixtureModel, float: trained model and final loss
    """

    # create a torch tensor to be augmented
    features = torch.from_numpy(data)
    # create the DMM
    model = DirichletMixtureModel(n_clusters, n_transpositions, n_dimensions)
    augmented_features = model.augment_tensor(features)
    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = np.zeros(n_iter)

    for it in range(n_iter):
        # compute the negative log likelihood
        optimizer.zero_grad()
        cost = model.neg_log_likelihood(augmented_features)
        # one step of gradient descent
        cost.backward()
        optimizer.step()
        # save the loss
        loss[it] = cost.data.cpu().numpy()

    if plot:
        plt.figure()
        plt.plot(loss)
        plt.tight_layout()
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('loss as a function  of num_iterations')
        plt.show()

    return model, cost
