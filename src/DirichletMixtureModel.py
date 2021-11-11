import torch
from torch.nn import Module, Parameter
from torch.distributions import Dirichlet
import numpy as np
from typing import Tuple


class DirichletMixtureModel(Module):

    def __init__(self, num_clusters: int, num_transpositions: int = 12, dimension: int = 12) -> None:
        """
        Create a Dirichlet Mixture Model class.
        :param num_clusters: int, number of clusters to consider
        :param num_transpositions: int, number of possible transpositions of each vector
        :param dimension: int, dimension of the data points considered
        """
        super(DirichletMixtureModel, self).__init__()

        # save the parameters of the mixture model
        self.__num_clusters = num_clusters
        self.__num_transpositions = num_transpositions
        self.__dimension = dimension
        # Create the tensors for the parameters to optimize
        self.__alphas = 1 + torch.rand(num_clusters, dimension) * 1e-1
        self.__dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.__alphas = Parameter(self.__alphas.type(self.__dtype).contiguous())

    def augment_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """
        Augment the data tensor to consider the number of possible clusters and the number of transpositions.
        This methods makes num_clusters copies (done by broadcasting) of the data, then for each copy create num_transpositions variants of the data
        corresponding to the rotated vectors.
        This method is intended to be used only once before training, therefore saving computing time.
        :param X: torch.Tensor,  2-dimensional tensor, of shape (num_data_points, data_dimension)
        :return: torch.Tensor, augmented data, of shape (num_data_points, num_transpositions, 1, data_dimension)
        """
        # create a tensor for all (12) transpositions
        augmented_for_transp = torch.zeros((self.__num_transpositions, X.shape[0], X.shape[1]))
        # use torch.roll to simulate transposition on the data points
        for rotation in range(self.__num_transpositions):
            augmented_for_transp[rotation] = torch.roll(X, rotation, 1)
        # add a dimension for the number of clusters, and transpose to get the desired shape
        augmented_for_clusters = augmented_for_transp.view(1, self.__num_transpositions, X.shape[0], -1)
        augmented_for_clusters = augmented_for_clusters.transpose(0, 2)
        augmented_for_clusters.requires_grad_(True)
        return augmented_for_clusters.type(self.__dtype)

    def neg_log_likelihood(self, X: torch.Tensor) -> float:
        """
        Fit the dirichlet Model to the data tensor X. This corresponds to doing an MLE on the data in order to learn the parameters,
        assuming that conditional on the cluster and the transposition, each data point is distributed according to a Dirichlet
        probability density function.
        Negative Log-Likelihood of the data is computed, then torch autograd functions will be used for the gradient descent phase.
        :param X: torch.Tensor,  4-dimensional tensor, of shape (num_data_points, num_transpositions, 1, data_dimension)
        :return: float, negative log-likelihood of the data
        """
        # The alpha parameters of the Dirichlets distributions need to be positive, we use exponentiation to
        # make sure this is the case
        exp_alphas = torch.exp(self.__alphas)
        # create num_clusters Dirichlets distributions, this is done by calling the torch.Dirichlet with
        # a matrix of parameters, where each row is the alpha vector of a cluster
        dirichlets = Dirichlet(exp_alphas)
        # compute the log likelihood, using the logsumexp function, and keeping the correct constants for consistency
        log_probs = dirichlets.log_prob(X)
        log_likelihood = torch.logsumexp(log_probs, dim=(1, 2))
        num_datapoints = X.shape[0]
        return -(num_datapoints * np.log(1 / (self.__num_transpositions * self.__num_clusters)) + log_likelihood.sum())

    def get_cluster_probabilities(self, X: torch.Tensor) -> np.ndarray:
        """
        Return the final probabilities for each data point and each cluster, after training.
        Normalization constants are kept, in order to get meaningful probabilities.
        :param X: torch.Tensor,  4-dimensional tensor, of shape (num_data_points, num_transpositions, 1, data_dimension)
        :return: np.ndarray, of shape (num_data_points, num_clusters) containing the final probabilities.
        """
        exp_alphas = torch.exp(self.__alphas)
        dirichlets = Dirichlet(exp_alphas)
        probs = torch.exp(dirichlets.log_prob(X))
        # for each data_point, sum the probabilities on transpositions and clusters to get p(x)
        prob_x = probs.sum(dim=(1, 2)).view(X.shape[0], 1)
        # similarly sum on the transpositions to p(x|c)
        prob_x_knowing_c = probs.sum(dim=1)
        return (prob_x_knowing_c / prob_x).detach().numpy()

    def get_clusters(self, X: torch.Tensor) -> np.ndarray:
        """
        Return the cluster for each data point, after training.
        :param X: torch.Tensor,  4-dimensional tensor, of shape (num_data_points, num_transpositions, 1, data_dimension)
        :return: np.ndarray, of shape (num_data_points, 1) containing the cluster for each data point.
        """
        return np.argmax(self.get_cluster_probabilities(X), axis=1)

    def get_cluster_and_transposition_probabilities(self, X) -> np.ndarray:
        """
        returns probabilities of each cluster, transposition combination for each data_point.
        Knowing that the transposition is cluster specific, it is not meaningful to marginalize
        in order to obtain the transposition for each data_point
        :param X: torch.Tensor,  4-dimensional tensor, of shape (num_data_points, num_transpositions, 1, data_dimension)
        :return: np.ndarray, of shape (num_data_points, num_transpositions, 1, data_dimension)
        """
        exp_alphas = torch.exp(self.__alphas)
        dirichlets = Dirichlet(exp_alphas)
        probs = torch.exp(dirichlets.log_prob(X))
        return probs.detach().numpy()

    def get_cluster_and_transposition(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns for each data point, the most probable cluster, transposition pair.
        :param Tuple[np.ndarray, np.ndarray], where:
            Tuple[0]: np.ndarray of shape (num_data_points,1), most probable cluster
            Tuple[1]: np.ndarray of shape (num_data_points,1), most probable transposition
        """
        probs = self.get_cluster_and_transposition_probabilities(X)
        dims = probs.shape
        argmaxs = probs.reshape((dims[0], -1)).argmax(axis=1)
        return argmaxs % self.__num_clusters, argmaxs // self.__num_clusters

    def get_alphas(self) -> np.ndarray:
        """
        :return: np.ndarray, of shape(num_clusters, data_dimension), copy of the alpha parameters for the cluster
        """
        return self.__alphas.detach().numpy()
