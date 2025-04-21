import math
import copy
import numpy as np
from scipy.special import betainc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from rashomon_sets.ellipsoid_tools import (
    hyperplane_ellipsoid_intersection,
    in_ellipsoid,
)
import time
import cvxpy as cp


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TorchLogisticRegression(nn.Module):
    def __init__(self, input_size=2000, num_classes=200, trained_weights=None):
        super(TorchLogisticRegression, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size, num_classes, bias=False)
        if trained_weights is not None:
            self.linear.weight.data = trained_weights.reshape(num_classes, input_size)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.linear(x)
        return out

    def predict_proba(self, x):
        probs = self.forward(x)
        return nn.functional.softmax(probs, dim=1)

    def predict(self, x):
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds

    def get_params(self):
        return self.linear.weight.data.reshape(-1).detach()


class TorchLogisticRegressionPositiveOnly(nn.Module):
    def __init__(self, prototype_class_identity, trained_weights=None):
        super(TorchLogisticRegressionPositiveOnly, self).__init__()
        """
        Args:
            prototype_class_identity : tensor (num_prototypes, num_classes) -- An indicator
                matrix in which index [i, j] is 1 if prototype i is from class j
        """
        self.prototype_class_identity = prototype_class_identity
        if trained_weights is None:
            self.linear_weights = torch.nn.Parameter(
                torch.ones(prototype_class_identity.shape[0])
            )
        else:
            self.linear_weights = torch.nn.Parameter(trained_weights)

    def predict_with_coeffs(self, x, coeff):
        # x is (num_rows, num_ftrs))
        x = x.to(torch.float32)
        # self.linear_weights is a vector of num_prototypes
        # We form a (num_prototypes, num_classes) matrix where
        # inter class connections are 0
        overall_params = coeff.unsqueeze(1) * self.prototype_class_identity
        out = x @ overall_params
        return out

    def forward(self, x):
        # x is (num_rows, num_ftrs))
        x = x.to(torch.float32)
        # self.linear_weights is a vector of num_prototypes
        # We form a (num_prototypes, num_classes) matrix where
        # inter class connections are 0
        overall_params = (
            self.linear_weights.unsqueeze(1) * self.prototype_class_identity
        )
        out = x @ overall_params
        return out

    def predict_proba(self, x):
        probs = self.forward(x)
        return nn.functional.softmax(probs, dim=1)

    def predict(self, x):
        probs = self.predict_proba(x)
        preds = torch.argmax(probs, dim=1)
        return preds

    def get_params(self):
        return self.linear_weights.data.reshape(-1).detach()


class MultiClassLogisticRSet:
    """
    A class to define the linear regression Rashomon set
    under logistic loss
    Args:
        rashomon_bound_multiplier: float -- If not none, the maximum loss allowed
            in the Rashomon set will be optimal * rashomon_bound_multiplier
        absolute_rashomon_bound: float -- If not none, the maximum loss allowed
            in the Rashomon set will be absolute_rashomon_bound
        num_models: int -- The number of models to sample from the Rashomon set
        reg: str ('l1', 'l2', or None) -- Which type of regularization to apply
        lam: float -- The weight of the regularization
        max_iter: int -- The maximum number of iterations to perform when computing
            our optimal linear layer
        compute_hessian_batched: bool -- If true, compute the hessian wrt multiple
            samples at a time. Memory inefficient, but fast
        directly_compute_hessian: bool -- If True, we'll directly compute and store
            the hessian; else, we'll store the dataset and do repeat computation
        lr_for_opt: float -- The learning rate to use in stochasic gradient descent
    """

    def __init__(
        self,
        rashomon_bound_multiplier=1.05,
        absolute_rashomon_bound=None,
        num_models=0,
        reg="l2",
        lam=0.0001,
        max_iter=10_000,
        compute_hessian_batched=False,
        directly_compute_hessian=True,
        device=torch.device("cpu"),
        lr_for_opt=0.1,
        opt_tol=5e-5,
        verbose=False
    ):
        # Whether to produce additional logging
        self.verbose = verbose

        # The matrix defining the shape of our ellipsoid
        self.stretch_matrix = None

        # The center of our ellipsoid
        self.center = None

        # The volume of the ellipsoid; stored to save
        # repear computation
        self.volume = 0

        # The loss threshold for which models to include
        # in the R-Set
        self.rashomon_bound_multiplier = rashomon_bound_multiplier

        self.absolute_rashomon_bound = absolute_rashomon_bound

        # The optimal loss value obtained for reference
        self.opt_loss = -1
        self.opt_tol = opt_tol

        # The learning rate to use for SGD
        self.lr_for_opt = lr_for_opt

        # Whether we'll use a subsample of rows to estimate the hessian
        self.compute_hessian_batched = compute_hessian_batched

        self.directly_compute_hessian = directly_compute_hessian

        self.device = device

        # The max iterations for logistic regression fit
        self.max_iter = max_iter

        # A flag to indicate whether this set is empty
        self.empty_set = False

        self.num_models = num_models

        self.reg = reg

        self.lam = lam

    def rset_predict(self, X):
        preds = torch.stack([m.predict(X) for m in self.sampled_models], dim=-1)
        return preds

    def rset_predict_proba(self, X):
        preds = torch.stack([m.predict_proba(X) for m in self.sampled_models], dim=-1)
        return preds

    def fit_optimal_model(self, X, Y, model):
        """
        Builds the optimal prediction head for the given data.
        Args:
            X -- an n x d array of features
            Y -- an n x 1 array of labels
        """

        # assert (X[:, 0] == 1).all(), "Error: Please add a column of all ones to the distances dataset for bias"

        self.X_input = X.to(self.device)
        self.Y_input = Y.to(self.device)

        if type(Y) is not torch.Tensor:
            Y = torch.tensor(Y).type(torch.LongTensor)
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
        train_dataset = TensorDataset(X, Y)
        train_loader = DataLoader(train_dataset, batch_size=X.shape[0], shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_for_opt)

        if self.verbose:
            print(
                "Accuracy of model before training: ",
                accuracy_score(
                    self.Y_input.detach().cpu().numpy(),
                    model.predict(self.X_input).detach().cpu().numpy(),
                ),
            )

        # Train the model
        start = time.time()
        old_loss = None
        for epoch in tqdm(range(self.max_iter), desc="Running SGD to find optimal model"):
            for i, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to the device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(inputs)
                all_params = model.linear_weights
                if self.reg == "l1":
                    loss = criterion(outputs, labels) + self.lam * torch.norm(
                        all_params, p=1
                    )
                elif self.reg == "l2":
                    loss = criterion(outputs, labels) + self.lam * torch.norm(
                        all_params, p=2
                    )
                else:
                    loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print training loss for each epoch
            if (epoch + 1) % 10 == 0 and self.verbose:
                print(
                    "Epoch [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, self.max_iter, loss.item()
                    )
                )

            if old_loss is not None:
                if old_loss - loss.item() < self.opt_tol:
                    break
            old_loss = loss.item()

        if self.verbose:
            print(f"Performed SGD in {time.time() - start} seconds")
            print(f"Final loss value was {loss}")
        self.optimal_model = model

        # model.linear.weight comes out as (num_classes, num_ftrs)
        if self.verbose:
            print(
                "Accuracy of optimal model: ",
                accuracy_score(
                    self.Y_input.detach().cpu().numpy(),
                    self.optimal_model.predict(self.X_input).detach().cpu().numpy(),
                ),
            )
        self.center = self.optimal_model.get_params()
        # TODO: Add some asserts checking that indices correspond as expected

        self.opt_loss = criterion(self.optimal_model(self.X_input), self.Y_input)

        if self.reg == "l2":
            self.opt_loss += self.lam * torch.norm(self.center, p=2)
        elif self.reg == "l1":
            self.opt_loss += self.lam * torch.norm(self.center, p=1)

        if self.absolute_rashomon_bound is None:
            self.absolute_rashomon_bound = (
                self.rashomon_bound_multiplier * self.opt_loss
            )
        if self.verbose:
            print(f"Found absolute_rashomon_bound loss of {self.absolute_rashomon_bound} given multiplier {self.rashomon_bound_multiplier}")

        # TODO: Document -- rename this, rather than converting from mult to add
        self.relative_rashomon_bound = self.absolute_rashomon_bound - self.opt_loss

    def fit(self, X, Y):
        """
        Builds the linear regression RSet for the given data.
        Args:
            X -- an n x d array of features
            Y -- an n x 1 array of labels
        """

        # Fit our logistic regression model ================
        model = TorchLogisticRegression(
            input_size=X.shape[1], num_classes=torch.unique(Y).shape[0]
        ).to(self.device)
        self.fit_optimal_model(X, Y, model)
        with torch.no_grad():
            # I'm going to be working with the hessian from here:
            # https://stats.stackexchange.com/questions/525042/derivation-of-hessian-for-multinomial-logistic-regression-in-b%C3%B6hning-1992
            # predicted_probabilities is a (num_rows, num_classes) matrix

            predicted_probabilities = self.optimal_model.predict_proba(self.X_input)
            num_rows, num_classes = predicted_probabilities.shape
            num_ftrs = self.X_input.shape[1]

            if not self.directly_compute_hessian:
                pass
            elif self.compute_hessian_batched:
                # I'm going to compute the hessian for each sample, then average at the end
                X_dummy_dim = self.X_input.reshape((num_rows, num_ftrs, 1))
                X_dummy_dim_trans = self.X_input.reshape((num_rows, 1, num_ftrs))

                # all_xxT is (num_rows, num_ftrs, num_ftrs)
                all_xxT = torch.bmm(X_dummy_dim, X_dummy_dim_trans)

                # Build our probability matrices
                # Initially, use the formula for off diagonal eles to fill everything in
                prob_matrices = torch.bmm(
                    predicted_probabilities.reshape((num_rows, num_classes, 1)),
                    predicted_probabilities.reshape((num_rows, 1, num_classes)),
                )
                for c in range(num_classes):
                    prob_matrices[:, c, c] = -1 * (
                        predicted_probabilities[:, c]
                        * (1 - predicted_probabilities[:, c])
                    )

                self.stretch_matrix = torch.zeros(
                    (num_ftrs * num_classes, num_ftrs * num_classes)
                )
                for i in range(num_rows):
                    self.stretch_matrix += torch.kron(prob_matrices[i], all_xxT[i])
            else:
                self.stretch_matrix = None
                X = self.X_input

                for i in tqdm(range(num_rows)):
                    # I'm going to compute the hessian for each sample, then average at the end
                    X_dummy_dim = X[i].reshape((num_ftrs, 1))
                    X_dummy_dim_trans = X[i].reshape((1, num_ftrs))

                    # all_xxT is (num_ftrs, num_ftrs)
                    all_xxT = torch.matmul(X_dummy_dim, X_dummy_dim_trans)

                    # Build our probability matrices
                    # Initially, use the formula for off diagonal eles to fill everything in
                    prob_matrices = torch.matmul(
                        predicted_probabilities[i].reshape((num_classes, 1)),
                        predicted_probabilities[i].reshape((1, num_classes)),
                    )
                    for c in range(num_classes):
                        prob_matrices[c, c] = -1 * (
                            predicted_probabilities[i, c]
                            * (1 - predicted_probabilities[i, c])
                        )

                    if self.stretch_matrix is None:
                        self.stretch_matrix = torch.kron(prob_matrices, all_xxT)
                    else:
                        self.stretch_matrix += torch.kron(prob_matrices, all_xxT)

                    del X_dummy_dim, X_dummy_dim_trans, all_xxT, prob_matrices

                self.stretch_matrix = self.stretch_matrix

            if self.directly_compute_hessian:
                self.stretch_matrix = self._adjust_hessian(self.stretch_matrix)

            # Finally, computing the volume for our ellipsoid------
            # self.volume = self._get_volume(self.stretch_matrix)
            self.sampled_models = [self._sample_model() for _ in range(self.num_models)]

    def _adjust_hessian(self, hessian):
        # TODO: Double check this -- I'm negating the Hessian because it was coming out
        # as negative semi-definite, meaning something definitely went wrong because this should
        # be a convex problem. Need to double check my math here

        if self.reg == "l2":
            reg_hessian = torch.diag(
                torch.tensor(
                    [
                        2 / torch.norm(self.center, p=2)
                        for _ in range(self.center.shape[0])
                    ],
                    device=self.center.device,
                )
            )
            reg_hessian = reg_hessian - 4 * self.center.reshape(
                -1, 1
            ) @ self.center.reshape(1, -1) / (torch.norm(self.center, p=2) ** 3)

            hessian = hessian + reg_hessian.to(self.device) * self.lam
        # We actually don't need to adjust the hessian for L1, since the second derivitive of abs val is 0

        hessian = hessian / self.relative_rashomon_bound

        return hessian

    def _sample_model_with_constraints(self, required_protos):
        '''
        required_protos should be a list of (index, min_coef) tuples
        '''
        n = self.stretch_matrix.shape[0]
        v = cp.Variable(n)
        constraints = []
        for i, m in required_protos:
            constraints.append(v[i] >= m)
        prob = cp.Problem(cp.Minimize(cp.quad_form(v - self.center.cpu().detach().numpy(), self.stretch_matrix.cpu().detach().numpy(), assume_PSD=True)), constraints)
        prob.solve()
        if prob.value > 1:
            print("infeasible")
            return None
        else:
            return self.wrap_sampled_params(torch.tensor(v.value, device=self.stretch_matrix.device, dtype=self.stretch_matrix.dtype))

    def _sample_model(self, target_dist_to_edge=None, direction_to_step=None):
        """
        After fitting a Rashomon set, this function samples a random coefficient vector from
        somewhere in the Rashomon ellipsoid. In particular, we pick a random direction, and find
        the point on the surface of the ellipsoid that direction hits. We then interpolate between
        the center and that point according to target_dist_to_edge.
        Args:
            target_dist_to_edge : float -- What proportion of the distance between the center
                and the edge of the ellipse we should walk. If 0, walk to the edge; if 1, return
                the center.
        """

        with torch.no_grad():
            if target_dist_to_edge is None:
                target_dist_to_edge = (1 - torch.rand(1)).item()

            """
            Let S = H / (rashomon_thresh - opt_loss). We want to find a step length a in [0, 1] such that
            (0.5 * (a * direction_to_step + center - center)^T S (a * direction_to_step + center - center)) = 1
            => (a * direction_to_step)^T S (a * direction_to_step) = 2

            This is quadratic in a, so we can use quadratic formula. Here, the (overloaded) quadradic formula coeffs are
            a = (direction_to_step^T S direction_to_step)
            b = 0
            c = -2
            """
            if direction_to_step is None:
                # If I normalize this, it might help add some stability
                direction_to_step = (
                    torch.rand(self.center.shape[0]).to(self.device) - 0.5
                )
                # direction_to_step = direction_to_step / torch.norm(direction_to_step)

            # Since we're dealing with convex problems, our hessian should be semi-definite positive,
            # so this should be >= 0
            if self.directly_compute_hessian:
                a_for_quad = (
                    direction_to_step.T @ self.stretch_matrix @ direction_to_step
                )
            else:
                """
                self.prob_matrices will be a (n, num_classes, num_classes) tensor
                self.all_xxT will be a (n, num_ftrs, num_ftrs) tensor
                We're gonna leverage the fact that (A kron B) vec(V) = vec(BVA^T)
                First, form the (num_classes, num_ftrs) matrix V
                """
                predicted_probabilities = self.optimal_model.predict_proba(self.X_input)

                # Build our probability matrices
                # Initially, use the formula for off diagonal eles to fill everything in
                a_for_quad = 0
                for i in range(predicted_probabilities.shape[0]):
                    prob_matrices = torch.mm(
                        predicted_probabilities[i].reshape(
                            (predicted_probabilities.shape[1], 1)
                        ),
                        predicted_probabilities[i].reshape(
                            (1, predicted_probabilities.shape[1])
                        ),
                    )
                    for c in range(predicted_probabilities.shape[1]):
                        prob_matrices[c, c] = -1 * (
                            predicted_probabilities[i, c]
                            * (1 - predicted_probabilities[i, c])
                        )

                    # I'm going to compute the hessian for each sample, then average at the end
                    X_dummy_dim = self.X_input[i].reshape((self.X_input.shape[1], 1))
                    X_dummy_dim_trans = self.X_input[i].reshape(
                        (1, self.X_input.shape[1])
                    )

                    # all_xxT is (num_rows, num_ftrs, num_ftrs)
                    all_xxT = torch.mm(X_dummy_dim, X_dummy_dim_trans).to(torch.float32)

                    direction_to_step = direction_to_step.to(self.device)
                    direction_to_step_expanded = direction_to_step.view(
                        (-1, self.X_input.shape[1])
                    ).permute(1, 0)
                    tmp = torch.mm(
                        direction_to_step_expanded, prob_matrices.permute(1, 0)
                    )
                    # This mean is averaging our hessian over samples
                    tmp = torch.mm(all_xxT, tmp).permute(1, 0).flatten()

                    a_for_quad += direction_to_step.T @ tmp
                a_for_quad = -self._adjust_hessian(a_for_quad)

            b_for_quad = 0
            c_for_quad = -2
            assert (
                a_for_quad > 0
            ), f"Error: Found a_for_quad = {a_for_quad}, which should be impossible for a convex problem"

            # Find length of step to take using quadratic formula
            len_of_step_along_dir = (
                -b_for_quad + torch.sqrt(b_for_quad**2 - 4 * a_for_quad * c_for_quad)
            ) / (2 * a_for_quad)
            new_param_vec = (
                direction_to_step * len_of_step_along_dir * (1 - target_dist_to_edge)
                + self.center
            )

            return self.wrap_sampled_params(new_param_vec)

    def _drop_prototype(
        self, target_variable: int, inplace: bool = True, resample: bool = True
    ):
        """
        Filter our Rashomon set down to the subset that excludes
        the given variables
        Args:
            target_variables : int -- the prototype index
                corresponding to the prototype that should be
                avoided
            inplace : bool -- If true, directly modify this rset
            resample : bool -- If true, sample a new set of models
                from our filtered rset
        Returns:
            rset : MultiClassLogisticRSet -- The filtered Rset
            do_intersect : bool -- Whether or not we were able to
                remove the specified variable
        """
        if inplace:
            rset = self
        else:
            rset = copy.deepcopy(self)

        normal_vector = torch.zeros_like(rset.center)
        normal_vector[target_variable] = 1

        (
            new_stretch_matrix,
            new_center,
            do_intersect,
        ) = hyperplane_ellipsoid_intersection(
            rset.stretch_matrix, rset.center.view(-1, 1), normal_vector.view(-1, 1), 0
        )
        rset.center = new_center.view(rset.center.shape)
        rset.stretch_matrix = new_stretch_matrix
        rset.optimal_model = TorchLogisticRegressionPositiveOnly(
            rset.prototype_class_identity, trained_weights=rset.center
        )

        if resample:
            rset.sampled_models = [self._sample_model() for _ in range(rset.num_models)]
        return rset, do_intersect

    def _get_dim_k_extrema(self, axis):
        k_coord_vec = np.zeros(self.center.shape[0])
        k_coord_vec[axis] = 1

        a = (
            0.5 * k_coord_vec.T @ self.stretch_matrix @ k_coord_vec
        )  # np.linalg.norm(self.X_input @ k_coord_vec) ** 2
        b = 0  # 2 * np.dot(self.X_input @ k_coord_vec, self.X_input @ self.center - self.Y_input).item()
        c = (
            -1
        )  # np.linalg.norm(self.X_input @ self.center - self.Y_input) ** 2 - self.absolute_rashomon_bound

        alpha_vals = [
            (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a),
            (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a),
        ]
        extrema = [
            (alpha_vals[0] * k_coord_vec + self.center.reshape(-1))[axis],
            (alpha_vals[1] * k_coord_vec + self.center.reshape(-1))[axis],
        ]

        return min(extrema), max(extrema)

    def _get_volume(self, stretch_matrix):
        """
        Helper function to get the volume of the ellipsoid defined by
        a given stretch matrix
        """
        # NOTE: This is a utility I set up for some earlier VI work. It technically works,
        # but will cause numerical overflow issues for large num_dims (which happens all the
        # time for proto-rsets)
        num_dims = stretch_matrix.shape[0]

        # Finally, computing the volume for our ellipsoid------
        # First, get the volume of a d dimensional unit ball
        # See https://en.wikipedia.org/wiki/Volume_of_an_n-ball for formula
        d_ball_vol = (np.pi ** (num_dims / 2)) / math.gamma((num_dims / 2) + 1)

        # Pull this together to get our overall volume,
        # see e.g. https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15859-f11/www/notes/lecture08.pdf
        volume = d_ball_vol  # * (np.linalg.det(np.linalg.inv(stretch_matrix)) ** 0.5)

        return volume

    def wrap_sampled_params(self, params):
        """
        Wraps a set of sampled parameters in the appropriate last
        layer class
        """
        return TorchLogisticRegression(
            input_size=self.optimal_model.input_size,
            num_classes=self.optimal_model.num_classes,
            trained_weights=params,
        )

    def is_in_rashomon_set(self, model):
        return in_ellipsoid(model.get_params(), self.stretch_matrix, self.center)

    def get_volume_under_k(self, k, axis):
        """
        Gets the volume of the portion of the hyper-ellipsoid below
        value k along axis axis. NOTE: This method does not handle
        when the plane is outside our sphere. Should check this explicitly.
        Input:
            k -- float; the value which we want to split our ellipsoid
                 on
            axis -- int; the axis along which we want to compare
        Output:
            The volume of the hyper-ellipsoid that is less than
            k along the given axis
        """
        # NOTE: This is a utility I set up for some earlier VI work. It technically works,
        # but will cause numerical overflow issues for large num_dims (which happens all the
        # time for proto-rsets
        if self.empty_set:
            print("Returning none")
            return None

        # Check whether this k is outside our elipsoid
        extrema = self._get_dim_k_extrema(axis)
        if k <= min(extrema):
            return 0
        elif k >= max(extrema):
            return self.volume

        # Phi should be the angle between k and the axis of interest
        # To compute this, we need to first map k from the space of the
        # ellipse to that of a hypersphere
        # plane_basis = np.concatenate((np.identity(n_dims)[:, :axis], np.identity(n_dims)[:, axis+1:]), axis=1)
        point_on_plane = self.center.copy()  # np.zeros(self.center.shape[0])
        point_on_plane[axis] = k

        ortho_vec = self.center.copy()
        if k >= self.center[axis]:
            ortho_vec[axis] = max(extrema)
        else:
            ortho_vec[axis] = min(extrema)
        # ortho_vec[axis] = 1
        # transformed_ortho_vec = self.half_stretch_matrix @ null_space(plane_basis.T)
        # transformed_ortho_vec = transformed_ortho_vec / np.linalg.norm(transformed_ortho_vec)
        # transformed_ortho_vec = transformed_ortho_vec.reshape(-1)

        """
        We now just need a unit vector on our transformed plane
        """
        # dir_on_plane = np.sum(plane_basis, axis=1)
        # dir_on_plane = dir_on_plane / (np.linalg.norm(dir_on_plane))# * 0.92)
        dir_on_plane = np.zeros(self.center.shape[0])  # self.center
        if axis > 0:
            dir_on_plane[0] = 1
        else:
            dir_on_plane[1] = 1

        # Compute how long of a step in the chosen direction
        # we need to take in order to intersect the hypersphere
        a_for_quad = 0.5 * dir_on_plane.T @ self.stretch_matrix @ dir_on_plane
        b_for_quad = (
            0.5 * 2 * dir_on_plane.T @ self.stretch_matrix @ point_on_plane
            - 0.5 * 2 * dir_on_plane.T @ self.stretch_matrix @ self.center
        )
        c_for_quad = (
            0.5 * point_on_plane.T @ self.stretch_matrix @ point_on_plane
            - 0.5 * 2 * point_on_plane.T @ self.stretch_matrix @ self.center
            + 0.5 * self.center @ self.stretch_matrix @ self.center
            - 1
        )

        # Find length of step to take using quadratic formula
        # print(f"b_for_quad {b_for_quad}, b_for_quad**2 {b_for_quad**2} ,4*a_for_quad*c_for_quad {4*a_for_quad*c_for_quad} ")
        len_of_step_along_dir = (
            -b_for_quad + np.sqrt(b_for_quad**2 - 4 * a_for_quad * c_for_quad)
        ) / (2 * a_for_quad)

        # Use this information to grab one intersect between
        # the transformed hyperplane and the sphere
        # print("point_on_plane", point_on_plane)
        # print("len_of_step_along_dir", len_of_step_along_dir)
        # print("dir_on_plane", dir_on_plane)
        intersection_point_no_shift_no_trans = point_on_plane + (
            len_of_step_along_dir * dir_on_plane
        )
        intersection_point = intersection_point_no_shift_no_trans - self.center
        transformed_ortho_vec = ortho_vec - self.center

        # NOTE: I think the problem is that one of these vectors is not adjusted for the offset
        # Now get the angle between the intersection point and the normal
        # print("intersection poitn: ", intersection_point / np.linalg.norm(intersection_point))
        # print("ortho vec: ", transformed_ortho_vec / np.linalg.norm(transformed_ortho_vec))
        cos_angle = np.dot(
            intersection_point / np.linalg.norm(intersection_point),
            transformed_ortho_vec / np.linalg.norm(transformed_ortho_vec),
        )
        angle = np.arccos(cos_angle)
        # print(f"Angle is {angle}, from cos angle {cos_angle}")
        if angle > np.pi / 2:
            # Since the formula we use is for the smaller cap,
            # we need to make sure the angle < pi/2
            angle = angle - np.pi / 2

        # Decide whether we want the smaller or the
        # larger hemisphere for this k
        use_small_half = False
        if k < self.center[axis]:
            use_small_half = True

        # Get the volume of the hypercap of the hypersphere
        # see https://scialert.net/fulltext/?doi=ajms.2011.66.70
        num_dims = self.center.shape[0]
        d_ball_vol = (np.pi ** (num_dims / 2)) / math.gamma((num_dims / 2) + 1)
        if use_small_half:
            cap_vol = (
                1
                / 2
                * d_ball_vol
                * betainc((num_dims + 1) / 2, 1 / 2, np.sin(angle) ** 2)
            )
        else:
            cap_vol = d_ball_vol - 1 / 2 * d_ball_vol * betainc(
                (num_dims + 1) / 2, 1 / 2, np.sin(angle) ** 2
            )
            # d_ball_vol - betainc((n_dims + 1) / 2, 1 / 2, np.sin(angle)**2)

        # print(f"cap_vol: {cap_vol}")
        # Then per https://math.stackexchange.com/questions/596289/volume-of-ellipsoid-bounded-by-two-planes,
        # the desired volume of the ellipsoid is just (np.linalg.det(np.linalg.inv(stretch_matrix)) ** 0.5)
        # times cap_vol
        return cap_vol  # (np.linalg.det(np.linalg.inv(self.stretch_matrix)) ** 0.5) * cap_vol


class CorrectClassOnlyMultiClassLogisticRSet(MultiClassLogisticRSet):
    """
    A class to define the linear regression Rashomon set
    under logistic loss, with a reduced parametric form that says
    prototypes may only connect to their own class
    Args:
        prototype_class_identity : tensor (num_prototypes, num_classes) -- An indicator
            matrix in which index [i, j] is 1 if prototype i is from class j
        rashomon_bound_multiplier: float -- If not none, the maximum loss allowed
            in the Rashomon set will be optimal * rashomon_bound_multiplier
        absolute_rashomon_bound: float -- If not none, the maximum loss allowed
            in the Rashomon set will be absolute_rashomon_bound
        num_models: int -- The number of models to sample from the Rashomon set
        reg: str ('l1', 'l2', or None) -- Which type of regularization to apply
        lam: float -- The weight of the regularization
        max_iter: int -- The maximum number of iterations to perform when computing
            our optimal linear layer
        compute_hessian_batched: bool -- If true, compute the hessian wrt multiple
            samples at a time. Memory inefficient, but fast
        directly_compute_hessian: bool -- If True, we'll directly compute and store
            the hessian; else, we'll store the dataset and do repeat computation
        lr_for_opt: float -- The learning rate to use in stochasic gradient descent
    """

    def __init__(
        self,
        prototype_class_identity,
        rashomon_bound_multiplier=1.05,
        absolute_rashomon_bound=None,
        num_models=0,
        reg="l2",
        lam=0.0001,
        max_iter=10_000,
        compute_hessian_batched=False,
        directly_compute_hessian=True,
        device=torch.device("cpu"),
        lr_for_opt=0.1,
        opt_tol=5e-5,
        verbose=False
    ):
        super(CorrectClassOnlyMultiClassLogisticRSet, self).__init__(
            rashomon_bound_multiplier=rashomon_bound_multiplier,
            absolute_rashomon_bound=absolute_rashomon_bound,
            num_models=num_models,
            reg=reg,
            lam=lam,
            max_iter=max_iter,
            compute_hessian_batched=compute_hessian_batched,
            directly_compute_hessian=directly_compute_hessian,
            device=device,
            lr_for_opt=lr_for_opt,
            verbose=verbose
        )
        self.prototype_class_identity = prototype_class_identity

    def wrap_sampled_params(self, params):
        """
        Wraps a set of sampled parameters in the appropriate last
        layer class
        """
        return TorchLogisticRegressionPositiveOnly(
            self.prototype_class_identity, trained_weights=params
        )

    def fit(self, X, Y):
        """
        Builds the linear regression RSet for the given data.
        Args:
            X -- an n x d array of features
            Y -- an n x 1 array of labels
        """

        # Fit our logistic regression model ================
        model = TorchLogisticRegressionPositiveOnly(self.prototype_class_identity).to(
            self.device
        )
        self.fit_optimal_model(X, Y, model)

        def f(coeff):
            preds = self.optimal_model.predict_with_coeffs(self.X_input, coeff)
            return F.cross_entropy(preds, self.Y_input)

        self.stretch_matrix = torch.autograd.functional.hessian(
            f, self.optimal_model.linear_weights
        )
        self.stretch_matrix = self._adjust_hessian(self.stretch_matrix)

        # Finally, computing the volume for our ellipsoid------
        # self.volume = self._get_volume(self.stretch_matrix)
        self.sampled_models = [self._sample_model() for _ in range(self.num_models)]


if __name__ == "__main__":
    X = np.random.rand(100, 5)
    y = np.round(np.sum(X, axis=1))
    y = (y - y.min()).astype(int)
    X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    print(f"y: {y}")
    model = LogisticRegression(fit_intercept=False)
    model = model.fit(X, y)
    print(model.coef_.reshape(-1))

    print("Accuracy of optimal outside: ", accuracy_score(y, model.predict(X)))

    rset = MultiClassLogisticRSet(
        rashomon_bound_multiplier=1.1,
        num_models=10,
        reg="l1",
        lam=1,
    )

    rset.fit(X, y)
    print("rset.center: ", rset.center)

    preds = rset.rset_predict(X)
    print(
        f"Train Accuracies: {[accuracy_score(y, preds[:, i]) for i in range(preds.shape[1])]}"
    )
