import torch


def in_ellipsoid(pt, stretch_matrix, center):
    """
    Checks if the given point is in the given ellipsoid
    Args:
        pt : torch.tensor (num_ftrs, 1) -- the point to check
            for inclusion
        stretch_matrix : torch.tensor (num_ftrs, num_ftrs) --
            the shape matrix for the ellipsoid of interest
        center : torch.tensor (num_ftrs, 1) --
            the center of the ellipsoid of interest
    """
    return (pt - center).T @ stretch_matrix @ (pt - center) <= 1


def on_plane(pt, hyperplane_coeffs, hyperplane_target, tol=1e-5):
    """
    Checks if the given point is on the given hyperplane
    Args:
        pt : torch.tensor (num_ftrs, 1) -- the point to check
            for inclusion
        hyperplane_coeffs : torch.tensor (num_ftrs, 1) --
            the normal vector for the hyperplane of interest
        hyperplane_target : float -- the offset for the hyperplane
        tol : float -- how much floating point error to allow
    """
    return torch.abs(hyperplane_coeffs.T @ pt - hyperplane_target) <= tol


def hyperplane_ellipsoid_distance(
    stretch_matrix,
    center,
    hyperplane_coeffs,
    hyperplane_target
):
    """
    Compute the distance between the given ellipsoid and hyperplane.
    Formula is given in equation (2.10) of the Ellipsoidal Toolbox
    (https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf)
    Args:
        stretch_matrix : torch.tensor (num_ftrs, num_ftrs) --
            the shape matrix for the ellipsoid of interest
        center : torch.tensor (num_ftrs, 1) --
            the center of the ellipsoid of interest
        hyperplane_coeffs : torch.tensor (num_ftrs, 1) --
            the normal vector for the hyperplane of interest
        hyperplane_target : float -- the offset for the hyperplane
    """
    numerator = torch.abs(
        hyperplane_target - center.T @ hyperplane_coeffs
    ) - (hyperplane_coeffs.T @ torch.inverse(stretch_matrix) @ hyperplane_coeffs) ** 0.5
    return numerator / (hyperplane_coeffs.T @ hyperplane_coeffs) ** 0.5


def do_hyperplane_ellipsoid_intersect(
    stretch_matrix,
    center,
    hyperplane_coeffs,
    hyperplane_target
):
    """
    Checks if the given ellipsoid and hyperplane intersect
    Args:
        stretch_matrix : torch.tensor (num_ftrs, num_ftrs) --
            the shape matrix for the ellipsoid of interest
        center : torch.tensor (num_ftrs, 1) --
            the center of the ellipsoid of interest
        hyperplane_coeffs : torch.tensor (num_ftrs, 1) --
            the normal vector for the hyperplane of interest
        hyperplane_target : float -- the offset for the hyperplane
    """
    return hyperplane_ellipsoid_distance(
        stretch_matrix,
        center,
        hyperplane_coeffs,
        hyperplane_target
    ) <= 0


def get_alignment_matrix(v, w):
    '''
    Construct a matrix that will align w with v
    Note that I am using the Orthogonal Procrustes soluton
    (https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
    rather than (2.18) and (2.19) from page 11 of ellipsoidal
    toolbox
    Args:
        v : torch.tensor (num_ftrs, 1) -- the target to
            align w with
        w : torch.tensor (num_ftrs, 1) -- the vector to
            align to v
    Returns:
        R : torch.tensor (num_ftrs, num_ftrs) -- a rotation
            matrix such that cos( angle(Rw, v) ) = 1
    '''
    v = torch.nn.functional.normalize(v)
    w = torch.nn.functional.normalize(w)
    M = v @ w.T
    U, S, V = torch.svd(M)
    R = U @ V.T

    return torch.det(R) * U @ V.T


def regularize(Q, epsilon=1e-9):
    """
    Regularize - regularization of singular symmetric matrix.
    """
    m, n = Q.shape
    r = torch.linalg.matrix_rank(Q)

    if r < n:
        if torch.allclose(Q, Q.T):
            return Q + epsilon * torch.eye(n, device=Q.device)
        else:
            U, S, V = torch.svd(Q)
            return Q + (epsilon * U @ V.T)
    else:
        return Q


def hyperplane_ellipsoid_intersection(
    stretch_matrix,
    center,
    hyperplane_coeffs,
    hyperplane_target
):
    """
    Compute the intersection between the given ellipsoid and
    hyperplane. See section 2.2.4 of Ellipsoidal Toolbox
    (https://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-2006-46.pdf)
    Args:
        stretch_matrix : torch.tensor (num_ftrs, num_ftrs) --
            the shape matrix for the ellipsoid of interest
        center : torch.tensor (num_ftrs, 1) --
            the center of the ellipsoid of interest
        hyperplane_coeffs : torch.tensor (num_ftrs, 1) --
            the normal vector for the hyperplane of interest
        hyperplane_target : float -- the offset for the hyperplane
    Returns:
        stretch_matrix -- the adjusted shape matrix
        center -- the center of the new ellipsoid
        Success -- whether or not the hyperplane and ellipsoid intersect
    """
    try:
        if not do_hyperplane_ellipsoid_intersect(
            stretch_matrix,
            center,
            hyperplane_coeffs,
            hyperplane_target
        ):
            print("Error: The given ellipsoid and plane do not intersect")
            return stretch_matrix, center, False
    except torch._C._LinAlgError:
        print("ERROR Inverting matrix")
        print(f"stretch_matrix: {stretch_matrix}")
        print(f"stretch_matrix[:, 0]: {stretch_matrix[:, 0]}")
        return stretch_matrix, center, False

    inverse_stretch_matrix = torch.inverse(stretch_matrix)

    # =======================================
    # Convert our coordinate system such that
    # the hyperplane coef vector becomes [1, 0, 0, ...]
    basis_vec = torch.zeros_like(hyperplane_coeffs)
    basis_vec[0] = 1
    # NOTE: I swapped the variable ordering from the book
    S = get_alignment_matrix(basis_vec, hyperplane_coeffs)

    f = S @ hyperplane_coeffs * hyperplane_target / (
        (hyperplane_coeffs.T @ hyperplane_coeffs)
    )

    new_center = S @ center - f

    new_inverse_stretch_matrix = S @ inverse_stretch_matrix @ S.T

    # =======================================
    # Compute the intersection in this coordinate system

    new_stretch_matrix = torch.inverse(regularize(new_inverse_stretch_matrix))
    w = new_stretch_matrix[1:, 0]
    w11 = new_stretch_matrix[0, 0]
    W = torch.inverse(regularize(new_stretch_matrix[1:, 1:]))

    h = new_center[0, 0] ** 2 * (w11 - w.T @ W @ w)

    z = new_center + new_center[0, 0] * torch.concat([torch.tensor([-1], device=W.device), W @ w]).view(-1, 1)
    tmp = torch.zeros(W.shape[0] + 1, W.shape[1] + 1, device=W.device)
    tmp[1:, 1:] = W
    Z = (1 - h) * tmp
    z = S.T @ z + hyperplane_coeffs * hyperplane_target / (
        (hyperplane_coeffs.T @ hyperplane_coeffs)
    )
    Z = S.T @ Z @ S

    return torch.inverse(regularize(Z)), z, True
