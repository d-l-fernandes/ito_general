import jax.numpy as jnp
import jax.ops as ops
from jax.scipy.linalg import solve_triangular


def matrix_diag_transform(a, f):
    diag_tensor = diag_part(a)
    diag_tensor = f(diag_tensor)
    return set_diag(a, diag_tensor)


def diag_part(a):
    return jnp.diagonal(a, axis1=-2, axis2=-1)


def set_diag(a, diag_tensor):
    inner_dim = a.shape[-1]
    return ops.index_update(a, ops.index[..., tuple(range(inner_dim)), tuple(range(inner_dim))], diag_tensor)


def diag(diag_tensor):
    shape = tuple(list(diag_tensor.shape) + [diag_tensor.shape[-1]])
    a = jnp.zeros(shape)

    return set_diag(a, diag_tensor)


def kl_divergence_multivariate_normal(a_mean, a_scale_tril, b_mean, b_scale_tril, lower=True):

    def log_abs_determinant(scale_tril_arg):
        diag_scale_tril = jnp.diagonal(scale_tril_arg, axis1=-2, axis2=-1)
        return 2 * jnp.sum(jnp.log(diag_scale_tril), axis=-1)

    def squared_frobenius_norm(x):
        """Helper to make KL calculation slightly more readable."""
        return jnp.sum(jnp.square(x), axis=(-2, -1))

    if b_scale_tril.shape[0] == 1:
        tiles = tuple([b_mean.shape[0]] + [1 for _ in range(len(b_scale_tril.shape) - 1)])
        scale_tril = jnp.tile(b_scale_tril, tiles)
    else:
        scale_tril = b_scale_tril

    b_inv_a = solve_triangular(b_scale_tril, a_scale_tril, lower=lower)
    kl = 0.5 * (log_abs_determinant(b_scale_tril) - log_abs_determinant(a_scale_tril)
                - a_scale_tril.shape[-1] + squared_frobenius_norm(b_inv_a) +
                squared_frobenius_norm(solve_triangular(scale_tril, (b_mean - a_mean)[..., jnp.newaxis],
                                                        lower=lower)))

    return kl


def log_prob_multivariate_normal(loc, scale_tril, value):
    def _batch_mahalanobis(bL, bx):
        # NB: The following procedure handles the case: bL.shape = (i, 1, n, n), bx.shape = (i, j, n)
        # because we don't want to broadcast bL to the shape (i, j, n, n).

        # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
        # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tril_solve
        sample_ndim = bx.ndim - bL.ndim + 1  # size of sample_shape
        out_shape = jnp.shape(bx)[:-1]  # shape of output
        # Reshape bx with the shape (..., 1, i, j, 1, n)
        bx_new_shape = out_shape[:sample_ndim]
        for (sL, sx) in zip(bL.shape[:-2], out_shape[sample_ndim:]):
            bx_new_shape += (sx // sL, sL)
        bx_new_shape += (-1,)
        bx = jnp.reshape(bx, bx_new_shape)
        # Permute bx to make it have shape (..., 1, j, i, 1, n)
        permute_dims = (tuple(range(sample_ndim))
                        + tuple(range(sample_ndim, bx.ndim - 1, 2))
                        + tuple(range(sample_ndim + 1, bx.ndim - 1, 2))
                        + (bx.ndim - 1,))
        bx = jnp.transpose(bx, permute_dims)

        # reshape to (-1, i, 1, n)
        xt = jnp.reshape(bx, (-1,) + bL.shape[:-1])
        # permute to (i, 1, n, -1)
        xt = jnp.moveaxis(xt, 0, -1)
        solve_bL_bx = solve_triangular(bL, xt, lower=True)  # shape: (i, 1, n, -1)
        M = jnp.sum(solve_bL_bx ** 2, axis=-2)  # shape: (i, 1, -1)
        # permute back to (-1, i, 1)
        M = jnp.moveaxis(M, -1, 0)
        # reshape back to (..., 1, j, i, 1)
        M = jnp.reshape(M, bx.shape[:-1])
        # permute back to (..., 1, i, j, 1)
        permute_inv_dims = tuple(range(sample_ndim))
        for i in range(bL.ndim - 2):
            permute_inv_dims += (sample_ndim + i, len(out_shape) + i)
        M = jnp.transpose(M, permute_inv_dims)
        return jnp.reshape(M, out_shape)

    mahalanobis = _batch_mahalanobis(scale_tril, value - loc)
    half_log_det = jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)).sum(-1)
    normalize_term = half_log_det + 0.5 * scale_tril.shape[-1] * jnp.log(2 * jnp.pi)
    return - 0.5 * mahalanobis - normalize_term
