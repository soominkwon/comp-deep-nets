import jax.numpy as jnp
from jax.nn import log_softmax

def create_loss(target, mask=None, reduction="mean"):

    def loss_fn(output):
        residual = output - target

        if mask is not None:
            residual *= mask

        if reduction == "mean":
            # return 1/2 * jnp.mean(residual**2)
            return 1/2 * jnp.sum(residual**2) / jnp.sum(mask) if mask is not None else 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
        
    return loss_fn


# def create_loss_ms(labels, forward, reduction="sum"):
#     """
#     Constructs loss function for matrix sensing (ms).
#     """

#     def loss_fn(output):

#         # Compute residual
#         residual = forward(output) - labels

#         if reduction == "mean":
#             return 1/2 * jnp.mean(residual**2)
#         elif reduction == "sum":
#             return 1/2 * jnp.sum(residual**2)
#         else:
#             raise ValueError("Reduction type not implemented")
        
#     return loss_fn


# Zekai:
def create_loss_ms(labels, sensing_matrices, reduction="sum"):
    """
    Constructs loss function for matrix sensing (ms).
    """

    def loss_fn(output):

        # Compute residual
        residual = (sensing_matrices * output).sum((-2,-1)) - labels

        if reduction == "mean":
            return 1/2 * jnp.mean(residual**2)
        elif reduction == "sum":
            return 1/2 * jnp.sum(residual**2)
        else:
            raise ValueError("Reduction type not implemented")
        
    return loss_fn


def reconstruction_loss(target):
    """
    Reconstruction loss function.
    """
    def loss_fn(output):
        norm = jnp.linalg.norm(target, ord='fro')
        return jnp.linalg.norm(target - output, ord='fro') / norm

    return loss_fn

def create_loss_ce(target, reduction="mean"):

    def loss_fn(output):
        # residual = output - target # (k, n)
        # taking cross entropy along the first dimension
        return -jnp.sum(target * log_softmax(output, axis=0)) / target.shape[1]
        
    return loss_fn