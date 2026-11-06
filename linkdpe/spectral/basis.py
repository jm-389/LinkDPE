import numpy as np

from .scales import ppr_response


def diffusion_factors(U, lam, scales):
    responses = np.stack([ppr_response(lam, a) for a in scales])
    factors = U[None] * np.sqrt(responses).astype(np.float32)[:, None, :]
    norms = np.linalg.norm(factors, axis=2, keepdims=True)
    factors = np.divide(factors, norms, out=np.zeros_like(factors), where=norms > 0)
    return factors.astype(np.float32), responses.astype(np.float32)
