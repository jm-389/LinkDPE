import math

import torch
import torch.nn as nn

from ..spectral import diffusion_factors, select_scales, truncated_eigendecomposition
from .chebyshev import chebyshev_basis


class LinkDPE(nn.Module):

    def __init__(self, factors, responses, pe_dim, cheby_order):
        super().__init__()
        m = factors.size(0)
        self.register_buffer('factors', factors)
        self.register_buffer('cheby', torch.stack([chebyshev_basis(r, cheby_order) for r in responses]))
        self.coeffs = nn.Parameter(torch.empty(m, cheby_order, pe_dim))
        self.gate_logits = nn.Parameter(torch.empty(m))
        self.out_dim = m * pe_dim
        self.scales = None
        self.reset_parameters()

    @classmethod
    def from_graph(cls, edge_index, num_nodes, k, max_scales, num_candidates, response_range, tol,
                   pe_dim, cheby_order):
        U, lam = truncated_eigendecomposition(edge_index, num_nodes, k)
        scales = select_scales(lam, max_scales, num_candidates, response_range, tol)
        factors, responses = diffusion_factors(U, lam, scales)
        module = cls(torch.from_numpy(factors), torch.from_numpy(responses), pe_dim, cheby_order)
        module.scales = scales
        return module

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.coeffs.size(1))
        nn.init.uniform_(self.coeffs, -bound, bound)
        nn.init.zeros_(self.gate_logits)

    def spectral_weights(self):
        return self.cheby @ self.coeffs

    def gates(self):
        return torch.sigmoid(self.gate_logits)

    def forward(self):
        P = self.factors @ self.spectral_weights()
        P = self.gates()[:, None, None] * P
        return P.transpose(0, 1).reshape(P.size(1), -1)

    def regularization(self, spectral_coef, gate_coef):
        W = self.spectral_weights().reshape(-1)
        return spectral_coef * (W.abs().sum() + W.norm()) + gate_coef * self.gates().sum()
