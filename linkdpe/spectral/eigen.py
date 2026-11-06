import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackError, eigsh


def normalized_adjacency(edge_index, num_nodes):
    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).flatten()
    d_inv_sqrt = np.zeros_like(deg, dtype=float)
    d_inv_sqrt[deg > 0] = deg[deg > 0] ** -0.5
    D = sp.diags(d_inv_sqrt)
    return (D @ A @ D).tocsr()


def truncated_eigendecomposition(edge_index, num_nodes, k):
    A_hat = normalized_adjacency(edge_index, num_nodes)
    _, labels = connected_components(A_hat, directed=False)
    lcc = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    A_lcc = A_hat[np.ix_(lcc, lcc)]

    n = len(lcc)
    k = min(k, n - 2)
    try:
        lam, U_lcc = eigsh(A_lcc, k=k, which='LA', ncv=min(n, max(2 * k + 1, k + 64)))
    except ArpackError:
        lam, U_lcc = np.linalg.eigh(A_lcc.toarray())
    order = np.argsort(lam)[::-1][:k]

    U = np.zeros((num_nodes, k), dtype=np.float32)
    U[lcc] = U_lcc[:, order]
    return U, lam[order].astype(np.float32)
