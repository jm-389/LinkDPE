import numpy as np


def ppr_response(lam, alpha):
    return alpha / (1.0 - (1.0 - alpha) * np.asarray(lam, dtype=np.float64))


def candidate_scales(lam, num_candidates, response_range):
    gap = 1.0 - float(np.min(lam))
    if gap <= 0:
        raise ValueError("retained eigenvalues have no spread below 1; increase k")
    lo, hi = response_range
    if not 0 < lo < hi < 1:
        raise ValueError(f"need 0 < lo < hi < 1, got response_range = {response_range}")
    b = np.logspace(np.log10((1.0 / hi - 1.0) / gap), np.log10((1.0 / lo - 1.0) / gap), num_candidates)
    return np.sort(1.0 / (1.0 + b))


def response_matrix(lam, candidates):
    R = np.stack([ppr_response(lam, a) for a in candidates], axis=1)
    return R / np.linalg.norm(R, axis=0, keepdims=True)


def dpp_select(R_hat, max_m, tol):
    K, C = R_hat.shape
    if not 0 < max_m <= min(K, C):
        raise ValueError(f"need 0 < max_m <= min(K, C) = {min(K, C)}, got max_m = {max_m}")
    K_scale = R_hat.T @ R_hat

    sel = [int(np.argmin(K_scale.mean(axis=1)))]
    residual = R_hat.copy()
    while len(sel) < max_m:
        v = residual[:, sel[-1]]
        residual = residual - np.outer(v, v @ residual) / (v @ v)
        gain = np.linalg.norm(residual, axis=0)
        gain[sel] = -np.inf
        best = int(np.argmax(gain))
        if gain[best] < tol:
            break
        sel.append(best)

    def logdet(idx):
        sign, ld = np.linalg.slogdet(K_scale[np.ix_(idx, idx)])
        return ld if sign > 0 else -np.inf

    m = len(sel)
    unselected = [j for j in range(C) if j not in sel]
    improved = True
    while improved:
        improved = False
        current = logdet(sel)
        for pos in range(m):
            best, best_j = current, None
            for j in unselected:
                ld = logdet(sel[:pos] + [j] + sel[pos + 1:])
                if ld > best + 1e-8:
                    best, best_j = ld, j
            if best_j is not None:
                unselected.remove(best_j)
                unselected.append(sel[pos])
                sel[pos], current, improved = best_j, best, True

    return sorted(sel)


def select_scales(lam, max_scales, num_candidates, response_range, tol):
    candidates = candidate_scales(lam, num_candidates, response_range)
    idx = dpp_select(response_matrix(lam, candidates), max_scales, tol)
    return [float(candidates[i]) for i in idx]
