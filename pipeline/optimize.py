"""Pequenos utilitários de otimização estatística sem dependência de SciPy.

Se SciPy estiver instalado, suas implementações são usadas; caso contrário há
fallbacks em NumPy puro / biblioteca padrão para que a pipeline rode em qualquer
ambiente com NumPy.
"""

from __future__ import annotations

import numpy as np
from statistics import NormalDist

try:  # opcional
    from scipy.optimize import linear_sum_assignment as _sp_lsa
    from scipy.stats import chi2 as _sp_chi2
except Exception:  # pragma: no cover - depende do ambiente
    _sp_lsa = None
    _sp_chi2 = None


Array = np.ndarray


def _hungarian_numpy(cost: Array) -> tuple[Array, Array]:
    """Kuhn-Munkres O(n^3) para matriz retangular (minimizar custo)."""
    C = np.asarray(cost, dtype=float)
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    rows = [int(p[j]) - 1 for j in range(1, m + 1) if p[j] != 0]
    cols = [j - 1 for j in range(1, m + 1) if p[j] != 0]
    rows_a, cols_a = np.array(rows, dtype=int), np.array(cols, dtype=int)
    return (cols_a, rows_a) if transposed else (rows_a, cols_a)


def linear_sum_assignment(cost: Array) -> tuple[Array, Array]:
    """Índices (linhas, colunas) do emparelhamento de custo mínimo."""
    C = np.asarray(cost, dtype=float)
    if C.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if _sp_lsa is not None:
        r, c = _sp_lsa(C)
        return np.asarray(r), np.asarray(c)
    return _hungarian_numpy(C)


def chi2_ppf(q: float, df: int = 4) -> float:
    """Quantil da qui-quadrado (SciPy se disponível; tabela/Wilson-Hilferty)."""
    if _sp_chi2 is not None:
        return float(_sp_chi2.ppf(q, df))
    exact = {(0.99, 2): 9.21034, (0.95, 2): 5.99146,
             (0.99, 4): 13.27670, (0.95, 4): 9.48773}
    for (q0, d0), value in exact.items():
        if abs(q - q0) < 1e-9 and df == d0:
            return value
    if df == 2:
        return float(-2.0 * np.log(1.0 - q))
    z = NormalDist().inv_cdf(q)
    k = float(df)
    return float(k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3)
