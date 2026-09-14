"""Hub-Spectral Activation: source-edge fitting and frozen task readouts."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
import torch


class HSA:
    """Fit on two source hub edges, then score frozen target representations.

    Inputs must follow the feature normalization used for their experiment.
    Calibration uses equal-count source marginals and five fixed derangements.
    """

    def fit(self, a, h_a, b, h_b):
        arrays = [np.asarray(x) for x in (a, h_a, b, h_b)]
        if any(x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() for x in arrays):
            raise ValueError("source features must be finite matrices with at least two rows")
        a, h_a, b, h_b = arrays
        if len(a) != len(b) or len(a) != len(h_a) or len(b) != len(h_b):
            raise ValueError("source calibration requires equal row counts")
        if a.shape[1] != b.shape[1] or h_a.shape[1] != h_b.shape[1]:
            raise ValueError("target dimensions and hub dimensions must match respectively")
        if any(np.trace(np.cov(x, rowvar=False)) <= 0 for x in arrays):
            raise ValueError("source features must have positive total variance")
        self.model = _fit_relation(a, h_a, b, h_b, lam=0.5, kappa=2.0,
                                   null_seed=142, max_rank=256)
        self.calibration = calibrate_null_stds(self.model, a, b)
        left = residual_covariance(leg_covariances(a, h_a), lam=0.5)
        right = residual_covariance(leg_covariances(b, h_b), lam=0.5)
        null = coordinate_permutation_null(left, right, n_permutations=256, seed=42)
        self.gate = visibility_preservation_gate(residual_affinity(left, right), null,
                                                quantile=0.95)
        return self

    def retrieval_coordinates(self, a, b, device="cpu"):
        coordinates = prepare_coordinates(self.model, a, b)
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in coordinates.items()}

    def retrieval_scores(self, coordinates, start, stop, reverse=False):
        left, right = ("b", "a") if reverse else ("a", "b")
        carrier_left = coordinates["hub_" + left][start:stop]
        carrier_right = coordinates["hub_" + right]
        spectrum = torch.as_tensor(self.model["hub_spectrum"], dtype=torch.float32,
                                   device=carrier_left.device)
        carrier = _gaussian_llr_score_block_torch(carrier_left, carrier_right, spectrum)
        residual = coordinates["residual_" + left][start:stop] @ coordinates["residual_" + right].T
        return carrier / self.calibration["hub_std"] + self.gate["gate"] * residual / self.calibration["residual_std"]

    def classify(self, prototypes, queries):
        scores, _ = visibility_preserving_hsa_class_scores(
            self.model, prototypes, queries, gate=self.gate["gate"],
            residual_null_std=self.calibration["residual_std"],
            carrier_null_std=self.calibration["hub_std"])
        return np.asarray(scores, dtype=np.float32)



def derangement(n: int, seed: int) -> np.ndarray:
    size = int(n)
    if size < 2:
        raise ValueError("a derangement requires at least two rows")
    rng = np.random.default_rng(int(seed))
    target = np.arange(size)
    while True:
        permutation = rng.permutation(size)
        if np.all(permutation != target):
            return permutation.astype(np.int64, copy=False)


def leg_covariances(x: np.ndarray, h: np.ndarray) -> dict[str, Any]:
    values = np.asarray(x)
    hub = np.asarray(h)
    if values.ndim != 2 or hub.ndim != 2:
        raise ValueError("leg inputs must be matrices")
    if len(values) != len(hub) or len(values) < 2:
        raise ValueError("leg inputs require equal row counts of at least two")
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    if not np.issubdtype(hub.dtype, np.floating):
        hub = hub.astype(np.float32)
    n = int(len(values))
    mean_x = values.mean(axis=0, dtype=np.float64)
    mean_h = hub.mean(axis=0, dtype=np.float64)
    covariance_xx = np.asarray(values.T @ values, dtype=np.float64) / n
    covariance_xh = np.asarray(values.T @ hub, dtype=np.float64) / n
    covariance_hh = np.asarray(hub.T @ hub, dtype=np.float64) / n
    covariance_xx -= np.outer(mean_x, mean_x)
    covariance_xh -= np.outer(mean_x, mean_h)
    covariance_hh -= np.outer(mean_h, mean_h)
    return {
        "n": n,
        "mean_x": mean_x,
        "mean_h": mean_h,
        "Cxx": 0.5 * (covariance_xx + covariance_xx.T),
        "Cxh": covariance_xh,
        "Chh": 0.5 * (covariance_hh + covariance_hh.T),
    }


def _ridge_solve(covariance: np.ndarray, right: np.ndarray, lam: float) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    dimension = int(matrix.shape[0])
    ridge = float(lam) * float(np.trace(matrix)) / max(dimension, 1)
    return np.linalg.solve(matrix + ridge * np.eye(dimension), right)


def hub_mediated_cross(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    lam: float,
) -> np.ndarray:
    hub_covariance = 0.5 * (
        np.asarray(left["Chh"], dtype=np.float64)
        + np.asarray(right["Chh"], dtype=np.float64)
    )
    return np.asarray(left["Cxh"]) @ _ridge_solve(
        hub_covariance,
        np.asarray(right["Cxh"]).T,
        lam,
    )


def residual_covariance(stats: dict[str, Any], *, lam: float) -> np.ndarray:
    residual = np.asarray(stats["Cxx"]) - np.asarray(stats["Cxh"]) @ _ridge_solve(
        np.asarray(stats["Chh"]),
        np.asarray(stats["Cxh"]).T,
        lam,
    )
    return 0.5 * (residual + residual.T)


def residual_affinity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-20:
        return 0.0
    return float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def structural_hub_rank(h: np.ndarray, *, max_rank: int = 256) -> int:
    values = np.asarray(h)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("hub values must be a nonempty matrix")
    stop_after = min(int(max_rank) + 1, int(values.shape[1]) + 1)
    states_seen: set[bytes] = set()
    for start in range(0, len(values), 512):
        quantized = np.rint(values[start : start + 512] * 1000.0).astype(
            np.int16,
            copy=False,
        )
        for row in quantized:
            states_seen.add(row.tobytes())
            if len(states_seen) >= stop_after:
                return max(1, min(int(max_rank), len(states_seen) - 1, int(values.shape[1])))
    states = len(states_seen)
    return max(1, min(int(max_rank), states - 1, int(values.shape[1])))


def _inverse_sqrt(covariance: np.ndarray, lam: float) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    dimension = int(matrix.shape[0])
    ridge = float(lam) * float(np.trace(matrix)) / max(dimension, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix + ridge * np.eye(dimension)
    )
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T


def _fit_relation(
    a: np.ndarray,
    h_a: np.ndarray,
    b: np.ndarray,
    h_b: np.ndarray,
    *,
    lam: float,
    kappa: float,
    null_seed: int,
    max_rank: int = 256,
) -> dict[str, Any]:
    left = leg_covariances(a, h_a)
    right = leg_covariances(b, h_b)
    cross = hub_mediated_cross(left, right, lam=lam)
    raw_left, raw_spectrum, raw_right_t = np.linalg.svd(
        cross,
        full_matrices=False,
    )

    whiten_a = _inverse_sqrt(left["Cxx"], lam)
    whiten_b = _inverse_sqrt(right["Cxx"], lam)
    whitened_cross = whiten_a @ cross @ whiten_b.T
    hub_u, hub_spectrum, hub_vt = np.linalg.svd(
        whitened_cross,
        full_matrices=False,
    )

    permutation = np.random.default_rng(int(null_seed)).permutation(len(a))
    null_left = leg_covariances(np.asarray(a), np.asarray(h_a)[permutation])
    null_cross = hub_mediated_cross(null_left, right, lam=lam)
    null_spectrum = np.linalg.svd(
        whiten_a @ null_cross @ whiten_b.T,
        compute_uv=False,
    )
    null_top = float(null_spectrum[0]) if len(null_spectrum) else 0.0
    analytic_rank = int(
        np.sum(np.square(hub_spectrum) > float(kappa) * null_top**2)
    )
    analytic_rank = max(
        1,
        min(int(max_rank), analytic_rank, len(hub_spectrum)),
    )
    hub_rank = min(
        structural_hub_rank(h_a, max_rank=max_rank),
        structural_hub_rank(h_b, max_rank=max_rank),
        len(hub_spectrum),
    )

    residual_a = residual_covariance(left, lam=lam)
    residual_b = residual_covariance(right, lam=lam)
    affinity = residual_affinity(residual_a, residual_b)
    residual_weight = float(np.clip((1.0 + affinity) / 2.0, 0.0, 1.0))
    return {
        "mean_a": np.asarray(left["mean_x"]),
        "mean_b": np.asarray(right["mean_x"]),
        "raw_left": raw_left[:, :analytic_rank],
        "raw_right": raw_right_t[:analytic_rank].T,
        "raw_spectrum": raw_spectrum[:analytic_rank],
        "hub_left": whiten_a @ hub_u[:, :hub_rank],
        "hub_right": whiten_b @ hub_vt[:hub_rank].T,
        "hub_spectrum": hub_spectrum[:hub_rank],
        "analytic_rank": analytic_rank,
        "hub_rank": int(hub_rank),
        "null_top_singular": null_top,
        "residual_affinity": affinity,
        "residual_weight": residual_weight,
    }


def _l2_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return array / np.maximum(
        np.linalg.norm(array, axis=1, keepdims=True),
        1e-12,
    )


def prepare_coordinates(
    model: dict[str, Any],
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, np.ndarray]:
    centered_a = np.asarray(a) - np.asarray(model["mean_a"])
    centered_b = np.asarray(b) - np.asarray(model["mean_b"])
    raw_a = centered_a @ np.asarray(model["raw_left"])
    raw_b = centered_b @ np.asarray(model["raw_right"])
    residual_a = _l2_rows(
        centered_a - raw_a @ np.asarray(model["raw_left"]).T
    )
    residual_b = _l2_rows(
        centered_b - raw_b @ np.asarray(model["raw_right"]).T
    )
    hub_a = centered_a @ np.asarray(model["hub_left"])
    hub_b = centered_b @ np.asarray(model["hub_right"])
    return {
        "residual_a": residual_a,
        "residual_b": residual_b,
        "hub_a": hub_a,
        "hub_b": hub_b,
    }


def _gaussian_llr_diagonal(
    left: np.ndarray,
    right: np.ndarray,
    spectrum: np.ndarray,
) -> np.ndarray:
    rho = np.clip(np.asarray(spectrum), 0.0, 1.0 - 1e-6)
    denominator = 1.0 - np.square(rho)
    cross_weight = rho / denominator
    energy_weight = np.square(rho) / (2.0 * denominator)
    log_normalizer = -0.5 * np.log(denominator).sum()
    return (
        np.sum(np.asarray(left) * np.asarray(right) * cross_weight, axis=1)
        - np.sum(np.square(left) * energy_weight, axis=1)
        - np.sum(np.square(right) * energy_weight, axis=1)
        + log_normalizer
    )


def _gaussian_llr_score_block_torch(
    left: torch.Tensor,
    right: torch.Tensor,
    spectrum: torch.Tensor,
) -> torch.Tensor:
    rho = torch.clamp(spectrum, min=0.0, max=1.0 - 1e-6)
    denominator = 1.0 - torch.square(rho)
    cross_weight = rho / denominator
    energy_weight = torch.square(rho) / (2.0 * denominator)
    log_normalizer = -0.5 * torch.log(denominator).sum()
    cross = (left * cross_weight) @ right.T
    left_energy = torch.sum(torch.square(left) * energy_weight, dim=1)
    right_energy = torch.sum(torch.square(right) * energy_weight, dim=1)
    return (
        cross
        - left_energy[:, None]
        - right_energy[None, :]
        + log_normalizer
    )


def calibrate_null_stds(
    model: dict[str, Any],
    a: np.ndarray,
    b: np.ndarray,
    *,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    batch_size: int = 1024,
) -> dict[str, Any]:
    left = np.asarray(a)
    right = np.asarray(b)
    if len(left) != len(right):
        raise ValueError("calibration requires equal row counts")
    residual_stds: list[float] = []
    hub_stds: list[float] = []
    for seed in seeds:
        permutation = derangement(len(left), int(seed))
        residual_values: list[np.ndarray] = []
        hub_values: list[np.ndarray] = []
        step = max(1, int(batch_size))
        for start in range(0, len(left), step):
            end = min(len(left), start + step)
            coordinates = prepare_coordinates(
                model,
                left[start:end],
                right[permutation[start:end]],
            )
            residual_values.append(
                np.sum(
                    coordinates["residual_a"] * coordinates["residual_b"],
                    axis=1,
                )
            )
            hub_values.append(
                _gaussian_llr_diagonal(
                    coordinates["hub_a"],
                    coordinates["hub_b"],
                    model["hub_spectrum"],
                )
            )
        residual_stds.append(
            max(float(np.std(np.concatenate(residual_values))), 1e-6)
        )
        hub_stds.append(
            max(float(np.std(np.concatenate(hub_values))), 1e-6)
        )
    return {
        "seeds": [int(seed) for seed in seeds],
        "residual_stds": residual_stds,
        "hub_stds": hub_stds,
        "residual_std": float(np.mean(residual_stds)),
        "hub_std": float(np.mean(hub_stds)),
    }


def _square_finite_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains nonfinite values")
    return matrix


def frobenius_affinity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the clipped Frobenius cosine between two square matrices."""

    a = _square_finite_matrix("left", left)
    b = _square_finite_matrix("right", right)
    if a.shape != b.shape:
        raise ValueError("left and right must have the same square shape")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-20:
        return 0.0
    return float(np.clip(np.sum(a * b) / denominator, -1.0, 1.0))


def coordinate_permutation_null(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    """Break shared feature coordinates while preserving each covariance spectrum."""

    a = _square_finite_matrix("left", left)
    b = _square_finite_matrix("right", right)
    if a.shape != b.shape:
        raise ValueError("left and right must have the same square shape")
    count = int(n_permutations)
    if count < 1:
        raise ValueError("n_permutations must be positive")
    rng = np.random.default_rng(int(seed))
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        permutation = rng.permutation(a.shape[0])
        permuted = b[np.ix_(permutation, permutation)]
        values[index] = frobenius_affinity(a, permuted)
    return values


def visibility_preservation_gate(
    observed_affinity: float,
    null_values: np.ndarray,
    *,
    quantile: float,
) -> dict[str, Any]:
    """Map affinity beyond a coordinate-null quantile to a bounded gate."""

    observed = float(observed_affinity)
    values = np.asarray(null_values, dtype=np.float64)
    level = float(quantile)
    if not np.isfinite(observed):
        raise ValueError("observed_affinity must be finite")
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("null_values must be a nonempty finite vector")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    threshold = float(np.quantile(values, level))
    denominator = max(1.0 - threshold, 1e-12)
    gate = float(np.clip((observed - threshold) / denominator, 0.0, 1.0))
    p_value = float((1 + np.sum(values >= observed)) / (len(values) + 1))
    return {
        "gate": gate,
        "observed_affinity": observed,
        "null_quantile": threshold,
        "null_mean": float(np.mean(values)),
        "null_std": float(np.std(values)),
        "empirical_p_value": p_value,
        "quantile": level,
        "n_permutations": int(len(values)),
    }


def l2_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("row normalization requires a two-dimensional matrix")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def topk_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float]:
    matrix = np.asarray(scores)
    targets = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError("scores must be a two-dimensional matrix")
    if len(matrix) != len(targets):
        raise ValueError("scores and labels must contain the same number of rows")
    if len(targets) == 0:
        raise ValueError("classification requires at least one sample")
    if targets.min() < 0 or targets.max() >= matrix.shape[1]:
        raise ValueError("labels must index score-matrix columns")

    order = np.argsort(-matrix, axis=1)
    result: dict[str, float] = {}
    for k in ks:
        width = min(max(1, int(k)), matrix.shape[1])
        hits = np.any(order[:, :width] == targets[:, None], axis=1)
        result[f"top{int(k)}"] = float(hits.mean())
        class_means = [
            float(hits[targets == class_index].mean())
            for class_index in np.unique(targets)
        ]
        result[f"macro_top{int(k)}"] = float(np.mean(class_means))
    return result


def gaussian_hsa_class_scores(
    model: dict[str, Any],
    prototypes: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    class_values = np.asarray(prototypes, dtype=np.float64)
    query_values = np.asarray(queries, dtype=np.float64)
    class_coordinates = (
        class_values - np.asarray(model["mean_a"], dtype=np.float64)
    ) @ np.asarray(model["hub_left"], dtype=np.float64)
    query_coordinates = (
        query_values - np.asarray(model["mean_b"], dtype=np.float64)
    ) @ np.asarray(model["hub_right"], dtype=np.float64)
    spectrum = np.clip(
        np.asarray(model["hub_spectrum"], dtype=np.float64), 0.0, 1.0 - 1e-6
    )
    denominator = 1.0 - np.square(spectrum)
    cross_weight = spectrum / denominator
    energy_weight = np.square(spectrum) / (2.0 * denominator)
    cross = (query_coordinates * cross_weight) @ class_coordinates.T
    query_energy = np.sum(
        np.square(query_coordinates) * energy_weight, axis=1
    )
    class_energy = np.sum(
        np.square(class_coordinates) * energy_weight, axis=1
    )
    return cross - query_energy[:, None] - class_energy[None, :]


def visibility_preserving_hsa_class_scores(
    model: dict[str, Any],
    prototypes: np.ndarray,
    queries: np.ndarray,
    *,
    gate: float,
    residual_null_std: float,
    carrier_null_std: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    class_values = np.asarray(prototypes, dtype=np.float64)
    query_values = np.asarray(queries, dtype=np.float64)
    centered_classes = class_values - np.asarray(model["mean_a"], dtype=np.float64)
    centered_queries = query_values - np.asarray(model["mean_b"], dtype=np.float64)
    raw_left = np.asarray(model["raw_left"], dtype=np.float64)
    raw_right = np.asarray(model["raw_right"], dtype=np.float64)
    residual_classes = l2_rows(centered_classes - (centered_classes @ raw_left) @ raw_left.T)
    residual_queries = l2_rows(centered_queries - (centered_queries @ raw_right) @ raw_right.T)
    residual = np.asarray(residual_queries, dtype=np.float64) @ np.asarray(
        residual_classes, dtype=np.float64
    ).T
    carrier = gaussian_hsa_class_scores(model, class_values, query_values)
    residual_scale = max(float(residual_null_std), 1e-6)
    carrier_scale = max(float(carrier_null_std), 1e-6)
    weight = float(gate)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("gate must lie in [0, 1]")
    scores = carrier / carrier_scale + weight * residual / residual_scale
    return scores, {"carrier": carrier, "residual": residual}
