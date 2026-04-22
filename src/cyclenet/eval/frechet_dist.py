import numpy as np


def covariance_matrix(feats: np.ndarray) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {feats.shape}")
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    return np.cov(feats, rowvar=False).astype(np.float64)


def symmetric_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(fake_feats: np.ndarray, real_feats: np.ndarray) -> float:
    fake = np.asarray(fake_feats, dtype=np.float64)
    real = np.asarray(real_feats, dtype=np.float64)
    if fake.ndim != 2 or real.ndim != 2:
        raise ValueError(
            f"Expected 2D feature arrays, got fake={fake.shape}, real={real.shape}"
        )
    if fake.shape[1] != real.shape[1]:
        raise ValueError(
            f"Feature dimensions must match, got fake={fake.shape[1]} and real={real.shape[1]}"
        )
    if len(fake) == 0 or len(real) == 0:
        raise ValueError("Feature arrays must be non-empty.")

    mu_fake = fake.mean(axis=0)
    mu_real = real.mean(axis=0)
    cov_fake = covariance_matrix(fake)
    cov_real = covariance_matrix(real)

    cov_real_sqrt = symmetric_matrix_sqrt(cov_real)
    middle = cov_real_sqrt @ cov_fake @ cov_real_sqrt
    cov_mean = symmetric_matrix_sqrt(middle)

    diff = mu_fake - mu_real
    dist = diff @ diff + np.trace(cov_fake + cov_real - 2.0 * cov_mean)
    return float(max(dist, 0.0))