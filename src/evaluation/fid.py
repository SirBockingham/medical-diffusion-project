import numpy as np


def _mean_and_covariance(features: np.ndarray):
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    covariance = np.cov(features, rowvar=False, ddof=1)
    return mean, covariance



def _trace_sqrt_product(sigma_a: np.ndarray, sigma_b: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(sigma_a @ sigma_b)
    return float(np.sqrt(eigenvalues.astype(np.complex128)).real.sum())



def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Different feature dimensions: real {real_features.shape[1]}, "
            f"generated {fake_features.shape[1]}."
        )
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("For covariance estimation at least 2 samples are needed on each side.")
 
    mean_real, cov_real = _mean_and_covariance(real_features)
    mean_fake, cov_fake = _mean_and_covariance(fake_features)
 
    mean_term = float(((mean_real - mean_fake) ** 2).sum())
    trace_term = float(np.trace(cov_real) + np.trace(cov_fake))
    cross_term = _trace_sqrt_product(cov_real, cov_fake)
 
    fid_value = mean_term + trace_term - 2.0 * cross_term
 
    return max(fid_value, 0.0)