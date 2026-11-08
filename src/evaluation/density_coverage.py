import numpy as np

DEFAULT_CHUNK_SIZE = 1024



def _squared_norms(features: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", features, features)



def _pairwise_distances(
        block: np.ndarray,
        others: np.ndarray,
        block_sq_norms: np.ndarray,
        others_sq_norms: np.ndarray
    ) -> np.ndarray:
        squared = (
            block_sq_norms[:, None]
            + others_sq_norms[None, :]
            - 2.0 * (block @ others.T)
        )
        np.maximum(squared, 0.0, out=squared)
        return np.sqrt(squared)
    
    
    
def nearest_neighbour_radii(
        features: np.ndarray,
        k: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    sample_count = features.shape[0]
    if sample_count <= k:
        raise ValueError(
            f"At least k+1 = {k + 1} samples needed for determining the k-th neighbor, but there is only {sample_count}"
        )
        
    sq_norms = _squared_norms(features)
    radii = np.empty(sample_count, dtype=np.float64)
 
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        distances = _pairwise_distances(
            features[start:end], features, sq_norms[start:end], sq_norms
        )
 
        row_indices = np.arange(end - start)
        distances[row_indices, start + row_indices] = np.inf
 
        radii[start:end] = np.partition(distances, k - 1, axis=1)[:, k - 1]
 
    return radii



def compute_density_coverage(
        real_features: np.ndarray,
        fake_features: np.ndarray,
        k: int = 5,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, float]:
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("The properties need to be in a [samplecount, dimension] form.")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            f"Different property-dimension: real {real_features.shape[1]}, "
            f"generated {fake_features.shape[1]}."
        )
        
    real = np.asarray(real_features, dtype=np.float64)
    fake = np.asarray(fake_features, dtype=np.float64)
 
    real_count = real.shape[0]
    fake_count = fake.shape[0]
 
    real_radii = nearest_neighbour_radii(real, k, chunk_size)
    fake_radii = nearest_neighbour_radii(fake, k, chunk_size)
 
    real_sq_norms = _squared_norms(real)
    fake_sq_norms = _squared_norms(fake)
    
    
    density_counts = np.zeros(fake_count, dtype=np.int64)  
    fake_inside_any = np.zeros(fake_count, dtype=bool)     
    real_covered = np.zeros(real_count, dtype=bool)        
    real_reached = np.zeros(real_count, dtype=bool)        
 
    for start in range(0, real_count, chunk_size):
        end = min(start + chunk_size, real_count)
 
        distances = _pairwise_distances(
            real[start:end], fake, real_sq_norms[start:end], fake_sq_norms
        )
 
        inside = distances < real_radii[start:end, None]
 
        density_counts += inside.sum(axis=0)
        fake_inside_any |= inside.any(axis=0)
 
        # coverage
        real_covered[start:end] = distances.min(axis=1) < real_radii[start:end]

        #recall
        real_reached[start:end] = (distances < fake_radii[None, :]).any(axis=1)
 
    return {
        "precision": float(fake_inside_any.mean()),
        "recall": float(real_reached.mean()),
        "density": float(density_counts.sum() / (k * fake_count)),
        "coverage": float(real_covered.mean()),
        "k": k,
        "num_real": real_count,
        "num_fake": fake_count,
    }