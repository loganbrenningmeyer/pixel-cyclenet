import umap
import numpy as np
from sklearn.decomposition import PCA


class UmapProjector:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)

    def fit(self, feats: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(feats)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return self.reducer.transform(feats)


class PcaProjector:
    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = PCA(n_components=n_components, random_state=random_state)

    def fit(self, feats: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(feats)

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return self.reducer.transform(feats)
