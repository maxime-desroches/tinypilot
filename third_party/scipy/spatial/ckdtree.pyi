import numpy as np

class cKDTreeNode:
    @property
    def data_points(self) -> np.ndarray: ...
    @property
    def indices(self) -> np.ndarray: ...

class cKDTree: ...
