import jax
from jax.sharding import Mesh
import numpy as np

def get_global_mesh():
    """Dynamically creates or returns the mesh based on available devices."""
    devices = np.array(jax.devices())
    # This works whether you have 1 device (test) or 1024 (TPU Pod)
    return Mesh(devices, axis_names=('data',))