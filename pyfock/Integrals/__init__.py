"""
This submodule provides highly optimized and symmetry-aware routines for 
evaluating all essential one-electron and two-electron integrals needed in 
electronic structure calculations. The integrals are computed over Gaussian 
type orbitals (GTOs). Pure Python/NumPy implementations are used with 
optional MLX acceleration for Apple Silicon.

Included modules and functionality:
-----------------------------------
- One-electron integrals:
    * Overlap integrals
    * Kinetic energy integrals
    * Nuclear attraction integrals
    * Dipole moment integrals
    * Gradients of one-electron integrals
- Two-electron integrals:
    * Full 4-center two electron repulsion integrals using Rys quadrature
    * 3-center and 2-center two electron integrals for density fitting
    * Schwarz screening utilities for integral pruning
- Exchange-correlation evaluation routines compatible with numerical grids
- Modular helper functions for computing factorials, boys functions, contraction coefficients, etc.

Note: CuPy/CUDA support has been replaced with MLX for Apple Silicon acceleration.

Usage:
------

Example:
    from pyfock.Integrals import overlap_mat_symm
    S = overlap_mat_symm(basis)

"""

# Try to import MLX for Apple Silicon acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

from .mmd_nuc_mat_symm import mmd_nuc_mat_symm
from .nuc_mat_symm import nuc_mat_symm
from .kin_mat_symm import kin_mat_symm
from .overlap_mat_symm import overlap_mat_symm
from .cross_overlap_mat_symm import cross_overlap_mat_symm
from .dipole_moment_mat_symm import dipole_moment_mat_symm
from .conv_4c2e_symm import conv_4c2e_symm
from .mmd_4c2e_symm import mmd_4c2e_symm
from .rys_4c2e_symm import rys_4c2e_symm, rys_4c2e_symm_old
from .conv_3c2e_symm import conv_3c2e_symm
from .rys_3c2e_symm import rys_3c2e_symm
from .conv_2c2e_symm import conv_2c2e_symm
from .rys_2c2e_symm import rys_2c2e_symm
from .rys_3c2e_tri import rys_3c2e_tri
from .rys_nuc_mat_symm import rys_nuc_mat_symm
from .schwarz_helpers import rys_3c2e_tri_schwarz
from .eval_xc_1 import eval_xc_1
from .eval_xc_2 import eval_xc_2
from .eval_xc_3 import eval_xc_3
from . import bf_val_helpers
from . import rys_helpers
from . import schwarz_helpers
from . import integral_helpers
from .overlap_mat_grad_symm import overlap_mat_grad_symm
from .kin_mat_grad_symm import kin_mat_grad_symm
from .nuc_mat_grad_symm import nuc_mat_grad_symm

# CuPy modules are disabled for MLX compatibility
# These were GPU-accelerated versions using NVIDIA CUDA
# If you need GPU acceleration, consider using MLX on Apple Silicon
# For now, use the non-cupy versions of these functions instead

# Placeholder variables for compatibility
nuc_mat_symm_cupy = None
kin_mat_symm_cupy = None
kin_mat_symm_shell_cupy = None
overlap_mat_symm_cupy = None
dipole_moment_mat_symm_cupy = None
rys_3c2e_symm_cupy = None
rys_3c2e_symm_cupy_fp32 = None
rys_2c2e_symm_cupy = None
eval_xc_1_cupy = None
eval_xc_2_cupy = None
eval_xc_3_cupy = None

__all__ = ['integral_helpers', 'mmd_nuc_mat_symm', 'nuc_mat_symm', 'kin_mat_symm', 'overlap_mat_symm', 'conv_4c2e_symm', 'mmd_4c2e_symm', 'rys_helpers', 'rys_4c2e_symm',
     'rys_4c2e_symm_old', 'conv_3c2e_symm', 'rys_3c2e_symm', 'conv_2c2e_symm', 'rys_2c2e_symm', 'rys_3c2e_tri', 'rys_nuc_mat_symm', 'schwarz_helpers', 'rys_3c2e_tri_schwarz',
        'bf_val_helpers', 'eval_xc_1', 'eval_xc_2', 'eval_xc_3', 'dipole_moment_mat_symm',
        'overlap_mat_grad_symm', 'kin_mat_grad_symm', 'nuc_mat_grad_symm', 'cross_overlap_mat_symm']