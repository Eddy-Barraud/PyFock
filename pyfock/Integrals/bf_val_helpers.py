import numpy as np
import math

# Try to import MLX for Apple Silicon acceleration
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


def eval_bfs_and_grad(basis, coord, deriv=1, parallel=True, non_zero_indices=None):
    """Evaluate basis functions and their gradients on a grid."""
    # We convert the required properties to numpy arrays
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])

    # Convert lists to numpy arrays
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
        bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

    if non_zero_indices is not None:
        bf_values, bf_grad_values = eval_bfs_and_grad_sparse_internal(
            bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], 
            bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
    else:
        bf_values, bf_grad_values = eval_bfs_and_grad_internal(
            bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], 
            bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        
    return bf_values, bf_grad_values


def eval_bfs(basis, coord, parallel=True, non_zero_indices=None):
    """Evaluate basis functions on a grid."""
    # We convert the required properties to numpy arrays
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])

    # Convert lists to numpy arrays
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
        bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

    if non_zero_indices is not None:
        bf_values = eval_bfs_sparse_internal(
            bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], 
            bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, non_zero_indices)
    else:
        bf_values = eval_bfs_internal(
            bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], 
            bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
    
    return bf_values


def eval_rho(bf_values, densmat):
    """
    Evaluates the value of density on a grid using the values of the basis functions 
    at those grid points and the density matrix.
    rho at the grid point m is given as:
    rho^m = sum_{mu,nu} D_{mu nu} mu^m nu^m
    """
    ncoords = bf_values.shape[0] 
    nbfs = bf_values.shape[1]
    rho = np.zeros((ncoords,))

    # Loop over grid points
    for m in range(ncoords):
        rho_temp = 0.0
        for i in range(nbfs):
            mu = bf_values[m,i]
            if abs(mu) < 1.0e-8:
                continue
            for j in range(i+1):
                dens = densmat[i,j]
                if abs(dens) < 1.0e-8:
                    continue
                if i == j:  # Diagonal terms
                    nu = mu
                    rho_temp += dens*mu*nu 
                else:  # Non-diagonal terms
                    nu = bf_values[m,j]
                    if abs(nu) < 1.0e-8:
                        continue
                    rho_temp += 2*dens*mu*nu 
        rho[m] = rho_temp
    return rho 


def eval_gto(alpha, coeff, lmn, x, y, z, exponent_dist_sq):
    """
    Evaluates the value of a given Gaussian primitive 
    with given values of alpha (exponent), coefficient, and angular momentum.
    """
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp_val = math.exp(-alpha*exponent_dist_sq)
    value = coeff*xl*ym*zn*exp_val
    return value


def eval_bfs_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    """Evaluates the value of all the given Basis functions on the grid (coord)."""
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    # Loop over grid points
    for k in range(ncoord):
        coord_grid = coord[k]
        # Loop over BFs
        for i in range(nao):
            value = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2) > bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i][ik] 
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                value += eval_gto(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
            result[k,i] = value

    return result


def eval_bfs_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    """Evaluates the values of the specific (significant) basis functions on a set of grid points."""
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    # Loop over grid points
    for k in range(ncoord):
        coord_grid = coord[k]
        # Loop over BFs
        for i in range(nao):
            ibf = bf_indices[i]
            value = 0.0
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[ibf]):
                dik = bfs_coeffs[ibf][ik] 
                Nik = bfs_prim_norms[ibf][ik]
                alphaik = bfs_expnts[ibf][ik]
                value += eval_gto(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
            result[k,i] = value

    return result


def eval_bfs_and_grad_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    """Evaluates the value of all the given Basis functions and their gradients on the grid."""
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    # Loop over grid points
    for k in range(ncoord):
        coord_grid = coord[k]
        # Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2) > bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i, ik] 
                Nik = bfs_prim_norms[i, ik]
                alphaik = bfs_expnts[i, ik]
                a,b,c,d = eval_gto_and_grad(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
                value_ao = value_ao + a
                valuex += b
                valuey += c
                valuez += d
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
    return result1, result2


def eval_bfs_and_grad_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    """Evaluates the value of specific Basis functions and their gradients on the grid."""
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    # Loop over grid points
    for k in range(ncoord):
        coord_grid = coord[k]
        # Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            ibf = bf_indices[i]
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            exponent_dist_sq = x**2 + y**2 + z**2
            for ik in range(bfs_nprim[ibf]):
                dik = bfs_coeffs[ibf, ik] 
                Nik = bfs_prim_norms[ibf, ik]
                alphaik = bfs_expnts[ibf, ik]
                a,b,c,d = eval_gto_and_grad(alphaik, Ni*Nik*dik, lmni, x, y, z, exponent_dist_sq)
                value_ao += a
                valuex += b
                valuey += c
                valuez += d
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
    return result1, result2


def eval_bfs_and_grad_sparse_internal_serial(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    """Serial version of eval_bfs_and_grad_sparse_internal."""
    return eval_bfs_and_grad_sparse_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices)


def eval_gto_and_grad(alpha, coeff, lmn, x, y, z, exponent_dist_sq):
    """
    A very low-level way to calculate the ao values as well as their gradients simultaneously.
    """
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp_val = math.exp(-alpha*(exponent_dist_sq))
    factor2 = coeff*exp_val

    # AO Value
    value0 = factor2*xl*ym*zn
    # Grad x
    if np.abs(x-0) < 1e-14:
        value1 = 0.0
    else:
        xl = x**(lmn[0]-1)
        factor = (lmn[0]-2*alpha*x**2)
        value1 = factor2*xl*ym*zn*factor
    # Grad y
    if np.abs(y-0) < 1e-14:
        value2 = 0.0
    else:
        xl = x**lmn[0]
        ym = y**(lmn[1]-1)
        factor = (lmn[1]-2*alpha*y**2)
        value2 = factor2*xl*ym*zn*factor 
    # Grad z 
    if np.abs(z-0) < 1e-14:
        value3 = 0.0
    else:
        zn = z**(lmn[2]-1)
        xl = x**lmn[0]
        ym = y**lmn[1]  
        factor = (lmn[2]-2*alpha*z**2)
        value3 = factor2*xl*ym*zn*factor
        
    return value0, value1, value2, value3


def eval_bfs_sparse_vectorized_internal(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord, bf_indices):
    """Vectorized version of sparse BF evaluation."""
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    for i in range(nao):
        ibf = bf_indices[i]
        coord_bf = bfs_coords[ibf]
        x = coord[:,0] - coord_bf[0]
        y = coord[:,1] - coord_bf[1]
        z = coord[:,2] - coord_bf[2]
        Ni = bfs_contr_prim_norms[ibf]
        lmni = bfs_lmn[ibf]
        exp_sq_term = x**2 + y**2 + z**2
        xl = x**lmni[0]
        ym = y**lmni[1]
        zn = z**lmni[2]
        value = np.zeros(coord.shape[0], dtype=np.float64)
        for ik in range(bfs_nprim[ibf]):
            dik = bfs_coeffs[ibf][ik] 
            Nik = bfs_prim_norms[ibf][ik]
            alphaik = bfs_expnts[ibf][ik]
            value += eval_gto_vectorize(alphaik, Ni*Nik*dik, exp_sq_term, xl, ym, zn)
        result[:,i] = value

    return result


def eval_gto_vectorize(alpha, coeff, exp_sq_term, xl, ym, zn):
    """Vectorized GTO evaluation."""
    exp_val = np.exp(-alpha*(exp_sq_term))
    value = coeff*xl*ym*zn*exp_val
    return value


def calc_norm(x, y, z):
    """Calculate squared norm."""
    return x**2 + y**2 + z**2


def xlymzn(x, y, z, l, m, n):
    """Calculate x^l * y^m * z^n."""
    return (x**l) * (y**m) * (z**n)


def nonzero_ao_indices_batch(coords, bfs_coords, bfs_radius_cutoff):
    """Find indices of basis functions with non-zero contributions to a batch of grid points."""
    nbfs = bfs_coords.shape[0]
    ncoords = coords.shape[0]
    count = 0
    indices = np.zeros((nbfs), dtype='uint16')
    # Loop over the basis functions 
    for ibf in range(nbfs):
        coord_bf = bfs_coords[ibf]
        cutoff = bfs_radius_cutoff[ibf]
        # Loop over the grid points and check if the value of the basis function is greater than the threshold
        for igrd in range(ncoords):
            coord_grid = coords[igrd]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2) < cutoff):
                indices[count] = ibf
                count = count + 1
                break

    return indices, count


def nonzero_ao_indices(basis, coords, blocksize, nblocks, ngrids):
    """
    For a given set of grids and the batch/block size,
    calculate the list of indices for each block that corresponds to the basis functions 
    which have non-zero contributions to those batches/blocks.
    """
    bfs_coords = np.array([basis.bfs_coords])
    bfs_radius_cutoff = np.zeros([basis.bfs_nao])
    for i in range(basis.bfs_nao):
        bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
    # Calculate the value of basis functions for all grid points in batches
    list_nonzero_indices = []
    count_nonzero_indices = []
    # Loop over batches
    for iblock in range(nblocks+1):
        offset = iblock*blocksize
        coords_block = coords[offset : min(offset+blocksize,ngrids)]   
        nonzero_indices, count = nonzero_ao_indices_batch(coords_block, bfs_coords[0], bfs_radius_cutoff)
        list_nonzero_indices.append(nonzero_indices)
        count_nonzero_indices.append(count)
    return list_nonzero_indices, count_nonzero_indices
