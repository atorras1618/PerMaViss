"""
    module_persistence_homology.py

    This module implements the persistence module homology
"""
import numpy as np

from .persistence_algebra import barcode_basis

from .image_kernel import image_kernel

###############################################################################
# Find pivot of array

def _pivot(l):
    """ 
    Given a 1D array of integers, return the index of the last nonzero entry.  
    Returns -1 if the list is zero.  
    INPUT: 
    -l: list on integers. 
    OUTPUT: 
    -index of last nonzero entry. 
    """ 
    l_bool = np.nonzero(l) 
    if len(l_bool[0]) > 0: 
        return l_bool[0][-1] 
     
    return -1 

assert _pivot(np.array([0,1,0,1,0]))==3
assert _pivot(np.array([0,0,0]))==-1


################################################################################
# Quotient of barcode bases
# This function has serious theoretical Issues. Fix it.
 
def quotient(M, N, p):
    """
    Assuming that N generates a submodule of M, we compute 
    a barcode basis for the qoutient M / N
    OUTPUT: Q - barcode base for the quotient M / N
    """
    if N.dim == 0:
        return M
    
    # Check that the barcode bases coincide
    if M.prev_basis != N.prev_basis:
        raise ValueError
    
    # We create N_M, which is not a basis, but will be used as the domain
    # basis in image_kernel so that the quotient mod N can be performed
    N_M_barcode = np.concatenate((N.barcode, M.barcode), axis=0)
    N_M = barcode_basis(N_M_barcode)
    # create a matrix (N|M) in M.prev_basis
    matrix_N_M = np.concatenate((N.coordinates, M.coordinates), axis=1)
    # Compute quotient barcodes for M mod N
    Q, Ker, PreIm = image_kernel(N_M, M.prev_basis, matrix_N_M, p, start_index=N.dim, prev_basis=M)
    return Q


################################################################################
# Persistence module homology mod p

def module_persistence_homology(D, Base, p):
    """
    TO DO: Include a check for well-defined persistence morphisms. 
    Given the differentials of a chain of tame persistence modules, we compute 
    barcode bases for the homology of the chain. 
    INPUT:
        -D: list of differentials of the chain complex.
        -Base: list containing barcode bases for each dimension
        -p: prime number to perform arithmetic mod p
    OUTPUT:
        -Hom: cycles mod boundaries of differentials, starting with: birth rad, death rad.
            If a cycle does not die we put max_rad as death radius.
        -Im: boundaries of differentials
        -PreIm: preimage matrices, 'how to go back from boundaries'
    """
    dim = len(Base)
    # lists of homology, boundaries and preboundaries
    Hom = []
    Im = []  
    PreIm = []  
    for d in range(dim):
        Hom.append([])
        Im.append([])
        PreIm.append([])

    Im[dim - 1] = barcode_basis([], Base[d])
    # Find bases for Kernel and Image, starting from D[dim-1] and ending on D[1].
    for d in range(dim - 1, 0, -1):
        # Handle trivial cases
        if Base[d].dim == 0:
            Im[d-1] = barcode_basis([],Base[d-1])
            Hom[d] = barcode_basis([],Base[d])
        elif Base[d-1].dim == 0:
            Im[d-1] = Base[d-1]
            Hom[d] = Base[d]
        else:
            # compute barcode bases for image and kernel
            Im[d - 1], Ker, PreIm[d] = image_kernel(Base[d], Base[d-1], D[d], p) 
            # Perform quotient of Ker by Im[d]
            Hom[d] = quotient(Ker, Im[d], p)
   
 
    # Trivial case dim 0
    if Base[0].dim == 0:
        Hom[0] = Base[0] 
    else:    
        # All the generators from Base[0] lie in the kernel, since D[0]=0
        Ker = barcode_basis(Base[0].barcode, Base[0], np.identity(Base[0].dim)) 
        Ker.sort()
        if Ker.prev_basis != Im[0].prev_basis:
            print(Ker.prev_basis)
            print(Im[0].prev_basis)
            assert False

        Hom[0] = quotient(Ker, Im[0], p)

    # Return lists of barcode bases for Hom, Im and PreIm
    return Hom, Im, PreIm

###############################################################################
# Test 1

A_bas = barcode_basis([[1,4],[2,4],[3,5]])
B_bas = barcode_basis([[0,4],[0,2],[1,4],[1,3]])
C_bas = barcode_basis([[0,3],[0,2],[0,2]])
Base = [C_bas, B_bas, A_bas]
f = np.array([[0,1,1],[1,0,0],[0,4,1],[0,0,0]])
g = np.array([[0,0,0,1],[4,0,1,3],[1,0,1,4]])
D = [0, g,f]
Hom, Im, PreIm = module_persistence_homology(D, Base, 5)
H0 = np.array([[0,1],[0,1]])
H1 = np.array([[0,1],[2,3]])
H2 = np.array([[2,4],[4,5]])
H = [H0,H1,H2]
PreIm0 = []
PreIm1 = np.array([[1,2,4],[0,0,0],[0,4,1],[0,1,0]])
PreIm2 = np.array([[1,0,0],[0,1,1],[0,0,1]])
PreImages = [PreIm0, PreIm1, PreIm2]
for j in range(3):
    assert np.array_equal(Hom[j].barcode, H[j])
    assert np.array_equal(PreIm[j], PreImages[j])


###############################################################################
# Test 2

C_bas = barcode_basis([[0,5],[0,5],[0,4],[0,2]])
B_bas = barcode_basis([[2,6],[2,7],[3,8],[4,12],[4,10],[4,9]])
A_bas = barcode_basis([[4,9],[5,10],[6,12]])
Base = [C_bas, B_bas, A_bas]
f = np.transpose(np.array([[1,2,3,0,0,0],[1,4,1,0,4,1],[0,1,2,3,4,1]]))
g = np.transpose(np.array([[1,2,3,0],[2,3,1,0],[3,2,1,0],[1,0,0,0],[4,0,0,0],[3,3,0,0]]))
D = [0, g, f]
Hom, Im, PreIm = module_persistence_homology(D, Base, 5)

#     print("H[{}]".format(j))
#     print(Hom[j])

# Write tests checking for PreImages
# assert False

# Test for unordered bases
A = barcode_basis(np.array(
[[ 0.,   2. ],
 [ 0.,   1.2],
 [ 0.,   0.4],
 [ 0.,   0.4]]))
B = barcode_basis(np.array(
[[ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4],
 [ 0.,   2. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   1. ],
 [ 0.,   0.4],
 [ 0.,   0.4]]))
Base = [B, A]
F = np.array(
[[ 1.,  0.,  0.,  0.],
 [ 4.,  4.,  0.,  0.],
 [ 4.,  4.,  0.,  0.],
 [ 0.,  1.,  0.,  0.],
 [ 0.,  1.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.],
 [ 0.,  0.,  1.,  0.],
 [ 0.,  0.,  0.,  1.],
 [ 4.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  4.,  0.,  0.],
 [ 0.,  1.,  0.,  0.],
 [ 0.,  0.,  4.,  0.],
 [ 0.,  0.,  0.,  4.]])
D = [0, F]
Hom, Im, PreIm = module_persistence_homology(D, Base, 5)

