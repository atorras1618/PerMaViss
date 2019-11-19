import sparseGauss
from operator import itemgetter


def image_kernel(F, A, B):
    """
    INPUT:
        -F:     differential in terms of domain and codomain bases. 
        -A:     list of birth&death radius of domain generators. 
        -B:     list of birth&death radius of codomain generators. 
    OUTPUT:
        -K:     basis for kernels. 
        -I:     basis for images. 
    """
    # Generate list of values where something is born or death in
    # domain or codomain. 
    a_dict = {}
    for g in A:
        a_dict.append(g[0], g[1])
    
    for g in B: 
        a_dict.append(g[0], g[1])

    a.sort()

def persistent_homology(D, R, max_rad):
    """
    Given the differentials of a chain of persistence modules, we compute the 
    sequence of homology groups. 
    INPUT:
        -D: list of differentials of the chain of persistence modules. 
        -R: list containing birth and death radious of each barcode generator. 
	-max_rad: maximum radius of filtration. We will assume the initial radius is 
		  always 0. 
    OUTPUT:
        -Z: graded list. Each entry Z[n] contains a basis of representatives for 
	    homology classess. These will be stored starting with: birth rad, death rad.
            If a cycle does not die we put max_rad as death radius.
        -B: graded matrices containing bases of boundaries of differentials. Each
	    column will start with the radius where the generator is born. 
        -P: preimage matrices. These correspond to the combinations generating 
	    each of the entries in B. 
    """
    dim = len(D)
    Z = []  # list of cylces
    B = []  # list of bounadaries
    P = []  # list of preboundaries
    for d in range(dim + 1):
        Z.append([])
        B.append([])
        P.append([])

    return


def changes(l):
    """
    Returns a list of changes in a list of barcodes.
    """ 
    changes = []
    for i, gen in enumerate(l): 
        introduce_change(changes, [g[0], i])

    return changes

   
def introduce_change(changes, new_change):
    """
    This adds a new change to a list of changes.
    :param changes: list of lists, where each list starts with the radius of
    change, followed by the indexes of the cycles.
    :param new_change: list [rad, index1, index2, ...]
    :return: changes: modified list of changes.
    """

    changed = False
    for c in changes:
        if c[0] == new_change[0]:
            c.append(new_change[1])
            changed = True
            break

    if not changed:
        changes.append([new_change[0], new_change[1]])
        changes.sort()

    return changes
