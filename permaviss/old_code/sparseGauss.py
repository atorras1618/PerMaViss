

def sum_sparse(a, b):
    """Adds two sparse vectors and returns the result ordered"""
    s = []  #initialize solution list
    for n in a:
        if b.count(n)==0:
            s.append(n)
    for n in b:
        if a.count(n)==0:
            s.append(n)
    return sorted(s)


def gauss(A, init):
    """
    Reduce matrix gaussian. Returns reduced matrix and transicion matrix
    INPUT:
        -A: matrix to be reduced.
        -init: initial index of reduction; earlier columns will not be reduced.
    OUTPUT:
        -R: reduced A from init. 
        -T: transition basis
    """
    T = []  # initialize transition sparse matrix
    R = A[:][:]  # initialize reduced matrix equal to A
    """Copy entries of A to R, so that we do not modify the original matrix A"""

    for n in range(len(R)):
        T.append([n])

    """Perform now gaussian elimination and obtain transition matrix T.
    T is the matrix from the new basis to the old one"""
    for i in range(init, len(R)):
        unreduced = True
        while (unreduced):
            unreduced = False
            for j in range(len(R))[:i]:
                if (len(R[j]) > 0) & (len(R[i]) > 0):
                    if max(R[i]) == max(R[j]):
                        """If the pivots coincide sum the lower column to the higher one"""
                        R[i] = sum_sparse(R[i], R[j])
                        T[i] = sum_sparse(T[i], T[j])
                        unreduced = True
                        break

    """Reorder transition basis"""
    for i in range(len(T)):
        T[i] = sorted(T[i])

    return R, T


def gaussSolve(A, b):
    """ Given a vector of lists A, and a given list b we return a list s.
    These are a sparse representation of the problem As=b."""
    R = A[:][:]  # sparse matrix to be reduced
    a = b[:]  # auxiliary b so that we do not modify the original
    s = []  # initialize solution list
    """First we reduce A with gaussian Elimination"""
    R, T = gauss(R, 0)
    """We asssume that a solution exists. Otherwise there will be a loop"""
    while (len(a) > 0):
        for i in range(len(R)):
            if R[i]:
                if (R[i][-1] == a[-1]):
                    a = sum_sparse(R[i], a)
                    s.append(i)
                    break
    
    return compose(T, s)


def compose(A, b):
    """Returns the sparse vector corresponding to A*b.
    Where A is a sparse matrix and b is a sparse vector. We assume 
    that len(A) is bigger or equal to max(b)"""
    s = []  # initialize solution list
    for i in b:
        s = sum_sparse(s, A[i])

    return sorted(s)
