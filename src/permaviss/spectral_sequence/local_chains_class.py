"""
This module takes care of chains stored on first page of the spectral sequence.

The format will be, for a certain position (n_deg, deg):
References on each open cover of intersection degree `n_deg`
Local chain coordinates as rows for each reference.
"""

import numpy as np

class local_chains(object):
    """Chains in local forms, references and local coordinates.
    """
    def __init__(self, *args):
        """Initialize local_chains
        """
        if len(args) == 2:
            if len(references) != len(local_coordinates):
                raise ValueError
            for idx, ref in enumerate(args[0]):
                if len(ref) != np.size(local_coordinates, 0):
                    raise ValueError
                self.ref = np.copy(ref)
                self.coord = np.copy(args[1][idx])
        elif isinstance(args[0], int):
            # initialize empty references and coordinates
            self.ref, self.coord = [], []
            for i in range(no_covers):
                self.ref.append([])
                self.coord.append([])

    def add_entry(self, index, ref, coord):
        if istype(coord, list):
            if len(coord) == len(ref) == 1:
                return
            else:
                raise ValueError
        elif len(ref) == self.size(coord, 0):
            self.ref[index] = ref
            self.coord[index] = coord
        else:
            raise ValueError

    def __add__(self, A, B):
        """Given two local chains, adds then and returns the result.

        Assumes that references are the same.
        """
        new_ref, new_chains = [], []
        for idx, A_ref in enumerate(A[0]):
            A_coord, B_ref, B_coord = A[1][idx], B[0][idx], B[1][idx]
            C_ref = np.unique(np.append(A_ref, B_ref))
            new_ref.append(C_ref.astype(int))
            if min(len(A_ref), len(B_ref)) > 0:
                C_coord = np.zeros((len(C_ref), np.size(A_coord, 1)))
                C_coord[np.isin(C_ref, A_ref)] += A_coord
                C_coord[np.isin(C_ref, B_ref)] += B_coord
            elif len(A_ref) > 0:
                C_coord = A_coord
            elif len(B_coord) > 0:
                C_coord = B_coord
            else:
                C_coord = []
            new_chains.append(C_coord)
            # end if
        # end for
        return [new_ref, new_chains]
# end local_chains class


###############################################################################
# local_sums


def local_sums(chains, sums):
    """Sum chains as indicated by "sums"

    Each sum is given by rows in sums. These store the coefficients that
    the chain entries need to be added.

    chains are given in localized form
    """
    no_sums = np.size(sums, axis=0)
    new_ref = []
    new_coord = []
    for local_idx, ref in enumerate(chains[0]):
        if len(ref) > 0:
            new_ref.append(np.array(range(no_sums)).astype(int))
            new_coord.append(np.matmul(
                chains[1][local_idx].T, (sums.T)[ref]).T)
        else:
            new_ref.append([])
            new_coord.append([])
    # end for
    return [new_ref, new_coord]

###############################################################################
# copy_seq_local


def copy_seq_local(seq_chains):
    """ Given a sequence of local chains, makes a copy and returns it.
    """
    copy_seq = []
    for chains in seq_chains:
        copy_seq.append(local_chains(chains.ref, chains.coord))
    return copy_seq
