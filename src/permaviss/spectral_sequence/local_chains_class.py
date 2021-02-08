"""
This module takes care of chains stored on first page of the spectral sequence.

The format will be, for a certain position (n_deg, deg):
References on each open cover of intersection degree `n_deg`
Local chain coordinates as rows for each reference.
"""

import numpy as np


class local_chains(object):
    """Chains in local forms, references and local coordinates.

    Attributes
    ----------
    p : int or None
        Prime number to be used for computations.
    """
    p = None

    def __init__(self, *args):
        """Initialize local_chains.

        Parameters
        ----------
        *args : int or two `:obj:list`
            If one parameter is given, it has to be an `int` indicating the
            number of local entries to save.
            If two parameters are given, these have to be two `:obj:list`. The
            first stores references while the second, `:obj:Numpy Array` with
            local coordinates.

        Attributes
        ----------
        ref : `:obj:list`
            List of lists containing local references.
        coord : `:obj:list`
            List of `:obj:Numpy Array` indicating local coordinates. The
            rows correspond to the number of indices in references. This is
            what we mean by `local_chains` standard format.

        Raises
        ------
        ValueError
            Whenever more than two arguments are given in *args.
            If `len(args[0])` does not match `len(args[1])`
            If the number of elements on the `k` entry in `args[0]` does not
            match the number of rows on the `k` entry in `args[0]`.

        """
        if len(args) == 2:
            if len(args[0]) != len(args[1]):
                raise ValueError
            self.ref, self.coord = [], []
            for idx, ref in enumerate(args[0]):
                if len(ref) != np.size(args[1][idx], 0):
                    raise ValueError
                self.ref.append(np.copy(ref))
                self.coord.append(np.copy(args[1][idx]))
        elif isinstance(args[0], int):
            # initialize empty references and coordinates
            self.ref, self.coord = [], []
            for i in range(args[0]):
                self.ref.append([])
                self.coord.append([])
        else:
            raise ValueError

    ###########################################################################
    # add_entry

    def add_entry(self, index, ref, coord):
        """ Add a references and coordinates at some index.

        Checks that the input is correct.

        Parameters
        ----------
        index : int
            Position in local_chains object.
        ref : :obj:`list`
            Reference list for current index.
        coord : :obj:`Numpy Array` or :obj:`list`
            Local coordinates where expressions are indexed by rows. If it is
            a list it is the empty list [].

        Raises
        ------
        ValueError
            If the given pair of ref and coord do not respect the format rules.
        """
        # include trivial case of adding [] and [] at index
        if isinstance(coord, list):
            if len(coord) == len(ref) == 0:
                return
            else:
                print(coord)
                print(ref)
                raise ValueError
        # check that number of expressions match references
        elif len(ref) == np.size(coord, 0):
            self.ref[index] = ref
            self.coord[index] = coord
        else:
            raise ValueError

    ###########################################################################
    # __add__

    def __add__(self, B):
        """Given two local chains, adds then and returns the result.

        Assumes that references are the same

        Parameters
        ----------
        B : `:class:local_chains` object
        """
        new_ref, new_chains = [], []
        A = self
        for idx, A_ref in enumerate(A.ref):
            A_coord, B_ref, B_coord = A.coord[idx], B.ref[idx], B.coord[idx]
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
        return local_chains(new_ref, new_chains)

    ###########################################################################
    # minus

    def minus(self):
        """ Minus all local coordinates of self.
        """
        for idx, coordinates in enumerate(self.coord):
            if len(coordinates) > 0:
                self.coord[idx] = -coordinates % self.p

    ###########################################################################
    # local_sums

    def sums(chains, coord_sums):
        """Sum chains as indicated by "coord_sums"

        Parameters
        ----------
        chains : :class:`local_chains` object
            Chains in local_chains format to be added.
        coord_sums : :obj:`Numpy Array`
            Each sum is given by rows in coord_sums. These store the
            coefficients that the chain entries need to be added.

        """
        no_sums = np.size(coord_sums, axis=0)
        new_ref = []
        new_coord = []
        for local_idx, ref in enumerate(chains.ref):
            if len(ref) > 0:
                new_ref.append(np.array(range(no_sums)).astype(int))
                new_coord.append(np.matmul(
                    chains.coord[local_idx].T, (coord_sums.T)[ref]).T)
            else:
                new_ref.append([])
                new_coord.append([])
        # end for
        return local_chains(new_ref, new_coord)

    ###########################################################################
    # copy_seq_local

    def copy_seq(seq_chains):
        """ Given a sequence of local chains, makes a copy and returns it.
        """
        copy_seq = []
        for chains in seq_chains:
            copy_seq.append(local_chains(chains.ref, chains.coord))
        return copy_seq


# end local_chains class
