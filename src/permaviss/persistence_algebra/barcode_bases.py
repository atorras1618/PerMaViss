"""
    barcode_bases.py

"""
import numpy as np


class barcode_basis(object):
    """This class implements barcode bases.

    Associated bars and coordinates are given for some barcode base.
    Each coordinate is stored in a column, whereas bars are stored on rows.
    This generates a barcode basis with all the input data.
    There is the exception of broken barcode bases, which come up
    when solving the extension problem.

    Note
    ----
    Barcode bases are assumed to be well defined in the sense that:

        1)They are linearly independent with respect to boxplus operation

        2)They generate the respective module or submodule.


    Parameters
    ----------
    bars : :obj:`Numpy Array (dim, 2)`
        Each entry is a pair specifying birth and death radius of a bar.
    prev_basis : reference to a previously defined :class:`barcode_basis`.
        This will be the basis in which the coordinates are given.
    coordinates : :obj:`Numpy Array (dim, prev_basis.dim)`
        Cooridnates of this basis in terms of `prev_basis`
    store_well_defined : bool, default is `False`
        Whether we want to store the indices of well defined bars.
        That is, whether we wish to store indices of bars where the
        birth radius is strictly smaller than the death radius.
    broken_basis : bool, default is `False`
        Whether the barcode basis is `broken`. This appears when solving
        the extension problem. A barcode base is `broken` if it is not natural.
    broken_differentials : :obj:`Numpy Array (dim, dim)`
        Matrix of broken differentials.
        These give coefficients of a barcode generator in term of other
        generators. This is used when the given generator dies, and we
        write it in terms of other generators that are still alive.

    Returns
    -------
    :obj:`barcode_basis`

    Raises
    ------
    ValueError
        If `prev_coord.dim` is different to the number of rows in `coordinates`

    ValueError
        If the number of rows in `bar` is different to the number of columns
        in `coordinates`.

    ValueError
        If `broken_basis = True` but `broken_differentials` is not given.

    Examples
    --------

        >>> import numpy as np
        >>> bars = np.array([[0,2],[0,1],[2,3],[2,2]])
        >>> base1 = barcode_basis(bars, store_well_defined=True)
        >>> base1.dim
        3
        >>> base1.well_defined
        array([ True,  True,  True, False], dtype=bool)
        >>> print(base1)
        Barcode basis
        [[0 2]
         [0 1]
         [2 3]]


        >>> bars = np.array([[0,2],[2,3]])
        >>> coordinates = np.array([[1,0],
        ...                         [1,0],
        ...                         [0,2]])
        >>> base2 = barcode_basis(bars, prev_basis, coordinates)
        >>> base2.dim
        2
        >>> print(base2)
        Barcode basis
        [[0 2]
         [2 3]]
        [[1 0]
         [1 0]
         [0 2]]

        >>> bars = np.array([[0,2],[0,1],[1,3]])
        >>> broken_differential = np.array(
        ... [[0,0,0],
        ...  [0,0,0],
        ...  [0,1,0]])
        >>> base3 = barcode_basis(bars, broken_basis=True,
        ... broken_differentials=broken_differential)
        >>> base3.dim
        3
        >>> print(base3)
        Barcode basis
        [[0 2]
         [0 1]
         [1 3]]


    """
    def __init__(self, bars, prev_basis=None, coordinates=np.array([[]]),
                 store_well_defined=False, broken_basis=False,
                 broken_differentials=None):
        """Constructor method
        """
        # When the basis is broken, the death_differentials must be given
        if broken_basis and (broken_differentials is None):
            raise ValueError

        self.broken_basis = broken_basis

        # Store given data
        self.barcode = np.copy(bars)
        self.prev_basis = prev_basis
        self.coordinates = coordinates
        # Assume the barcode basis is not sorted
        self.sorted = False
        # Take out trivial bars and also bad defined bars
        if len(bars) == 0:
            self.dim = 0
        else:
            well_defined = self.barcode[:, 0] < self.barcode[:, 1]
            self.barcode = self.barcode[well_defined]
            self.dim = np.sum(well_defined)

        # Add coordinates, checking for mistakes on the data.
        if (prev_basis is not None) and (np.size(coordinates, 1) > 0):
            # Check that coordinates match the dimension of the previous basis
            if np.size(coordinates, 0) != self.prev_basis.dim:
                print("Error: given coordinates do not match dimension of" +
                      "prev_basis")
                raise ValueError

            # check that same amount of coordinates as bars were given
            if np.size(bars, 0) != np.size(coordinates, 1):
                print("Error:given {} bars but {} coordinates".format(
                        len(bars), np.size(coordinates, 1)))
                raise ValueError

            self.coordinates = self.coordinates[:, well_defined]

        if store_well_defined:
            self.well_defined = well_defined

        # make sure the broken differentials are given as a square
        # matrix of dimension self.dim
        if broken_basis:
            if np.size(broken_differentials, 0) == self.dim and np.size(
                    broken_differentials, 1) == self.dim:
                self.broken_differentials = broken_differentials
            else:
                raise ValueError

    def __str__(self):
        """Printing function for barcode bases.
        """
        print("Barcode basis")
        print(self.barcode)
        if self.prev_basis is not None:
            print(self.coordinates)

        return ""

    def sort(self, precision=7, send_order=False):
        """Sorts a barcode basis according to the standard barcode order.

        That is, from smaller birth radius to bigger, and from
        bigger death radius to smaller. A precision up to n zeros is an
        optional argument.

        Parameters
        ----------
        precision : int, default is 7
            Number of zeros of precision.
        send_order : bool, default is False
            Whether we want to generate a :obj:`Numpy Array` storing
            the original order.

        Returns
        -------
        :obj:`Numpy Array`
            One dimensional array storing original order of barcodes.

        Example
        -------

            >>> bars = np.array([[1,2],[0,4],[-1,3],[1,4]])
            >>> base4 = barcode_basis(bars)
            >>> base4.sort(send_order=True)
            array([2, 1, 3, 0])
            >>> base4 = barcode_basis(bars)
            >>> print(base4)
            Barcode basis
            [[ 1  2]
             [ 0  4]
             [-1  3]
             [ 1  4]]
            >>> base4.sort(send_order=True)
            array([2, 1, 3, 0])
            >>> print(base4)
            Barcode basis
            [[-1  3]
             [ 0  4]
             [ 1  4]
             [ 1  2]]

        """

        aux_bars = np.around(self.barcode, decimals=precision)
        order = np.lexsort((-aux_bars[:, 1], aux_bars[:, 0]))
        if self.prev_basis is not None:
            self.coordinates = self.coordinates[:, order]

        self.barcode = self.barcode[order]
        self.sorted = True
        if self.broken_basis:
            self.broken_differentials = self.broken_differentials[order]
            self.broken_differentials = self.broken_differentials[:, order]

        if send_order:
            return order

    ###########################################################################
    # Functions used by Image_kernel:

    def changes_list(self):
        """Returns an array with the values of changes occurring in the basis.

        Returns
        -------
        :obj:`Numpy Array`
            One-dimensional array with radii where either a bar dies or is
            born in `self`.

        Example
        -------
            >>> print(base1)
            Barcode basis
            [[0 2]
             [0 1]
             [2 3]]
            >>> base1.changes_list()
            array([0, 1, 2, 3])

        """
        changes_radii = np.resize(self.barcode, (1, 2 * len(self.barcode)))
        return np.unique(changes_radii)

    def active(self, rad):
        """Returns array with active indices at rad.

        Option to restrict to generators from start to self.dim

        Parameters
        ----------
        rad : `float`
            Radius where we want to check which barcodes are active.

        Returns
        -------
        :obj:`Numpy Array`
            One-dimensional array with indices of active generators.

        Example
        -------
            >>> print(base3)
            Barcode basis
            [[0 2]
             [0 1]
             [1 3]]
            >>> base3.active(1.2)
            array([0, 2])


        """
        indices = np.array(range(self.dim))
        active_bool = np.logical_and(self.barcode[:, 0] <= rad,
                                     rad < self.barcode[:, 1])
        return indices[active_bool]

    def birth(self, rad):
        """Returns an array with the indices being born at rad.

        Parameters
        ----------
        rad : float
            Radius at which generators might be born

        Example
        -------
            >>> print(base4)
            Barcode basis
            [[-1  3]
             [ 0  4]
             [ 1  4]
             [ 1  2]]
            >>> base4.birth(1)
            array([2, 3])

        """
        indices = np.array(range(self.dim))
        birth_bool = (self.barcode[:, 0] == rad)
        return indices[birth_bool]

    def death(self, rad):
        """Returns an array with the indices dying at rad.

        Parameters
        ----------
        rad : float
            Radius at which generators might be dying

        Example
        -------
            >>> print(base4)
            Barcode basis
            [[-1  3]
             [ 0  4]
             [ 1  4]
             [ 1  2]]
            >>> base4.death(4)
            array([1, 2])

        """
        indices = np.array(range(self.dim))
        death_bool = (self.barcode[:, 1] == rad)
        return indices[death_bool]

    def update_broken(self, A, rad):
        """Updates a matrix A, using the broken differentials of the
        generators dying at rad.

        We assume that the broken barcode basis is indexing the rows of A.

        Parameters
        ----------
        A : :obj:`Numpy Array`
            columns represent coordinates in the barcode_basis.
        rad : float
            Radius at which we want to update the coordinates using the
            broken_differentials.

        Returns
        -------
        A : :obj:`Numpy Array`
            return updated matrix

        Example
        -------
            >>> print(base3)
            Barcode basis
            [[0 2]
             [0 1]
             [1 3]]
            >>> A = np.array(
            ... [[1,1,1],
            ...  [0,2,-1],
            ...  [0,1,1]])
            >>> base3.update_broken(A, 1)
            array([[1, 1, 1],
                   [0, 0, 0],
                   [0, 3, 0]])


        """
        dying_indices = self.death(rad)
        if len(dying_indices) > 0:
            sum_update = np.matmul(self.broken_differentials[:, dying_indices],
                                   A[dying_indices])
            A[dying_indices] = np.zeros((len(dying_indices), np.size(A, 1)))
            A = A + sum_update

        return A
