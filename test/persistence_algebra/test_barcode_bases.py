
import numpy as np

from permaviss.persistence_algebra.barcode_bases import barcode_basis


def test_barcode_basis():
    barcode = [[1, 2], [1, 4], [-0.25, 5], [0.1, 1]]
    bas = barcode_basis(barcode)
    order = bas.sort(send_order=True)
    assert np.array_equal(order[np.argsort(order)], range(4))
    assert np.array_equal(np.array([
        [-0.25, 5.], [0.1, 1.], [1., 4.], [1., 2.]]), bas.barcode)
    assert np.array_equal(bas.active(1.1), np.array([0, 2, 3]))
    assert np.array_equal(bas.birth(1), np.array([2,3]))
    assert np.array_equal(bas.death(4), np.array([2]))
    assert np.array_equal(bas.changes_list(),
                          np.array([-0.25, 0.1, 1, 2, 4, 5]))
