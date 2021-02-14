
import numpy as np

from permaviss.persistence_algebra.barcode_bases import barcode_basis

from permaviss.persistence_algebra.image_kernel import image_kernel


def test_image_kernel_1():
    A = barcode_basis([[1, 5], [1, 4], [2, 5]])
    B = barcode_basis([[0, 5], [0, 3], [1, 4]])
    F = np.array([[0, 0, 1], [1, 0, 0], [1, 1, 1]])
    Im = np.array([[0, 0, 1], [4, 4, 0], [4, 0, 0]])
    Ker = np.array([[4, 1, 0]]).T
    PreIm = np.array([[4, 4, 0], [0, 1, 4], [0, 0, 1]])
    res_Im, res_Ker, res_PreIm = image_kernel(A, B, F, 5)
    print(res_Im.coordinates)
    print(Im)
    assert np.array_equal(res_Im.coordinates, Im)
    assert np.array_equal(res_Ker.coordinates, Ker)
    assert np.array_equal(res_PreIm, PreIm)


def test_image_kernel_2():
    A = barcode_basis([[1, 8], [1, 5], [2, 5], [4, 8]])
    B = barcode_basis([[-1, 3], [0, 4], [0, 3.5], [2, 5], [2, 4], [3, 8]])
    F = np.array([
        [4, 1, 1, 0],
        [1, 4, 1, 0],
        [1, 1, 4, 0],
        [0, 0, 1, 4],
        [0, 0, 4, 1],
        [0, 0, 0, 1]])
    res_Im, res_Ker, res_PreIm = image_kernel(A, B, F, 5)
    Im = np.array([
     [4, 0, 1, 0],
     [1, 0, 1, 0],
     [1, 2, 4, 0],
     [0, 0, 1, 4],
     [0, 0, 4, 1],
     [0, 0, 0, 1]])
    Ker = np.array([
     [1, 0],
     [1, 4],
     [0, 0],
     [0, 0]])
    PreIm = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.array_equal(res_Im.coordinates, Im)
    assert np.array_equal(res_Ker.coordinates, Ker)
    assert np.array_equal(res_PreIm, PreIm)


def test_image_kernel_3():
    A = barcode_basis([[0, 4], [0, 2], [1, 4], [1, 3]])
    B = barcode_basis([[0, 3], [0, 2], [0, 2]])
    F = np.array([[0, 0, 0, 1], [4, 0, 1, 3], [1, 0, 1, 4]])
    res_Im, res_Ker, PreIm = image_kernel(A, B, F, 5)
    Ker_basis = np.array([[0, 2], [2, 4],  [2, 4]])
    Im_basis = np.array([[0, 2], [1, 3], [1, 2]])
    res_PreIm = np.array([[1, 2, 4], [0, 0, 0], [0, 4, 1], [0, 1, 0]])
    assert np.array_equal(Im_basis, res_Im.barcode)
    assert np.array_equal(Ker_basis, res_Ker.barcode)
    assert np.array_equal(res_PreIm, PreIm)


def test_image_kernel_4():
    A = barcode_basis([[1, 4], [2, 4], [3, 5]])
    B = barcode_basis([[0, 4], [0, 2], [1, 4], [1, 3]])
    F = np.array([[0, 1, 1], [1, 0, 0], [0, 4, 1], [0, 0, 0]])
    res_Im, res_Ker, PreIm = image_kernel(A, B, F, 5)
    Ker_basis = np.array([[2, 4], [4, 5]])
    Im_basis = np.array([[1, 2], [2, 4], [3, 4]])
    assert np.array_equal(Im_basis, res_Im.barcode)
    assert np.array_equal(Ker_basis, res_Ker.barcode)


def test_image_kernel_5():
    A = barcode_basis([[0, 4], [0, 2], [1, 4], [1, 3]])
    A_rel = barcode_basis([[0, 2], [1, 4], [1, 3]])
    B = barcode_basis([[0, 3], [0, 2], [0, 2]])
    F = np.array([[0, 0, 0, 1], [4, 0, 1, 3], [1, 0, 1, 4]])
    res_Im, res_Ker, PreIm = image_kernel(
        A, B, F, 5, start_index=1, prev_basis=A_rel)
    Ker_basis = np.array([[0, 2], [2, 4]])
    Im_basis = np.array([[1, 3], [1, 2]])
    res_PreIm = np.array([[0, 0], [4, 1], [1, 0]])
    assert np.array_equal(Im_basis, res_Im.barcode)
    assert np.array_equal(Ker_basis, res_Ker.barcode)
    assert np.array_equal(res_PreIm, PreIm)


def test_image_kernel_6():
    # Test for unordered bases
    start_idx = 4
    A = np.array([
        [0.,   2.],
        [0.,   1.],
        [0.,   0.4],
        [0.,   0.4],
        [0.,   2.],
        [0.,   2.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   0.4],
        [0.,   0.4],
        [0.,   0.4],
        [0.,   0.4]])
    A_dim = np.size(A, 0)
    shuffleA = np.append(range(4,  A_dim, 2), range(5, A_dim, 2), axis=0)
    shifted_shuffleA = np.append(range(4), shuffleA, axis=0)
    A_shuffle = barcode_basis(A[shifted_shuffleA])
    A_rel = barcode_basis(A[start_idx:])
    A = barcode_basis(A)
    B = np.array([
        [0.,   2.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   0.4],
        [0.,   0.4],
        [0.,   2.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   0.4],
        [0.,   0.4]])
    B_dim = np.size(B, 0)
    shuffleB = np.append(range(0, B_dim, 2), range(1, B_dim, 2), axis=0)
    B_shuffle = barcode_basis(B[shuffleB])
    B = barcode_basis(B)
    F = np.array([
        [1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [4.,  4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [4.,  4.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
        [4.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
        [0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
        [0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    F_shuffle = F[shuffleB]
    F_shuffle = F_shuffle[:, shifted_shuffleA]
    Im = np.array([
        [0.,   2.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   1.],
        [0.,   0.4],
        [0.,   0.4]])
    res_Im_sh, res_Ker_sh, PreIm_sh = image_kernel(
        A_shuffle, B_shuffle, F_shuffle, 5,
        start_index=start_idx, prev_basis=A_rel)
    res_Im, res_Ker, PreIm = image_kernel(
        A, B, F, 5,  start_index=start_idx, prev_basis=A_rel)
    shuffleA = shuffleA - start_idx
    assert np.array_equal(res_Im_sh.barcode, Im)
    assert np.array_equal(res_Im.barcode, Im)
    assert np.array_equal(res_Ker_sh.coordinates,
                          res_Ker.coordinates[shuffleA])


def test_image_kernel_7():
    A = barcode_basis(np.array([
        [0.20065389,  0.21080812],
        [0.25135865,  0.29062369],
        [0.25135865,  0.27869416],
        [0.25403887,  0.3],
        [0.28256212,  0.3],
        [0.29679942,  0.3],
        [0.24715565,  0.29062369]]))
    B = barcode_basis(np.array([
        [0.20065389,  0.21080812],
        [0.25135865,  0.29062369],
        [0.25135865,  0.27869416],
        [0.25403887,  0.3],
        [0.28256212,  0.3],
        [0.29679942,  0.3],
        [0.24715565,  0.25135865]]))
    F = np.array([
        [1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  1.,  0.,  0.,  0.,  0.,  3.],
        [0.,  0.,  1.,  0.,  0.,  0.,  3.],
        [0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [0.,  0.,  0.,  0.,  1.,  0.,  0.],
        [0.,  0.,  0.,  0.,  0.,  1.,  0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    res_Im, res_Ker, PreIm = image_kernel(A, B, F, 5)
    Im_bar = np.array([
        [0.20065389,  0.21080812],
        [0.24715565,  0.29062369],
        [0.25135865,  0.27869416],
        [0.25403887,  0.3],
        [0.28256212,  0.3],
        [0.29679942,  0.3]])
    assert np.array_equal(Im_bar, res_Im.barcode)


def test_image_kernel_8():
    # Test for quotients
    A = barcode_basis(np.array([[2.5, 5], [0, 5], [1, 4], [2, 3]]))
    B = barcode_basis(np.array([[0, 5], [1, 4], [2, 3]]))
    F = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1]])
    res_Im, res_Ker, PreIm = image_kernel(A, B, F, 5, start_index=1)
    Im_bar = np.array([
        [0.,  4.],
        [1.,  3.],
        [2.,  2.5]])
    Im_coord = np.array([
        [1.,  1.,  1.],
        [0.,  1.,  1.],
        [0.,  0.,  1.]])
    assert np.array_equal(Im_bar, res_Im.barcode)
    assert np.array_equal(Im_coord, res_Im.coordinates)


def test_image_kernel_9():
    # Test for dying dependent classes in Kernel after adding them
    # through the image reduction.
    A = barcode_basis([[1, 8], [2, 8], [3, 5]])
    B = barcode_basis([[0, 5], [1, 8], [2, 5]])
    F = np.array([[1, 0, -1], [1, -1, 0], [0, 1, -1]])
    res_Im, res_Ker, res_PreIm = image_kernel(A, B, F, 5)
    # expected results
    Im_barcode = np.array([[1., 8.], [2., 5.]])
    Im_coord = np.array([[1., 1.], [1., 0.], [0., 1.]])
    Ker_barcode = np.array([[3., 8.]])
    Ker_coord = np.array([[1.], [1.], [1.]])
    PreIm = np.array([[1., 1.], [0., 1.], [0., 0.]])
    print(res_Im)
    print(res_Ker)
    print(res_PreIm)
    assert np.array_equal(Im_barcode, res_Im.barcode)
    assert np.array_equal(Im_coord, res_Im.coordinates)
    assert np.array_equal(Ker_barcode, res_Ker.barcode)
    assert np.array_equal(Ker_coord, res_Ker.coordinates)
    assert np.array_equal(res_PreIm, PreIm)


if __name__ == "__main__":
    test_image_kernel_1()
    test_image_kernel_2()
    test_image_kernel_3()
    test_image_kernel_4()
    test_image_kernel_5()
    test_image_kernel_6()
    test_image_kernel_7()
    test_image_kernel_8()
    test_image_kernel_9()
