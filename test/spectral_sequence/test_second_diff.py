#import numpy as np
#import scipy.spatial.distance as dist
#
#from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
#
#from permaviss.simplicial_complexes.vietoris_rips import vietoris_rips
#from permaviss.simplicial_complexes.differentials import complex_differentials
#from permaviss.persistence_algebra.PH_classic import persistent_homology
#
#
#def test_second_diff():
#    X = np.array([
#        [-2.13714531e-01,   4.25223397e-01,   2.74083041e-01],
#        [4.01913345e-01,  -4.72134869e-01,  -4.73859599e-01],
#        [2.38914429e-01,  -4.31470540e-01,   4.83776997e-01],
#        [-4.89370820e-01,  -2.07968729e-01,  -3.89636353e-01],
#        [4.24500295e-01,   4.58964614e-01,  -4.98237341e-01],
#        [-4.76572637e-01,  -4.06088514e-01,   4.05202146e-01],
#        [4.61652748e-01,   2.34792814e-01,   2.04225883e-01],
#        [-3.52711052e-01,   4.26715183e-01,  -4.80003363e-01],
#        [-2.81490512e-02,  -1.06009136e-01,   2.02445947e-02],
#        [4.96239456e-01,  -4.80397512e-01,   2.17295778e-02],
#        [1.11144441e-01,  -6.45898285e-02,  -4.82885781e-01],
#        [1.11734990e-01,   4.44344773e-01,  -1.07659800e-01],
#        [-5.32530741e-02,   1.75106037e-02,   4.95661087e-01],
#        [4.80731798e-01,  -3.59756275e-02,  -1.96771412e-01],
#        [-4.95507778e-01,  -1.03466987e-02,   3.13125635e-02],
#        [-1.63373706e-02,  -4.69398375e-01,  -2.68753162e-01],
#        [-4.45299445e-01,  -4.85796964e-01,  -6.68902706e-02],
#        [-4.72004853e-01,   9.70583788e-02,   4.96291646e-01],
#        [-7.82471997e-02,  -4.97384572e-01,   1.68513920e-01],
#        [2.00439090e-01,   4.26702152e-01,   4.68103994e-01],
#        [-4.70900282e-01,   4.92768146e-01,  -4.51163885e-02],
#        [4.86542849e-01,  -1.65661358e-01,   3.36148725e-01],
#        [-1.91356565e-01,   1.68627422e-01,  -1.86292084e-01],
#        [5.29252474e-02,   2.94990466e-01,  -4.29486145e-01],
#        [4.78790968e-01,   4.65502419e-01,  -5.58868179e-02],
#        [9.46372648e-02,   1.58764102e-01,   1.97332579e-01],
#        [3.11750791e-01,   8.74414056e-02,   4.71358652e-01],
#        [-2.66839044e-01,  -1.86830845e-01,   2.36774532e-01],
#        [2.11860827e-01,  -2.41622535e-01,  -1.88999919e-01],
#        [2.10742604e-01,  -2.43169589e-01,   2.00870011e-01],
#        [-2.55390495e-01,  -3.33269483e-03,  -4.46772579e-01],
#        [-2.89947388e-01,  -2.19377661e-01,  -1.12359682e-01],
#        [-1.45230521e-01,  -3.75468394e-01,   4.51457520e-01],
#        [-4.84274992e-01,   1.89424136e-01,  -2.76887012e-01],
#        [-2.18059968e-01,   1.32276711e-01,   1.96619070e-01],
#        [-2.93920729e-01,  -4.78747614e-01,  -3.89491137e-01],
#        [4.94302449e-01,   2.63667032e-01,  -2.82673654e-01],
#        [1.70233344e-01,   1.53557436e-01,  -9.51354726e-02],
#        [4.22679639e-01,  -1.79520845e-01,  -4.60215694e-01],
#        [-4.61283014e-01,   3.95823689e-01,   4.19869086e-01],
#        [1.14713860e-01,   4.97274281e-01,   2.01893714e-01],
#        [-1.80772158e-01,   4.48634933e-01,  -9.97324867e-03],
#        [-7.59471010e-02,   3.05404646e-01,   4.90944708e-01],
#        [-1.31765376e-01,  -2.64189775e-01,  -4.71488836e-01],
#        [3.50626151e-01,   8.53619684e-03,   6.31736864e-02],
#        [3.27342370e-01,   1.12736500e-01,  -4.67830272e-01],
#        [2.26969648e-01,  -4.80957040e-01,  -5.41255144e-02],
#        [-3.59343891e-01,  -1.74093080e-01,   4.86893652e-01],
#        [4.66999261e-01,  -4.16984835e-01,   2.72692653e-01],
#        [-6.91211582e-02,  -4.87917144e-02,  -2.46868978e-01],
#        [4.34318398e-01,  -3.52282399e-01,  -2.30194602e-01],
#        [-1.85350528e-01,   4.05890330e-01,  -2.62470454e-01],
#        [-3.90897570e-01,   2.57630258e-01,  -5.62473929e-02],
#        [-4.88922065e-01,  -8.98057939e-02,   2.88336801e-01],
#        [3.48242761e-02,  -2.03828999e-01,   4.29397943e-01],
#        [1.23360188e-01,  -4.05973253e-01,  -4.60472677e-01],
#        [4.34598592e-01,   4.59320146e-01,   4.07317279e-01],
#        [2.76543740e-01,  -1.52277597e-01,   4.94491146e-01],
#        [2.72507857e-01,   4.53784173e-01,  -3.08576698e-01],
#        [-4.05842123e-02,  -3.26184908e-02,   2.51198145e-01],
#        [-1.21954911e-03,  -3.63745861e-01,  -5.33832257e-02],
#        [-4.32475653e-01,  -2.91193914e-01,   9.91502909e-02],
#        [1.60707196e-01,  -4.96748331e-01,   1.73419212e-01],
#        [-2.95159529e-01,  -4.48162501e-01,   2.60348915e-01],
#        [3.67709790e-01,   4.49301866e-01,   1.76591979e-01],
#        [-2.05662982e-01,  -4.82046093e-01,  -1.29367220e-01],
#        [4.35391399e-01,   1.92287067e-01,  -5.30268064e-02],
#        [2.03230577e-01,  -3.90130422e-03,  -2.78116020e-01],
#        [-4.08775879e-02,   2.20556669e-01,   1.90769388e-02],
#        [-4.47807882e-01,  -8.13703532e-02,  -2.05568991e-01],
#        [-4.90648535e-01,   3.99916522e-01,   1.56957368e-01],
#        [2.40234266e-01,  -3.81754425e-02,   2.69566353e-01],
#        [-1.49435190e-01,   1.81721485e-01,  -4.07422016e-01],
#        [-1.95551049e-01,  -3.59765989e-01,   5.05509993e-02],
#        [-3.12584419e-01,   2.36738768e-01,   4.56346884e-01],
#        [-2.79559496e-01,  -1.41068778e-05,   3.52485705e-01],
#        [2.43368926e-01,  -4.53663433e-01,  -2.64863495e-01],
#        [2.50806322e-01,   3.13136489e-01,   2.98814784e-01],
#        [4.64167960e-01,  -1.59744390e-01,   1.06879739e-01],
#        [4.84447427e-01,   4.69861875e-01,  -2.93875544e-01],
#        [-4.84200093e-01,   2.13061888e-01,  -4.81443616e-01],
#        [1.10900037e-02,  -2.66281055e-01,  -2.66423714e-01],
#        [-1.41919528e-01,   4.52894129e-01,  -4.57095104e-01],
#        [-9.40200734e-03,   2.65552917e-01,  -2.36584584e-01],
#        [4.35401772e-01,  -4.85193017e-01,   4.90717018e-01],
#        [-4.55376104e-02,   3.04497817e-01,   2.07928691e-01],
#        [1.85612046e-02,  -3.73082920e-01,   3.19525063e-01],
#        [2.57120177e-01,   2.21216275e-01,   1.01338061e-01],
#        [3.28704483e-01,  -3.79758582e-01,   1.17556594e-01],
#        [-4.89856589e-01,   4.28481106e-01,  -3.36128640e-01],
#        [-4.75262567e-01,  -4.44026774e-01,  -4.63118046e-01]])
#    max_r = 0.39
#    max_dim = 4
#    max_div = 2
#    overlap = max_r
#    p = 5
#    # compute ordinary persistent homology
#    Dist = dist.squareform(dist.pdist(X))
#    C, R = vietoris_rips(Dist, max_r, max_dim)
#    Diff = complex_differentials(C, p)
#    PerHom, _, _ = persistent_homology(Diff, R, max_r, p)
#    # compute spectral sequence
#    MV_ss = create_MV_ss(X, max_r, max_dim, max_div, overlap, p)
#    # Check that computed barcodes coincide
#    for it, PH in enumerate(MV_ss.persistent_homology):
#        assert np.array_equal(PH.barcode, PerHom[it].barcode)
#
