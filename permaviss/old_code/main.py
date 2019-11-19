import matplotlib.pyplot as plt

from vietorisRips import *
from homology import *
from exPointClouds import *
from MayerVietorisSS import spectral_sequence



def main():
    X = [[0,0],[0,1],[1,1],[1,0]]
    max_dim = 3
    max_r = 2
    max_div = 2
    overlap = 2.2
    spectralSequence(X, max_r, max_dim, max_div, overlap)

    """Dist = computeDistanceMatrix(X)

    max_rad = 0.4
    max_dim = 3
    C = filteredRipsComplex(Dist, max_rad, max_dim)

    D = complexDifferentials(C)

    Z, B, P = persistentHomology(D, C, max_rad)

    for d in range(len(Z)):
        print(len(Z[d]))



    for pt in X:
        plt.plot(pt[0], pt[1], 'ro')

    plt.axis('equal')
    plt.title('pointCloud')
    plt.show()

    for d in range(len(Z)-2):
        if Z[d]:
            B = []
            D = []
            y = 0
            step = 1 / len(Z[d])
            for c in Z[d]:
                B.append(c[0])
                D.append(c[1])
                plt.plot([B, D], [y, y], 'r')
                y += step

            plt.xlim(0, max_rad)
            plt.ylim(-step, 1)
            plt.title(d)
            plt.show()"""

if __name__ == '__main__':
    main()
