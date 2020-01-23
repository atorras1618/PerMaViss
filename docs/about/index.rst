
About
*****

Quickstart
==========

The main function which we use is :meth:`permaviss.spectral_sequence.MV_spectral_seq.create_MV_ss`. 
We start by taking 100 points in a noisy circle of radius 1

    >>> from permaviss.sample_point_clouds.examples import random_circle
    >>> point_cloud = random_circle(100, 1, epsilon=0.2)

Now we set the parameters for spectral sequence. These are 

- a prime number `p`, 

- the maximum dimension of the Rips Complex `max_dim`, 

- the maximum radius of filtration `max_r`, 

- the number of divisions `max_div` along the maximum range in `point_cloud`,

- and the `overlap` between different covering regions. 

In our case, we set the parameters to cover our circle with 9 covering regions.
Notice that  in order for the algorithm to give the correct result we need `overlap > max_r`. 

    >>> p = 3
    >>> max_dim = 3
    >>> max_r = 0.2
    >>> max_div = 3
    >>> overlap = max_r * 1.01

Then, we compute the spectral sequence, notice that the method prints the successive page ranks. 

    >>> from permaviss.spectral_sequence.MV_spectral_seq import create_MV_ss
    >>> MV_ss = create_MV_ss(point_cloud, max_r, max_dim, max_div, overlap, p)
    PAGE: 1
    [[  0   0   0   0   0]
     [  7   0   0   0   0]
     [133  33   0   0   0]]
    PAGE: 2
    [[  0   0   0   0   0]
     [  7   0   0   0   0]
     [100   0   0   0   0]]
    PAGE: 3
    [[  0   0   0   0   0]
     [  7   0   0   0   0]
     [100   0   0   0   0]]
    PAGE: 4
    [[  0   0   0   0   0]
     [  7   0   0   0   0]
     [100   0   0   0   0]]

We can inspect the obtained barcodes on the 1st dimension.

    >>> MV_ss.persistent_homology[1].barcode
    array([[ 0.08218822,  0.09287436],
           [ 0.0874977 ,  0.11781674],
           [ 0.10459203,  0.12520266],
           [ 0.14999507,  0.18220508],
           [ 0.15036084,  0.15760192],
           [ 0.16260913,  0.1695936 ],
           [ 0.16462541,  0.16942819]])

Notice that in this case, there was no need to solve the extension problem. See the examples section for nontrivial extensions. 


.. include:: ../../DISCLAIMER.rst

How to cite
===========

Álvaro Torras Casas. (2020, January 20). PerMaViss: Persistence Mayer Vietoris spectral sequence (Version v0.0.2). Zenodo. http://doi.org/10.5281/zenodo.3613870


Reference
=========

This module is written using the algorithm in `Distributing Persistent Homology via Spectral Sequences <https://arxiv.org/abs/1907.05228>`_.

