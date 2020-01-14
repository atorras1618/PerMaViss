.. PerMaViss documentation master file, created by
   sphinx-quickstart on Tue Jan  7 17:02:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PerMaViss' documentation!
====================================

PerMaViss is a Python3 library implementing the Persistence Mayer Vietoris spectral sequence.

Suppose that you start from a point cloud in the `n` dimensional real space. Then, PerMaViss 
creates a cover by hypercubes covering this point cloud. It computes the Persistence Mayer Vietoris 
spectral sequence associated to this cover, and solves the extension problem. The resulting barcodes will be the same as if we computed persistent homology directly on our data. 

Here you will find examples using this. 


.. toctree::
   :maxdepth: 3
   :caption: Contents

   about/index.rst
   install.rst
   examples/index.rst
   reference/index.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
