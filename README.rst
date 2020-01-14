
PerMaViss 
*********

Welcome to PerMaViss! This is a Python3 implementation of the Persistence Mayer Vietoris spectral sequence. 
For a mathematical description of the procedure, see `Distributing Persistent Homology via Spectral Sequences <https://arxiv.org/abs/1907.05228>`_

.. image:: docs/examples/3Dextension.png
   :width: 700 
   :align: center
   

Dependencies
============

PerMaViss requires:

- Python3
- NumPy
- Scipy
- Math

Optional for examples and notebooks:

- Matplotlib
- mpl_toolkits


Installation
============

Permaviss is built on Python 3, and relies only on `NumPy <http://www.numpy.org/>`_, `Math <https://docs.python.org/2/library/math.html>`_ and `Scipy <https://www.scipy.org/>`_. 
Additionally, Matplotlib and mpl_toolkits are used for the tutorials. 

To install, clone from GitHub repository and install::

    $ git clone https://github.com/atorras1618/PerMaViss
    $ cd PerMaViss
    $ pip3 install -e .


DISCLAIMER
==========

**The main purpose of this library is to explore how the Persistent Mayer Vietoris spectral sequence can be used for computing persistent homology.**

**This does not pretend to be an optimal library. Also, it does not parallelize the computations of persistent homology after the first page. Thus, this is slower than most other persistent homology computations.**

**This library is still on development and is still highly undertested. If you notice any issues, please email
atorras1618@gmail.com**

**This library is published under the standard MIT licence. Thus:
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.**

