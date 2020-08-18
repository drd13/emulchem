========
emulchem
========


.. image:: https://img.shields.io/pypi/v/emulchem.svg
        :target: https://pypi.python.org/pypi/emulchem

.. image:: https://readthedocs.org/projects/emulchem/badge/?version=latest
        :target: https://emulchem.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




A collection of statistical emulators for the UCLCHEM and RADEX astronomical codes. The UCLCHEM emulators can be used to obtain chemical abundances for various molecules under varying physical conditions. The RADEX emulator can be used to estimate the strength of molecular lines. Both emulators are built using neural networks. For more detail on the emulation procedure and its applications, please refer to the associated paper. Please cite the paper if you end up using it.


Installation
------------

The module can be installed either through pip::

   pip install emulchem

or directly from the repository::

    git clone https://github.com/drd13/emulchem.git
    cd emulchem
    pip install .


Usage
-----

We give here a quick demonstration of our package. Check the `Documentation
<https://emulchem.readthedocs.io>`_ for a more thorough explanation.

The emulators can be accessed through our ``ChemistryEmulator`` and ``RadexEmulator`` objects::

   import emulchem
   CS = emulchem.ChemistryEmulator(specie="CS")

Predictions can be obtained through calling the ``get_prediction`` method::

    radfield = 10
    zeta = 10
    density = 10**5 
    temperature =  200
    metallicity = 1
    av = 10
    CS.get_prediction(radfield,zeta,density,av,temperature,metallicity)

Units for input parameters are shown through ``help``::

    help(CS.get_prediction)

If the requested prediction lies outside the emulator range an error will be raised.
 


Citation
--------

.. code-block:: tex


 @article{ emulchem,
	author = {{de Mijolla, D.} and {Viti, S.} and {Holdship, J.} and {Manolopoulou, I.} and {Yates, J.}},
	title = {Incorporating astrochemistry into molecular line modelling via emulation},
	DOI= "10.1051/0004-6361/201935973",
	url= "https://doi.org/10.1051/0004-6361/201935973",
	journal = {A\&A},
	year = 2019,
	volume = 630,
	pages = "A117",
 }

License
-------

The software is released, as is, under an MIT license.

Documentation
-------------

Documentation can be found at: https://emulchem.readthedocs.io.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

If you use this code please cite the associated emulator paper.
