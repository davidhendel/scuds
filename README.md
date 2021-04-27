# SCUDS: A machine-vision method for automatic classification of stellar halo substructure

Identify and characterize halo substructure (or other linear features) using Subspace--Constrained Mean Shift. See [Hendel et al. 2019](https://ui.adsabs.harvard.edu/#abs/2018arXiv181110613H/abstract) for a complete description of the method and an example of its use.

## Requirements

- Python 3
- helit
- numpy
- scipy
- astropy
- transforms3d

### Helit

The Subspace--Constrained Mean Shift computation is done by the 'helit/ms' subpackage of https://github.com/thaines/helit. 

To use this package you will need to replace the ms_c.c file in the helit/ms directory with the one in this repository. This allows output of the principle curve eigendirections but restricts usage to 2d (for now). Ensure that you build helit/ms/setup.py with Python 3 as the C api has changed.

If you use this package please also cite helit and the work it was developed for:

```
  @MISC{helit,
    author = {T. S. F. Haines},
    title = {\url{https://github.com/thaines/helit}},
    year = {2010--},
  }

  @INPROCEEDINGS{HainesTOG2016,
    author       = {Haines, Tom S.F. and Mac Aodha, Oisin and Brostow, Gabriel J.},
    title        = {{My Text in Your Handwriting}},
    booktitle    = {Transactions on Graphics},
    year         = {2016},
  }
```

## Usage

SCUDS contains scripts that can be used with any 2d point distribution but was originally designed to analyse the simulations of [Hendel & Johnston 2015](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454.2472H/abstract) and therefore contains several routines specific to that work. An example using simple XY particle data can be found in mspy_example.ipynb.
