Identify and characterize halo substructure (or other linear features) using Subspace--Constrained Mean Shift. See https://ui.adsabs.harvard.edu/#abs/2018arXiv181110613H/abstract for a complete description and examples.

Requires Python 2.x

## Helit

The Subspace--Constrained Mean Shift computation is done by the 'helit/ms' subpackage of https://github.com/thaines/helit. 

To use this package you will need to replace the ms_c.c file in the helit/ms directory with the one in this repository. This allows output of the principle curve eigendirections but restricts usage to 2d (for now).

If you use this package please also cite helit and the work it was developed for, 

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
