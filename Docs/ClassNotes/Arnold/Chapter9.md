---
jupytext:
  formats: ipynb,md:myst,py:light
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (phys-521-2022)
  language: python
  name: phys-521-2022
---

Arnold 9: Canonical Formalism
=============================

```{margin}
We will put comments with page numbers etc. in the margin here.  These page numbers
refer to the second print edition of {cite:p}`Arnold:1989`.
```

```{contents} Contents
:local:
:depth: 3
```

```{code-cell}
:tags: [hide-cell]

import mmf_setup

mmf_setup.nbinit()
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%pylab inline --no-import-all
import manim.utils.ipython_magic
!manim --version
```

This document contains some random notes and discussions about Chapter 9 of
{cite:p}`Arnold:1989`.

## §44 The integral invariant of Pointcaré-Cartan

```{margin}
**P233:** The [2-cycle](https://en.wikipedia.org/wiki/Homology_(mathematics)#Origins)
$\sigma$ is the 2D surface with boundary $\partial \sigma = \gamma_1 - \gamma_2$.
Imagine the vortex tube as a hose: if you cut the hose along the curves $\gamma_{1,2}$,
then $\sigma$ would be the piece of hose between.  The orientation of the boundary is
important, which is why there is a minus sign.
```
In 3D, every vector field has a curl $\vect{r} = \vect{nabla}\times \vect{v}$, and one
can find the integral curves of these, which Arnold calls *vortex lines*.  Since
$\vect{\nabla}\cdot \vect{r} = 0$, these vortex lines have no "source", hence flux is
conserved.  They need not form closed loops.



Here we plot some

```{code-cell}
def v(x, y, z):
    """Return a 3D vector field."""
    return (y, z, x)
    
    

```
