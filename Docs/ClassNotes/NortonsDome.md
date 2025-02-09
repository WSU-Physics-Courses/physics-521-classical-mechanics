---
jupytext:
  formats: ipynb,md:myst,py:light
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (python)
  language: python
  name: python3
---

```{code-cell}
:tags: [hide-cell]

import mmf_setup; mmf_setup.nbinit()
import os
from pathlib import Path
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
```

(sec:NortonsDome)=
Norton's Dome
=============

[John Norton][] proposed considering Newtonian dynamics on a dome with height
\begin{gather*}
  h(r) = -\frac{2\alpha}{3g}r^{3/2}, \qquad 
  V(r) = mgh(r) = -\frac{2\alpha m}{3}r^{3/2}, \qquad
  \ddot{r} = \frac{-V'(r)}{m} = \alpha r^{1/2}.
\end{gather*}
Choosing units so that $m = \alpha = 1$, we thus have
\begin{gather*}
  \ddot{r} = \diff{\dot{r}}{r}\dot{r} = \sqrt{r}, \\
  p = m\dot{r} = \sqrt{2m[E - V(r)]} 
               = \sqrt{2[E + \frac{2}{3}r^{3/2}]}.
\end{gather*}
Now consider solutions that start at rest, infinitesimally off of the apex, so $E = 0$.
We can then solve
\begin{gather*}
  \dot{r} = \frac{2}{\sqrt{3}}r^{3/4},\qquad
  \int_{0}^{r}r^{-3/4}\d{r} = \frac{2}{\sqrt{3}}\int_{t}^{0}\d{t}\qquad
  r(t) = \left(\frac{t}{2\sqrt{3}}\right)^{4} = \frac{t^4}{144}.
\end{gather*}
The key fact is that the particle takes a finite amount of time to reach a particular
radius.  Said another way, a particle can be shot towards the apex from a distance $r$
with a particular velocity such that it reaches $r=0$ in a finite amount of time
$t=\sqrt[4]{144r}$, at which point is does what?  This is John's point: does it stay
there indefinitely?  Or roll of at some indeterminant period of time $T$?

Contrast this with an inverted harmonic oscillator:
\begin{gather*}
  V(r) = -\frac{r^2}{2},\qquad
  \dot{r} = r,\qquad
  \int_{0}^{r}\frac{\d{r}}{r} =\ln(r) - \ln(0) = t.
\end{gather*}
In this case, it takes a particle shot with appropriate velocity infinite time to reach
the top - it approaches, but never reaches the top.

[John Norton]: <https://sites.pitt.edu/~jdnorton/Goodies/Dome/index.html>

