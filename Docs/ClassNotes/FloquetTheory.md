---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell}
:tags: [hide-cell]

import os
import mmf_setup;mmf_setup.nbinit()
from pathlib import Path
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
```

(sec:FloquetTheory)=
## Floquet Theory

This analysis falls under the broader term of [Floquet theory][] which derives from the
following theorem due to Floquet concerning the solutions to a linear first-order
differential equation of the form
\begin{gather*}
  \dot{\vect{x}} = \mat{A}(t)\vect{x}, \qquad \mat{A}(t+T) = \mat{A}(t),
\end{gather*}
where the time-dependence is periodic with period $T$.  If $\mat{\phi}(t)$ is a
**fundamental matrix solution**  -- i.e. all columns are linearly independent solutions
-- then
\begin{gather*}
  \mat{\phi}(t + T) = \mat{\phi}(t)\underbrace{\mat{\phi}^{-1}(0)\mat{\phi}(T)}_{e^{T\mat{B}}}.
\end{gather*}
The matrix $\mat{\phi}^{-1}(0)\mat{\phi}(T) = e^{T\mat{B}}$ is called the **monodromy
matrix**, and for any $\mat{B}$ that satisfies this (there may be several), the
solutions can be expressed in terms of a periodic function matrix-valued function $\mat{P}(t)$:
:::{margin}
Here $\mat{B}$ is generally complex, but there is also a real matrix $\mat{R}$
\begin{gather*}
    \mat{\phi}(t) = \mat{Q}(t)e^{t\mat{R}},\\
    \mat{Q}(t+2T) = \mat{Q},
\end{gather*}
where $\mat{Q}$ now has period $2T$.
:::
\begin{gather*}
  \mat{\phi}(t) = \mat{P}(t)e^{t\mat{B}}, \qquad \mat{P}(t+T) = \mat{P}(t).
\end{gather*}
*Note: this is a generalization of [Bloch's theorem][] which states that eigenfunctions for a
periodic potential expressed as periodic solutions times a phase factor $e^{\I
\vect{k}\cdot\vect{x}}$.* 


For the previous system, we note that, about the unstable equilibrium point, the system
can be expressed as
\begin{gather*}
  \underbrace{\diff{}{t}
    \begin{pmatrix}
      \theta\\
      \dot{\theta}
    \end{pmatrix}
  }_{\dot{\vect{x}}}
  =
  \underbrace{
    \begin{pmatrix}
    0 & 1\\
    \omega^2(t)
    \end{pmatrix}
  }_{\mat{A}(t)}
  \underbrace{
    \begin{pmatrix}
      \theta\\
      \dot{\theta}
    \end{pmatrix}
  }_{\vect{x}}
\end{gather*}

{ref}`sec:FloquetTheory`.

[Floquet theory]: <https://en.wikipedia.org/wiki/Floquet_theory>
[Bloch's theorem]: <https://en.wikipedia.org/wiki/Bloch's_theorem>
