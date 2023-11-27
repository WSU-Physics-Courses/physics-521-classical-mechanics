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

import mmf_setup;mmf_setup.nbinit()
from pathlib import Path
import os
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

# Chapter 6

## Coins

Suppose a player starts with initial capital $K$.  With each play, with probability $p$,
the player wins, increasing his capital to $K+1$.  If the player looses (with
probability $q=1-p$), then his capital shrinks to $K-1$.  Play continues until either
the player loses all his capital ($K=0$), called "ultimate ruin", or the player wins all
of the bank's capital ($K=B$).

Schroeder introduces the "probability of ultimate ruin" $q_{K}$ which is the probability
that the player will eventually lose all of his money.  Schroeder describes this by a
difference equation
\begin{gather*}
  q_{K} = pq_{K+1} + qq_{K-1}, \qquad
  0 < K < B,\\
  q_0 = 1, \qquad 
  q_B = 0,
\end{gather*}
where $B$ is the total capital in the game (i.e. the capital of the bank).  The initial
conditions are:
* $q_0 = 1$: the player has already lost since he starts with no capital $K=0$ --
  certain ruin -- and
* $q_B = 0$: the player has all the money and has won -- no chance of ruin.

:::::{admonition} Do It! Solve for $q_K$.
:class: dropdown

\begin{gather*}
  q_K = \frac{(q/p)^{B} -  (q/p)^{K}}{(q/p)^{B} - 1}.
\end{gather*}


:::::

## Claude Shannon's Outguessing Machine

https://github.com/AnandChowdhary/claude
