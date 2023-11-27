---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell} ipython3
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

(sec:PerturbationTheory)=
# Perturbation Theory

Here we work through chapter 8 of {cite}`Percival:1982` which provides a nice
introduction to the application of perturbation theory in dynamical systems.

## Example 8.1 from {cite}`Percival:1982`

\begin{gather*}
  \dot{x} = x + \epsilon x^2, \qquad x(0) = A.
\end{gather*}

This example demonstrates how a perturbation can qualitatively change the nature of the
solution. While the unperturbed solution is valid for all times, the perturbed solution
diverges at a finite time $t_c = \ln(1 + 1/\epsilon A)$ due to the appearance of a pole. 

:::::{admonition} Do It! Solve this, both exactly, and as a series in $\epsilon$.
:class: dropdown

:::{margin}
\begin{gather*}
  \int_{A}^{x} \frac{\d{x}}{x + \epsilon x^2} = t\\
  \left.-\ln\left(1 + \frac{1}{\epsilon x}\right)\right|_{A}^{x} = t\\
  x(t) = \frac{A e^{t}}{1 - \epsilon A (e^{t} - 1)}.
\end{gather*}
:::
This is a separable system with solution
\begin{gather*}
  x(t) = \frac{A e^{t}}{1 - \epsilon A (e^{t} - 1)}.
\end{gather*}
This admits a nice series expansion
\begin{gather*}
  x(t) = A e^{t}\Bigl(1 + \epsilon A (e^{t} - 1) 
     + \epsilon^2 A^2 (e^{t} - 1)^2 + O(\epsilon^2)\Bigr),
\end{gather*}
but will obviously fail to converge beyond the pole.

The naïve approach (which we call "naïve perturbation theory") expresses the solution as
\begin{gather*}
  x(t) = x_0(t) + \epsilon x_1(t) + \epsilon^2 x_2(t) + \cdots 
  = \sum_{n=0}^{\infty}\epsilon^{n} x_n(t),\qquad
  x^2 = \sum_{m,n=0}^{\infty}\epsilon^{m+n} x_m x_n,
\end{gather*}
Inserting this into the original equation and collecting terms of different orders in
$\epsilon$, we obtain the following equations, and corresponding solutions
\begin{align*}
  \dot{x}_0 &= x_0, & 
  x_0(t) &= Ae^{t},\\
  \dot{x}_1 &= x_1 + x_0^2 = x_1 + A^2e^{2t}, & 
  x_1(t) &= A^2e^{t}(e^{t} - 1),\\
  \dot{x}_2 &= x_2 + 2x_0x_1 = x_2 + 2A^3e^{2t}(e^{t} - 1), &
  x_2(t) &= A^3e^t(e^t-1)^2,\\
  \dot{x}_n &= x_n + \sum_{m=0}^{n-1}x_{m} x_{n-m-1}, &
  x_n(t) &= A^{n+1}e^{t}(e^{t}-1)^{n}.
\end{align*}
:::::

## Asymptotic Series

Here is another type of problem that can occur with perturbative solutions.  Consider
the following system:
\begin{gather*}
  \epsilon \dot{x} = -x + e^{-t}, \qquad x(0) = 1, \qquad
  x(t) = \frac{e^{-t} - \epsilon e^{-t/\epsilon}}{1-\epsilon}.
\end{gather*}

:::{admonition} Do It! Solve by Lagrange's [variation of parameters][].
:class: dropdown

Consider $\epsilon \dot{x} + x = f(t)$. The general solution to the homogeneous equation
is simply $x_h(t) = A e^{-t/\epsilon}$. Letting $A(t)$ be a function of time, and
plugging this back in, we have
\begin{gather*}
  x(t) = A(t) e^{-t/\epsilon}, \qquad
  \epsilon \dot{x}(t) + x(t) = \epsilon\dot{A}(t)e^{-t/\epsilon},\\
  \epsilon\dot{A}(t)e^{-t/\epsilon} = f(t),\qquad
  A(t) = \frac{1}{\epsilon}\int^{t}f(t)e^{t/\epsilon}\d{t}.
\end{gather*}
For our problem, $f(t) = -e^{-t}$ so we have
\begin{gather*}
  A(t) = -\frac{1}{\epsilon}\int^{t}e^{(-1+\epsilon^{-1})t}\d{t}
       = \frac{e^{(-1+\epsilon^{-1})t}}{1-\epsilon} + C.
\end{gather*}
:::

:::{admonition} Do It! Solve using naïve perturbation theory.
:class: dropdown

Letting $x = x_0 + \epsilon x_1 + \epsilon ^2 x_2 + \cdots$, we have the following:
\begin{align*}
  0 &= - x_0 + e^{-t}, &
  x_0(t) &= e^{-t},\\
  \dot{x}_0 &= -x_1, &
  x_1(t) &= e^{-t},\\
  \dot{x}_n &= -x_{n+1}, &
  x_n(t) &= e^{-t}.
\end{align*}
This series can be summed explicitly:
\begin{gather*}
  x(t) = \sum_{n=0}^{\infty} \epsilon^n x_n 
  = \left(\sum_{n=0}^{\infty} \epsilon^n\right)e^{-t}
  = \frac{e^{-t}}{1-\epsilon}.
\end{gather*}
Note however, that this is **not** the correct solution.  Even the initial condition is
violated.  What is missing is the homogeneous solution which must be added to restore
the initial condition.
:::

Here we are faced with an example where the series fails to converge to the correct
solution, even if we expand it to all orders.  This is due to the non-analytic pieces
$e^{-t/\epsilon}$ whose Taylor coefficients are all zero.  This is an example of an
**aymptotic series**.

```{code-cell}
:tags: [hide-input]

t = np.linspace(0, 2)

def x(t, epsilon):
    return (np.exp(-t) - epsilon * np.exp(-t/epsilon))/(1-epsilon)

def x_n(t, epsilon, N):
    return np.exp(-t)*sum([epsilon**n for n in range(N+1)])

fig, ax = plt.subplots()

ax.plot(t, x(t, epsilon=0.1))
for N in range(2):
    ax.plot(t, x_n(t, epsilon=0.1, N=N), label=f"${N=}$")
```



[variation of parameters]: <https://en.wikipedia.org/wiki/Variation_of_parameters>
