---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
---

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:linear-response)=
# Linear Response

Here we consider the linear response of a more complicated system described by the
Gross-Pitaevskii equation (GPE):
\begin{gather*}
  \I\hbar\dot{\psi}(\vect{x}, t) = \left(
    -\frac{\hbar^2\vect{\nabla}}{2m} + g\abs{\psi(\vect{x}, t)}^2 + V(\vect{x},
    t)\right)\psi(\vect{x}, t).
\end{gather*}
Although this is intended to represent a quantum many-body system (a Bose-Einstein
Condensate or BEC), $\psi(\vect{x}, t)$ is formally a complex-valued classical field.

In this document, we shall restrict ourselves to 1D.  The goal is to perform the
equivalent of a normal-modes analysis to see how the system will respond to small
perturbations.  Specifically, we will drive it with a small periodic potential
\begin{gather*}
  V(x, t) = \epsilon \sin(\omega t)f(x)
\end{gather*}
and see how it responds.

As usual, we must start with a stationary state $\psi_0(x, t) = e^{\mu
t/\I\hbar}\psi_0(x)$ such that
\begin{gather*}
  -\frac{\hbar^2\psi_0''(x)}{2m} + g\abs{\psi_0(x)}^2\psi_0(x) = \mu \psi_0(x).
\end{gather*}
We then look for states $\psi_{\pm n}(x)$ such that the GPE is satisfied to order
$\epsilon^2$ for
:::{margin}
The nonlinearity $g\abs{\psi}^2 = g\psi^*\psi$ in responsible for the need to include both terms
$e^{\pm\I\omega t}$.  If we insert one, the conjugation $\psi^*$ will introduce the other.
:::
\begin{gather*}
  \psi(x, t) = e^{\mu t/\I\hbar}\left(
    \psi_0(x) 
    + \epsilon e^{\I\omega t}\psi_{+}(x)
    + \epsilon e^{-\I\omega t}\psi_{-}(x)
  \right).
\end{gather*}

:::{margin}
Here we use
\begin{multline*}
\abs{\psi}^2 - \abs{\psi_0}^2\\
  =\epsilon(\psi_0^*\psi_+ + \psi_-^*\psi_0)e^{\I\omega t}\\
    + \epsilon(\psi_0^*\psi_- + \psi_+^*\psi_0)e^{-\I\omega t}\\
    + O(\epsilon^2)
\end{multline*}
:::
Inserting, expanding, collecting the coefficients of $e^{\I(\mu \pm\omega) t}$
respectively, and conjugating the second equation, we obtain the following equations to
linear order in $\epsilon$:
\begin{align*}
  (\mu - \omega)\psi_+ &= \frac{-\hbar^2}{2m}\psi_+'' 
  + g(\psi_0^*\psi_+ + \psi_-^*\psi_0)\psi_0 + (g\abs{\psi_0}^2 + V)\psi_+,\\
  (\mu + \omega)\psi_-^* &= \frac{-\hbar^2}{2m}\psi_-''^* 
  + g(\psi_0\psi_-^* + \psi_0^*\psi_+)\psi_0^* + (g\abs{\psi_0}^2 + V)\psi_-^*,\\
  \begin{pmatrix}
    \omega\\
    & -\omega
  \end{pmatrix}
  \begin{pmatrix}
    \psi_+\\
    \psi_-^*\\
  \end{pmatrix}
  &=
  \begin{pmatrix}
    \frac{-\hbar^2}{2m}\nabla^2 + 2g\abs{\psi_0}^2 - \mu + V & g\psi_0^2\\
    g(\psi_0^*)^2 & \frac{-\hbar^2}{2m}\nabla^2 + 2g\abs{\psi_0}^2 - \mu + V
  \end{pmatrix}
  \underbrace{
  \begin{pmatrix}
    \psi_+\\
    \psi_-^*\\
  \end{pmatrix}}_{\ket{\Psi}}.
\end{align*}











