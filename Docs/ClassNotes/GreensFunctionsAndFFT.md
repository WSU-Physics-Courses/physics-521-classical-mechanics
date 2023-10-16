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

```{code-cell}
:tags: [hide-cell]

%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import mmf_setup;mmf_setup.nbinit()
```

```{contents} Contents
:local:
:depth: 3
```

Green's Functions and the FFT
=============================

## Damped Harmonic Oscillator: Cosine Drive

In {ref}`sec:a0` we solved the following differential equation for a damped harmonic
oscillator with a cosine driving force:

\begin{gather*}
  \ddot{x} + 2 \lambda \dot{x} + \omega_0^2 x = \frac{f}{m} \cos(\gamma t).
\end{gather*}

From the standard theory of ODEs, the complete solution can be expressed as the sum of a
particular solution with the two general solution of the homogeneous equation without
the driving term.  The text points out that the algebra is simplified if we work with a
complex driving force and take the real part with an exponential drive:

:::{margin}
This particular trick of working with complex variables would be more subtle if the
equation was not linear, e.g., if it had terms like $x^2$.
:::
\begin{gather*}
  \ddot{x} + 2 \lambda \dot{x} + \omega_0^2 x = \frac{f}{m} \cos(\gamma t)
  = \frac{f}{m} \Re e^{\I \gamma t},\\
  \ddot{x} + 2 \lambda \dot{x} + \omega_0^2 x = e^{\I \gamma t}.
\end{gather*}

The general solution can be found by guessing a form $x(t) = A e^{\I\omega t}$ which
is a solution if

\begin{gather*}
  -\omega^2 + 2\I\lambda \omega_ + \omega^2 = 0, \qquad
  \omega_{\pm} = \I\lambda \pm \sqrt{\omega_0^2 - \lambda^2},
\end{gather*}

hence, 

\begin{gather*}
  x(t) = A_{+}e^{\I\omega_+ t} + A_{-}e^{\I\omega_- t}.
\end{gather*}

:::{note}

What happens if in the case of [critical damping] where $\omega_0^2 =
\lambda^2$?  In this case, the two solutions $\omega_+ = \omega_-$ become degenerate and
we only have one of the two general solutions.  You may have learned that in this case
you need to consider also the following form:

\begin{gather*}
  x(t) = A\; t e^{\I \omega t},
\end{gather*}

which has an extra factor of $t$.  Where does this come from, and how would one guess
this form if one did not recall to try this?  To gain some insight consider the
following argument:

The general solution before this not must be valid for $\omega_0^2 \approx \lambda^2$
when we have $\omega_{\pm} \approx \omega \pm \epsilon$ where $\epsilon$ is small.
Thus, this new form should somehow emerge in the limit $\omega_0^2 \rightarrow
\lambda^2$, $\epsilon \rightarrow 0$ and the two solutions $e^{\I\omega_{\pm} t}$ become
degenerate.

We thus look at the form of the difference between these, which goes to zero *(the sum
remains finite and remains a solution in this limit, so we need not explore it)*:

\begin{gather*}
  A(e^{\I \omega_+ t} - e^{\I  \omega_- t}) \approx
  A\Bigl(e^{\I \epsilon t} - e^{-\I  \epsilon t}\Bigr)e^{\I \omega t} \rightarrow 0.
\end{gather*}

You should recognize that the first term looks like a centered-difference form for a derivative:

\begin{gather*}
  f'(x) = \lim_{\epsilon \rightarrow 0}  \frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}.
\end{gather*}

We don't have the $2\epsilon$ denominator, but remember that $A$ is an arbitrary
time-independent constant, so we can easily take $A \rightarrow A/2\epsilon$ to get
this.

More simply, I have simply developed the instinct to try expending things as a Talor
series when I see small parameters:

\begin{gather*}
  e^{x} = 1 + x + \frac{x^2}{2} + \cdots + \frac{x^{n}}{n!} + \cdots,\\
  e^{\pm\I\epsilon t} = 1 \pm \I \epsilon t + \cdots,\\
  A(e^{\I \omega_+ t} - e^{\I  \omega_- t}) \approx
  A 2\I \epsilon t e^{\I \omega t} = 
  \tilde{A}\; t e^{\I \omega t},
\end{gather*}

where we have redefined the consent $\tilde{A} = 2\I \epsilon A$.  In other words, the
factor of $t$ comes from the derivative of the original $e^{\I\omega t}$ form with
respect to $\omega_{\pm} \approx \omega \pm \epsilon$.
:::

To this, we must add a particular solution, but in the case of the exponential version
of a cosine driving force, this has the same form, so we can use the same trial solution
(Ansatz):

\begin{gather*}
  x(t) = B e^{\I\omega t}, \qquad
  (-\omega^2 + 2 \I \lambda \omega + \omega_0^2)Be^{\I\omega t} = \frac{f}{m}e^{\I\gamma t},\\
  \omega = \gamma, \qquad
  B = \frac{f/m}{\omega_0^2 - \gamma^2 + 2 \I \lambda \gamma} = b e^{\I\delta}.
\end{gather*}

## Damped Harmonic Oscillator: General Drive

How do we generalize the problem if the driving force has a general functional form
$f(t)$ which is not just a pure cosine?  For linear equations, a very powerful and
general solution can be expressed in terms of [Green's function]s.  First write the
differential equation in terms of the linear operator $\op{L}$:

\begin{gather*}
  \underbrace{\left(\diff[2]{}{t} + 2\I \lambda \diff{}{t} + \omega_0^2\right)}_{\op{L}} x(t) 
  = \op{L} x(t) = f(t).
\end{gather*}

:::{margin}
Note that I said **a** Green's function since there are many different Green's functions
related to each other by adding general solutions of the homogeneous equation.  One
sometimes sees reference to **the** Green's function **satisfying** some specific **boundary
conditions** such as $G(0) = 0$ and $\dot{G}(0) = 0$.
:::
The idea of a Green's function is to find a solution $G(t)$ to the equation

\begin{gather*}
  \op{L}G(t) = \delta(t).
\end{gather*}

:::{important}
A Green's function is a particular solution to the problem in the case of a
delta-function driving force.  This should be very familiar to you: you hear a very good
approximation to a Green's function when you sharply strike an object, like a wine
glass, or a bell.
:::

The particular solution is then:

\begin{gather*}
  x(t) = \int G(t-\tau)f(\tau)\d{\tau}.
\end{gather*}

:::{margin}
It is convenient to think of the first property as matrix multiplication where $t$ and $\tau$ are
indices:
\begin{gather*}
  \d{\tau}\delta(t-\tau) \equiv [\mat{1}]_{t,\tau}.
\end{gather*}
In this way, the delta-function is the identity matrix:
\begin{gather*}
  \vec{f} = \mat{1}\vec{f}\\
  f_t = \sum_{\tau} \mat{1}_{t,\tau}f_{\tau}.
\end{gather*}
:::
This follows from the following two key properties:

1. One can express any driving force $f(t)$ in terms of delta functions:

   \begin{gather*}
     f(t) = \int\d{\tau}\;\delta(t - \tau)f(\tau).
   \end{gather*}

2. The equation is **linear**, so that

   \begin{gather*}
     \op{L}\Bigl( \alpha A(t) + \beta B(t)\Bigr) = 
     \alpha \op{L}A(t) + \beta \op{L}B(t).
   \end{gather*}

Combining these properties

\begin{align*}
  f(t) &= \int\d{\tau}\;\delta(t-\tau)f(\tau)\\
       &= \int\d{\tau}\;\op{L}G(t-\tau)f(\tau)\\
       &= \op{L}\underbrace{\int\d{\tau}\;G(t-\tau)f(\tau)}_{x(t)},\\
  x(t) &= \int\d{\tau}\;G(t-\tau)f(\tau).
\end{align*}

We may seem no further ahead because we still need to find the particular solution
$G(t)$ for a delta-function drive, but if we can find it, it will be worth the work,
since then we can solve *any* problem.

:::{margin}
This discussion can be completely expressed in terms of linear algebra, treating
functions as vectors in an infinite-dimensional vector space called [Hilbert space].  All we are
doing here is changing bases from the time basis -- whose basis vectors are
$\delta(t-\tau)$ -- to the frequency basis -- whose basis vectors are plane waves $e^{\I
\gamma t}$.  The same manipulations are extremely common in quantum mechanics, and I
recommend perusing a textbook like {cite:p}`Shankar:1994` for a discussion.
:::
## Fourier Analysis
With a little thought, we can reverse this process: above we have found particular
solutions $F_{\gamma}(t)$ to the equation

\begin{gather*}
  \op{L}F_{\gamma}(t) = e^{\I\gamma t}.
\end{gather*}

Thus, if we can express $\delta(t)$ as a linear combination of $e^{\I\gamma t}$, then we
can use the same approach.  This is the essence of **Fourier analysis**, and the
solution is well known:

\begin{gather*}
  \delta(t) = \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; e^{\I \gamma t}.
\end{gather*}

Hence, a Green's function is:

\begin{gather*}
  G(t) = \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; F_\gamma(t).
\end{gather*}

:::{admonition} Do it!  Derive this solution.
:class: dropdown

\begin{align*}
  \delta(t) &= \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; e^{\I \gamma t},\\
            &= \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; \op{L} F_\gamma(t),\\
            &= \op{L} \underbrace{\int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; F_\gamma(t)}_{G(t)},\\
G(t) &= \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; F_\gamma(t).
\end{align*}
:::

In our particular case, the solutions were expressed in terms of the [response
function] $\tilde{\chi}(\gamma)$:

\begin{gather*}
  B_\gamma(\omega) = \frac{f/m}{\omega_0^2 - \gamma^2 + 2 \I \lambda \gamma}
                   = \frac{f}{m}\tilde{\chi}(\gamma),\\
  F_{\gamma}(t) = \tilde{\chi}(\gamma)e^{\I\gamma t}.
\end{gather*}

Hence, the Green's function is just the Fourier transform of the [response function]

\begin{gather*}
  G(t) = \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; F_\gamma(t)  
  G(t) = \int_{-\infty}^{\infty} \frac{\d{\gamma}}{2\pi}\; \tilde{\chi}(\gamma)
         e^{\I\gamma t}
       = \chi(t).
\end{gather*}

:::{admonition} Do it!  Find $\chi(t)$.
:class: dropdown

I encourage you to find the explicit form of $\chi(t)$ for the damped harmonic
oscillator and explore it numerically.  See [step response] for a solution if you need help.
:::

:::{note}

We won't work out the details here, but if you like, you can further require that the
Green's function $G(t)$ satisfy specific boundary conditions.  For example, in this
case, having $G(0) = \dot{G}(0) = 0$ would ensure that any constructed particular
solution would not affect the initial conditions at time $t=0$.  Thus, one can solve the
initial value problem for the homogeneous problem, including the initial conditions
$x(0) = x_0$ and $\dot{x}(0) = v_0$, then immediately have the solution for any driving
force by adding the particular solution obtained by convolving these specific Green's
functions with any external driving force.

Without such care, one might need to re-solve the initial value problem each time.
:::




















[critical damping]: <https://en.wikipedia.org/wiki/Damping#Damping_ratio_definition>
[Green's function]: <https://en.wikipedia.org/wiki/Green%27s_function>
[Fourier analysis]: <https://en.wikipedia.org/wiki/Fourier_analysis>
[Hilbert space]: <https://en.wikipedia.org/wiki/Hilbert_space>
[response function]: https://en.wikipedia.org/wiki/Linear_response_function
