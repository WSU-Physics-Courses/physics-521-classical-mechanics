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
FIG_DIR = Path(mmf_setup.ROOT) / 'Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
try: from myst_nb import glue
except: glue = None
```

# Thermodynamics and Statistical Mechanics

## Introduction

For starters, I would like to point out the book {cite}`Sewell:2002`. This
is a very nice modern account of the formal aspects of thermodynamics and statistical
mechanics.  It is mathematical, but uses as little math as needed to make accurate
claims.

## Thermodynamics

### States

:::{margin}
There are subtleties when working with thermodynamics in that one must actually consider
systems with infinitely many degrees of freedom to have phases etc.  These systems admit
inequivalent representations of the operator algebra.  These subtleties are discussed in
{cite}`Sewell:2002` for example.
:::
Consider a quantum system with Hamiltonian $\op{H}$.  We can define
"states" of the system $\rho$ as functions on observables $\op{A}$
such that the expected outcome is $\rho(\op{A})$.  In the usual
quantum mechanical picture for finite systems, one can work with the
density matrix $\op{\rho}$ so that the expectation values
is
\begin{gather*}
  \braket{\op{A}}_{\rho} = \Tr[\op{A}\op{\rho}].
\end{gather*}

### Entropy
Entropy is a bit of an elusive concept.  One concrete way of
understanding entropy comes from information theory.  Suppose that
some experiment has a set of possible outcomes $\{e_1,e_2,\cdots\}$.
If we repeat the experiment $N$ times, then an outcome can be labelled
by the numbers $(n_1,n_2,\cdots)$ where $n_i$ describes how many times
outcome $e_i$ was obtained.

We assume that the states of the system can be described by a
probability measure on the system $p \equiv (p_1,p_2,\cdots)$ where,
\begin{gather*}
  \lim_{N \rightarrow \infty} \frac{n_i}{N}  =  p_i.
\end{gather*}
Hence $\sum_i p_i = 1$.  If this holds, the set of all states
$\mathcal{P}$ is convex and the extremal points are pure states
$\pi_i$ with exactly one non-zero $p_i = 1$.

The entropy of a state $p$ is a measure of the "impurity" of that
state:
\begin{gather*}
  S(p) = -\sum_{i} p_i \ln p_i.
\end{gather*}
Another useful concept is that of relative entropy:
\begin{gather*}
  S_{\text{rel}}(q|p) = \sum_{i}\left(p_i \ln p_i - p_i \ln q_i\right).
\end{gather*}
We shall see that this measures in some sense the "inaccessibility"
of the state $p$ from the state $q$.

To show this, consider an experiment repeated $N$ times on the state
$p$.  A given outcome is specified by the $n_i$.  The number of ways
that this outcome could be obtained is given by 
\begin{gather*}
  \label{eq:Pdef}
  P = \frac{N!}{n_1!n_2!\cdots}
\end{gather*}
:::{margin}
Note that experiments can be labelled, hence each outcome is distinguishable.
:::
where we simply count the permutations.

If the system were in a pure state $\pi_i$ then $n_i=N$ and all others
are zero so that $P=1$.  It is clear that if the system is not in a
pure state, $P>1$ and so $P$ provides a measure of how "impure" the
state is.  Now, consider the limit as $N\rightarrow \infty$.  Using
Sterling's approximation $N! \approx N (\ln N - 1)$,
\begin{gather*}
  \lim_{N \rightarrow \infty} \frac{\ln P}{N} 
  = S(p).
\end{gather*}

The interpretation of the relative entropy follows by defining the
probability that a given expreiment $n$ is obtained from a state $q$:
\begin{gather*}
  P_{q} = \frac{N!}{n_1!n_2!\cdots} q_1^{n_1}q_2^{n_2}\cdots.
\end{gather*}
Again, taking the limit of large $N$
\begin{gather*}
   -\lim_{N \rightarrow \infty} \frac{\ln P_q}{N} 
  = S_{\text{rel}}(p|q).
\end{gather*}
The quantum mechanical versions for finite systems are
\begin{subequations}
  \begin{align}
    S(\op{\rho}) &= -\Tr[\op{\rho}\ln \op{\rho}],\\
    S_{\text{rel}}(\op{\rho} | \op{\sigma}) &= 
    \Tr[\op{\rho}\ln \op{\rho}-\op{\rho}\ln \op{\sigma}].
  \end{align}
\end{subequations}

### Ensembles
Statistical mechanics is founded on the idea that one can describe a
complicated system with a small number of macroscopic variables which
represent measurable quantities.  The idea is that there are many
microscopic configurations (states) that will yield the same
macroscopic measurements and hence will be indistinguishable to the
observer.  We now make an assumption:\footnote{This assumption is
  justified by ergodic properties of the time-evolution of the system.
  In particular, it fails if there are conserved quantities.  See
  Section~\ref{sec:conserved-quantities}.}
\begin{assume}
  \label{assume:ensembles}
  We assume that all states with a given fixed set of
  measurable properties are equally likely to occur, and thus should
  be equally weighted in the ensemble.
\end{assume}
Using this assumption, one can perform well-defined statistical
averaging of quantities over the ensemble.
\subsubsection{Micro-canonical}
In the context where the energy is fixed, this known as the
microcanonical ensemble.  One can use the previous arguments where $N$
is the number of particles in the system to argue that the statistical
averages are dominated by the state with maximal entropy (we shall
perform this minimization later when we can justify the $N\rightarrow
\infty$ limit):
\begin{gather*}
  S(E) = \max_{\op{\rho} | \rho(\op{H}) = E} S(\op{\rho}).
\end{gather*}
If this is true, then one can simply work with the state of maximal
entropy.  However, this approach is typically only useful for gasses.
In other cases, one must include more states to approach the
appropriate distribution and this makes this approach difficult.
\subsubsection{Macro-canonical}
There is another reason to disfavour the microcanonical ensemble: It
is very difficult in practice to make a perfectly isolated system.
Instead it is easier to consider systems in contact with a thermal
bath. Emperically, we know that the composition of a thermal bath
makes little difference as long as it is large and has a
quasi-continuous spectrum.  This led Gibbs to suggest that one
consider a heat bath as $N-1$ copies of the system under
consideration.

One advantage of this approach is that the limit $N \rightarrow
\infty$ represents the limit of an ideal heat bath which is quite
realistic and attainable in practice.  In this setup, we imagine
distributing a fixed amount of energy $E$ over the ensemble of $N$
systems.  Again, we maximize the entropy of the collection of systems
subject to the constraints of fixed total ensemble energy $E$. We
should also only consider propertly normalized states $\Tr \op{\rho} =
1$.  To do this, we introduce a new function
\begin{gather*}
  f = S - \lambda \Tr \op{\rho} - \beta E
  = -\Tr\left(
    \op{\rho} \ln \op{\rho} +\lambda \op{\rho} +\beta \op{\rho} \op{H}
  \right).
\end{gather*}
which we maximize with respect to an unconstrained $\op{\rho}$.
Assuming that $S$ and $E$ are differentiable functions, this gives us
the constraint that
\begin{gather*}
  \op{\rho} = e^{-\lambda-1}e^{-\beta\op{H}}.
\end{gather*}
The normalization condition simply defines the partition function:
\begin{gather*}
  e^{1+\lambda} = Z = \Tr[e^{-\beta\op{H}}].
\end{gather*}
Using this to eliminate $\lambda$, we have the following state with
maximal entropy
\begin{gather*}
  \op{\rho}_{\beta} = \frac{e^{-\beta\op{H}}}{\Tr[e^{-\beta\op{H}}]}.
\end{gather*}
We can identify the parameter\footnote{We use units such that $k=1$
  here.} $\beta = 1/T$ in terms of the absolute termperature.

Note that we now have a well-defined state of maximal entropy at fixed
temperature.  We are justified in using this state for performing
statistical averages when considering systems coupled to an ideal heat
bath at fixed termperature.  This is justified by taking the
thermodynamica limit $N \rightarrow \infty$ where $N$ describes
properties of the heat bath rather than the system.  Unlike the
micro-canonical ensemble, this allows us to study any type of system,
as long as it is in equilibrium with a very large heat bath at
constant temperature.

Instead of maximizing the function $f$, one usually scales this and
instead minimizes the function $F = -f/\beta$ which is known as the
Helmholtz free energy.  Thus, we arive at the statistical formulation
of equilibrium thermodynamics:

> One can represent the equilibrium properties of a system coupled to
> an ideal thermal bath at termperature $T$ by the normalized state
> $\op{\rho}_\beta$ which minimizes the Helmholtz free energy $F$:
> \begin{gather*}\label{eq:Fmin}
   F(T) = \min_{\Tr\op{\rho}=1}
   \Tr[\op{\rho}\op{H} + T \op{\rho}\ln \op{\rho}].
  \end{gather*}
  
In terms of the partition function, we have\footnote{Note that this
  relationship is easy to remember in this form:
  \begin{gather*}
    \label{eq:F}
    e^{-\beta F} = \Tr[e^{-\beta \op{H}}].
  \end{gather*}}
\begin{gather*}
  F = -T \ln (Z).
\end{gather*}
Finally, one can view $\op{\rho}_U(\beta)$ as a function of $\beta$,
in which case, it satisfies the differential equation
\begin{gather*}
  \pdiff{\op{\rho}_U}{\beta} = -\op{H}\op{\rho}_U
\end{gather*}
with the initial condition
\begin{gather*}
  \op{\rho}_U(0)=\op{0}.
\end{gather*}

### Thermodyanmic Variables
***********This section needs work.**********

The first law of thermodynamics is that energy is conserved.  This can
be formulated by saying that there is a differential form for the heat
required by a system to change:
\begin{gather*}
  \d{Q} = \d{E} + P\d{V} + \vect{\Theta} \cdot \d{\vect{Q}}.
\end{gather*}
The second law says that for adiabatic changes of state, this is a
form $T\d{S}$ where $T=\beta^{-1}$ is the temperature and $S$ is
extensive (the thermodynamic entropy).  Combining these, we have
\begin{gather*}
  T\d{S} = \d{E} + P\d{V} + \vect{\Theta} \cdot \d{\vect{Q}}.
\end{gather*}
Putting $Q_0=E$, $\Theta_0=1$, $\theta_k = \beta \Theta_k$ and $p =
\beta P$ we have
\begin{gather*}
  \label{eq:dS1}
  \d{S} = p\d{V} +\vect{\theta}\cdot\vect{Q}.
\end{gather*}
\subsubsection{Conserved Quantities}
\label{sec:conserved-quantities}
The reason that the energy $E$ has been singled out in the previous
discussion is because it is a conserved quantity of the system (as
long as the Hamiltonian is time-independent).  If there are conserved
quantities such as particle number, volume etc. then the
assumption~\ref{assume:ensembles} that all microstates states are
equally likely is false. It is only possibly to justify this
assumption when one only considers states that have held fixed these
quantities.

Suppose one wants to study a box which contains $N$ particles and has
volume $V$.  The Hamiltonian for this system will conserve both $N$
and $V$, and so we must only consider states which have a definite
volume and particle number.  The formalism is the same as before, but
now one considers the Helmholtz free energy as a function of these
parameters as well: $F(T,V,N,\cdots)$ and one minimizes over
configurations where these quantities are well-defined.  This is how
standard thermodynamics proceeds.

Generically, these quantities are properties of the system as
specified in the Hamiltonian.  We shall refer to them as $Q_i$ where
$Q_0$ is the energy of the system.  To each of these we can defined
the thermodynamical conjugate $\theta_i$.  These appear as the
Lagrange multipliers introduced to enforce the appropriate constraint
while maximizing the entropy.  Thus, we have seen that $\theta_0 =
\beta$.  In this formalism, one finds that $p = \beta P$---the
reduce pressure---is conjugate to the volume $V$ etc.

Introducing all appropriate multipliers, we determine the
thermodynamic state by maximizing:
\begin{gather*}
  \label{eq:NonExtensivePotential}
  \max_{\Tr[\op{\rho}]=1} S - p V - \vect{\theta}\cdot \vect{Q}.
\end{gather*}
This procedure, however, fails for extensive systems as we shall see
in the next section.

### Extensivity
Consider two copies of a system, both in the same
thermodynamic state, but far from each other so that they only
exchange heat (as in Gibb's ensemble).  All additive quantities for
this combined system such as the entropy, energy, volume, particle
numbers etc. have twice the values of a single system.  Now consider
bringing the two systems into a single container of twice the volume.
If, for the resulting system, the additive quantities are all still
twice the value of a single system, then the system is said to be
*extensive*.

:::{margin}
See also {cite}`TE:2004,Tsallis:1988eu,Curado:1991jc,TBCP:2003` for a generalization of
the notion of entropy that is useful for non-extensive systems.
:::
Extensivity fails, for example, if there are long-range forces: In
this case, the energy of the combined system would be less/greater
than twice the original system depending on if the forces were
repulsive/attractive.  One usually makes the assumption of extensivity
to ensure that the assumption~\ref{assume:ensembles} holds. It does
not hold in general for non-extensive systems.


Extensivity also fails if there are substantial finite-size effects:
In this case, one alters the properties of the system by removing the
barier between them.  Often, however, extensivity can be returned by
considering very large systems.

:::{margin}
For two non-interacting systems, doubled system is
represented by $\op{\rho}\otimes \op{\rho}$ (this neglects possible
symmetrization conditions, but for spatially well-separated systems
the overlap is negligable) while the measurable operator is a linear
combination $\op{Q}_i \otimes \op{1} + \op{1} \otimes \op{Q}_i$.
The additivity follows trivially.
:::
Consider an extensive system: The entropy, volume and all of
measurable conserved quantities $Q_i$ are additive, and thus
extensive. The thermodyanmic entropy is thus
a homogenous function of its arguments:
\begin{gather*}
  S(\alpha V,\alpha \vect{Q}) = \alpha S(V,\vect{Q}).
\end{gather*}
:::{margin}
If $f(ax) = af(x)$ for all $x$, then
\begin{align*}
  \diff{f(ax)}{a} &= x\diff{f(ax)}{ax} = f(x) = \frac{f(ax)}{a} &
  &\Rightarrow &
  \diff{f}{x} &= \frac{f}{x}.
\end{align*}
This has the family of solutions $f(x) = \lambda x$.
:::
This implies that
\begin{gather*}
  S = p V + \vect{\theta}\cdot\vect{Q}
\end{gather*}
where $p$ and $\vect{\theta}$ are *intensive* quantities (independent of the size of the system).

Now, the thermodynamical law~(\ref{eq:dS1}) can be expressed in terms of the desities $s = S/V$ and $\vect{q} = \vect{Q}/V$:
\begin{gather*}
  V(\d{s}-\vect{\theta}\cdot\d{\vect{q}}) +
  (s-p-\vect{\theta}\cdot\vect{q})\d{V} = 0.
\end{gather*}
Since all of the volume dependence is explicit, this implies that
\begin{subequations}
  \begin{align}
    \d{s}&=\vect{\theta}\cdot\d{\vect{q}},\\
    p &= s-\vect{\theta}\cdot\vect{q}.
  \end{align}
\end{subequations}
From this we can see that ({eq}`eq:NonExtensivePotential`) is zero for
thermodynamics systems.  Instead, we use the Legendre transform of the
reduced pressure $p$:
\begin{gather*}
  p(\vect{\theta}) = \max_{\vect{q}} s(\vect{q}) - \vect{\theta}\cdot\vect{q}.
\end{gather*}

### Thermodynamic Potentials

:::{margin}
In finite systems, for example, there is always a unique ground state which is fully
symmetric due to the possibility of tunneling.  There is no possibility for spontaneous
symmetry breaking, or phase transitions.
:::
To carefully define the thermodynamic variables for quantum systems
and the resulting thermodynamics is a bit involved due to the
requirement of working with systems containing infinitely many degrees
of freedom. We refer the reader
to {cite}`Sewell:2002` Chapter 6.4 for a more thorough discussion.

We start with the Helmholtz free energy $F$ defined
by~(\ref{eq:Fmin}).  This is to be minimized at fixed temperature,
volume and particle number.  From this, we can Legendre transform to
one of several thermodynamic potentials to remove constraints by
introducing Lagrange multipliers.  One of the most useful is to form
the Gibbs free energy:
\begin{gather*}
  \Omega(T,\vect{\mu}) = \min \left(F-\vect{\mu}\cdot\vect{N}\right).
\end{gather*}
The minimization is over all physical states.  As a result, one can
prove that $\Omega$ is a convex function.  Here we prove convexity
over $\vect{\mu}$: the extension to include $T$ can be done similarly
by returning to the maximum entropy principle (where $T$ enters as a
Lagrange multiplier).  Let $\Omega(\rho,T,\vect{\mu}) =
F-\vect{\mu}\cdot\vect{N}$ be the minimand for an
arbitrary state $\rho$:
\begin{align*}
  x\Omega(T,\vect{\mu}_1) + (1-x) \Omega(T,\vect{\mu}_2) &\leq
  \min_\rho \left(
    x[F(\rho)-\vect{\mu}_1\cdot\vect{N}(\rho)] +
    (1-x)[F(\rho)-\vect{\mu}_2\cdot\vect{N}(\rho)] \right),\\
  &=\min_\rho
  F(\rho)-(x\vect{\mu}_1+(1-x)\vect{\mu}_2)\vect{N}(\rho),\\
  &= \Omega(T,x\vect{\mu}_1+(1-x)\vect{\mu}_2).
\end{align*}
From the convexity of $\Omega$, we can find a one-to-one mapping
between states which minimize $F$ for fixed particle number and
tangents to the surface $\Omega$.  In particular, if defined, the
gradient of the tangent is the particle number:
\begin{gather*}
  \vect{N}(T,\vect{\mu}) = -\pdiff{\Omega(T,\vect{\mu})}{\vect{\mu}}.
\end{gather*}
By a tangent to $\Omega$, we mean a plane that intersects $\Omega$ and
for which no point of $\Omega$ lies above the plane.  If $\Omega$ is
differentiable, then there is exactly one such plane with the gradient
given above: this indicates that the thermodyanmic state is a pure
phase with well-defined particle number.  If $\Omega$ is not
differentiable, then there are many such planes.  In this case, each
plane describes a different mixed phase.  The mixture is composed of
the pure states that intersect at the mixture.

### Example: Free Fermi Gas
The Hamiltonian for a gas of non-interacting fermions is given in
second quantized form as
\begin{gather*}
  \op{H} = \int\dbar^{3}{\vect{p}}\;\frac{p^2}{2m}
  \op{a}^\dagger e_{\vect{p}}\op{a}_{\vect{p}}.
\end{gather*}
The states have energy $p^2/(2m)$, and degeneracy $4\pi p^2$.
