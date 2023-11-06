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
:tags: [hide-input]

import mmf_setup;mmf_setup.nbinit(quiet=True)
```

# Assignment 2: Accelerated Frames

Due: Friday 30 September 2022 at 11:59pm.

## Merry-go-round

Consider a free particle moving in 2D ($x$-$y$ plane) with constant velocity. Using
the force law for motion in an accelerated frame, compute the trajectory of this "free"
particle in a moving frame with $\omega(t) = \alpha t$ (i.e. with angle $\theta(t) =
\alpha t^2/2$).  Show that your trajectory explicitly satisfies Newton's law in the
accelerating frame including all three correction terms.

It is fairly straightforward to apply formula (8.2) twice, obtaining the three
correction terms in (8.3), however, it is very easy to make a mistake with the signs
(e.g. by forgetting which term is the derivative in the "inertial" frame and which is
the derivative in the "body" frame).  Use simple arguments to quickly but carefully
justify the signs of the corrections.  You really should memorize Eq. (8.2) and these
methods for future use.  *(For example, when working with motion on the earth, think
about sitting on the north pole and repeating these arguments.)*  Don't try to remember
(8.3) as it is quick to derive from Eq. (8.2)

> \begin{gather*}
    \left(\frac{\d}{\d{t}}\right)_{\text{intertial}} =
    \left(\frac{\d}{\d{t}}\right)_{\text{body}} + \vect{\omega}\times \tag{8.2}
  \end{gather*}

+++

## Larmor's Theorem: Problem 2.1

Do problem 2.1 from {cite:p}`Fetter:2003`:

> **2.1** **Larmor's theorem**
>
> 1. The Lorentz force implies the equation of motion $m\ddot{\vect{r}} = e(\vect{E} +
>     c^{-1}\vect{v}\times\vect{B})$.  Prove that the effect of a weak uniform magnetic
>     field $\vect{B}$ on the motion of a charged particle in a central electric field
>     $\vect{E} = E(\norm{\vect{r}})\uvect{r}$ can be removed by transforming to a
>     coordinate system rotating with an angular frequency $\omega_L =
>     -(e/2mc)\vect{B}$. (State precisely what "weak" means.)
> 2. Extend this result to a system of particles of given ratio $e/m$ interacting
>     through potentials $V_{ij}(\norm{\vect{r}_i - \vect{r}_j})$.

In class you have seen how trajectories and dynamics in a rotating frame can look much
more complicated than in an inertial frame.  Here you will explore how the motion of a
charged particle in a weak electric field looks like motion in a rotating frame.  Hence,
by removing the "rotation" you can simplify the problem to permit an analysis that is
more akin to mechanics in a non-rotating frame.

::::{note}
1. When justifying what makes something small ("weak"), be sure to compare quantities of
   the same dimension.  *(E.g., it make no sense to say that the mass of an electron $m$
   is small.  Compared to your mass, it is indeed small, but compared to the mass of an
   electron neutrino, it is huge!)*
   
   :::{warning}
   You may recall that charged particles in a constant magnetic field move in a
   helix, or circle with a characteristic [cyclotron frequency] $\omega_c = eB/mc =
   2\omega_L$.  This factor of two is not a mistake: the physics here is different.
   Rotating at the cyclotron frequency makes the motion of one particular set of
   particles very simple -- those that exactly orbit the axis of rotation are
   motionless.  However, other particles do not have a simple motion.  The transformation
   here to a frame roting with the Larmor frequency makes the motion of **all**
   particles *simpler* (within a particular approximation).  Think carefully about this.
   :::
2. The notation $E(\norm{\vect{r}}) = E(r)$ means that the magnitude of the electric
   field is a function of the distance $r$ from the origin.  This is not a product.
3. The point of the second part is to explain why the charge-to-mass ratio must be the
   same for all particles, and why the potentials must only depend on their separation.
   How would the analysis break down if these conditions were not satisfied?  You should
   not be solving equations for this part, but should be arguing why the method of
   analysis still makes sense.

[cyclotron frequency]: <https://en.wikipedia.org/wiki/Cyclotron_resonance>
::::

+++

## Motion on the Earth: Problem 2.2

Do problem 2.2 from {cite:p}`Fetter:2003`:

> **2.2** Assume that over the time interval of interest, the center of mass of the
>    earth moves with approximately constant velocity with respect to the fixed stars
>    and that $\vect{\omega}$, the angular velocity of the earth, is a
>    constant. Rederive the terrestrial equations of particle motion (11.8) and (11.6)
>    by writing Newton's law in a body-fixed frame with origin at the surface of the
>    earth (Fig. 11.2).
> \begin{gather*}
    \vect{g} = -(GM_eR_e^{-2} - \omega^2 R_e\sin^2\theta) \hat{\vect{r}}
             + \tfrac{1}{2}\sin 2\theta\; \omega^2 R_e \hat{\vect{\theta}} \tag{11.6}\\
  m\ddot{\vect{r}} = m \vect{g} - 2m \vect{\omega}\times \dot{\vect{r}} \tag{11.8}
\end{gather*}

Perhaps the most important daily experience we have with accelerating frames is the
motion due to the rotation of the eath.  Here you are asked to rederive two expressions
in the book: one for the local gravitational acceleration $\vect{g}$, and the other for
Newton's laws.  The $\vect{g}$ here is what you would measure if you were to drop an
object in the laboratory.  The corrections due to the motion of the earth would be
critical if you were, for example, using such a measurement to determine the
distribution of mass in the earth since that is only one component of the total
acceleration.

::::{note}
* The point of this problem is not to do a bunch of algebra: it is to figure out what
  the difference between Newton's laws expressed relative to an origin at the center
  of the earth, and those expressed in a terrestrial frame.  Hint: When considering
  the origin at the center of the earth, $\vect{r}$ is very long (radius of the earth
  plus any deviations in the lab).  This means that the centrifugal force is quite
  strong.  However, when derived in a terrestrial frame with the origin attached to
  the surface of the earth, $\vect{r}$ is short (few meters), so the centrifugal force
  is much smaller.  Obviously the motion of a particle in a terrestrial lab does not
  depend on where you place your origin.  How do you reconcile this apparent paradox?
::::

The corrections to the dynamical laws affect the motion of projectiles as you will use below.

+++

## Projectile Motion: Problem 2.5

> **2.5** A cannon is placed on the surface of the earth at colatitude (polar angle)
>    $\theta$ and pointed due east.
>
> 1. If the cannon barrel makes an angle $\alpha$ with the horizontal, show that the
>     lateral deflection of a projectile when it strikes the earth is
>     $(4V_0^3/g^2)\omega \cos\theta\;\sin^2\alpha \cos\alpha$, where $V_0$ is the
>     initial speed of the projectile and $\omega$ is the earth's angular-rotation
>     speed.  What is the direction of this deflection?
> 2. If $R$ is the range of the projectile for the case $\omega=0$, show that the change
>     in the range is given by $(2R^3/g)^{1/2} \omega \sin \theta \bigl[(\cot
>     \alpha)^{1/2} - \tfrac{1}{3}(\tan \alpha)^{3/2}\bigr]$. Neglect terms of order
>     $\omega^2$ throughout.

Here you can use the equations of motion you derived in problem 2.2 to analyze the
motion of a projectile.

I suggest (but don't require) that you try this problem two ways:

1. Use the equations of motion you derived above, expanding for small $\omega$ as
   suggested in the problem.
2. Treat the canon ball as an orbiting body and compare its elliptical motion in the
   central potential of the earth with the solid body rotation of the earth.
   (I.e. solve the problem in the inertial frame.)

If part 2. is too messy for you to do in a reasonable time, try problem 2.3 instead.  I
would like you to see how working in the accelerated frame is the same as working in the
inertial frame so you can choose whichever method is simpler when you come across a
real problem of this type.

> **2.3** An observer that rotates with the earth drops a particle from a height $h$
> above the earth's surface.  Analyze the motion from an inertial frame of reference and
> rederive the net eastward deflection of $\tfrac{1}{3}\omega g \sin \theta\;
> (2h/g)^{3/2}$, where $\theta$ is the observer's polar angle.

::::{note}
* Don't use the provided answers to "guide" you.  Work through the problem first on your
  own, then use these to check your work.  If you do not get the correct answer, then
  you probably made a mistake.  There are several places where it is easy to forget an
  important piece of physics in these problems that will spoil your calculation.  To
  check your answer with the form of the answer give, it might be best to plot the two
  results or look at them numerically to avoid doing unnecessary algebra.
* If you can't figure out what is missing, perhaps try a numerical solution to make sure
  that the formula presented here are indeed correct.
::::
