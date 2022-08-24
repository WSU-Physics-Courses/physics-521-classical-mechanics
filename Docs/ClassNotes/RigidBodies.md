---
jupytext:
  formats: ipynb,md:myst,py:light
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

Rigid Bodies
============

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

(phantom-torque)=
## Phantom Torque

Here I would like to try to clarify a point mentioned in class about what Turner and
Turner {cite:p}`Turner:2010a` call "phantom torque".  In the textbook,
{cite:p}`Fetter:2003` show in (27.1) that the rate of change of angular momentum
$\vect{L}$ in an inertial frame is equal to the external torque $\vect{\Gamma}^{(e)}$:

\begin{gather*}
  \left. \diff{\vect{L}}{t}\right|_{\text{inertial}} = \vect{\Gamma}^{(e)} \tag{27.1}
\end{gather*}

in two cases:

1. About an origin fixed in some inertial frame.
2. About the center of mass, even if it accelerates. Note: the "inertial" in (27.1)
   refers to the frame not rotating about the center of mass.

The paper by Turner and Turner {cite:p}`Turner:2010a` points out a potential fallacy if
the object is asymmetric such that .

Consider an object rotating with angle $\theta$ and momentum of inertia $I_p(\theta)$
about a point of contact where $I_p'(\theta) \neq 0$.  This dependence of the momentum
of inertia on the coordinate is responsible for the "phantom torque", but the actual
manifestation is subtle as we now show.  Note that for a rolling coin or ball which is
symmetric, $I_{p}(\theta) = I_{p}$ is constant, which is why this issue is often not
discussed with many classical textbook examples.

The example raised in {cite:p}`Turner:2010a` is that of a half-hoop rolling without
slipping along a flat surface with the motion described by the angle $\theta$ such that
$\dot{\theta}$ is the angular velocity.

```{code-cell}
:tags: [hide-input]

%%manim -v WARNING -qm RollingHoop
from manim import *
from types import SimpleNamespace


class Hoop:
    R = 4.0
    theta0 = 0.6
    r_cm = 2 / np.pi * R
    m = 1.0
    I_o = m * R ** 2
    I_cm = I_o - m * r_cm ** 2
    shift = UP

    def __init__(self, **kw):
        for k in kw:
            assert hasattr(self, k)
            setattr(self, k, kw.pop(k))
        assert not kw
        
        self.objects = self.annotations

    def get_arc(self, theta=None):
        """Return a (arc, points) for a hoop"""
        color = 'white'
        if theta is None:
            theta = self.theta0
            color = 'cyan'
            
        z_o = -self.R * theta + 0j
        z_cm = z_o - 1j * self.r_cm * np.exp(theta * 1j)
        z_p = z_o - 1j * self.R
        o, p, cm = [np.array([_z.real, _z.imag, 0]) for _z in [z_o, z_p, z_cm]]
        e1 = o + self.R * np.array([np.cos(theta), np.sin(theta), 0])

        arc = Arc(
            radius=self.R, start_angle=PI + theta, angle=PI, arc_center=o,
            color=color
        )
        arc.stroke_width = 10.0
        points = SimpleNamespace(O=o+self.shift, P=p+self.shift, C=cm+self.shift)
        arc.shift(self.shift)
        
        # These are the actual objects we display
        return (arc, points)

    @property
    def annotations(self):
        """Return a `Group` object with the hoop and all of the annotations."""
        arc, points = self.get_arc()
        
        objs = []
        P, C, O = points.P, points.C, points.O
        for _p0, _p1, _t in [
            (arc.get_end(), O, "R"),
            (C, O, r"a=\tfrac{2}{\pi}R"),
        ]:
            objs.append(BraceBetweenPoints(_p0, _p1))
            objs.append(objs[-1].get_tex(_t))
            #objs[-1].font_size = 40
        objs.extend(
            [
                LabeledDot("O", point=O),
                LabeledDot("P", point=P),
                LabeledDot("C", point=C),
            ]
        )
        
        OC = DashedLine(O, C, color='cyan')
        OP = DashedLine(O, P, color='cyan')
        theta = Angle(OP, OC, radius=0.7, dot_radius=2.0)
        theta_l = Tex(r"$\theta$")
        _m = theta.get_midpoint()
        theta_l.next_to(O + 1.5*(_m-O), 0)
        objs.extend([OC, OP, theta, theta_l])

        floor = Line((-6, -self.R, 0), (6, -self.R, 0))
        floor.shift(self.shift)
        floor.stroke_width = arc.stroke_width
        self.arc = arc
        annotations = Group(arc, floor, *objs)
        return annotations


class HoopAnimation(Animation):
    def __init__(self, hoop, **kw):
        super().__init__(hoop.get_arc(theta=hoop.theta0)[0], **kw)
        self.hoop = hoop
        
    def interpolate_mobject(self, alpha):
        # Fake dynamics
        theta = self.hoop.theta0 * np.cos(2*np.pi * alpha)
        self.mobject.points = self.hoop.get_arc(theta=theta)[0].points
        

class RollingHoop(Scene):
    def construct(self):
        hoop = Hoop()
        self.add(hoop.annotations)
        #self.play(HoopAnimation(hoop=hoop), run_time=3, rate_func=linear)
```

Here we have half of a hoop of radius $R$ and mass $m$, with center of mass at $C$ which
is distance $a=2R/\pi$ from the center of the hoop $O$.  The external torque is provided
by the downward acceleration due to gravity $g>0$.  The hoop rolls without slipping
along the floor with point of contact $P$.

```{margin}
The "safe" way to solve the problem is to use the Lagrangian approach, expressing the
energy relative to either the center of mass $C$, or the fixed-point $P$.  However, to
appreciate the difficulty here, try also solving the problem using torques.  You should
be able to get the correct answer if you work about the center of mass $C$, but it is very
challenging to get the correct answer working about the point of contact $P$.
```

If you have not solved this problem before, please stop for a moment and try it with the
coordinate $\theta$ being the angle of rotation with $\theta = 0$ the equilibrium
position with $O$, $C$, and $P$ vertically aligned.

````{admonition} Solution
:class: dropdown

For the hoop, the center of mass is located at $a = 2R/\pi$, and the moment of inertia
about the contact point $P$ can be found by using the
parallel axis theorem twice, starting from $I_{O} = mR^2 = I_{C} + ma^2$ and $I_{P}(\theta) =
I_{C} + m(R^2 + a^2 - 2aR\cos\theta) = 2mR(R-a\cos\theta)$.  This gives the following
Lagrangian and solution:

\begin{gather*}
  L(\theta, \dot{\theta}) = mR(R-a\cos\theta)\dot{\theta}^2 + mga\cos\theta,\\
  \diff{}{t}\pdiff{L}{\dot{\theta}} = \pdiff{L}{\theta},\\
  2mR(R-a\cos\theta)\ddot{\theta} = -mRa\sin\theta\dot{\theta}^2 - mga\sin\theta,\\
  \left(1 - \frac{2}{\pi}\cos\theta\right)\ddot{\theta} =
  -\frac{\sin\theta}{\pi} \dot{\theta}^2 - \frac{\sin\theta}{\pi}\frac{g}{R}.
\end{gather*}

This is (13) from {cite:p}`Turner:2010a`.
````

A careful way of proceeding is to use a Lagrangian.  Here we work about the
instantaneously stationary point $P$, so the kinetic energy is purely rotational:

$$
  L(\theta, \dot{\theta}) = \frac{1}{2}I_{p}(\theta)\dot{\theta}^2 - V(\theta),\\
  p_\theta = L = \pdiff{L}{\dot{\theta}} = I_{p}(\theta)\dot{\theta}, \qquad
  \dot{L} = \pdiff{L}{\theta}.
$$

Consider carefully the terms in the last equation of motion $\dot{L} = \partial
L/\partial \theta$:

$$
  \overbrace{I_{p}'(\theta)\dot{\theta}^2}^{a} + I_{p}(\theta)\ddot{\theta} =
  \overbrace{\frac{1}{2}I'_{p}(\theta)\dot{\theta}^2}^{b} - V'(\theta),\\
  I_{p}(\theta)\ddot{\theta} = \underbrace{- V'(\theta)}_{\Gamma^{(e)}}
 + \underbrace{\frac{-I'_{p}(\theta)}{2}\dot{\theta}^2}_{\text{phantom torque}}.
$$

The two terms $a$ and $b$ indicated give rise to the "phantom torque" $b-a$, but notice
that the point is subtle.  {cite:p}`Turner:2010a` point out that both terms are missing
if one uses the $F = ma$ form $\Gamma = I \dot{\omega}$, but even using (27.1) one
may be misled into finding the piece $a$ but not $b$:

\begin{gather*}
  \left.\diff{\vect{L}}{t}\right|_{\text{inertial}}
  = \mat{I}\cdot \ddot{\vect{\theta}} + \underbrace{\dot{\mat{I}}\cdot \dot{\vect{\theta}}}_{a}
  = \vect{\Gamma}^{(e)}.
\end{gather*}

```{admonition} Incomplete Discussion
:class: dropdown

*This is an incomplete discussion about what goes wrong and how to fix it. One account
is given in {cite:p}`Jensen:2011` but, while correct, I do not find any of the four
forms he discusses very intuitive.*

The simple attempts fail because the reference point $P$ may be
accelerating.  Instead with need the distance from $C$ to the point $P_0=P$ in a locally
inertial frame where $P_0$ does not depend on $\theta$.  (In this problem, $P_0$ is
fixed since the hoop does not slip against the stationary floor).
In this problem, $P_0$ lies along the floor at
distance $-R\theta_0$ from the equilibrium position where $\theta_0 = \theta$ at the
time we are discussing, but **does not change**.  The error in blind calculation of
$\dot{I} = I_{P}'(\theta)\dot{\theta}$ is that the expression for $I_{P}(\theta)$ does
not keep this $\theta_0$ fixed.  Doing this properly, we find that the the center of
mass is at

$$
  x_{C} = a\sin\theta - \theta R, \qquad
  y_{C} = -a\cos\theta.
$$

Thus, the distance to the fixed point $P_0$ from the center of mass $C$ is:

$$
  \sqrt{\Bigl(x_{C}(\theta) + R\theta_0\Bigr)^2 + y_{C}^2(\theta)}.
$$

The moment of inertia about the point $P_0$ is thus:

$$
  I_{P_0}(\theta, \theta_0)
  = I_{C} + m\Bigl(x_{C}(\theta) + \theta_0 R\Bigr)^2 + my_{C}^2(\theta)\\
  = I_{P}(\theta) - 2mRx_{C}(\theta)(\theta - \theta_0) - mR^2(\theta^2 - \theta_0^2).
$$

Note that $I_{P}(\theta_0) = I_{P_0}(\theta_0)$, but the correct variation is

\begin{gather*}
  \dot{I} = \left.\diff{I_{P_0}}{t}\right|_{\text{inertial}} =
  \pdiff{I_{P_0}(\theta, \theta_0)}{\theta}\dot{\theta},\\
  \left.\pdiff{I_{P_0}(\theta, \theta_0)}{\theta} \right|_{\theta=\theta_0}=
  2m\bigl(x_{C}(\theta_0) + \theta_0R\bigr)x_{C}'(\theta_0) + 2m
  y_{C}(\theta_0)y_{C}'(\theta_0)\\
  = I_{P}'(\theta_0) - 2mR\bigl(x_{C}(\theta_0) + \theta_0R\bigr)
\end{gather*}

$$
  I_{P}'(\theta) =
  2m\bigl(x_{C}(\theta) + \theta R\bigr)(x_{C}'(\theta) + R) + 2m y_{C}(\theta)y_{C}'(\theta)\\
$$

```
