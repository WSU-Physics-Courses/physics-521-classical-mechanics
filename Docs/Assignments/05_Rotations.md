---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (phys-521-2021)
  language: python
  name: phys-521-2021
---

# Assignment 5: Rotations

+++

**Due: 11:59pm Friday 19 November 2021**

+++

## Rolling Coin
Consider a coin rolling in a circle.  Treat the coin as a flat disk of radius $R$ and
mass $M$.  Assume that the coin rolls without slipping.  Determine all properties of the
motion (period of the orbit, radius of the circle, angle of the coin, angular velocities
etc.)  Be sure to clearly sketch the geometry of your problem, perform a dimensional
analysis, and check your solution by considering various limits.

+++

## Rolling Hoop (Phantom Torques)

```{code-cell} ipython3
:tags: [hide-cell]

import manim.utils.ipython_magic
```

Solve for the equations of motion for a rolling half-circle as discussed in the class
notes {ref}`phantom-torque` and shown below:


```{code-cell} ipython3
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
            (C, O, r"a"),
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
is distance $a$ from the center of the hoop $O$.  The external torque is provided
by the downward acceleration due to gravity $g>0$.  The hoop rolls without slipping
along the floor with point of contact $P$.  The aim of this problem is to obtain the
equations of motion for $\theta(t)$.

1. Compute the moments of inertia $I_{O}$, $I_{C}$, and $I_{P}(\theta)$ and relate these
   to each other using the parallel axis theorem.  Express your answer in terms of $m$,
   $R$, $a$, and $\theta$.  *(For this problem, show that $a=2R/m$, but express
   everything in terms of $a$.  This allows you to generalize to symmetric but unequal
   mass distributions.)* 
2. Solve for the equations of motion using the Lagrangian framework about the two points
   where the rotational and transitional motions can be decoupled.  Show that this gives
   the same equations of motion, with the parallel axis theorem adjusting the moments of
   inertia exactly as needed to correct the equations.
3. Solve the equations of motion using Newton's law about the center-of-mass $C$:

   $$
     \diff{\vect{L}}{t} = \vect{\tau}.
   $$
   
   Draw all forces and calculate the torque $\vect{\tau}$ in terms of these.
   
4. :::{margin}
   Alternative corrections to Newton's law are discussed in {cite:p}`Jensen:2011`
   (available [on
   Perusal](https://app.perusall.com/courses/2021-fall-physics-521-pullm-1-01-01645-classical-mechanics-i/rules-for-rolling-as-a-rotation-about-the-instantaneous-point-of-contact-73799225)
   for discussion), but I 
   do not find any of these very intuitive.  Please let me know if you find any of these
   intuitive.
   :::

   Try solving the equations of motion using Newton's law about the instantaneously
   stationary point $P$.  You should find that a simple application of Newton's law does
   **not** give the correct answer.  Instead, you will need to correct Newton's law to
   include a "phantom torque" {cite:p}`Turner:2010a`.
   
   Show that these corrections arise from the $\theta$ dependence of $I_{P}(\theta)$.
   For this reason, one can use $P$ for the analysis if $I_{P}$ is independent of
   $\theta$ as it is for a complete hoop, balls, cylinders etc.

+++

## Nutation

:::{margin}
You might rightly worry that using $\dot{\vec{L}} = \vec{\tau}$ here might cause
problems due to the phantom-torque issue discussed in the previous problem, however, in
this case the momentum of inertia about the pivot point does not depend on the angles,
so, as you showed in part 4 above, no phantom torques are needed.
:::
Using the parametrization of the Euler angles in Landau and Lifshitz Volume 1, describe
nutation in the small amplitude limit (i.e. treat this as an application of the
normal-mode theory discussed in class.)  Clearly state all steps in formulating the
problem, i.e. deriving the equations (use the Lagrangian approach, and conserved
quantities to obtain an effective 1D theory with an appropriate effective potential),
find the "stationary" solution - this corresponds to simple **precession**.  Check this
by considering the Newton's law in the form of the time-evolution of the angular
momentum $\dot{\vec{L}} = \vec{\Omega}\times\vec{L} = \vec{\tau}$.  Then, describe the
deviations from the stationary solution to derive the normal modes.  Try to identify all
of the modes shown in Fig. 49, and check your answer numerically using the analytic
solution (7).  (References are version available [on
Perusall](https://app.perusall.com/courses/2021-fall-physics-521-pullm-1-01-01645-classical-mechanics-i/ll_6-rigidbody).)
Let me know if you need any help setting up the numerical checks.  I recommend using the
[`scipy.integration.quad`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
routine.
