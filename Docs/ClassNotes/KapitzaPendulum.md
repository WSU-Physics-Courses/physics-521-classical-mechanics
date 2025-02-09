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
try: from myst_nb import glue
except: glue = None
```

(sec:KapitzaPendulum)=
# Kapitza's Pendulum

Here we consider the high-frequency limit of parametric oscillations of a pendulum:
\begin{gather*}
  \ddot{\theta} + 2\lambda \dot{\theta} + \omega^2(t)\sin\theta = 0, \qquad
  \omega^2(t) = \omega_0^2(1 + a \sin \omega t), \qquad
  \omega \gg \omega_0.
\end{gather*}
This problem was considered by [Pyotr Kaptitza][Kapitza's pendulum], who showed that
one can use such a mechanism to stabilize the motion about unstable points.

## Simplified Analysis

We start with a simplified analysis, getting at the essence of the problem without
placing too much attention on rigor.  This analysis is based on the following
assumptions:
:::{margin}
We express the "fast" part in terms of $\phi = \omega t$ so that $f(\phi)$, $f'(\phi)$,
and $f''(\phi)$ all are the same order.  The "fastness" will appear through factors of
the large drive frequency $\omega$:
\begin{gather*}
  \dot{f} = \omega f', \quad
  \ddot{f} = \omega^2 f'',
\end{gather*}
etc.
:::
1. Separation of scales: $\omega \gg \omega_0$.  We decompose the motion into a fast
   part $f(\omega t)$ and a slow part $s(t)$:
   \begin{gather*}
     \theta(t) = s(t) + f(\omega t), \qquad \braket{f}(\omega t) = 0,\qquad
     \braket{A} = \frac{1}{T}\int_{t}^{t+T}\!\!\!\!\!\!\!\!A(\tau)\d{\tau}.
   \end{gather*}
   To recover the slow components, we average over one fast period $T = 2\pi / \omega$.
   We remove any slow component from $f(\omega t)$, placing them in $s(t)$ so that
   $\braket{f}(t) = 0$.
2. Linearization: we assume that the amplitude of the fast part $\abs{f} \ll 1$ is
   small.  Note: we **do not** consider the amplitude of the drive $a$ to be small.  The
   final effects will be $O(a^2)$, so we must at least keep this to quadratic order.

The idea will be to consider separately the fast motion $f(\omega t)$ by treating the slow variables
as constants, and then the slow motion $s(t)$ by averaging over fast variables.  This
will yield an effective potential for the slow motion that permits stabilization.

:::{margin}
Assuming that the fluctuation part $f$ is small, we expand $\cos f \rightarrow 1$ and
$\sin f \rightarrow f$.
:::
Expanding the original equation, we have
\begin{gather*}
  0 = \ddot{s} + \omega^2 f'' + 2\lambda (\dot{s} + \omega f') 
  + \omega_0^2(1+ a \sin \omega t)\underbrace{(\sin s\cos f + \cos s \sin f)}_{\sin(s+f)},\\
  \approx \ddot{s} + \omega^2 f'' + 2\lambda (\dot{s} + \omega f') 
  + \omega_0^2(1 + a \sin \omega t)(\sin s + f\cos s)
\end{gather*}

### Fast Motion

:::{margin}
Alternatively, and more rigorously, one should separate the fast and slow equations,
showing that the slow equation will naturally set this "constant" shift to zero if we
demand $\braket{f} = 0$.  If $f(\phi)$ is periodic (as we shall see), then all
derivatives also average to zero: $\braket{f'} = \braket{f''} = 0$ etc.  To see this, consider
the Fourier series
\begin{gather*}
  f(\theta) = \sum_{n} c_n e^{n \I \theta}.
\end{gather*}
Only the $n=0$ term has a non-zero average, but this vanishes for all derivatives.
:::
We first consider the fast motion $f(\phi)$ where $\phi = \omega t$.  Since the division
between $f$ and $s$ is arbitrary, we can choose to adjust $s$ so that $\omega_0^2 \sin
s + 2\lambda \dot{s} + \ddot{s} = 0$.  As we shall see, this ensures that $\braket{f} =
0$ as discussed above.  We also assume that $f\ll 1$, $\abs{\omega} \gg \lambda$ and
$\abs{\omega} \gg \abs{\omega_0}$ so we can neglect the damping and resonant terms.  The
resulting equation is
\begin{gather*}
  \omega^2 f''(\phi) \approx  - a\omega_0^2 \sin s\sin \phi, \qquad
  f(\omega t) \approx \frac{a\omega_0^2 \sin s}{\omega^2}\sin \omega t.
\end{gather*}
Note that $f$ is periodic, so all averages $\braket{f} = \braket{f'} = \braket{f''} =
0$ as desired.

:::{admonition} Details.
:class: dropdown
More explicitly, to linear order in $f \sim f' \sim f''$ we can write
\begin{gather*}
  \omega^2 f'' 
  + {\color{red}
    2\omega \lambda f'
    + f\omega_0^2\cos s}
  \approx
  - a\omega_0^2 \sin \omega t (\sin s -  {\color{red}f\cos s})
  - {\color{red}( \omega_0^2\sin s + 2\lambda \dot{s} + \ddot{s})}.
\end{gather*}
The second two (red) terms on the left-hand side are parametrically small compared to
$\omega^2 f''$ under the assumption that the drive frequency $\abs{\omega}$ is much
larger than the damping or resonant frequency.  Under the assumption that $\abs{f} \ll
1$, we can neglect $f\cos s$ with respect to $\sin s$ in the first term on the right.
Note that this approximation assumes $f$ is small: it is often not that small, and the
resulting approximation shown numerically below can benefit from keeping these terms.

Finally, the last term is set to zero by suitable choosing $s$.  To see this explicitly,
imaging keeping this constant $\beta = \omega_0^2 \sin s + 2\lambda \dot{s} + \ddot{s}$.
The solution would become:
\begin{gather*}
  \omega^2 f'' = -\alpha \sin \omega t  - \beta, \qquad
  f(\omega t) = \frac{\alpha}{\omega^2}\sin\omega t + \frac{\beta}{\omega^2} t^2.
\end{gather*}
Thus, the presence of this constant gives an overall slow trend to $f(\omega t)$ with
$\braket{f} = \beta T^2/3\omega^2 \neq 0$.  To remove this, we must adjust $s$ slowly
exactly so that $\beta = 0$.

:::
### Slow Motion

We now averaging the dynamical equation:
\begin{gather*}
  0 = \ddot{s} + 2\lambda \dot{s} 
  + \underbrace{\omega_0^2\sin s + \omega_0^2a \cos s \overbrace{\braket{f \sin \omega
  t}}^{\frac{a\omega_0^2}{2\omega^2}\sin s}}_{U_{\text{eff}}'(s)},\\
  U'_{\text{eff}}(s) = \omega_0^2\sin s + \frac{a^2\omega_0^4}{4\omega^2}
  \underbrace{\sin(2s)}_{2\cos s \sin s}.
\end{gather*}
Thus, the slow degree of freedom experiences damped motion in an effective potential of
the form
\begin{gather*}
  U_{\text{eff}}(s) = -\omega_0^2\cos s - \frac{a^2\omega_0^4}{8\omega^2}\cos(2s).
\end{gather*}

:::::{admonition} Do It! Demonstrate numerically that this is correct.
:::{solution}
To numerically check the results, we simulate both the full theory, and the effective
theory.  The results are plotted below.
:::
:::::

```{code-cell}
:tags: [hide-input]
# Here is a numerical check
from scipy.integrate import solve_ivp

class Kapitza:
    w0 = 1      # Natural frequency
    w = 10.0    # Drive frequency
    a = 20.0    # Drive amplitude
    lam = 0.01  # Damping
    phi = 0.0   # Angle of drive
    corrected = False

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        pass
    
    def pack(self, q, dq, qe, dqe):
        y = np.ravel([q, dq, qe, dqe])
        return y

    def unpack(self, y):
        q, dq, qe, dqe = y
        return q, dq, qe, dqe
        
    def compute_dy_dt(self, t, y):
        """Return dy_dt."""
        q, dq, qe, dqe = self.unpack(y)
        w0, w, a, phi, lam = map(self.__getattribute__, ['w0', 'w', 'a', 'phi', 'lam'])
        th = w * t
        Fq = - w0**2 * np.sin(q) - a * w0**2 * np.sin(q - phi) * np.sin(th)
        ddq = -2*lam*dq + Fq
        ddqe = -2*lam*dq - self.get_Ueff(qe, d=1)
        return self.pack(dq, ddq, dqe, ddqe)
        
    def get_Ueff(self, q, c2=None, d=0):
        """Return the dth derivative of Veff(q)"""
        w, w0, a, phi = self.w, self.w0, self.a, self.phi
        
        c1 = -w0**2
        if c2 is None:
            c2 = - a**2 * w0**4 / 8 / w**2
        q_phi = q - phi
        if self.corrected and d == 1:
            _tmp = (1 - (w0/w)**2*np.cos(q_phi))
            _cor = _tmp/(_tmp**2 + 4*(self.lam/w)**2)
            return - (c1 * np.sin(q) + 2 * c2 * np.sin(2*q_phi) * _cor)
        return (c1 * (1j)**d * np.exp(1j*q) + c2 * (2j)**d * np.exp(2j*(q_phi))).real
    
    def solve(self, q0=None,  dq0=0.1, periods=6, points=10, **kw):
        """Return a dense solution.
        
        Arguments
        ---------
        periods : int
            Number of slow periods to solve.
        points : int
            Number of points per fast cycle.
        """
        if q0 is None:
            q0 = np.pi + self.phi
            
        y0 = self.pack(q0, dq0, q0, dq0)
        Ts = 2*np.pi / self.w0  # Slow periods
        Tf = 2*np.pi / self.w   # Fast periods
        dt = Tf / points
        T = periods * Ts
        Nf = int(np.ceil(T/Tf)) # Number of fast periods
        t_eval = np.arange(Nf*points + 1)*dt
        kwargs = dict(
            y0=y0, t_span=(0, T), t_eval=t_eval, method="BDF", dense_output=True)
        kwargs.update(kw)
        self.sol = solve_ivp(self.compute_dy_dt, **kwargs)
        assert self.sol.success
        
        # Compute some useful quantities by averaging over periods
        q, dq, qe, dqe = self.unpack(self.sol.y)
        t_ = self.sol.t[:-1].reshape((Nf, points)).mean(axis=1)
        q_ = q[:-1].reshape((Nf, points)).mean(axis=1)
        self.sol.t_ = t_
        self.sol.q_ = q_
        self.sol.q, self.sol.qe = q, qe
        return self.sol
        
for (kw, dq0) in [(dict(phi=0.05*np.pi, lam=0.01), 0.4), 
                  (dict(phi=0, lam=0.01), 0.6)]:
    s = Kapitza(**kw)
    sc = Kapitza(corrected=True, **kw)
    sol = s.solve(dq0=dq0)
    solc = sc.solve(dq0=dq0)
    t_unit = 2*np.pi / s.w0
    q_unit = np.pi

    fig, axs = plt.subplots(
        1, 2, 
        figsize=(8*1.2, 2*1.2),
        sharey=True, width_ratios=(1, 0.2), 
        gridspec_kw=dict(wspace=0.02))

    for ax in axs:
        ax.grid(True)
        # These are wrong... only approximate
        #[ax.axhline(phi_min/q_unit, lw=1, ls=":", c="grey")
        # for phi_min in [s.phi, np.pi + 2*s.phi]]

    ax = axs[0]
    ax.plot(sol.t/t_unit, sol.q / q_unit, 'C0-', lw=1, label='Exact')
    ax.plot(sol.t_/t_unit, sol.q_ / q_unit, 'C0-', label='Averaged')
    ax.plot(sol.t/t_unit, sol.qe / q_unit, 'C1--', label='Effective')
    ax.plot(solc.t/t_unit, solc.qe / q_unit, 'C1-', label='Corrected')
    ax.set(xlabel=r"$\omega_0 t / 2\pi$", ylabel=r"$\theta / \pi$")
    ax.legend()

    ax = axs[1]
    q0, q1 = sol.q.min(), sol.q.max()
    dq = 0.1*(q1-q0)
    q0 -= dq
    q1 += dq

    q0 = min([q0, s.phi, 0])
    q1 = max(s.phi + np.pi, q1)
    q = np.linspace(q0, q1)
    ax.plot(s.get_Ueff(q, c2=0), q/q_unit, 'C0-', label=r"$U_{0}$")
    ax.plot(s.get_Ueff(q), q/q_unit, 'C1-', label=r"$U_{\mathrm{eff}}$")
    ax.set(xlim=(-2.1, 1.1), xlabel=r"$U(q)$");
    #ax.legend(loc='lower right', bbox_to_anchor=(0, 0));
    ax.legend(loc='upper left');
    title = [
        rf"$\omega/\omega_0={s.w/s.w0:.2g}$",
        rf"$a={s.a:.2g}$",
        rf"$\lambda/\omega_0={s.lam/s.w0:.2g}$",
        rf"$\phi={s.phi/np.pi:.2g}\pi$",
    ]
    plt.suptitle(", ".join(title));
```

We see that our approach does indeed correctly stabilize the inverted pendulum, and that
the effective potential is roughly correct, however, there are quantitative
disagreement.  If you play with these equations, you will find that these disagreements
are mostly an issue with larger amplitude excitations.

:::::{admonition} Do It! Improve the effective potential.

:::{hint}
:class: dropdown

One approximation that is not so difficult to improve is to keep the linear pieces in
the fast response equation.

\begin{gather*}
  \omega^2 f''(\phi) + 2 \lambda \omega f' + \omega_0^2 \cos s f 
  \approx  - a\omega_0^2 \sin s\sin \phi.
\end{gather*}

This correction is shown in the plot, and seems to help, but not perfectly.
:::

:::{solution}

Following the hint, we have

\begin{gather*}
  f(\omega t) \approx 
  \Im \frac{a\omega_0^2 \sin s e^{\I\omega t}}
           {\omega^2 - 2\I \omega \lambda - \omega_0^2 \cos s}\\
  =
  a\omega_0^2 \sin s
  \frac{(\omega^2 - \omega_0^2 \cos s)\sin \omega t - 2\omega\lambda\cos \omega t}
       {(\omega^2 - \omega_0^2 \cos s)^2 + 4\omega^2\lambda^2},\\
  \braket{f\sin \omega t} = \frac{a\omega_0^2 \sin s}{2\omega^2}
  \frac{1 - \tfrac{\omega_0^2}{\omega^2} \cos s}
       {(1 - \tfrac{\omega_0^2}{\omega^2} \cos s)^2 + 4\frac{\lambda^2}{\omega^2}}.
\end{gather*}

Another approach would be to numerically simulate the full system, then try to match the
effective theory.
:::::

:::::{admonition} Do It!

Show that driving at a different angle $\phi$ gives
\begin{gather*}
  U_{\text{eff}}(s) = 
  -\omega_0^2\cos s 
  - \frac{a^2\omega_0^4}{8\omega^2}\cos\bigl(2(s - \phi)\bigr).
\end{gather*}
as shown in the video.  Analyze the stability as a function of $\phi$.

<iframe width="560" height="315" src="https://www.youtube.com/embed/jS-rzZJovm4?si=TGe0bVfKgJoRpkOf&amp;start=138" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

:::{solution}
Here is a start. First we find the equilibrium positions.  In units where $\omega_0 = 1$:

\begin{gather*}
  U_{\text{eff}}(s) = 
  \sin s + \frac{\alpha}{2} \sin\bigl(2(s - \phi)\bigr),\\
  U'_{\text{eff}}(s) = 
  \sin s + \alpha \sin\bigl(2(s - \phi)\bigr),\\
  U''_{\text{eff}}(s) = 
  -\cos s - 2\alpha \cos\bigl(2(s - \phi)\bigr),\\
\end{gather*}

\begin{gather*}
  U'_{\text{eff}}(s) = 0,\qquad x = \sin s,\\
  \sin s = 2\alpha (\sin s \cos\phi + \cos s \sin \phi)(\cos s \cos \phi - \sin s \sin
  \phi),\\
  x = 2\alpha (x \cos\phi + \sqrt{1-x^2} \sin \phi)(\sqrt{1-x^2} \cos \phi - x \sin \phi).
\end{gather*}

This gives a quartic equation for $\sin s$.
:::
:::::

For a related approach, please see {ref}`sec:FloquetTheory`.

## Exact Solution
We start by considering the 

[Kapitza's pendulum]: <https://en.wikipedia.org/wiki/Kapitza's_pendulum>
