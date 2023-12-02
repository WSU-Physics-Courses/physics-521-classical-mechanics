---
jupytext:
  formats: ipynb,md:myst
  notebook_metadata_filter: language_info.pygments_lexer,language_info.name
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (phys-521)
  language: python
  name: phys-521
language_info:
  name: python
  pygments_lexer: ipython3
---

```{code-cell} ipython3
:tags: [hide-cell]

from pathlib import Path
import os
import mmf_setup; mmf_setup.set_path()
FIG_DIR = Path(mmf_setup.ROOT) / '../Docs/_build/figures/'
os.makedirs(FIG_DIR, exist_ok=True)
import mmf_setup;mmf_setup.nbinit(console_logging=False)
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:DuffingOscillator)=
# Duffing Oscillator

:::{margin} Wikipedia

Wikipedia has an article about the [Duffing equation][].  We will track some of that
discussion here as a check, and reproduce their figures to ensure that the
normalizations are correct.  Our formulation agrees with their main article presentation
with the identification:
\begin{gather*}
  q \equiv x, \quad
  a \equiv \alpha, \quad
  b \equiv \beta, \\
  2\lambda \equiv \delta, \quad
  f \equiv \gamma, \quad
  \omega \equiv \omega.
\end{gather*}
:::
Here we consider the following problem:
\begin{gather*}
  \ddot{q} + 2\lambda \dot{q} + V'(q) = f\cos \omega t, \qquad
  V(q) = \frac{a}{2}q^2 + \frac{b}{4}q^4, \qquad b\geq 0.
\end{gather*}
For positive $a$, this is a driven harmonic oscillator with a quartic interaction.
Without driving, the quartic interaction will make the oscillation frequency amplitude
dependent, increasing with larger amplitude.

The interesting behavior occurs for negative $a$.  In this case, the potential develops
two minima, breaking the ground state:

```{code-cell} ipython3
:tags: [hide-input]

q = np.linspace(-1, 1)
b = 1
fig, ax = plt.subplots()
for a in [-0.2, 0, 0.2]:
    V = a*q**2/2 + b*q**4/4
    ax.plot(q, V, label=f"${a=:.2g}$")
ax.set(xlabel="$q$", ylabel="$V(q)$", ylim=(-0.05, 0.1))
ax.legend();
```

## Poincaré Map

In the case of a driven oscillator with $a>0$, we expect the solution to dampen to have
a periodic response with the same period $T = 2\pi/\omega$ as the driving term.  To
characterize the behavior, it is thus useful to consider how $(q, p)$ behaves at regular
time intervals $t_0 + nT$.  Such a portrait of the behavior is an example of a [Poincaré
map][], which represents the intersection of the full orbit $\{(t, q(t), p(t))\}$ with a
Poincaré section.

```{code-cell} ipython3
import numpy as np
from scipy.integrate import solve_ivp

class Duffing:
    a = 1.0
    b = 1.0
    f = 0.1
    lam = 0.1
    w = 1.2
    max_steps = 100
    phi0 = 0  # Initial phase, set to -np.pi/2 to drive with sin(wt)
    
    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])

    def show(self):
        print(f"alpha={self.a}, beta={self.b}, gamma={self.f}, delta={2*self.lam}, w={self.w}")
        a_ = abs(self.a)
        b_ = a_**3/self.f
        lam_ = w_ = np.sqrt(a_)
        print(f"alpha={self.a/a_}, beta={self.b/b_}, gamma={1}, lam=eta={self.lam/lam_}, w={self.w/w_}")

    @property
    def T(self):
        return 2*np.pi / self.w

    def V(self, q, d=0):
        if d == 0:
            return self.a*q**2/2 + self.b*q**4/4
        elif d == 1:
            return self.a*q + self.b*q**3

    def compute_dy_dt(self, t, y):
        q, dq = y
        ddq = self.f*np.cos(self.w*t + self.phi0) - self.V(q, d=1) - 2*self.lam*dq
        return dq, ddq

    def cycle(self, y, cycles=1, **kw):
        """Evolve for `cycles` cycle."""
        sol = solve_ivp(self.compute_dy_dt, y0=y, t_span=(0, self.T*cycles), 
                        dense_output=True, **kw)
        assert sol.success
        return sol
     
    def step(self, y, **kw):
        sol = self.cycle(y, **kw)
        return sol.y[:, -1]

    def get_cycle(self, y0=(0, 0), cycles=1, **kw):
        y = self.step(y0, **kw)
        for s in range(self.max_steps):
            y0, y = y, self.step(y, **kw)
            if np.allclose(y, y0):
                break
        sol = self.cycle(y, cycles=cycles, **kw)
        return sol
    
    def get_chi(self, y0=(0, 0), **kw):
        sol = self.get_cycle(y0=y0, **kw)
        t = np.linspace(0, self.T, 1000)
        q, dq = sol.sol(t)
        return (q.max() - q.min())/self.f
    
    def get_poincare(self, y0=(0, 0), N=4000, frames=100, **kw):
        """Return the data `(q, dq)` for Poincaré sections.
        
        Arguments
        ---------
        N : int
            Number of points in each section.
        frames : int
            Number of frames.
        
        Returns
        -------
        q, dq : array
            Position and velocity with shape=(N, frames)
        """
        sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, self.T*self.max_steps), **kw)
        y0 = sol.y[:, -1]
        ts = (self.T * np.arange(N*frames)) / frames
        sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, ts.max()), t_eval=ts, **kw)
        q, dq = sol.y.reshape(2, N, frames)
        return q, dq
        
    def plot_poincare(self, frame=0, data=None, y0=(1.0, 1.0), N=4000, frames=1, ms=0.5,
                      normalize=False,
                      interact=False, **kw):
        if data is None:
            data = self.get_poincare(y0=y0, N=N, frames=frames, **kw)
            N, frames = data[0].shape
        q, dq = data
        q_scale = dq_scale = 1.0
        if normalize:
            q_scale = abs(self.f/self.a)
            dq_scale = q_scale * np.sqrt(abs(self.a))
        
        q = q / q_scale
        dq = dq / dq_scale
        
        if interact:
            import ipywidgets
            @ipywidgets.interact(frame=(0, frames-1))
            def _go(frame=0):
                plt.plot(q[:, frame], dq[:, frame], '.', ms=ms)
                plt.axis([q.min(), q.max(), dq.min(), dq.max()])
        else:
            plt.plot(q[:, frame], dq[:, frame], '.', ms=ms)
        ax = plt.gca()
        ax.set(xlabel="$q$", ylabel=r"$\dot{q}$")
        return data
        '''
        sol = self.get_cycle(y0=y0, cycles=cycles, **kw)
        t = np.linspace(0, self.T*cycles, N)
        q, dq = sol.sol(t)
        plt.plot(q, dq, '.', ms=ms)
        '''
    
    def wikiFig1(self):
        """Plot first figure on Wikipedia."""
        gs = gridspec()
        

#ws = np.linspace(0.1, 2, 200)
#as_ = [0.1, 0.2, 0.3, 0.5, 1]
#for a in as_:
#    chis = [Duffing(w=w, a=a).get_chi() for w in ws]
#    plt.plot(ws, chis, label=f"{a=}")
#plt.legend()
```

```{code-cell} ipython3
%pylab notebook
ax = plt.figure().add_subplot(projection='3d')
q, dq = data
Ns, Nt = q.shape
th = 2*np.pi * np.arange(Nt)/Nt
r = q + 5
z = dq
th = th[None, :] + 0*q
x, y, z = map(np.ravel, (r*np.cos(th), r*np.sin(th), z))
ax.plot(x, y, z, lw=0.5)
```

```{code-cell} ipython3
%pylab notebook
ax = plt.figure().add_subplot(projection='3d')

alpha = 0.5
beta = 0.0625
gamma = 0.1
F0 = 2.5
w = 2.0
d = Duffing(a=-2*alpha, b=4*beta, lam=gamma/2, f=F0, w=w)
d.show()
a = 0.21
a = -0.326
for n, y0 in enumerate([(-0.1, 0), (0,0), (1.0, 0), (-1.0, 0)]):
    kw = dict(y0=y0, method="BDF", atol=1e-8)
    data = Duffing(a=a, b=4*beta, lam=gamma/2, f=F0, w=w, max_steps=500).plot_poincare(frames=100, N=100, **kw)
    q, dq = data
    Ns, Nt = q.shape
    th = 2*np.pi * np.arange(Nt)/Nt
    r = q + 5
    z = dq
    th = th[None, :] + 0*q
    x, y, z = map(np.ravel, (r*np.cos(th), r*np.sin(th), z))
    ax.plot(x, y, z, lw=0.5, c=f"C{n}")
    plt.plot(data[0].ravel(), data[1].ravel(), f'C{n}', lw=0.1)
#ax, bx, ay, by = plt.axis()
#plt.axis([min(ax, -2), max(bx, 2), min(ay, -2), max(by, 2)])
```

```{code-cell} ipython3
alpha = 0.5
beta = 0.0625
gamma = 0.1
F0 = 2.5
w = 2.0
d = Duffing(a=-2*alpha, b=4*beta, lam=gamma/2, f=F0, w=w)
d.show()
a = 0.21
a = -0.326
for n, y0 in enumerate([(-0.1, 0), (0,0), (1.0, 0), (-1.0, 0)]):
    kw = dict(y0=y0, method="BDF", atol=1e-8)
    data = Duffing(a=a, b=4*beta, lam=gamma/2, f=F0, w=w, max_steps=500).plot_poincare(frames=100, N=100, **kw)
    plt.plot(data[0].ravel(), data[1].ravel(), f'C{n}', lw=0.1)
ax, bx, ay, by = plt.axis()
plt.axis([min(ax, -2), max(bx, 2), min(ay, -2), max(by, 2)])
```

```{code-cell} ipython3
from tqdm.auto import tqdm
data = []
for x0 in tqdm(np.linspace(-1.5, 1.5, 10)):
    kw = dict(y0=(x0, 0), method="BDF", atol=1e-8)
    data.append(
        (x0, Duffing(a=a, b=4*beta, lam=gamma/2, f=F0, w=w, max_steps=1000).plot_poincare(frames=100, N=100, **kw)))
    plt.plot(data[-1][1][0].ravel(), data[-1][1][1].ravel(), lw=0.1)
ax, bx, ay, by = plt.axis()
plt.axis([min(ax, -2), max(bx, 2), min(ay, -2), max(by, 2)])
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(40,1))
for d in data:
    plt.plot(d[1][0].ravel())
```

```{code-cell} ipython3
#cdist(data[0][1].ravel(), data[-1][1].ravel()
cdist(np.array(data[0][1]).reshape(2, 100**2).T, np.array(data[-1][1]).reshape(2, 100**2).T)
```

```{code-cell} ipython3
from scipy.cluster.vq import kmeans2
def dist(A, B):
    ax, ay = A
    bx, by = B
    a = (ax + 1j * ay).ravel()
    b = (bx + 1j * by).ravel()
    return a, b

D = np.array([abs(np.subtract(*dist(d1[1], d2[1]))).min() for d1 in data for d2 in data]).reshape(100, 100)
```

```{code-cell} ipython3
N = D.shape[0]
inds = set(range(N))
orbits = []
thresh = 0.12
while inds:
    orbit = np.where(D[min(inds), :] < thresh)[0]
    if len(orbit) == 0:
        break
    orbits.append(orbit)
    inds = inds.difference(orbit)
len(orbits)
```

```{code-cell} ipython3
for orbit in orbits:
    x, y = data[orbit[0]][1]
    plt.plot(x.ravel(), y.ravel(), '.')
```

## Poincaré Sections

Wiki Fig. 1:

0.5, 1/16, 0.1, 2.5, 2.0
a/2, b/4, lam?, f0, w,

```{code-cell} ipython3
# Wikipedia Fig 1.
alpha = 0.5
beta = 0.0625
gamma = 0.1
F0 = 2.5
w = 2.0
d = Duffing(a=-2*alpha, b=4*beta, lam=gamma/2, f=F0, w=w)
d.show()
data = Duffing(a=-2*alpha, b=4*beta, lam=gamma/2, f=F0, w=w).plot_poincare()
```

```{code-cell} ipython3
# Wikipedia Fig 2.
raise NotImplementedError
# I don't know how to reproduce this
alpha = 1
beta = 5
delta = 0.02
#beta, delta = delta, beta
gamma = 8
w = 0.5

d = Duffing(a=-alpha, b=beta, lam=delta/2, f=gamma, w=w, max_steps=100)
d.show()
data0 = d.plot_poincare(y0=(1.65, 0.0), interact=False, frames=100, N=4000, method='BDF')

#d = Duffing(a=-2*alpha, b=4*beta, lam=delta/2, f=gamma, w=w, max_steps=1000)
#d.show()
#data = d.plot_poincare(y0=(1.65, 0.0), interact=False, frames=100, N=4000)
```

```{code-cell} ipython3
d = Duffing(a=-alpha, b=beta, lam=delta/2, f=gamma, w=w, max_steps=1)
data1 = d.plot_poincare(y0=(1.0, 1.0), frames=1, N=1000, method='BDF')
```

```{code-cell} ipython3
d = Duffing(a=-alpha, b=beta, lam=delta/2, f=gamma, w=w, max_steps=100)
data100 = d.plot_poincare(y0=(1.0, 1.0), frames=1, N=1000-100, method='BDF')
```

```{code-cell} ipython3
data1[0][100, 0], data100[0][0, 0]
```

```{code-cell} ipython3
d.plot_poincare(data=data0, interact=True)
```

```{code-cell} ipython3
# Wikipedia Fig 2.
# I don't know how to reproduce this
alpha = 1
beta = 5
delta = 0.02
#beta, delta = delta, beta
gamma = 8
w = 0.5

alpha = 1
gamma = 1
beta = 40.0
delta = 0.02
w = 0.5

@interact(delta=(0, 100, 0.01), beta=(0, 100, 0.01), w=(0.1, 2.0), s=(1, 1000, 10), N=(100, 10000, 100))
def go(N=400, s=100, beta=40.0, delta=0.02, w=0.5):
    alpha, gamma = -1, 1
    d = Duffing(a=alpha, b=beta, lam=delta/2, f=gamma, w=w, max_steps=s)
    d.show()
    data = d.plot_poincare(y0=(1, 1.0), interact=True, frames=1000, N=N)
```

```{code-cell} ipython3
# Wikipedia Fig 3.
alpha = 1
beta = 1
delta = 0.02
gamma = 3
w = 1
d = Duffing(a=-alpha, b=beta, lam=delta/2, f=gamma, w=w, phi0=-np.pi/2)
d.show()
data = d.get_poincare(y0=(1.0, 1.0), N=10000, frames=100)
```

```{code-cell} ipython3
d = Duffing(a=-1, b=40.0, lam=0.01, f=1.0, w=0.5, max_steps=1)
data = d.plot_poincare(y0=(1,1), interact=True, N=400, frames=100)
```

```{code-cell} ipython3
plt.figure(figsize=(10,10))
data = d.plot_poincare(data=data, interact=True, normalize=False);
```

```{code-cell} ipython3
from ipywidgets import interact
alpha = 0.5
beta = 0.0625
gamma = 0.1
F0 = 2.5
w = 2.0
self = d = Duffing(a=-2*alpha, b=4*beta, #
                   lam=gamma/2, f=F0, w=w)
kw = {}
locals().update(y0=(0, 0), N=20000, frames=200, ms=0.5)
"""
We evolve for `N` full cycles with `frames` points within each cycle.
"""
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# Get initial point after self.max_steps cycles to reach the attractor.
sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, self.T*self.max_steps), **kw)
y0 = sol.y[:, -1]
ts = (self.T * np.arange(N*frames)) / frames
sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, ts.max()), t_eval=ts, method='BDF', **kw)
sol.y = sol.y.reshape(2, N, frames)
q, dq = sol.y

#plt.plot(q[:, frame], dq[:, frame], '.', ms=ms)
```

```{code-cell} ipython3
from ipywidgets import interact
alpha = 0.5
beta = 0.0625
gamma = 0.1
F0 = 2.5
w = 2.0
self = d = Duffing(a=-2*alpha, b=4*beta, #
                   lam=gamma/2, f=F0, w=w)
kw = {}
locals().update(y0=(0, 0), N=20000, frames=200, ms=0.5)
"""
We evolve for `N` full cycles with `frames` points within each cycle.
"""
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# Get initial point after self.max_steps cycles to reach the attractor.
sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, self.T*self.max_steps), **kw)
y0 = sol.y[:, -1]
ts = (self.T * np.arange(N*frames)) / frames
sol = solve_ivp(self.compute_dy_dt, y0=y0, t_span=(0, ts.max()), t_eval=ts, method='BDF', **kw)
sol.y = sol.y.reshape(2, N, frames)
q, dq = sol.y

#plt.plot(q[:, frame], dq[:, frame], '.', ms=ms)
```

```{code-cell} ipython3
@interact(frame=(0,frames-1), points=(0,N-1))
def go(frame=0, points=N-1):
    #print(frame)
    plt.plot(q[:points, frame], dq[:points, frame], '.', ms=ms)
    plt.axis([q.min(), q.max(), dq.min(), dq.max()])
```

```{code-cell} ipython3
d.plot_poincare(cycles=10000)
#d=Duffing(a=-1, b=1, lam=0.3/2, f=0.6, w=1.5, max_steps=100);d.plot_poincare(cycles=10000)
```

# Dimensionless Form

WLOG, we can choose units so that $\alpha = -1$ (time) and $f=1$ (distance) so that there are only three parameters:

\begin{gather}
  \ddot{q} + 2\lambda \dot{q} - q + \beta q^3 = \cos(\omega t)
\end{gather}

+++

## Linear Response

[Duffing equation]: <https://en.wikipedia.org/wiki/Duffing_equation>

```{code-cell} ipython3
def potential_well(x, a, b):
    return - a * x ** 2 + b * x ** 4

def draw_potential_well(potential_well, F, x0, xmin, xmax, ax, safety_factor=0.03):
    x = np.linspace(xmin, xmax, 200)
    y = potential_well(x)
    ymin, ymax = min(y), max(y)
    ymin -= (ymax - ymin) * safety_factor
    ymax += (ymax - ymin) * safety_factor
    
    ax.plot(x, y, color='gray')

    arrow_props = {'width': (ymax-ymin) * 5e-3, 'head_width': (ymax-ymin) * 2e-2, 
                   'head_length': (xmax-xmin) * 2e-2, 'length_includes_head': True,
                   'facecolor': '#4a5a90', 'edgecolor': 'none'}

    ax.arrow(x0, potential_well(x0), F, 0, **arrow_props)
    
    ax.scatter(x0, potential_well(x0), color='#938fba', s=100)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax

# Code modified from https://github.com/vkulkar/Duffing by Vikram Kulkarni

import numpy as np
import matplotlib.pyplot as plt

# parameters (mass = 1)
a, b = 0.5, 1/16 # potential coefficients
gamma = 0.1 # damping coefficient
F_0 = 2.5 # driving force
omega = 2.0 # driving angular frequency
period = 2*np.pi/omega
cycles, steps_per_cycle = 100000, 65
h = period/steps_per_cycle # time step

# length of the simulation
T = period * cycles
t = np.arange(0,T,h)

def x_2(x,v):    return -gamma*v + 2.0*a*x - 4.0*b*x*x*x
def x_3(x2,x,v):     return -gamma*x2 + 2.0*a*v -12.0*b*x*x*v
def x_4(x3,x2,x,v):    return -gamma*x3 + 2.0*a*x2 -12.0*b*x*x*x2 - 24.0*b*v*v*x
def x_5(x4,x3,x2,x,v):    return -gamma*x4 + 2*a*x3 -12.0*b*(x*x*x3 + 2.0*x2*x*v) -24.0*b*(v*v*v+2*x*v*x2)

# Trigonometric terms in derivatives
x2F =  F_0*np.cos(omega*t)
x3F = -F_0*omega*np.sin(omega*t)
x4F = -F_0*omega*omega*np.cos(omega*t)
x5F =  F_0*omega*omega*omega*np.sin(omega*t)

# Taylor series coefficients
coef1 = 1/2  *h**2
coef2 = 1/6  *h**3
coef3 = 1/24 *h**4
coef4 = 1/120*h**5

# initial conditions
x, v = 0.5, 0.0

position = np.zeros(len(t))
velocity = np.zeros(len(t))
position[0] = x

for i in range(1,len(t)):
    d2 = x_2(x,v) + x2F[i]
    d3 = x_3(d2,x,v) + x3F[i]
    d4 = x_4(d3,d2,x,v) + x4F[i]
    d5 = x_5(d4,d3,d2,x,v) + x5F[i]
    # Taylor series expansion for x,v. Order h^5
    x += v*h + coef1*d2 + coef2*d3 + coef3*d4 + coef4*d5
    v += d2*h + coef1*d3 + coef2*d4 + coef3*d5
    position[i] = x
    velocity[i] = v

def get_lims(tensor, safety_factor = 0.03):
    if tensor.shape[1] != 2:
        tensor = tensor.T
    xmin, xmax = min(tensor[:,0]), max(tensor[:,0])
    ymin, ymax = min(tensor[:,1]), max(tensor[:,1])
    xmin -= (xmax - xmin) * safety_factor
    xmax += (xmax - xmin) * safety_factor
    ymin -= (ymax - ymin) * safety_factor
    ymax += (ymax - ymin) * safety_factor
    return xmin, xmax, ymin, ymax

def plotting(trail, tmin, tmax, t, position, velocity):
    xmin, xmax, ymin, ymax = get_lims(np.array([position, velocity]))

    fig, axs = plt.subplot_mosaic("AAAAAB;AAAAAC;AAAAAC;AAAAAD", figsize=(20, 16))

    # Poincare plot
    ax=axs['A']
    poincare_plot = np.array([position, velocity]).T[(tmax%steps_per_cycle)::steps_per_cycle,:]
    ax.scatter(poincare_plot[:,0],poincare_plot[:,1], color='#938fba', s=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('$x$',{'fontsize':16})
    ax.set_ylabel('$\dot x$',{'fontsize':16})
    # ax.set_title(r'Poincare Plot (Phase space at time = $\frac{2\pi N}{\omega}$, N = 1,2,3...)',{'fontsize':16})
    ax.tick_params(axis='both',labelsize=16)
    
    # Vector field plot
    x_axis = np.linspace(xmin, xmax, 20)
    y_axis = np.linspace(ymin, ymax, 20)
    x_values, y_values = np.meshgrid(x_axis, y_axis)
    
    dx = 1.0*y_values
    dy = x_2(x_values, y_values) + x2F[tmax]
    arrow_lengths = np.sqrt(dx**2 + dy**2)
    alpha_values = 1 - (arrow_lengths / np.max(arrow_lengths))**0.4
    ax.quiver(x_values, y_values, dx, dy, color='blue', linewidth=0.5, alpha=alpha_values)

    # Potential well plot
    ax = axs['B']
    draw_potential_well(potential_well=(lambda x: potential_well(x, a, b)), 
                        F=x2F[tmax], x0=position[tmax], xmin=xmin, xmax=xmax, ax=ax, safety_factor=0.03)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('$x$',{'fontsize':16})
    ax.set_ylabel('$V(x)$',{'fontsize':16})

    # Trajectory plot
    ax = axs['C']
    ax.plot(position[max(0, tmin):tmax], t[max(0, tmin):tmax], color='#4a5a90', linewidth=1)
    ax.scatter(position[tmax], t[tmax], color='#938fba')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim((tmin-2)*h, (tmax+2)*h)
    # ax.set_title('Trajectory of the oscillator',{'fontsize':16})
    ax.set_xlabel('$x$',{'fontsize':16})
    ax.set_ylabel('$t$',{'fontsize':16})
    ax.tick_params(axis='both',labelsize=16)

    # Phase space plot
    ax=axs['D']
    for j in range(max(0, tmin), tmax):
        alpha = (j - (tmax - trail)) / trail
        ax.plot(position[j-1:j+1], velocity[j-1:j+1], '.-', markersize=2, color='#4a5a90', alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_title('Phase space',{'fontsize':16})
    ax.set_xlim([-4.5,4.5])
    ax.set_xlabel('$x$',{'fontsize':16})
    ax.set_ylabel('$\dot x$',{'fontsize':16})
    ax.tick_params(axis='both',labelsize=16)

    fig.suptitle(fr'Duffing, $(\alpha, \beta, \gamma, F_0, \omega) = ({a}, {b}, {gamma}, {F_0}, {omega})$', fontsize=20,y=0.92)
    return fig, ax
```

```{code-cell} ipython3
trail = 3 * steps_per_cycle
i = 0
tmax = 0
tmin = tmax-trail
plotting(trail=trail, tmin=tmin, tmax=tmax, t=t, position=position, velocity=velocity)
```

```{code-cell} ipython3
import imageio.v3 as iio
import os
from natsort import natsorted
import moviepy.editor as mp

def export_to_video(dir_paths, fps=12):
    for dir_path in dir_paths:
        file_names = natsorted((fn for fn in os.listdir(dir_path) if fn.endswith('.png')))

        # Create a list of image files and set the frame rate
        images = []

        # Iterate over the file names and append the images to the list
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            images.append(iio.imread(file_path))

        filename = dir_path[2:]
        iio.imwrite(f"{filename}.gif", images, duration=1000/fps, rewind=True)
        clip = mp.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(f"{filename}.mp4")
    return

from tqdm import trange
import os
for i, tmax in enumerate(trange(0, 10 * steps_per_cycle)):
    trail = 3 * steps_per_cycle
    tmin = tmax-trail
    fig, ax = plotting(trail=trail, tmin=tmin, tmax=tmax, t=t, position=position, velocity=velocity)
    
    dir_path = "./duffing"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig.savefig(f"{dir_path}/{i}.png")
    plt.close()

export_to_video(["./duffing"], fps=12)
```

```{code-cell} ipython3

```

[Poincaré map]: <https://en.wikipedia.org/wiki/Poincar%C3%A9_map>
