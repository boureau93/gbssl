import numpy as np
from scipy.optimize import root_scalar as rs
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Times"],
    "font.size": 16})

def f(x,r,q,alpha):
    return x*(r+1./q)-r*np.log(q+2*x)-(1-r)*np.log(q+x)-np.log(alpha)

def f_prime(x,r,q,alpha):
    return r+1./q-r*(2/(q+2*x))-(1-r)/(q+x)

fig, ax = plt.subplots(2,sharex="col",figsize=(8,6))

r_range = np.arange(0.01,1.01,0.01)
q_range = [2,4,8,16]

for q in q_range:
    beta_range = []
    for r in r_range:
        beta_range.append(rs(f,(r,q,1.),x0=1.,fprime=f_prime,
            xtol=0.001,method="newton").root)
    ax[0].plot(r_range,beta_range,label="q = "+str(q))

for q in q_range:
    beta_range = []
    for r in r_range:
        beta_range.append(rs(f,(r,q,(1.+q)/(2.*q)),x0=1.,fprime=f_prime,
            xtol=0.001,method="newton").root)
    ax[1].plot(r_range,beta_range,label="q = "+str(q))

ax[0].set_ylabel(r"$\beta^{*}_{\gamma=1}$")
ax[0].set_xlabel(r"$r_{l}$")
ax[1].set_ylabel(r"$\beta^{*}_{\gamma=\frac{1+q}{2q}}$")
ax[1].set_xlabel(r"$r_{l}$")

ax[0].grid(True)
ax[1].grid(True)
ax[0].legend()
ax[1].legend()

plt.show()