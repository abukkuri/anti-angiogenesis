from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

m = GEKKO() #initialize gekko
nt = 1010 #time steps

m.time = np.linspace(0,800,nt) #having nt linearly spaced time points between 0 and end time.

# Variables

#Initial conditions
x = m.Var(value=0.8)
y = m.Var(value=0.0006)
z = m.Var(value=0)
v = m.Var(value=0)
w = m.Var(value=0)
OF = m.Var(value=0)
uther = m.Var(value=0,lb=0,ub=1) #Control is initially 0 with a lower bound of 0 and an upper bound of 1
vther = m.Var(value=0,lb=0,ub=1) #Control is initially 0 with a lower bound of 0 and an upper bound of 1
p = np.zeros(nt) #mark final time point
p[-1] = 1.0 #all zeros except the end, which is 1
final = m.Param(value=p) #final depends on integration limits

#Parameter Values
alpha1 = 6.8*10**3
alpha2 = 0.01
alpha3 = 0.002
r = 0.002
p5 = 0.032
q1 = 7.2*10**(-3)
q2 = 7.2*10**(-4)
beta = 0.004
d4 = 0.0132
d5 = 0.136
q3 = 0.01
p3 = 1.8
p2 = 0.002
a3 = 0.49
gamma = 0.15
S1 = 0.017 #u: immunotherapy
S2 = .07 #v: anti-angio

alpha = 1.2
b1 = 1.5
b2 = 0
theta = 1.3
psi = 3

# Equations
m.Equation(x.dt() == alpha1*x*(1-x)-q1*x*y)
m.Equation(y.dt() == alpha2*y*(1-y/(1+gamma*z))-q2*x*y-q3*y*v+p2*y*z)
m.Equation(z.dt() == beta*y+alpha3*z*(1-z)-(p3*z*w)/(a3+z))
m.Equation(v.dt() == uther*S1+r*y-d4*v)
m.Equation(w.dt() == vther*S2-(p5*z*w)/(a3+z)-d5*w)

m.Equation(OF.dt() == alpha*v+theta*x-.5*b1*uther**2-b2*.5*vther**2-psi*y)

m.Obj(-OF*final) #Objective functional

m.options.IMODE = 6 #optimal control mode
m.solve(disp=False) #solve

plt.figure(figsize=(4,3)) #plot results
plt.subplot(2,1,1)
plt.plot(m.time,x.value,'k-',label=r'$X$')
plt.plot(m.time,y.value,'b-',label=r'$Y$')
plt.plot(m.time,z.value,'g-',label=r'$Z$')
plt.plot(m.time,v.value,'r-',label=r'$V$')
plt.plot(m.time,w.value,'y-',label=r'$W$')
plt.legend()
plt.ylabel('CV')
plt.subplot(2,1,2)
plt.plot(m.time,uther.value,'r--',label=r'$u$')
plt.plot(m.time,vther.value,'g--',label=r'$v$')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
