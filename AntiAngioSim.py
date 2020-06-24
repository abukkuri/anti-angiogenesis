import numpy as n
import pylab as plab
import scipy.integrate as integrate
import math as math

def dx(x,y):
    return alpha1*x*(1-x)-q1*x*y

def dy(x,y,z,v):
    return alpha2*y*(1-y/(1+gamma*z))-q2*x*y-q3*y*v+p2*y*z

def dz(y,z,w):
    return beta*y+alpha3*z*(1-z)-(p3*z*w)/(a3+z)

def dv(y,v):
    return uther*S1+r*y-d4*v

def dw(z,w):
    return vther*S2-(p5*z*w)/(a3+z)-d5*w

def derivs(state, t):

    #print t, state
    x,y,z,v,w = state
    deltax = dx(x,y)
    deltay = dy(x,y,z,v)
    deltaz = dz(y,z,w)
    deltav = dv(y,v)
    deltaw = dw(z,w)

    return deltax, deltay, deltaz, deltav, deltaw

t = n.arange(0,3000,.1)

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
S1 = 0.02
S2 = 0.07
uther = 0 #Change this for therapy
vther = 0 #Change this for therapy

#Initial Conditions
x0 = .8
y0 = .0006
z0 = 0
v0 = 0
w0 = 0

isv0 = [x0, y0, z0, v0, w0]  #The initial state vector
isv = integrate.odeint(derivs, isv0, t)
x = isv[:, 0]
y = isv[:, 1]
z = isv[:, 2]
v = isv[:, 3]
w = isv[:, 4]

plab.figure()
plab.plot(t, x, label = 'Host Cells')
plab.plot(t, y, label = 'Tumor Cells')
plab.plot(t, z, label = 'Endothelial Cells')
plab.plot(t, v, label = 'Effector Cells')
plab.plot(t, w, label = 'Anti-Angiogenesis')
plab.xlabel('Time')
plab.ylabel('Number of Cells')
plab.title('Cellular Dynamics')
plab.grid()
plab.legend()
plab.show()