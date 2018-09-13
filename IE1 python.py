
import matplotlib.pyplot as plt
from numpy import *


#distances in m, Force in N, uncertainities in m
#distances corrected negating initial error, adding depths and thickness
z_a = array([0.03865,0.03665,0.03465,0.03265,0.03065,0.02865,0.02665,0.02465,0.02265,0.02065,0.01865,0.01665])
z_r = array([0.04065,0.03865,0.03665,0.03465,0.03265,0.03065,0.02865,0.02665,0.02465,0.02265,0.02065,0.01965])


F_a = array([0,7,8,10,15,18,29,35,51,73,113,171])*0.0098
F_r = array([0,0,-5,-7,-8,-12,-15,-21,-30,-42,-60,-81])*0.0098



u_h12 = 0.0003
u_h1 = 0.00002
u_h2 = 0.00002
u_t = 0.0001

u_z = sqrt(u_h12**2 + u_h1**2 + u_h2**2 + u_t**2)

u_z4_a = 4*u_z/z_a**5

u_z4_r = 4*u_z/z_r**5

u_F = 0.001

#plt.errorbar(z_a,F_a,xerr=u_z4_a,yerr=u_F,)
#plt.errorbar(z_r,F_r,xerr=u_z4_r,yerr=u_F,)
plt.plot(1/(z_a**4),F_a)
#plt.plot(1/(z_r**4),F_r)


