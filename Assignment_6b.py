import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b):
    return a*x**b



F = np.array([0,7,8,10,15,18,29,35,51,73,113,171])*0.0098

z = np.array([0.03865,0.03665,0.03465,0.03265,0.03065,0.02865,0.02665,0.02465,0.02265,0.02065,0.01865,0.01665])
z4_1 = 1/z**4


mu_0 = 4*np.pi*1e-7
V = np.pi*0.005*0.005*0.005




u_h12 = 0.0003
u_h1 = 0.00002
u_h2 = 0.00002
u_t = 0.0001

uz = np.sqrt(u_h12**2 + u_h1**2 + u_h2**2 + u_t**2)

uz4_1 = 4*uz/z**5

uF = np.array([1,1,1,1,1,1,1,1,1,1,1,1])* 0.001


plt.errorbar(z4_1,F,xerr=uz4_1,yerr=uF,linestyle='none',\
             elinewidth=1,ecolor='b',capsize=2,marker = 'o', markersize=3)

plt.axis([0,1.5e7,0,1.85])   #plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel('1/z^4 (m^-4)')
plt.ylabel('F (N)')
plt.title('MEL COSTA')


popt,pcov = curve_fit(func,z,F, sigma = uF, absolute_sigma =True)
perr = np.sqrt(np.diag(pcov))

print('\na :{0:5.9f} +/-{1:5.9f} Nm^4'.format(popt[0],perr[0]))
print('\nb :{0:5.3f} +/-{1:5.3f} '.format(popt[1],perr[1]))

x = np.linspace(0.04,0.015,100)
y = func(x,popt[0],popt[1])
plt.plot(x**popt[1],y,'r',linewidth = 1)

plt.savefig('np_python_assignment6b.pdf')
plt.show()



chi_sq = sum(((func(z,popt[0],popt[1])-F)/uF)**2)
DFE = len(z)-len(popt)
RMSE = np.sqrt(chi_sq/DFE)
print('\nRoot mean square error :{0:5.2f}'.format(RMSE))


m = np.sqrt((2*np.pi*popt[0])/(3*mu_0))
um = 0.5*np.sqrt((2*np.pi)/(3*mu_0*popt[0]))*perr[0]
print('\nm :{0:5.3f} +/-{1:5.3f} Am^2'.format(m,um))

Br = mu_0*m/V
uBr = mu_0*um/V
print('\nBr :{0:5.3f} +/-{1:5.3f} A/m'.format(Br,uBr))