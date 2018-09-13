import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b):
    return np.sqrt(2*(x-b)/a)



t = np.array([0.200,0.285,0.350,0.400,0.450,0.495,0.535,0.550])
s = np.array([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.5])
ut =np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
#ut =10*np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
us = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])


plt.errorbar(s,t,xerr=us,yerr=ut,linestyle='none',\
             elinewidth=1,ecolor='b',capsize=2,marker = 'o', markersize=3)

plt.axis([0,1.8,0,0.65])   #plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel('s (m)')
plt.ylabel('t (s)')
plt.title('MEL COSTA')


popt,pcov = curve_fit(func,s,t,p0=[10,0], sigma = ut, absolute_sigma =True)
perr = np.sqrt(np.diag(pcov))

print('\ng :{0:5.2f} +/-{1:5.2f} m/s^2'.format(popt[0],perr[0]))
print('\ns0 :{0:5.2f} +/-{1:5.2f} m/s^2'.format(popt[1],perr[1]))

x = np.linspace(0.1,1.6,100)
y = func(x,popt[0],popt[1])
plt.plot(x,y,'r',linewidth = 1)

plt.savefig('np_python_assignment2.pdf')
plt.show()



chi_sq = sum(((func(s,popt[0],popt[1])-t)/ut)**2)
DFE = len(s)-len(popt)
RMSE = np.sqrt(chi_sq/DFE)

print('\nRoot mean square error :{0:5.2f}'.format(RMSE))


