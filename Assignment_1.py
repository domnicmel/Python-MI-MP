
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a):
    return a*x




r = np.array([1,2,3,4,5])
F = np.array([1.4,2.9,4.1,5.6,7.7])
ur = np.array([0.02,0.02,0.02,0.02,0.02])
uF = np.array([0.2,0.2,0.3,0.2,0.6])

plt.errorbar(r,F,xerr=ur,yerr=uF,linestyle='none',\
             elinewidth=1,ecolor='b',capsize=2)

plt.axis([0,5.5,0,10.0])   #plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel('r (cm)')
plt.ylabel('F (N)')
plt.title('MEL COSTA')


popt,pcov = curve_fit(func,r,F, sigma = uF, absolute_sigma =True)
perr = np.sqrt(np.diag(pcov))

print('\nk :{0:5.2f} +/-{1:5.2f} N/m'.format(popt[0],perr[0]))

x = np.linspace(0,5.5,2)
y = func(x,popt[0])
plt.plot(x,y,'r',linewidth = 1)

plt.savefig('np_opgave1.pdf')
plt.show()




