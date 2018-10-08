

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma


def func(x,a):
    return (a**x)*np.exp(-a)/gamma(x+1)
    
def func_rc0(x,a0,a1,a2):
    return a0 + a1*x + a2*x*x
    

N = [6,5,8,7,6,5,7,10,11,10,9,8,4,12,10,6,7,7,4,4,5,13,5,13,8,10,13,8,8,9,6,7,7,12,7,5,5,13,8,9,8,6,3,8,8,\
10,11,7,5,6,8,10,15,5,3,9,12,7,12,10,8,6,6,11,13,5,6,4,6,3,6,7,10,9,3,10,9,7,4,4,7,4,9,5,10,5,7,9,7,13,8,6,\
7,7,11,7,5,6,7,5,3,8,9,12,4,9,4,8,8,12,5,11,3,9,3,8,11,3,3,10,7,5,10,8,7,8,8,6,8,7,10,9,8,6,7,8,2,6,5,13,\
10,5,8,8,10,7,11,12,8,5,7,5,8,10,7,7,11,7,5,10,12,7,4,8,6,5,8,4,6,5,7,6,7,10,7,5,7,11,9,4,4,6,7,6,7,12,12,7,\
7,8,14,6,3,6,7,9,9,8,11,4]

print (len(N),min(N),max(N), sum(N))


mu = float(sum(N))/float(len(N))
sd = np.sqrt(mu)

n =plt.hist(N, bins=15, range = [0.5,15.5], rwidth=0.9)
un = np.ones(15)*sd

kf = np.linspace(0.5,15.5,100)
Pois = len(N)*(mu**kf)*np.exp(-mu)/gamma(kf+1)
plt.plot(kf,Pois,color= 'k', linestyle = 'dashed')


x1 = np.linspace(1,15,15)
popt,pcov = curve_fit(func,x1,n[0][:])
Pois1 = len(N)*(popt**kf)*np.exp(-popt)/gamma(kf+1)

plt.plot(kf,Pois1,color = 'r',linestyle ='-.')
plt.axis([0,16,0,40])   #plt.axis([xmin,xmax,ymin,ymax])
plt.xlabel('$k$')
plt.ylabel('$P_{\mu}(k)$')
plt.title('Poisson distribution')

m_e = np.array([11.2,17.2,22.6,25.3,29.1])
m_f = np.array([12.0,19.1,25.5,28.5,33.0])
m_tot = (m_f - m_e)*0.001

tk = np.array([1.44,2.32,3.50,4.09,5.11])*0.001
utk = 0.0001*np.array(np.ones(5))

N_20 = np.array([1505.,2015.,2024.,2023.,1968.])
N_b = np.array([203.,203.,203.,203.,203.])
Dt = 20.*60.


r_c = (N_20 - N_b)/(Dt*m_tot)


popt_rc,pcov_rc = curve_fit(func_rc0,tk,r_c)
perr_rc = np.sqrt(np.diag(pcov_rc))


plt.figure(2)
plt.plot(tk,r_c)

#
#
#mu_c = popt*rc0/(eph*G*f*rc)


