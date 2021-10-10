import numpy as np
import scipy as sp
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

#--------------------------------------------------------------------
def lorentzfunc(x,A,c,gamma):
    return A/(1+((x-c)/gamma)**2)



#--------------------------------------------------------------------


def sq(time_list, obs_data, fitted_data):
    if len(time_list)!=len(obs_data) or len(time_list)!=len(fitted_data):
        print("You don't have data for all input time frame.")
        return 0
    else:
        chi_sqr = 0
        for i in range(len(time_list)):
            chi_sqr += (obs_data[i] - fitted_data[i])**2 / fitted_data[i]

        return chi_sqr

# Newton's law model
def DZ_dt_Newton(Z, t,args):
    h = args[0]
    g = args[1]
    b = args[2]
    return [Z[1], -Z[1]**2/Z[0] - g +g*h/Z[0] -b*Z[1]/Z[0] ]


def Newton_model(time_list, h, b, g=980):
    
    params = (h,g,b)
    # solve Newton model:
    t_soln = time_list
    Z_soln_Newton = sp.integrate.odeint(DZ_dt_Newton, [0.02, 0], t_soln, args=(params,))   
    
    z_soln_Newton = Z_soln_Newton[:,0]      # fluid height

    return z_soln_Newton

def DZ_dt_Lor(Z, t,args):
    
    Omeg = args[2]
    if Z[1]>0:
        return [Z[1], 1/Z[0] - 1 -Omeg*Z[1] - (Z[1])**2/Z[0]]
    else:
        return [Z[1], 1/Z[0] - 1 -Omeg*Z[1] ]

    
def Lorenceu_model(time_list, h, omega, g=980):
    
    params = (h,g,omega)
    
    t_soln = time_list
    Z_soln = sp.integrate.odeint(DZ_dt_Lor, [0.02, 0.00], t_soln, args=(params,))   
    
    z_soln = Z_soln[:,0] * h     # fluid height
 
    return z_soln

# Finding optimum b
def minimizing_chi_sqr(time_list, obs_data, h, b_guess, limit=12):
    b_list = np.linspace(b_guess-limit, b_guess+limit, 20*limit+10)
    chi_list = np.zeros(len(b_list))

    for i in range(len(b_list)):
        fitted_data = Newton_model(time_list, h, b_list[i])
        chi_list[i] = sq(time_list, obs_data, fitted_data)

    b_opt_index = np.argmin(chi_list)

    return b_list[b_opt_index]

# Finding optimum omega
def min_chi_sqr(time_list, obs_data, h, o_guess, limit=0.15):
    o_list = np.linspace(o_guess-limit, o_guess+limit+0.1, 30)
    chi_list = np.zeros(len(o_list))

    for i in range(len(o_list)):
        fitted_data = Lorenceu_model(time_list, h, o_list[i])
        chi_list[i] = sq(time_list, obs_data, fitted_data)
        

    o_opt_index = np.argmin(chi_list)

    return o_list[o_opt_index]




# user-modified area: our data:

#filename= 'set7.dat'
filename='Set-7_(2)Eth.txt' #------------------------------
#timeshift = 0.5333333333333333 # --shifted so that t=0 is the time in video that the cap is released.
timeshift = 0.067   #-------------------------------
data = np.genfromtxt(filename,delimiter='\t',skip_header=2 )
time_data = data[:,0]
z_data = data[:,1]

time_data_clean = time_data[np.isfinite(z_data)]-timeshift # cleaned data includes the timeshift, and removes any instances of infinite.

z_data_clean = z_data[np.isfinite(z_data)]

# cleansing data to start rise from t=0,z=0
time_data_clean=time_data_clean[2:]   # ----------------
z_data_clean=z_data_clean[2:] * 100   # to cm -----------

plt.plot(time_data_clean,z_data_clean,'r.')
plt.xlabel('time in s ',fontsize=15)
plt.ylabel('fluid level in cm ',fontsize=15)
plt.show()

# calculate the initial starting level of the fluid in the submersed straw using fluid statics:
rho = 0.894 # g/cm^3--------------------------
eta = 1.095e-3 # Pa*sec
g= 980
P_atm = 1030 # g/cm^2
h = 8 # cm -------------------------
r=1.1/2  #cm -----------------
H = 13.9  # length of straw, cm



# fiting newton model
time_axis1=time_data_clean
z_data1=z_data_clean

bl = minimizing_chi_sqr(time_axis1,z_data1,h,22)

plt.plot(time_axis1,z_data1,'b.',label=f'Data (h={h} cm)') 
z_soln_Newton=Newton_model(time_axis1,h,bl)
print("b = ",bl)
print("The damping co-efficient b'=",bl*rho*math.pi*r**2)
plt.plot(time_axis1,z_soln_Newton,'r',label='Fitted Newtonian model')



# fiting lorenceu model
time_Lor=time_axis1*(h*1e-2/9.8)**(-0.5)   # dimensionless data

ol=min_chi_sqr(time_Lor, z_data1, h, 0.16)

z_soln_Lor=Lorenceu_model(time_Lor, h, ol)
print("The found value of omega =",ol)
omeg = 16*eta*(9.2e-2)**0.5/(rho*1000*(0.010/2)**2 * 9.8**0.5)
print("Theoretical value of omega =",omeg)
plt.plot(time_Lor*(h*1e-2/9.8)**0.5,z_soln_Lor,'g',label='Fitted Lorenceau model')
plt.xlabel('time (sec)',fontsize=15)
plt.ylabel('fluid level (cm)',fontsize=15)
plt.legend()
plt.show()


t_soln_sec = time_Lor*(h*1e-2/9.8)**0.5
plt.clf()
z_soln = z_data1

# For the original data-----------------------------
# perform the discrete Fourier transform of the data

Z_FFT = np.fft.fft(z_soln-np.mean(z_soln))
z2 = Z_FFT * np.conjugate(Z_FFT)
pow = abs(z2[1:len(Z_FFT)//2] + z2[:len(Z_FFT)//2:-1])
pow = pow/np.max(pow)
DT = t_soln_sec[1]-t_soln_sec[0]   # sample time
freq = (np.fft.fftfreq(t_soln_sec.shape[0])/DT)[1:len(Z_FFT)//2]

# check the power spectrum:

plt.plot(freq,pow,'r.-')

plt.xlabel('frequency',fontsize=15)
plt.ylabel('Power density',fontsize=15)

plt.title('Power spectrum curve (Original data)',fontsize=20)
plt.xlim([0,6])
# plt.savefig('power spectrum.png',dpi=400)
plt.show()



f_est = 1/(2*np.pi)*np.sqrt(g/h)
print('Small oscillation frequency = %2.3f Hz'%(f_est))

# For the fitted lorenceu data -----------------------
time_lFit = np.arange(0,300,0.01)
t_soln_sec = time_lFit*(h*1e-2/9.8)**0.5
z_soln= Lorenceu_model(time_lFit, h, ol)
Z_FFT = np.fft.fft(z_soln-np.mean(z_soln))
z2 = Z_FFT * np.conjugate(Z_FFT)
pow = abs(z2[1:len(Z_FFT)//2] + z2[:len(Z_FFT)//2:-1])
pow = pow/np.max(pow)
DT = t_soln_sec[1]-t_soln_sec[0]   # sample time
freq1 = (np.fft.fftfreq(t_soln_sec.shape[0])/DT)[1:len(Z_FFT)//2]

# check the power spectrum:

plt.plot(freq1,pow,'r.-')

plt.xlabel('frequency',fontsize=15)
plt.ylabel('Power density',fontsize=15)

plt.title('Power spectrum curve (Fitted Lorenceu data)',fontsize=20)
plt.xlim([0,6])
# plt.savefig('power spectrum.png',dpi=400)
plt.show()

m = np.argmax(pow)
#print("Frequency from the Lorenceu data power curve: ",freq1[m],'Hz')

# LORENTZIAN FITTING
#Experimental x and y data points    

 
#Plot experimental data points
plt.plot(freq1, pow, 'bo', label=f'Lorenceu fitted data (h={h} cm)')
 
# Initial guess for the parameters
#initialGuess = [1.0,1.0]    
 
#Perform the curve-fit
popt, pcov = curve_fit(lorentzfunc, freq1, pow)
print("Parameters:",popt)
pop=popt
print("Frequency from the fitted data power curve: ",pop[1],'Hz') 

#Plot the fitted function
plt.plot(freq1, lorentzfunc(freq1, *popt), 'r', label='fit params: A=%5.3f, $x_0$=%5.3f, $\gamma$=%5.3f' % tuple(popt))
plt.xlim([0,6])
plt.xlabel('frequency',fontsize=15)
plt.ylabel('Power density',fontsize=15)
plt.legend(title='Lorentzian fit',fontsize=8)
plt.show()

print("error in frequency calculation = ",(f_est - pop[1]) * 100/f_est)
