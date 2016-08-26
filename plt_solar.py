import matplotlib.pyplot as plt
import numpy as np

wvlength1,power1 = np.loadtxt("/home/kyle_robertson/schoolwork/gradschool/nanowire/code/ASTMG173.csv",
                            skiprows=2,delimiter=",",usecols=(0,3),unpack=True)
freq1 = (2.998*10E8/10E-9)/wvlength1

freq2,power2 = np.loadtxt("/home/kyle_robertson/schoolwork/gradschool/nanowire/code/Input_sun_power.txt",
                            skiprows=1,unpack=True)
#plt.plot(freq1,power1,'b-',freq2,power2,'r-')
plt.figure(1)
plt.plot(freq1,power1,'b-')

plt.figure(2)
plt.plot(freq2,power2,'r-')
plt.show()
