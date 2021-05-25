''' Run a MCMC monte carlo simulation for 2D Ising model. The parameters of the MCMC simulation (e.g. numsweeps) are defined here.
This file outputs a list of square microstate data for each temperature. Each element of the outputed list contains an array of
shape (numsweeps,L,L) of microstate data for that temperature.'''

import numpy as np
import Thermo_Functions
import csv
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import random
import math

np.set_printoptions(threshold=sys.maxsize)

def monte_carlo(L,numsweeps,RELAX_SWEEPS,temperature):


    lattice = 2 * np.random.randint(2, size=(L,L)) -1 #initialise lattice with 1s and -1s


    lattice_storage = np.zeros(shape=(numsweeps,L,L)) #numpy array that will store the microstates

    #lattice_storage[0,:,:] = lattice

    #a sweep involves changing a random spin and deciding whether to keep the resulting microstate, L*L times
    for sweep in range(numsweeps+RELAX_SWEEPS):
        start_time = time.time()
        '''perform the lattice update, which involves testing N randomly selected spins'''

        lattice = Thermo_Functions.lattice_update(lattice,L,temperature)

        #only store the lattice if thermal equilibrium has been reached
        if sweep>=RELAX_SWEEPS:
            lattice_storage[sweep-RELAX_SWEEPS,:,:] = lattice
        #print("Lattice Update time --- %s seconds ---" % (time.time() - start_time))

    return lattice_storage



def main():

    L = 32 # length of lattice
    N=L**2 #number of spins in lattice
    numsweeps = 100000 #number of microstates to store for each temperature
    RELAX_SWEEPS = 500 #number of sweeps to reach thermal equilibrim
    numtemps = 1# number of temperatures to survey
    #temperature_list = np.linspace(2.0,2.09,numtemps) #list of temperatures to sample
    #temperature_list = [2.1,2.15,2.2,2.25,2.269185314,2.3,2.35,2.4]
    temperature_list = [2.10]

    storage = []
    for temperature in temperature_list:
        microstate_data_at_temperature = monte_carlo(L,numsweeps,RELAX_SWEEPS,temperature)
        # where_minus = np.where(microstate_data_at_temperature == -1)
        # microstate_data_at_temperature[where_minus] = 0
        #print(microstate_data_at_temperature[-1])

        # plt.imshow(microstate_data_at_temperature[0], interpolation='nearest')
        # plt.show()

        #flatten each microstate array
        microstate_data_at_temperature = np.reshape(microstate_data_at_temperature,(numsweeps,N))
        storage.append(microstate_data_at_temperature)

    #store your Ising data in a csv file
    df = pd.DataFrame() #create a data frame to hold the data
    for i in range(len(storage)):
        df1 = pd.DataFrame(storage[i]) #create a data frame for each temperature
        df = df.append(df1,ignore_index=True) #add this data frame to df

    df.to_csv('L32_100KIsing_data_210.csv',index=False) #make csv file


main()
