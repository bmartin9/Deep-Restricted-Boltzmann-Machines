"""
Compute the thermodynamics produced by Holographic DBMs
"""

import numpy as np
import pandas
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '/Users/Brendan/FifthYear/Project/Version2')
import Thermo_Functions

L = 32
h1 = 16
h2 = 8
num_samples = 5000
MCMC_num_samples = 10000
hol_numsweeps = 5000
MCMC_numsweeps = 100000
numtemps=3
temperature_list = [2.1,2.269185314,2.4]

MCMC_array = np.zeros((numtemps,MCMC_num_samples,L,L))
Hol_array = np.zeros((numtemps,num_samples,L,L))
Hol_h1_array = np.zeros((numtemps,num_samples,h1,h1))
Hol_h2_array = np.zeros((numtemps,num_samples,h2,h2))

df = pandas.read_csv('experiment3_visible_units210.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,L,L))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,L,L))
Hol_array[0]= ising_data

df = pandas.read_csv('experiment3_visible_units2269.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,L,L))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,L,L))
Hol_array[1]= ising_data

df = pandas.read_csv('experiment3_visible_units240.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,L,L))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,L,L))
Hol_array[2]= ising_data

df = pandas.read_csv('/Users/Brendan/FifthYear/Project/Version2/L32_100KIsing_data_210.csv')
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(MCMC_numsweeps,L,L))
ising_data = ising_data[:MCMC_num_samples]
ising_data = np.reshape(ising_data,(MCMC_num_samples,L,L))
MCMC_array[0]= ising_data

df = pandas.read_csv('/Users/Brendan/FifthYear/Project/Version2/L32_100KIsing_data_2269.csv')
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(MCMC_numsweeps,L,L))
ising_data = ising_data[:MCMC_num_samples]
ising_data = np.reshape(ising_data,(MCMC_num_samples,L,L))
MCMC_array[1]= ising_data

df = pandas.read_csv('/Users/Brendan/FifthYear/Project/Version2/L32_100KIsing_data_240.csv')
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(MCMC_numsweeps,L,L))
ising_data = ising_data[:MCMC_num_samples]
ising_data = np.reshape(ising_data,(MCMC_num_samples,L,L))
MCMC_array[2]= ising_data


df = pandas.read_csv('experiment3_h1_units210.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h1,h1))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h1,h1))
Hol_h1_array[0]= ising_data

df = pandas.read_csv('experiment3_h1_units2269.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h1,h1))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h1,h1))
Hol_h1_array[1]= ising_data

df = pandas.read_csv('experiment3_h1_units240.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h1,h1))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h1,h1))
Hol_h1_array[2]= ising_data

df = pandas.read_csv('experiment3_h2_units210.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h2,h2))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h2,h2))
Hol_h2_array[0]= ising_data

df = pandas.read_csv('experiment3_h2_units2269.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h2,h2))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h2,h2))
Hol_h2_array[1]= ising_data

df = pandas.read_csv('experiment3_h2_units240.csv',index_col=0)#read in the ising data from your csv file
ising_data = df.values#convert to numpy array
ising_data = ising_data.astype('float32')
ising_data = np.reshape(ising_data,(hol_numsweeps,h2,h2))
ising_data = ising_data[:num_samples]
ising_data = np.reshape(ising_data,(num_samples,h2,h2))
Hol_h2_array[2]= ising_data


MCMC_plot_list = Thermo_Functions.raw_variables(MCMC_array,L,temperature_list,numtemps)
RBM_plot_lists = Thermo_Functions.raw_variables(Hol_array,L,temperature_list,numtemps)
h1_RBM_plot_lists = Thermo_Functions.raw_variables(Hol_h1_array,h1,temperature_list,numtemps)
h2_RBM_plot_lists = Thermo_Functions.raw_variables(Hol_h2_array,h2,temperature_list,numtemps)



#do plots
MCMCplt1 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[1],MCMC_plot_list[5],'b','MCMC',mark="*") #mag plot of MCMC
shallow_E = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[1],RBM_plot_lists[5],col='r',lab='visible',mark="o")
Thermo_Functions.thermodynamic_plot(temperature_list,h1_RBM_plot_lists[1],h1_RBM_plot_lists[5],col='g',lab='first hidden',mark="v")
Thermo_Functions.thermodynamic_plot(temperature_list,h2_RBM_plot_lists[1],h2_RBM_plot_lists[5],col='m',lab='second hidden',mark="p")
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.legend()
plt.savefig('E.pdf')
plt.close()

MCMCplt2 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[2],MCMC_plot_list[6],'b','MCMC',mark="*") #mag plot of MCMC
shallow_M = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[2],RBM_plot_lists[6],col='r',lab='visible',mark="o")
Thermo_Functions.thermodynamic_plot(temperature_list,h1_RBM_plot_lists[2],h1_RBM_plot_lists[6],col='g',lab='first hidden',mark="v")
Thermo_Functions.thermodynamic_plot(temperature_list,h2_RBM_plot_lists[2],h2_RBM_plot_lists[6],col='m',lab='second hidden',mark="p")
plt.xlabel('Temperature')
plt.ylabel('Magnetisation')
plt.legend()
plt.savefig('M.pdf')
plt.close()

MCMCplt3 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[3],MCMC_plot_list[7],'b','MCMC',mark="*") #mag plot of MCMC
shallow_C = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[3],RBM_plot_lists[7],col='r',lab='visible',mark="o")
Thermo_Functions.thermodynamic_plot(temperature_list,h1_RBM_plot_lists[3],h1_RBM_plot_lists[7],col='g',lab='first hidden',mark="v")
Thermo_Functions.thermodynamic_plot(temperature_list,h2_RBM_plot_lists[3],h2_RBM_plot_lists[7],col='m',lab='second hidden',mark="p")
plt.xlabel('Temperature')
plt.ylabel('Heat Capacity')
plt.legend()
plt.savefig('C.pdf')
plt.close()

MCMCplt4 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[4],MCMC_plot_list[8],'b','MCMC',mark="*") #mag plot of MCMC
shallow_X = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[4],RBM_plot_lists[8],col='r',lab='visible',mark="o")
Thermo_Functions.thermodynamic_plot(temperature_list,h1_RBM_plot_lists[4],h1_RBM_plot_lists[8],col='g',lab='first hidden',mark="v")
Thermo_Functions.thermodynamic_plot(temperature_list,h2_RBM_plot_lists[4],h2_RBM_plot_lists[8],col='m',lab='second hidden',mark="p")
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')
plt.legend()
plt.savefig('X.pdf')
plt.close()
