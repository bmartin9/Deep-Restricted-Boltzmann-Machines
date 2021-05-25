"""
Compute the thermodynamics produced by shallow fully connected RBMs
"""

import RBMClass
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import pandas
import time
import matplotlib.pyplot as plt
import NonTrainableIndices
import Thermo_Functions

def CSV_reader(file,N,numtemps,numsweeps):
    """
    Read in Ising data and convert to 0s and 1s. Shuffle Data.
    Output a Torch tensor

    Args:
        file (str): Ising data. Must be given in form 'file.csv'
        N (int): Length of each microstate

    Returns:
        Torch tensor of size (numtemps,numsweeps,N)
    """
    df = pandas.read_csv(file)#read in the ising data from your csv file
    ising_data = df.values#convert to numpy array
    ising_data = ising_data.astype('float32')
    ising_data = np.reshape(ising_data,(numtemps,numsweeps,N)) #reshape numpy array to separate out the data at different temps
    #change every -1 to a 0 in the data
    where_minus = np.where(ising_data == -1)
    ising_data[where_minus] = 0
    for i in range(numtemps):
        np.random.shuffle(ising_data[i])
    ising_tensor = torch.from_numpy(ising_data)
    return ising_tensor


L=16
N=16**2
nv = N
nh = nv
st_dev = 1/np.sqrt(nv+nh)
numsweeps = 50000
numtemps = 10
batch_size=10
train_size = 400
test_size = 400
num_batches = int(train_size/batch_size) #must always be whole number
num_epochs = 5
k=2 #number of Gibbs sampling steps
temperatures = torch.linspace(2.2,2.3,numtemps)

start_time = time.time()
ising_tensor = CSV_reader('Ising_data_narrow50K.csv',N,numtemps,numsweeps)
print("Read CSV --- %s seconds ---" % (time.time() - start_time))

dreams = torch.zeros(numtemps,test_size,N,requires_grad=False) #dreamed microstates

weight_mask = NonTrainableIndices.DIT_mask(L) #weight mask for DIT connections
v_mask = torch.zeros(nv,requires_grad=False)
h_mask = torch.zeros(nh,requires_grad=False)

recons = []
epoch_list = [z for z in range(num_epochs)]

for i in range(numtemps):
    shallowRBM = RBMClass.RBM(nv,nh)

    #initialise parameters according to a normal distribution around zero
    shallowRBM.W = nn.Parameter(torch.normal(torch.zeros(nh,nv),std = st_dev))
    shallowRBM.v_bias = nn.Parameter(torch.zeros(nv))
    shallowRBM.h_bias = nn.Parameter(torch.zeros(nh))

    train_set = ising_tensor[i,:train_size,:]
    train_set = torch.reshape(train_set,(num_batches,batch_size,N)) #separate training set into batches

    reconstruction_errors = []

    for epoch in range(num_epochs):
        start_time = time.time()
        batch_reconstruction = 0
        for mini in range(num_batches):
            shallowRBM.parameter_update(train_set[mini],k,option=False)
            mini_error = shallowRBM.batch_reconstruction_error(train_set[mini])
            batch_reconstruction+=mini_error
        print("Time per epoch --- %s seconds ---" % (time.time() - start_time))
        reconstruction_errors.append(batch_reconstruction)
        print('Epoch: '+str(epoch + 1))

    recons.append(reconstruction_errors)


    #sample from the model
    sample_tensor= torch.zeros(test_size,N,requires_grad=False) #tensor to hold dreamed states at this temperature
    test_set = ising_tensor[i,train_size:train_size+test_size,:]
    for j in range(test_size):
        test_state = test_set[j]
        sample = shallowRBM.Gibbs_v(test_state,k)
        sample = torch.where(sample==0,-torch.ones(N),sample)
        sample_tensor[j] = sample

    dreams[i] = sample_tensor
    print("temperature: "+str(i+1))

#reshape dreamed states to into square lattices
dreams = torch.reshape(dreams,(numtemps,test_size,L,L))
dreams = dreams.detach().numpy()

RBM_plot_lists = Thermo_Functions.raw_variables(dreams,L,temperatures,numtemps)

ising_tensor = torch.where(ising_tensor==0,-torch.ones(N),ising_tensor) #change the 0s back to -1
ising_tensor = torch.reshape(ising_tensor,(numtemps,numsweeps,L,L)) #convert MCMC data to square shape again
ising_array = ising_tensor.detach().numpy()
MCMC_plot_array = ising_array[:,:test_size,:,:] #no need to calculate the thermodynamics over 50000 microstates
MCMC_plot_list = Thermo_Functions.raw_variables(MCMC_plot_array,L,temperatures,numtemps)

temperature_list = temperatures.numpy()
#do plots
MCMCplt1 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[1],MCMC_plot_list[5],'b','MCMC',mark="*") #mag plot of MCMC
shallow_E = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[1],RBM_plot_lists[5],col='r',lab='RBM',mark="o")
plt.title('RBM vs MCMC',fontsize=18)
plt.xlabel('Temperature',fontsize=18)
plt.ylabel('Energy',fontsize=20)
plt.legend()
plt.savefig('E.pdf')
plt.close()

MCMCplt2 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[2],MCMC_plot_list[6],'b','MCMC',mark="*") #mag plot of MCMC
shallow_M = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[2],RBM_plot_lists[6],col='r',lab='RBM',mark="o")
plt.title('RBM vs MCMC',fontsize=18)
plt.xlabel('Temperature',fontsize=18)
plt.ylabel('Magnetisation',fontsize=18)
plt.legend()
plt.savefig('M.pdf')
plt.close()

MCMCplt3 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[3],MCMC_plot_list[7],'b','MCMC',mark="*") #mag plot of MCMC
shallow_C = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[3],RBM_plot_lists[7],col='r',lab='RBM',mark="o")
plt.title('RBM vs MCMC',fontsize=18)
plt.xlabel('Temperature',fontsize=18)
plt.ylabel('Heat Capacity',fontsize=18)
plt.legend()
plt.savefig('C.pdf')
plt.close()

MCMCplt4 = Thermo_Functions.thermodynamic_plot(temperature_list,MCMC_plot_list[4],MCMC_plot_list[8],'b','MCMC',mark="*") #mag plot of MCMC
shallow_X = Thermo_Functions.thermodynamic_plot(temperature_list,RBM_plot_lists[4],RBM_plot_lists[8],col='r',lab='RBM',mark="o")
plt.title('RBM vs MCMC',fontsize=18)
plt.xlabel('Temperature',fontsize=18)
plt.ylabel('Susceptibility',fontsize=18)
plt.legend()
plt.savefig('X.pdf')
plt.close()

# plt.plot(epoch_list,recons[0])
# plt.title('Reconstruction Errors',fontsize=18)
# plt.xlabel('Epoch',fontsize=18)
# plt.ylabel('Error',fontsize=18)
# plt.savefig('Recon22.pdf')
# plt.close()
#
# plt.plot(epoch_list,recons[5])
# plt.title('Reconstruction Errors',fontsize=18)
# plt.xlabel('Epoch',fontsize=18)
# plt.ylabel('Error',fontsize=18)
# plt.savefig('Recon225.pdf')
# plt.close()
