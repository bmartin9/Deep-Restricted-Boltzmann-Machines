"""
Decimation Weight == 2
Block Biases == -1
Visible Biases : Trainable
Bulk weights : Trainable
Bulk Biases : Trainable
"""

import DBMClass_2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import time
import pandas
import matplotlib.pyplot as plt

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
    L=int(np.sqrt(N))
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

def visible_non_zero_indices(N):
    """
    Create a weight mask containing the values 1000, 1, and 0.
    Locations in the weight mask containing a 1 correspond to bulk nodes.
    Locations in the weight mask containing a 1000 correspond to block nodes.
    Locations in the weight mask containing a 0 correspond to non trainable weights.

    Args:
         (int): the number of visible nodes of your visible input lattice.

    Returns:
        numpy array of shape (num_hidden_units,num_visible_units)
    """

    L = int(np.sqrt(N))
    num_block_nodes = N/4
    num_bulk_nodes = N/4
    num_hidden_nodes = int(num_block_nodes + num_bulk_nodes)

    mask = np.zeros((int(L/2),L,L,L))

    for i in range(int(L/2)):
        for j in range(L):
            #block nodes are even
            visible_row = 2*i
            if j%2==0:
                mask[i][j][visible_row][j] = 2

            else:
                next_visible_row = visible_row +1
                mask[i][j][visible_row][j-1] = 1
                mask[i][j][visible_row][j] = 1
                mask[i][j][next_visible_row][j-1] = 1
                mask[i][j][next_visible_row][j] = 1

    weight_mask = np.reshape(mask,(num_hidden_nodes,N))

    weight_mask = weight_mask.astype('float32')

    weight_mask = torch.from_numpy(weight_mask)

    return weight_mask


def hidden_non_zero_indices(nh):
    """
    Create a weight mask containing the values 1000, 1, and 0.
    Locations in the weight mask containing a 1 correspond to bulk nodes.
    Locations in the weight mask containing a 1000 correspond to block nodes.
    Locations in the weight mask containing a 0 correspond to non trainable weights.

    Args:
         nh (int): the number of hidden nodes of your input lattice.

    Returns:
        numpy array of shape (num_hidden_units,num_visible_units)
    """

    N = int(nh/2)

    L = int(np.sqrt(N))
    num_block_nodes = N/4
    num_bulk_nodes = N/4
    num_hidden_nodes = int(num_block_nodes + num_bulk_nodes)

    mask = np.zeros((int(L/2),L,L,2*L))

    for i in range(int(L/2)):
        for j in range(L):
            #block nodes are even
            visible_row = 2*i
            if j%2==0:
                mask[i][j][visible_row][2*j] = 2

            else:
                next_visible_row = visible_row +1
                mask[i][j][visible_row][2*(j-1)] = 1
                mask[i][j][visible_row][2*(j-1)+2] = 1
                mask[i][j][next_visible_row][2*(j-1)] = 1
                mask[i][j][next_visible_row][2*(j-1)+2] = 1

    weight_mask = np.reshape(mask,(num_hidden_nodes,nh))

    weight_mask = weight_mask.astype('float32')

    weight_mask = torch.from_numpy(weight_mask)

    return weight_mask



def main():

    L=32
    N=L**2
    numsweeps = 100000
    numtemps = 1
    batch_size=40
    train_size = 100000
    test_size = 0
    num_batches = int(train_size/batch_size) #must always be whole number
    num_epochs = 15
    k=3 #number of Gibbs sampling steps
    temperature = 2.1


    architecture = [1024,512,128,32,8,2]

    DBM = DBMClass_2.DBM(architecture)

    #set your weight masks
    weight_mask_list=[visible_non_zero_indices(architecture[0])]
    for i in range(DBM.numlayers -2):
        weight_mask_list.append(hidden_non_zero_indices(DBM.architecture[i+1]))

    #set your bias masks
    bias_mask_list = []
    visible_layer_bias_mask = torch.ones((N))
    bias_mask_list.append(visible_layer_bias_mask)
    for i in range(DBM.numlayers-1):
        a = torch.ones(DBM.architecture[i+1])
        num_bulk_nodes = int(DBM.architecture[i+1]/2)
        for j in range(num_bulk_nodes):
            block_index = 2*j
            a[block_index] = -1

        bias_mask_list.append(a)




    # initialise the weights around 0 (Normally distributed) everywhere except taking the value 100 at block connections - DO THIS WHEN YOU DONT HAVE
    #WEIGHT CSV FILES TO READ IN
    for i in range(DBM.numlayers -1):
        #bulk_weights = 10*torch.ones((DBM.architecture[i+1],DBM.architecture[i]))
        bulk_weights = torch.normal(torch.zeros((DBM.architecture[i+1],DBM.architecture[i])),1)
        B = torch.where(weight_mask_list[i]==1,bulk_weights,DBM.weights[i])
        W  = torch.where(weight_mask_list[i]==2,weight_mask_list[i],DBM.weights[i])
        combine = B+W
        DBM.weights[i] = nn.Parameter(combine)


    # initialise the biases as 0 for bulk nodes and -50 for block connections - DO THIS WHEN YOU DONT HAVE
    #WEIGHT CSV FILES TO READ IN
    for l in range(DBM.numlayers):
        bias = torch.where(bias_mask_list[l]==-1,bias_mask_list[l],DBM.bias[l])
        DBM.bias[l] = nn.Parameter(bias)


    #weight_list = []

    # Read in weights and biases - DO THIS WHEN YOU DO HAVE
    # WEIGHT CSV FILES TO READ IN
    # W_visible = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32visible_weights210.csv',index_col=0)#read in the weights from weights.csv
    # W_visible = W_visible.values#convert to numpy array
    # W_visible = W_visible.astype('float32')
    # W_visible = torch.from_numpy(W_visible)
    # weight_list.append(W_visible)
    #
    # W_h1 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h1_weights210.csv',index_col=0)#read in the weights from weights.csv
    # W_h1 = W_h1.values#convert to numpy array
    # W_h1 = W_h1.astype('float32')
    # W_h1 = torch.from_numpy(W_h1)
    # weight_list.append(W_h1)
    #
    # W_h2 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h2_weights210.csv',index_col=0)#read in the weights from weights.csv
    # W_h2 = W_h2.values#convert to numpy array
    # W_h2 = W_h2.astype('float32')
    # W_h2 = torch.from_numpy(W_h2)
    # weight_list.append(W_h2)
    #
    # W_h3 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h3_weights210.csv',index_col=0)#read in the weights from weights.csv
    # W_h3 = W_h3.values#convert to numpy array
    # W_h3 = W_h3.astype('float32')
    # W_h3 = torch.from_numpy(W_h3)
    # weight_list.append(W_h3)
    #
    # W_h4 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h4_weights210.csv',index_col=0)#read in the weights from weights.csv
    # W_h4 = W_h4.values#convert to numpy array
    # W_h4 = W_h4.astype('float32')
    # W_h4 = torch.from_numpy(W_h4)
    # weight_list.append(W_h4)
    #
    #
    # for i in range(DBM.numlayers -1):
    #     DBM.weights[i] = nn.Parameter(weight_list[i])

    #### Read in Biases ####
    # bias_list = []
    #
    # b_visible = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32visible_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_visible = b_visible.values#convert to numpy array
    # b_visible = b_visible.astype('float32')
    # b_visible = np.reshape(b_visible,(DBM.architecture[0]))
    # b_visible = torch.from_numpy(b_visible)
    # bias_list.append(b_visible)
    #
    # b_h1 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h1_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_h1 = b_h1.values#convert to numpy array
    # b_h1 = b_h1.astype('float32')
    # b_h1 = np.reshape(b_h1,(DBM.architecture[1]))
    # b_h1 = torch.from_numpy(b_h1)
    # bias_list.append(b_h1)
    #
    # b_h2 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h2_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_h2 = b_h2.values#convert to numpy array
    # b_h2 = b_h2.astype('float32')
    # b_h2 = np.reshape(b_h2,(DBM.architecture[2]))
    # b_h2 = torch.from_numpy(b_h2)
    # bias_list.append(b_h2)
    #
    # b_h3 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h3_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_h3 = b_h3.values#convert to numpy array
    # b_h3 = b_h3.astype('float32')
    # b_h3 = np.reshape(b_h3,(DBM.architecture[3]))
    # b_h3 = torch.from_numpy(b_h3)
    # bias_list.append(b_h3)
    #
    # b_h4 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h4_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_h4 = b_h4.values#convert to numpy array
    # b_h4 = b_h4.astype('float32')
    # b_h4 = np.reshape(b_h4,(DBM.architecture[4]))
    # b_h4 = torch.from_numpy(b_h4)
    # bias_list.append(b_h4)
    #
    # b_h5 = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/new_bias_L32h5_bias210.csv',index_col=0)#read in the bias from bias.csv
    # b_h5 = b_h5.values#convert to numpy array
    # b_h5 = b_h5.astype('float32')
    # b_h5 = np.reshape(b_h5,(DBM.architecture[5]))
    # b_h5 = torch.from_numpy(b_h5)
    # bias_list.append(b_h5)
    #
    # for i in range(DBM.numlayers):
    #     DBM.bias[i] = nn.Parameter(bias_list[i])



    #Read in your Ising data
    start_time = time.time()
    ising_tensor = CSV_reader('L32_100KIsing_data_210.csv',N,numtemps,numsweeps)
    print("Read CSV --- %s seconds ---" % (time.time() - start_time))



    train_set = ising_tensor[0,:train_size,:]
    train_set = torch.reshape(train_set,(num_batches,batch_size,N)) #separate training set into batches


    reconstruction_array = np.zeros((num_epochs))


    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_reconstruction_error = 0
        for mini in range(num_batches):
            DBM.parameter_update(train_set[mini],k,weight_mask_list,bias_mask_list,option=True)
            r = DBM.batch_reconstruction_error(train_set[mini])
            epoch_reconstruction_error += r

        reconstruction_array[epoch] = epoch_reconstruction_error/num_batches

        print('Epoch: '+str(epoch + 1))
        print("Epoch time --- %s seconds ---" % (time.time() - start_time))



    numpy_weights_visible = DBM.weights[0].detach().numpy()
    visible_weights_df = pandas.DataFrame(numpy_weights_visible)
    visible_weights_df.to_csv('experiment3_visible_weights210.csv')

    numpy_weights_h1 = DBM.weights[1].detach().numpy()
    h1_weights_df = pandas.DataFrame(numpy_weights_h1)
    h1_weights_df.to_csv('experiment3_h1_weights210.csv')

    numpy_weights_h2 = DBM.weights[2].detach().numpy()
    h2_weights_df = pandas.DataFrame(numpy_weights_h2)
    h2_weights_df.to_csv('experiment3_h2_weights210.csv')

    numpy_weights_h3 = DBM.weights[3].detach().numpy()
    h3_weights_df = pandas.DataFrame(numpy_weights_h3)
    h3_weights_df.to_csv('experiment3_h3_weights210.csv')

    numpy_weights_h4 = DBM.weights[4].detach().numpy()
    h4_weights_df = pandas.DataFrame(numpy_weights_h4)
    h4_weights_df.to_csv('experiment3_h4_weights210.csv')

    #### Output reconstruction errors to csv ######

    recon_df = pandas.DataFrame(reconstruction_array)
    recon_df.to_csv('experiment3_recon210.csv')

    #### Output Biases to csv files ####
    numpy_visible_bias = DBM.bias[0].detach().numpy()
    visible_bias_df = pandas.DataFrame(numpy_visible_bias)
    visible_bias_df.to_csv('experiment3_visible_bias210.csv')

    numpy_bias_h1 = DBM.bias[1].detach().numpy()
    h1_bias_df = pandas.DataFrame(numpy_bias_h1)
    h1_bias_df.to_csv('experiment3_h1_bias210.csv')

    numpy_bias_h2 = DBM.bias[2].detach().numpy()
    h2_bias_df = pandas.DataFrame(numpy_bias_h2)
    h2_bias_df.to_csv('experiment3_h2_bias210.csv')

    numpy_bias_h3 = DBM.bias[3].detach().numpy()
    h3_bias_df = pandas.DataFrame(numpy_bias_h3)
    h3_bias_df.to_csv('experiment3_h3_bias210.csv')

    numpy_bias_h4 = DBM.bias[4].detach().numpy()
    h4_bias_df = pandas.DataFrame(numpy_bias_h4)
    h4_bias_df.to_csv('experiment3_h4_bias210.csv')

    numpy_bias_h5 = DBM.bias[5].detach().numpy()
    h5_bias_df = pandas.DataFrame(numpy_bias_h5)
    h5_bias_df.to_csv('experiment3_h5_bias210.csv')

    print(reconstruction_array)

if __name__ == "__main__":
    main()
