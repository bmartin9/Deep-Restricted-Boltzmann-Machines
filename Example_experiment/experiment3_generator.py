"""
Decimation Weight == 2
Block Biases == -1
Visible Biases == Trainable
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
    train_size = 0
    test_size = 5000
    num_batches = int(train_size/batch_size) #must always be whole number
    num_epochs = 15
    k=3 #number of Gibbs sampling steps
    temperature = 2.4


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
    # for i in range(DBM.numlayers -1):
    #     #bulk_weights = 10*torch.ones((DBM.architecture[i+1],DBM.architecture[i]))
    #     bulk_weights = torch.normal(torch.zeros((DBM.architecture[i+1],DBM.architecture[i])),1)
    #     B = torch.where(weight_mask_list[i]==1,bulk_weights,DBM.weights[i])
    #     W  = torch.where(weight_mask_list[i]==2,weight_mask_list[i],DBM.weights[i])
    #     combine = B+W
    #     DBM.weights[i] = nn.Parameter(combine)


    # initialise the biases as 0 for bulk nodes and -50 for block connections - DO THIS WHEN YOU DONT HAVE
    #WEIGHT CSV FILES TO READ IN
    # for l in range(DBM.numlayers):
    #     bias = torch.where(bias_mask_list[l]==-1,bias_mask_list[l],DBM.bias[l])
    #     DBM.bias[l] = nn.Parameter(bias)


    weight_list = []

    # Read in weights and biases - DO THIS WHEN YOU DO HAVE
    # WEIGHT CSV FILES TO READ IN
    W_visible = pandas.read_csv('experiment3_visible_weights240.csv',index_col=0)#read in the weights from weights.csv
    W_visible = W_visible.values#convert to numpy array
    W_visible = W_visible.astype('float32')
    W_visible = torch.from_numpy(W_visible)
    weight_list.append(W_visible)

    W_h1 = pandas.read_csv('experiment3_h1_weights240.csv',index_col=0)#read in the weights from weights.csv
    W_h1 = W_h1.values#convert to numpy array
    W_h1 = W_h1.astype('float32')
    W_h1 = torch.from_numpy(W_h1)
    weight_list.append(W_h1)

    W_h2 = pandas.read_csv('experiment3_h2_weights240.csv',index_col=0)#read in the weights from weights.csv
    W_h2 = W_h2.values#convert to numpy array
    W_h2 = W_h2.astype('float32')
    W_h2 = torch.from_numpy(W_h2)
    weight_list.append(W_h2)

    W_h3 = pandas.read_csv('experiment3_h3_weights240.csv',index_col=0)#read in the weights from weights.csv
    W_h3 = W_h3.values#convert to numpy array
    W_h3 = W_h3.astype('float32')
    W_h3 = torch.from_numpy(W_h3)
    weight_list.append(W_h3)

    W_h4 = pandas.read_csv('experiment3_h4_weights240.csv',index_col=0)#read in the weights from weights.csv
    W_h4 = W_h4.values#convert to numpy array
    W_h4 = W_h4.astype('float32')
    W_h4 = torch.from_numpy(W_h4)
    weight_list.append(W_h4)


    for i in range(DBM.numlayers -1):
        DBM.weights[i] = nn.Parameter(weight_list[i])

    #### Read in Biases ####
    bias_list = []

    b_visible = pandas.read_csv('experiment3_visible_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_visible = b_visible.values#convert to numpy array
    b_visible = b_visible.astype('float32')
    b_visible = np.reshape(b_visible,(DBM.architecture[0]))
    b_visible = torch.from_numpy(b_visible)
    bias_list.append(b_visible)

    b_h1 = pandas.read_csv('experiment3_h1_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_h1 = b_h1.values#convert to numpy array
    b_h1 = b_h1.astype('float32')
    b_h1 = np.reshape(b_h1,(DBM.architecture[1]))
    b_h1 = torch.from_numpy(b_h1)
    bias_list.append(b_h1)

    b_h2 = pandas.read_csv('experiment3_h2_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_h2 = b_h2.values#convert to numpy array
    b_h2 = b_h2.astype('float32')
    b_h2 = np.reshape(b_h2,(DBM.architecture[2]))
    b_h2 = torch.from_numpy(b_h2)
    bias_list.append(b_h2)

    b_h3 = pandas.read_csv('experiment3_h3_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_h3 = b_h3.values#convert to numpy array
    b_h3 = b_h3.astype('float32')
    b_h3 = np.reshape(b_h3,(DBM.architecture[3]))
    b_h3 = torch.from_numpy(b_h3)
    bias_list.append(b_h3)

    b_h4 = pandas.read_csv('experiment3_h4_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_h4 = b_h4.values#convert to numpy array
    b_h4 = b_h4.astype('float32')
    b_h4 = np.reshape(b_h4,(DBM.architecture[4]))
    b_h4 = torch.from_numpy(b_h4)
    bias_list.append(b_h4)

    b_h5 = pandas.read_csv('experiment3_h5_bias240.csv',index_col=0)#read in the bias from bias.csv
    b_h5 = b_h5.values#convert to numpy array
    b_h5 = b_h5.astype('float32')
    b_h5 = np.reshape(b_h5,(DBM.architecture[5]))
    b_h5 = torch.from_numpy(b_h5)
    bias_list.append(b_h5)

    for i in range(DBM.numlayers):
        DBM.bias[i] = nn.Parameter(bias_list[i])



    #Read in your Ising data
    start_time = time.time()
    ising_tensor = CSV_reader('/Users/Brendan/FifthYear/Project/Version2/L32_100KIsing_data_240.csv',N,numtemps,numsweeps)
    print("Read CSV --- %s seconds ---" % (time.time() - start_time))



    test_set = ising_tensor[0,:test_size,:]
    test_set = torch.reshape(test_set,(test_size,N)) #separate training set into batches

    all_unit_list = [np.zeros((test_size,architecture[0]))]
    for z in range(DBM.numlayers-1):
        placeholder = np.zeros((test_size,int(architecture[z+1]/2)))
        all_unit_list.append(placeholder)


    for i in range(test_size):
        visible_units = test_set[i]
        sample_list = DBM.alternating_update(visible_units,k) #get list of sampled units; this list has length DBM.numlayers

        visible_unit_tensor = sample_list[0]
        visible_unit_array = visible_unit_tensor.detach().numpy()
        #change every 0 to a -1 in the data
        where_zero_visible = np.where(visible_unit_array == 0)
        visible_unit_array[where_zero_visible] = -1
        all_unit_list[0][i] = visible_unit_array

        for j in range(DBM.numlayers-1):
            unit_tensor = sample_list[j+1]
            block_unit_tensor = torch.zeros(int(architecture[j+1]/2))
            for n in range(int(architecture[j+1]/2)):
                block_unit_tensor[n] = unit_tensor[2*n] #only pick out the block units; throw away the bulk units
            unit_array = block_unit_tensor.detach().numpy()
            #change every 0 to a -1 in the data
            where_zero = np.where(unit_array == 0)
            unit_array[where_zero] = -1
            all_unit_list[j+1][i] = unit_array

    visible_image = np.reshape(all_unit_list[0][-1],(32,32))
    plt.imshow(visible_image, interpolation='nearest')
    plt.show()

    h1_image = np.reshape(all_unit_list[1][-1],(16,16))
    plt.imshow(h1_image, interpolation='nearest')
    plt.show()

    h2_image = np.reshape(all_unit_list[2][-1],(8,8))
    plt.imshow(h2_image, interpolation='nearest')
    plt.show()

    h3_image = np.reshape(all_unit_list[3][-1],(4,4))
    plt.imshow(h3_image, interpolation='nearest')
    plt.show()



    visible_df = pandas.DataFrame(all_unit_list[0])
    visible_df.to_csv('experiment3_visible_units240.csv')

    h1_df = pandas.DataFrame(all_unit_list[1])
    h1_df.to_csv('experiment3_h1_units240.csv')

    h2_df = pandas.DataFrame(all_unit_list[2])
    h2_df.to_csv('experiment3_h2_units240.csv')

    h3_df = pandas.DataFrame(all_unit_list[3])
    h3_df.to_csv('experiment3_h3_units240.csv')

    h4_df = pandas.DataFrame(all_unit_list[4])
    h4_df.to_csv('experiment3_h4_units240.csv')

    h5_df = pandas.DataFrame(all_unit_list[5])
    h5_df.to_csv('experiment3_h5_units240.csv')




if __name__ == "__main__":
    main()
