"""
Use the MCRG Method to extract couplings from MCMC data.
"""

import numpy as np
import pandas
import time
import matplotlib.pyplot as plt

def find_NN(s,i,j,size):
    """
    return a list of neighbouring spins to
    spin (row,column). Use periodic boundary conditions

    Args:
        lattice (numpy array): numpy array of shape (L,L)
        row (int): row index of target spin
        col (int): column index of target spin
        L (int): length of lattice

    Returns:
        list (len(list)=4): the spins of the nearest neighbours to the target spin.
    """

    leftS = s[size-1,j] if i==0 else s[i-1,j]
    rightS = s[0,j] if i==size-1 else s[i+1,j]
    topS = s[i,size-1] if j==0 else s[i,j-1]
    bottomS = s[i,0] if j==size-1 else s[i,j+1]

    return [topS,bottomS,leftS,rightS]

def find_NextNN(s,i,j,size):
    """
    return a list of Next neighbouring spins to
    spin (row,column). Use periodic boundary conditions

    Args:
        lattice (numpy array): numpy array of shape (L,L)
        row (int): row index of target spin
        col (int): column index of target spin
        L (int): length of lattice

    Returns:
        list (len(list)=4): the spins of the next nearest neighbours to the target spin.
    """


    if j==size-1:
        bottomS = s[i,1]
    elif j ==size-2:
        bottomS = s[i,0]
    else:
        bottomS = s[i,j+2]

    if j==0:
        topS = s[i,size-2]
    elif j ==1:
        topS = s[i,size-1]
    else:
        topS = s[i,j-2]

    if i==0:
        leftS = s[size-2,j]
    elif 1 ==1:
        leftS = s[size-1,j]
    else:
        leftS = s[i-2,j]

    if i==size-1:
        rightS = s[1,j]
    elif i==size-2:
        rightS = s[0,j]
    else:
        rightS = s[i-2,j]

    return [topS,bottomS,leftS,rightS]


def find_diagonal_neighbours(s,i,j,L):
    """
    return a list of diagonal neighbouring spins (K_2) to
    spin (row,column). Use periodic boundary conditions

    Args:
        lattice (numpy array): numpy array of shape (L,L)
        row (int): row index of target spin
        col (int): column index of target spin
        L (int): length of lattice

    Returns:
        list (len(list)=4): the spins of the diagonal neighbours to the target spin.
    """

    if i==0 and j==0:
        up_left = s[L-1][L-1]
    elif i==0:
        up_left = s[i-1][L-1]
    elif j==0:
        up_left = s[L-1][j-1]
    else:
        up_left = s[i-1][j-1]

    if i==0 and j==L-1:
        up_right = s[L-1][0]
    elif i==0:
        up_right = s[L-1][j+1]
    elif j==L-1:
        up_right = s[i-1][0]
    else:
        up_right = s[i-1][j+1]

    if i==L-1 and j==L-1:
        down_right = s[0][0]
    elif i==L-1:
        down_right = s[0][j+1]
    elif j==L-1:
        down_right = s[i+1][0]
    else:
        down_right = s[i+1][j+1]

    if i==L-1 and j==0:
        down_left = s[0][L-1]
    elif i==L-1:
        down_left = s[0][j-1]
    elif j==0:
        down_left = s[i+1][L-1]
    else:
        down_left = s[i+1][j-1]



    return [down_left,up_left,up_right,down_right]

def Jacobian11(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat1*O_hat1)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func

def Jacobian12(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat1*O_hat2)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func

def Jacobian13(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat1*O_hat3)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func

def Jacobian22(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat2*O_hat2)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func

def Jacobian23(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat2*O_hat3)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func

def Jacobian33(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    """
    Calculate the funcion inside the loop over lattice spins for a jacobian element 11.
    Args:
        O_hat1 (int): Sum of surrounding NN spins
        O_hat2 (int): Sum of surrounding NNN spins

        x1 (float): coupling value corresponding to operator 1
        x2 (float): coupling value corresponding to operator 2


    Returns:
        Float
    """
    func = 0.5*(O_hat3*O_hat3)/((np.cosh(x1*O_hat1+x2*O_hat2+x3*O_hat3))**2)
    return func



def part_of_f1(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    func = 0.5 * O_hat1 * np.tanh(x1*O_hat1+x2*O_hat2+x3*O_hat3)
    return func

def part_of_f2(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    func = 0.5 * O_hat2 * np.tanh(x1*O_hat1+x2*O_hat2+x3*O_hat3)
    return func

def part_of_f3(O_hat1,O_hat2,O_hat3,x1,x2,x3):
    func = 0.5 * O_hat3 * np.tanh(x1*O_hat1+x2*O_hat2+x3*O_hat3)
    return func

def lattice_numbers(x1,x2,x3,lattice,L):
    O1 = 0
    O2 = 0
    O3 = 0
    f = np.zeros((3))
    J = np.zeros((3,3))

    # plt.imshow(lattice, interpolation='nearest')
    # plt.show()
    # plt.close()

    for i in range(L):
        for j in range(L):
            spin = lattice[i][j]
            NN = find_NN(lattice,i,j,L)
            NextNN = find_NextNN(lattice,i,j,L)
            diag_neighbours = find_diagonal_neighbours(lattice,i,j,L)
            O_hat1 = -np.sum(NN)
            O_hat2 = -np.sum(NextNN)
            O_hat3 = -np.sum(diag_neighbours)
            energyNN = O_hat1*spin
            O1+=energyNN
            energyNextNN = O_hat2*spin
            O2+=energyNextNN
            energy_diag = O_hat3*spin
            O3+=energy_diag

            f[0] += part_of_f1(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            f[1] += part_of_f2(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            f[2] += part_of_f3(O_hat1,O_hat2,O_hat3,x1,x2,x3)

            J11 = Jacobian11(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            J12 = Jacobian12(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            J13 = Jacobian13(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            J22 = Jacobian22(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            J23 = Jacobian23(O_hat1,O_hat2,O_hat3,x1,x2,x3)
            J33 = Jacobian33(O_hat1,O_hat2,O_hat3,x1,x2,x3)

            J[0][0] += J11
            J[1][1] += J22
            J[2][2] += J33
            J[0][1] += J12
            J[1][0] += J12
            J[0][2] += J13
            J[2][0] += J13
            J[1][2] += J23
            J[2][1] += J23


    #add part of f to O
    O1=0.5*O1
    O2=0.5*O2
    O3=0.5*O3
    f[0] = f[0] + O1
    f[1] = f[1] + O2
    f[2] = f[2] + O3
    return [f,J]

def mean_lattice_numbers(x1,x2,x3,lattices,num_samples,L):

    f_store = []
    J_store = []
    for k in range(num_samples):
        number_list = lattice_numbers(x1,x2,x3,lattices[k],L)
        f_store.append(number_list[0])
        J_store.append(number_list[1])

    f_av = np.mean(f_store,axis=0)
    J_av = np.mean(J_store,axis=0)

    return [f_av,J_av]

def main():

    L = 4
    num_samples = 1000
    max_steps = 10
    tolerance = 0.000001 #if your gradient update is less than this, stop updating
    numtemps =1
    numsweeps =5000


    #df = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_weights/30epochs2269_100K/L32_10Kvisible_units2269.csv',index_col=0)#read in the ising data from your csv file
    #df = pandas.read_csv('/Users/Brendan/FifthYear/Project/L32_params_new/3new_bias_L32_10Kh1_units210.csv',index_col=0)
    df = pandas.read_csv('experiment3_h3_units210.csv',index_col=0)
    ising_data = df.values#convert to numpy array
    ising_data = ising_data.astype('float32')
    ising_data = np.reshape(ising_data,(numsweeps,L,L))
    ising_data = ising_data[:+num_samples]
    ising_data = np.reshape(ising_data,(num_samples,L,L))
    # for l in range(num_samples):
    #     ising_data[l] = ising_data[l].transpose()
    #change every 0 to a -1 in the data
    # where_zero = np.where(ising_data == -1)
    # ising_data[where_zero] = 0


    print(ising_data[122][0])
    for l in range(10):
        plt.imshow(ising_data[l], interpolation='nearest')
        plt.show()
        plt.close()

    x = np.array([0.1,0.01,0.01])

    for gradient_step in range(max_steps):
        start_time = time.time()
        mean_numbers = mean_lattice_numbers(x[0],x[1],x[2],ising_data,num_samples,L)
        f_av = mean_numbers[0]
        J_av = mean_numbers[1]
        print("J: "+str(J_av))
        print("f: "+str(f_av))
        h = np.linalg.solve(J_av,-f_av)
        print("h: "+str(h))
        #h = np.linalg.inv(J_av).dot(-f_av)
        x_new = x + h
        print("Update time --- %s seconds ---" % (time.time() - start_time))
        print(x_new)
        if abs(f_av[0])<tolerance:
            x=x_new
            break
        else:
            x = x_new

    print("final coupling estimates: "+str(x))
    print("final vaue of f: "+str(mean_lattice_numbers(x[0],x[1],x[2],ising_data,num_samples,L)[0]))

if __name__ == '__main__':
    main()
