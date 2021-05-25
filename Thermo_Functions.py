"""
This file contains functions that are useful in calculating the thermodynamics
of the 2D Ising model.
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def find_neighbours(lattice,row,col,L):
    """ return a list of neighbouring spins to
        spin (row,column). Use periodic boundary conditions"""

    left = lattice[row, (col - 1) % L]
    down = lattice[(row + 1) % L, col]
    up = lattice[(row - 1) % L, col]
    right = lattice[row, (col + 1) % L]

    return [up,down,left,right]

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

    topS = s[size-1,j] if i==0 else s[i-1,j]
    bottomS = s[0,j] if i==size-1 else s[i+1,j]
    leftS = s[i,size-1] if j==0 else s[i,j-1]
    rightS = s[i,0] if j==size-1 else s[i,j+1]

    return [topS,bottomS,leftS,rightS]


def lattice_update(lattice,L,temp):
    beta = 1/temp

    for i in range(L):
        for j in range(L):

            row = np.random.randint(0, L)
            col = np.random.randint(0, L)
            spin = lattice[row, col]


            neighbours = find_NN(lattice,row,col,L)

            up = neighbours[0]
            down = neighbours[1]
            left = neighbours[2]
            right = neighbours[3]

            e_surrounding = (up + down + right + left)
            J = 1 #coupling value
            delta_E = 2 * J * spin * e_surrounding

            # check Glauber conditions for the spin -- negative delta_E or prob exp(-delta_E*beta)
            if delta_E <= 0 or random.random() < np.exp(-delta_E * beta):
                spin *= -1


            # replace value in the lattice
            lattice[row, col] = spin

    return lattice




def lattice_energy(lattice,L):
    """
    Returns the lattice energy per spin
    """
    E_lattice=0
    for i in range(L):
        for j in range(L):
            spin = lattice[i,j]

            neighbours = find_neighbours(lattice,i,j,L)

            up = neighbours[0]
            down = neighbours[1]
            left = neighbours[2]
            right = neighbours[3]

            e_surrounding = up + down + right + left

            J=1 #coupling value
            E_lattice += -J*spin*e_surrounding
    # divide by 2 to get energy calc
    return E_lattice/(2*L*L)

def average_energy(multiple_arrays,L):
    """
    Take an array of lattices and return a list of the average energies of all the lattices
    """
    av_list = []
    num_lattices = len(multiple_arrays)
    for i in range(num_lattices):
        av_en = lattice_energy(multiple_arrays[i],L)
        av_list.append(av_en)

    return av_list

def magnetisation(lattice,L):

    return abs(np.sum(lattice)/(L**2))

def average_mag(multiple_arrays,L):
    """
    Take an array of lattices and return a list of the average magnetisations of all the lattices
    """
    av_list = []
    num_lattices = len(multiple_arrays)
    for i in range(num_lattices):
        av_mag = magnetisation(multiple_arrays[i],L)
        av_list.append(av_mag)

    return av_list

def av_of_mag_list(multiple_arrays,L):
    """
    Take an array of lattices and return the average magnetisations over all the lattices
    """
    av_list = average_mag(multiple_arrays,L)
    return np.mean(av_list)

def error(list):
    return np.sqrt(np.var(list))/np.sqrt(len(list))

def thermodynamic_variables(E_list,M_list,temp):
    """
    Take in lists of lattice energies and lattice magnetisations
    and return <E>,<M>,C_v and X (the magnetic susceptibility) for a given temperature
    Also returns the errors in these values.
    The  error is calculated as the standard deviation over the sqrt of the number of elements of the list.
    """


    E_squared_list = [E**2 for E in E_list]
    M_squared_list = [M**2 for M in M_list]

    mean_energy = np.mean(E_list)

    mean_magnetisation = np.mean(M_list)

    mean_E_squared = np.mean(E_squared_list)
    c_v = (mean_E_squared - mean_energy**2)/((temp**2))

    mean_M_squared = np.mean(M_squared_list)
    X = (mean_M_squared - mean_magnetisation**2)/(temp)

    energy_err = error(E_list)
    mag_err = error(M_list)
    energy_squared_err = error(E_squared_list)
    mag_squared_err = error(M_squared_list)

    c_v_err = (energy_squared_err - 2*mean_energy*energy_err)/(temp**2)
    X_err = (mag_squared_err - 2*mean_magnetisation*mag_err)/temp

    return [mean_energy,mean_magnetisation,c_v,X,energy_err,mag_err,c_v_err,X_err]

def raw_variables(storage,L,temperature_list,numtemps):
    """ Given an array of microstates of shape (numtemps,num_test,L,L) return the average energy and error on this
    for each temperature. Do this for other thermodynamic variabales too"""

    #Plotting lists of E, M, C_v and X
    E_plot = []
    M_plot = []
    C_plot = []
    X_plot = []
    T_plot = []

    #Lists of errors in E, M, C_v and X
    E_err = []
    M_err = []
    C_err = []
    X_err = []

    for i in range(numtemps):

        microstates = storage[i]
        num_microstates = len(microstates)

        #work out thermodynamic variables on test guesses
        test_energies = []
        test_magnitisations = []

        for micro in range(num_microstates):
            test_energies.append(lattice_energy(microstates[micro],L))
            test_magnitisations.append(magnetisation(microstates[micro],L))

        #work out the nH,T RBM's guess for the thermodynamic variables and their errors
        variables = thermodynamic_variables(test_energies,test_magnitisations,temperature_list[i])

        #add thermodynamic variables for this temperature to plotting list
        E_plot.append(variables[0])
        M_plot.append(variables[1])
        C_plot.append(variables[2])
        X_plot.append(variables[3])
        T_plot.append(temperature_list[i])

        #add thermodynamic variable errors for this temperature to plotting list
        E_err.append(variables[4])
        M_err.append(variables[5])
        C_err.append(variables[6])
        X_err.append(variables[7])

    return [T_plot,E_plot,M_plot,C_plot,X_plot,E_err,M_err,C_err,X_err]

def thermodynamic_plot(temp_array,variable_array,y_errors,col,lab,mark):
    """
    Take in a list of thermodynamic variables (e.g a list of magnetisations) and their errors and
    output a plot.

    Args:
    temp_array (array): a one dimensional array of floats
    variable_array (array): a one dimensional array of floats
    y_errors (array): a one dimensional array of corresponding errors
    col (str): the color of the plot and the errorbars
    lab (str): the label of the plot
    mark (str): the marker used to plot points

    Returns:
    matplotlib errorbar plot
    """

    plot = plt.errorbar(temp_array,variable_array,yerr = y_errors,capsize=1, elinewidth=0.5, markeredgewidth=0.5,linewidth=0.5,ecolor=col,color=col,label=lab,marker=mark)
    return plot
