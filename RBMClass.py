"""
General RBM Class. Includes model initialization, Contrastive Divergence method, training method and
methods to calculate the loss over one training epoch.
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from memory_profiler import profile


class RBM(nn.Module):
    def __init__(self,nv,nh):
        """
        Args:
            nv (int): number visible units
            nh (int): number hidden units
        """
        super().__init__()


        self.nv = nv
        self.nh = nh
        self.W = nn.Parameter(torch.zeros(nh,nv))
        self.v_bias = nn.Parameter(torch.zeros(nv))
        self.h_bias = nn.Parameter(torch.zeros(nh))

    def random_probabilities(self,probabilities,random):
        """
        Given a tensor of probabilities (e.g. for the units of a layer),
        convert this into a tensor of zeros and ones. Do this by comparing to
        a tensor of random values of the same shape.

        Args:
            probabilities (tensor): tensor of probabilities (floats)
            random (tensor): tensor of float values to compare against probabilities

        Returns:
            tensor (same shape as probabilities) of zeros and ones
        """

        with torch.no_grad():
            difference = probabilities - random
            signs = torch.sign(difference) #get sign of the differences
            minus_to_zeros = F.relu(signs)
            del difference
            del signs
        return minus_to_zeros


    def prob_h_given_v(self,visible):
        """
        Calculate p(h|v,pararmeters)
        Args:
            visible (tensor): tensor of visible unit values (0s or 1s)

        Returns:
            tensor (floats) of shape (self.nh); the probabilities of the hidden nodes
        """
        with torch.no_grad():
            activation = F.linear(visible, self.W, self.h_bias) # W*v + h_bias
            probability = torch.sigmoid(activation) #p(h|v,pararmeters)
            del activation
        return probability

    def prob_v_given_h(self,hidden):
        """
        Calculate p(v|h,pararmeters)
        Args:
            hidden (tensor): tensor of hidden unit values (0s or 1s)

        Returns:
            tensor (floats) of shape (self.nv); the probabilities of the visible nodes
        """
        with torch.no_grad():
            activation = F.linear(hidden, torch.transpose(self.W,0,1), self.v_bias) # W*h + v_bias
            probability = torch.sigmoid(activation) #p(v|h,pararmeters)
            del activation

        return probability


    def h_given_v(self,visible):
        """
        Calculate the value of the hidden nodes given
        the visible nodes and the weights and bias parameters.

        Args:
            visible (tensor): tensor of visible unit values (0s or 1s)

        Returns:
            tensor of shape self.nh; the hidden units
        """
        with torch.no_grad():
            activation = F.linear(visible, self.W, self.h_bias) # W*v + h_bias
            probability = torch.sigmoid(activation) #p(h|v,pararmeters)
            random_field = torch.rand(probability.size())
            hidden_nodes = self.random_probabilities(probability,random_field)
            del activation
            del probability
            del random_field
        return hidden_nodes

    def v_given_h(self,hidden):
        """
        Calculate the value of the visible nodes given
        the hidden nodes and the weights and bias parameters.

        Args:
            hidden (tensor): tensor of hidden unit values (0s or 1s)

        Returns:
            tensor of shape self.nv - the visible units
        """
        with torch.no_grad():
            activation = F.linear(hidden, torch.transpose(self.W,0,1), self.v_bias) # W*h + v_bias
            probability = torch.sigmoid(activation) #p(v|h,pararmeters)
            random_field = torch.rand(probability.size())
            visible_nodes = self.random_probabilities(probability,random_field)
            del activation
            del probability
            del random_field
        return visible_nodes


    def Gibbs_v(self,v_0,k):
        """
        Starting from v_0, perform Gibbs sampling k times between layers
        to get v_k

        Args:
            v_0 (tensor): intitial values of the visible units
            k (int): The number of Gibbs sampling steps

        Return:
            v_k (tensor of same shape as v_0): v_0 --> h_1 --> ... -->v_k

        """
        with torch.no_grad():
            v = v_0
            h = torch.zeros(self.nh) #initialise the hidden units as zeros
            for step in range(k):
                h_new = self.h_given_v(v)
                v_new = self.v_given_h(h_new)
                h = h_new
                v = v_new
                del h_new
                del v_new

        del h

        return v

    def Gibbs_h(self,v_0,k):
        """
        Starting from v_0, perform Gibbs sampling k times between layers
        to get h_k

        Args:
            v_0 (tensor): intitial values of the visible units
            k (int): The number of Gibbs sampling steps

        Return:
            v_k (tensor of same shape as v_0): v_0 --> h_1 --> ... -->h_k

        """
        with torch.no_grad():
            v = v_0
            h = torch.zeros(self.nh) #initialise the hidden units as zeros
            for step in range(k):
                h_new = self.h_given_v(v)
                v_new = self.v_given_h(h_new)
                h = h_new
                v = n_new
                del h_new
                del v_new

        return h


    def weight_gradient(self,v_0,v_k,*args,w_mask=False):
        """
        Given two visible tensors, v_0 and v_k, representing states of the visible units,
        calculate an estimate of the gradient to the log likelihood wrt the weights using Contrastive
        Divergence. Include the possibility that you don't want to update certain parameters,
        specified in the last argument of this method. The default is that all pararmeters
        are trained.

        Args:
            v_0 (tensor): visible units at start of CD Markov Chain
            v_k (tensor): visible units at end of CD Markov Chain
            args[0] (tensor): weight_mask. binary tensor of 0 and 1. A value of 1 means that the corresponding weight parameter will be trained.
        Returns:
            tensor (same shape as self.W)
        """
        with torch.no_grad():
            #If you want to train all of your weight parameters
            if w_mask == False:
                probability_zero = torch.outer(self.prob_h_given_v(v_0),v_0) #p(h|v^(0))v^(0)
                probability_k = torch.outer(self.prob_h_given_v(v_k),v_k) #p(h|v^(k))v^(k)
                delta_W = probability_zero - probability_k #get derivatives of all parameters
                return delta_W

                del probability_zero
                del probability_k
                del delta_W
            #If you don't want to train all of your weight parameters
            else:
                weight_mask = args[0]
                probability_zero = torch.outer(self.prob_h_given_v(v_0),v_0)
                probability_k = torch.outer(self.prob_h_given_v(v_k),v_k)
                delta_W = probability_zero - probability_k #get derivatives of all parameters
                delta_W = torch.where(weight_mask==1,delta_W,weight_mask) #set the non trainable weight derivatives to zero
                del probability_zero
                del probability_k
                del weight_mask
                delta_W_np = delta_W.detach().numpy()
                del delta_W
                return delta_W_np





    def v_bias_gradient(self,v_0,v_k,*args,v_mask=False):
        """
        Given two visible tensors, v_0 and v_k, representing states of the visible units,
        calculate an estimate of the gradient to the log likelihood wrt the visible
        bias weight parameters using Contrastive Divergence.
        Include the possibility that you don't want to update certain parameters,
        specified in the last argument of this method. The default is that all pararmeters
        are trained.

        Args:
            v_0 (tensor): visible units at start of CD Markov Chain
            v_k (tensor): visible units at end of CD Markov Chain
            v_bias_mask (tensor): binary tensor of 0 and 1. A value of 1 means that the corresponding visible bias parameter will be trained.

        Returns:
            tensor of shape (nv)
        """
        with torch.no_grad():
            #If you want to train all of your weight parameters
            if v_mask == False:
                delta_v_bias = v_0 - v_k #get derivatives of all parameters
                return delta_v_bias
            #If you don't want to train all of your weight parameters
            else:
                v_bias_mask = args[0]
                delta_v_bias = v_0 - v_k #get derivatives of all parameters
                delta_v_bias = torch.where(v_bias_mask==1,delta_v_bias,v_bias_mask) #set the non trainable bias derivatives to zero
                del v_bias_mask
                return delta_v_bias


    def h_bias_gradient(self,v_0,v_k,*args,h_mask=False):
        """
        Given two visible tensors, v_0 and v_k, representing states of the visible units,
        calculate an estimate of the gradient to the log likelihood wrt the hidden
        bias weight parameters using Contrastive Divergence.
        Include the possibility that you don't want to update certain parameters,
        specified in the last argument of this method. The default is that all pararmeters
        are trained.

        Args:
            v_0 (tensor): visible units at start of CD Markov Chain
            v_k (tensor): visible units at end of CD Markov Chain
            h_bias_mask (tensor): binary tensor of 0 and 1. A value of 1 means that the corresponding hidden bias parameter will be trained.

        Returns:
            tensor of shape (nh)
        """
        with torch.no_grad():
            #If you want to train all of your weight parameters
            if h_mask == False:
                probability_zero = self.prob_h_given_v(v_0) #p(h|v^(0))
                probability_k = self.prob_h_given_v(v_k) #p(h|v^(k))
                delta_h_bias = probability_zero - probability_k #get derivatives of all parameters
                del probability_zero
                del probability_k
                return delta_h_bias
            #If you don't want to train all of your weight parameters
            else:
                h_bias_mask=args[0]
                probability_zero = self.prob_h_given_v(v_0) #p(h|v^(0))
                probability_k = self.prob_h_given_v(v_k) #p(h|v^(k))
                delta_h_bias = probability_zero - probability_k #get derivatives of all parameters
                delta_h_bias = torch.where(h_bias_mask==1,delta_h_bias,h_bias_mask) #set the non trainable bias derivatives to zero
                del probability_zero
                del probability_k
                del h_bias_mask
                return delta_h_bias


    def batch_weight_gradient(self,mini_batch,k,weight_mask=False):
        """
        Calculate an estimate to the gradient of log likelihood wrt the weight
        parameters over one mini-batch

        Args:
            mini_batch (tensor): a tensor of training mini batch data. The size of the
                                    tensor is (batch_size,nv)

            k (int): The number of Gibbs sampling steps

        Returns:
            tensor (shape = (nh,nv)). The batch gradient wrt the weights
        """

        batch_size = len(mini_batch)

        #for each training vector in your batch, make an update to the gradient
        for visible in mini_batch:
            vk = self.Gibbs_v(visible,k) #do CD_k for each training vector
            #If you want to train all of your weight parameters
            if weight_mask==False:
                grad_update = self.weight_gradient(visible,vk)
            #If you don't want to train all of your weight parameters
            else:
                grad_update = self.weight_gradient(visible,vk,weight_mask)

            self.W.grad += grad_update

            del vk
            del grad_update


    def batch_gradient_update(self,mini_batch,k,*args,option=False):
        """
        Calculate an estimate to the gradient of log likelihood wrt the weight,
        visible bias and hidden bias parameters over one mini-batch.

        Args:
            mini_batch (tensor): a tensor of training mini batch data. The size of the
                                    tensor is (batch_size,nv)

            k (int): The number of Gibbs sampling steps

            *args (tensor): Binary masks for the parameters to specify which parameters get trained.
                            weight_mask = args[0]
                            visible_mask = args[1]
                            hidden_mask = args[2]

        Returns:
            None. Updates self.parameter.grad
        """
        with torch.no_grad():
            batch_size = len(mini_batch)

            #initialise the gradient as zero

            weight_gradient = torch.zeros(self.nh,self.nv)
            v_bias_gradient = torch.zeros(self.nv)
            h_bias_gradient = torch.zeros(self.nh)

            if option == False:
                #for each training vector in your batch, make an update to the gradient
                for visible in mini_batch:
                    vk = self.Gibbs_v(visible,k) #do CD_k for each training vector
                    delta_W_np = self.weight_gradient(visible,vk)
                    delta_v_bias = self.v_bias_gradient(visible,vk)
                    delta_h_bias = self.h_bias_gradient(visible,vk)
                    weight_gradient += torch.from_numpy(delta_W_np)
                    v_bias_gradient += delta_v_bias
                    h_bias_gradient += delta_h_bias

                    del vk
                    del delta_W
                    del delta_v_bias
                    del delta_h_bias

                self.W.grad = -weight_gradient
                self.v_bias.grad = -v_bias_gradient
                self.h_bias.grad = -h_bias_gradient


            else:
                w_mask = args[0]
                visible_mask = args[1]
                hidden_mask = args[2]

                #for each training vector in your batch, make an update to the gradient
                for visible in mini_batch:
                    vk = self.Gibbs_v(visible,k) #do CD_k for each training vector
                    delta_W = self.weight_gradient(visible,vk,w_mask,w_mask=True)
                    delta_v_bias = self.v_bias_gradient(visible,vk,visible_mask,v_mask=True)
                    delta_h_bias = self.h_bias_gradient(visible,vk,hidden_mask,h_mask=True)
                    weight_gradient += delta_W
                    v_bias_gradient += delta_v_bias
                    h_bias_gradient += delta_h_bias

                    del vk
                    del delta_W
                    del delta_v_bias
                    del delta_h_bias

                # weight_gradient = torch.where(weight_mask==1,weight_gradient,weight_mask) #set the non trainable weight derivatives to zero
                # v_bias_gradient = torch.where(visible_mask==1,v_bias_gradient,visible_mask) #set the non trainable visible bias derivatives to zero
                # h_bias_gradient = torch.where(hidden_mask==1,h_bias_gradient,hidden_mask) #set the non trainable hidden bias derivatives to zero

                self.W.grad = -weight_gradient
                self.v_bias.grad = -v_bias_gradient
                self.h_bias.grad = -h_bias_gradient

                del w_mask
                del visible_mask
                del hidden_mask

            #del weight_gradient
            del v_bias_gradient
            del h_bias_gradient


    def parameter_update(self,mini_batch,k,*args,lr=0.01,mom=0.9,weight_decay=0.01,option=False):
        """
        Updates the model parameters over one mini-batch. Use Adam as an Optimiser.

        Args:
            mini_batch (tensor): a tensor of training mini batch data. The size of the
                                    tensor is (batch_size,nv)

            k (int): The number of Gibbs sampling steps

            lr  (float, optional) : learning rate (default: 1e-3)

            betas  (Tuple[float, float], optional) : coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))

            weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)

            *args (tensor): Binary masks for the parameters to specify which parameters get trained.
                            weight_mask = args[0]
                            visible_mask = args[1]
                            hidden_mask = args[2]

        Returns:
            None. Just updates the parameters.
        """
        with torch.no_grad():
            optimiser_class = optim.Adam(self.parameters(),lr=0.001)
            optimiser_class.zero_grad() #zero any previous gradient values
            if option == False:
                self.batch_gradient_update(mini_batch,k,option=False) #update the gradients

                optimiser_class.step() #update the parameters

            else:
                weight_mask = args[0]
                visible_mask = args[1]
                hidden_mask = args[2]
                self.batch_gradient_update(mini_batch,k,weight_mask,visible_mask,hidden_mask,option=True) #update the trainable gradients
                optimiser_class.step() #update the trainable parameters

            del optimiser_class

    def batch_reconstruction_error(self,mini_batch):
        """
        Calculate the reconstruction error over one mini-batch.

        Args:
            mini_batch (tensor): a tensor of training mini batch data. The size of the
                                    tensor is (batch_size,nv)

        Returns:
            float
        """
        with torch.no_grad():
            batch_error = 0

            for visible in mini_batch:
                v1 = self.Gibbs_v(visible,1)
                mse = F.mse_loss(visible,v1)
                batch_error+=float(mse)

            batch_size = len(mini_batch)
            mean_batch_error = batch_error/batch_size

        return mean_batch_error

    def energy(self,visible,k):
        """
        Given an input visible state, do CD_k and return the joint
        energy of the state (visble,h_k).

        Args:
            visible (tensor): tensor containing visible units values
            k (int): the number of Gibbs sampling steps to do

        Return:
            Float: the joint energy
        """

        h_k = self.Gibbs_h(visible,k)
        weight_energy = -torch.mm(torch.transpose(h_k),torch.mm(self.W,visible))
        visible_energy = torch.dot(self.v_bias,visible)
        hidden_energy = -torch.dot(self.h_bias,h_k)
        energy = weight_energy + visible_energy + hidden_energy

        return energy

    
