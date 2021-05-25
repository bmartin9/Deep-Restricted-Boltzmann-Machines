"""
Deep RBM Class. Includes model initialization, Contrastive Divergence method, training method and
methods to calculate the loss over one training epoch. Also includes Gibbs update over multiple layers
using the even vs odd layer update method. Includes a layer-wise pretraining method.
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

class DBM(nn.Module):
    def __init__(self,architecture):
        """
        Args:
            architecture (list) : list of number of nodes in each layer. e.g. [nv,nh1,nh2,nh3]
        """
        super().__init__()

        self.numlayers = len(architecture)
        self.nv = architecture[0]
        self.architecture = architecture

        self.weights = nn.ParameterList()
        self.bias = nn.ParameterList()
        visible_bias = nn.Parameter(torch.zeros(self.nv))
        self.bias.append(visible_bias)

        for i in range(self.numlayers -1):
            W  = nn.Parameter(torch.zeros(self.architecture[i+1],self.architecture[i]))
            h_bias = nn.Parameter(torch.zeros(self.architecture[i+1]))
            self.weights.append(W)
            self.bias.append(h_bias)



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

    def forward_probability(self,units,l):
        """
        Calculate p(h_(l+1)|h_(l),pararmeters_(l))
        Args:
            units (tensor): tensor of unit values (0s or 1s) of layer l

            l (int): the layer of units you are inputting

        Returns:
            tensor (floats) of shape (self.architecture[l+1]); the probabilities of the hidden nodes of layer (l+1)
        """
        with torch.no_grad():
            activation = F.linear(units, self.weights[l], self.bias[l+1]) # W*v + h_bias
            probability = torch.sigmoid(activation) #p(h|v,pararmeters)
            del activation
        return probability

    def backward_probability(self,units,l):
        """
        Calculate p(h_(l-1)|h_(l),pararmeters_(l-1))
        Args:
            units (tensor): tensor of unit values (0s or 1s) of layer l

            l (int): the layer of units you are inputting

        Returns:
            tensor (floats) of shape (self.architecture[l-1]); the probabilities of the nodes of layer (l-1)
        """
        with torch.no_grad():
            activation = F.linear(units, torch.transpose(self.weights[l-1],0,1), self.bias[l-1]) # W*h + v_bias
            probability = torch.sigmoid(activation) #p(v|h,pararmeters)
            del activation
        return probability

    def forward_units(self,units,l):
        """
        Given the values of the nodes for layer l,
        Calculate the value of the hidden nodes in the next hidden layer, l+1
        Args:
            units (tensor): tensor of unit values (0s or 1s) of layer l

            l (int): the layer of units you are inputting

        Returns:
            tensor (int) of shape (self.architecture[l+1]); the values of the hidden nodes of layer (l+1)
        """
        with torch.no_grad():
            activation = F.linear(units, self.weights[l], self.bias[l+1]) # W*v + h_bias
            probability = torch.sigmoid(activation) #p(h|v,pararmeters)
            random_field = torch.rand(probability.size())
            hidden_nodes = self.random_probabilities(probability,random_field)
            del activation
            del probability
            del random_field
        return hidden_nodes

    def backward_units(self,units,l):
        """
        Given the values of the nodes for layer l,
        Calculate the value of the nodes in the layer l-1
        Args:
            units (tensor): tensor of unit values (0s or 1s) of layer l

            l (int): the layer of units you are inputting

        Returns:
            tensor (int) of shape (self.architecture[l+1]); the values of the hidden nodes of layer (l+1)
        """
        with torch.no_grad():
            activation = F.linear(units, torch.transpose(self.weights[l-1],0,1), self.bias[l-1]) # W*h + v_bias
            probability = torch.sigmoid(activation) #p(v|h,pararmeters)
            random_field = torch.rand(probability.size())
            visible_nodes = self.random_probabilities(probability,random_field)
            del activation
            del probability
            del random_field
        return visible_nodes

    def intermediate_layer_probability(self,previous_layer,next_layer,l):
        """
        Given unit values of layers l-1 and l+1, calculate the probability of
        the units in layer l being +1.

        Args:
            previous_layer (tensor): unit values of layer l-1
            next_layer (tensor): unit values of layer l+1
            l (int): the layer for which you want the probabilities

        Returns:
            tensor (float): shape = (architecture[l]). tensor of probabilities.
        """
        with torch.no_grad():
            previous_activation = F.linear(previous_layer, self.weights[l-1], self.bias[l]) #activation due to layer l-1
            next_activation = F.linear(next_layer, torch.transpose(self.weights[l],0,1), self.bias[l]) #activation due to layer l+1
            probability = torch.sigmoid(previous_activation+next_activation)
            del previous_activation
            del next_activation
        return probability

    def intermediate_layer_units(self,previous_layer,next_layer,l):
        """
        Given unit values of layers l-1 and l+1, calculate the unit values of
        the units in layer l.

        Args:
            previous_layer (tensor): unit values of layer l-1
            next_layer (tensor): unit values of layer l+1
            l (int): the layer for which you want the unit values

        Returns:
            tensor (int): shape = (architecture[l]). tensor of unit values.
        """
        with torch.no_grad():
            previous_activation = F.linear(previous_layer, self.weights[l-1], self.bias[l]) #activation due to layer l-1
            next_activation = F.linear(next_layer, torch.transpose(self.weights[l],0,1), self.bias[l]) #activation due to layer l+1
            probability = torch.sigmoid(previous_activation+next_activation)
            random_field = torch.rand(probability.size())
            node_values = self.random_probabilities(probability,random_field)
            del previous_activation
            del next_activation
            del probability
            del random_field
        return node_values

    # def intermediate_layer_probability(self,previous_layer,next_layer,l):
    #     """
    #     Given unit values of layers l-1 and l+1, calculate the probability of
    #     the units in layer l.
    #
    #     Args:
    #         previous_layer (tensor): unit values of layer l-1
    #         next_layer (tensor): unit values of layer l+1
    #         l (int): the layer for which you want the unit values
    #
    #     Returns:
    #         tensor (float): shape = (architecture[l]). tensor of probabilities.
    #     """
    #     with torch.no_grad():
    #         previous_activation = F.linear(previous_layer, self.weights[l-1], self.bias[l]) #activation due to layer l-1
    #         next_activation = F.linear(next_layer, torch.transpose(self.weights[l],0,1), self.bias[l]) #activation due to layer l+1
    #         probability = torch.sigmoid(previous_activation+next_activation)
    #         del previous_activation
    #         del next_activation
    #     return probability


    def Gibbs_alternating_forward(self,even_unit_list):
        """
        Given a list of units for the even layers, calculate the values of the
        units in all of the odd layers.

        Args:
            even_unit_list (list(tensor)): each element in the list corresponds
            to a tensor of layer unit values. Only even layers are included.
            The first element of the list is a tensor of visible unit values; the
            second element is a tensor of unit values of hidden layer 2 etc.

        Returns:
            list(tensor). The unit values of the odd layers.
        """
        with torch.no_grad():
            num_odd = (self.numlayers)//2 #number of odd layers
            num_even = self.numlayers - num_odd #number of even layers

            odd_layer_list = [] #output list to hold the odd layer tensors of unit values

            # If the last hidden layer is an odd layer
            if self.numlayers %2 == 0:
                for i in range(num_odd-1): #i runs over all hidden odd layers except the last

                    previous_layer = even_unit_list[i]
                    next_layer = even_unit_list[i+1]
                    l = 2*i +1 #the actual odd hidden layer you want to get units for
                    hidden_units = self.intermediate_layer_units(previous_layer,next_layer,l)
                    odd_layer_list.append(hidden_units)

                last_even_layer_units = even_unit_list[-1]
                l = self.numlayers - 1 #index of last layer
                last_odd_layer_units = self.forward_units(last_even_layer_units,l-1)
                odd_layer_list.append(last_odd_layer_units)

            #if the last hidden layer is an even layer
            else:
                for i in range(num_odd): #i runs over all hidden odd layers
                    previous_layer = even_unit_list[i]
                    next_layer = even_unit_list[i+1]
                    l = 2*i +1 #the actual odd hidden layer you want to get units for
                    hidden_units = self.intermediate_layer_units(previous_layer,next_layer,l)
                    odd_layer_list.append(hidden_units)

        return odd_layer_list

    def Gibbs_alternating_backward(self,odd_unit_list):
        """
        Given a list of units for the odd layers, calculate the values of the
        units in all of the even layers.

        Args:
            odd_unit_list (list(tensor)): each element in the list corresponds
            to a tensor of layer unit values. Only odd layers are included.
            The first element of the list is a tensor of unit values of the first hidden layer; the
            second element is a tensor of unit values of hidden layer 3 etc.

        Returns:
            list(tensor). The unit values of the even layers.
        """
        with torch.no_grad():
            num_odd = (self.numlayers)//2 #number of odd layers
            num_even = self.numlayers - num_odd #number of even layers

            even_layer_list = [] #output list to hold the even layer tensors of unit values

            # If the last hidden layer is an even layer
            if self.numlayers %2 == 1:

                last_even_units = self.forward_units(odd_unit_list[-1],self.numlayers - 2) #last layer
                visible_units = self.backward_units(odd_unit_list[0],1) #visible layer
                even_layer_list.append(visible_units)
                #intermediate layers
                for i in range(num_even-2): #i runs over all even layers except the last and first
                    previous_layer = odd_unit_list[i]
                    next_layer = odd_unit_list[i+1]
                    l = 2*(i+1) #the actual even hidden layer you want to get units for
                    hidden_units = self.intermediate_layer_units(previous_layer,next_layer,l)
                    even_layer_list.append(hidden_units)

                even_layer_list.append(last_even_units)


            #if the last hidden layer is an odd layer
            else:
                visible_units = self.backward_units(odd_unit_list[0],1) #visible layer
                even_layer_list.append(visible_units)
                #intermediate layers
                for i in range(num_even - 1): #i runs over all hidden even layers
                    previous_layer = odd_unit_list[i]
                    next_layer = odd_unit_list[i+1]
                    l = 2*(i+1) #the actual odd hidden layer you want to get units for
                    hidden_units = self.intermediate_layer_units(previous_layer,next_layer,l)
                    even_layer_list.append(hidden_units)

        return even_layer_list

    def alternating_update(self,visible_units,k):
        """
        Given a tensor of visible units, sample the model using the alternating
        update method and return a list of unit layer tensors.

        Args:
            visible_units (tensor): tensor of visible units (0s and 1s)
            k (int): number of Gibbs sampling steps to perform

        Returns:
            List (tensor): length of list is self.numlayers
        """
        with torch.no_grad():
            num_odd = (self.numlayers)//2 #number of odd layers
            num_even = self.numlayers - num_odd #number of even layers

            even_unit_list = [visible_units]
            for i in range(num_even-1):
                l = 2*(i+1)
                layer_units = torch.zeros(self.architecture[l])
                even_unit_list.append(layer_units)

            for j in range(k):
                #forward pass
                odd_unit_list = self.Gibbs_alternating_forward(even_unit_list)
                even_unit_list = self.Gibbs_alternating_backward(odd_unit_list)

            layers = []

            if self.numlayers % 2 == 0:
                for layer in range(num_even):
                    layers.append(even_unit_list[layer])
                    layers.append(odd_unit_list[layer])

            else:
                layers.append(even_unit_list[0]) #visible
                for layer in range(num_even-1):
                    layers.append(odd_unit_list[layer])
                    layers.append(even_unit_list[layer+1])

        return layers

    def weight_gradient(self,v_0,v_k,l,*args,weight_mask=False):
        """
        Given two tensors, v_0 and v_k, representing states of the units in layer l,
        calculate an estimate of the gradient to the log likelihood wrt the weights connecting
        layer l to layer l+1 using Contrastive Divergence.
        Include the possibility that you don't want to update certain parameters,
        specified in the last argument of this method. The default is that all pararmeters
        are trained.

        Args:
            v_0 (tensor): layer l units at start of CD Markov Chain
            v_k (tensor): layer l units at end of CD Markov Chain
            weight_mask (tensor): binary tensor of 0 and 1. A value of 1 means that the corresponding weight parameter will be trained.
            l (int): the element of self.weights to be updated
        Returns:
            tensor (same shape as self.weights[l])
        """
        with torch.no_grad():
            #If you want to train all of your weight parameters
            if weight_mask == False:
                probability_zero = torch.outer(self.forward_probability(v_0,l),v_0) #p(h_{l+1}|v^(0))v^(0)
                probability_k = torch.outer(self.forward_probability(v_k,l),v_k) #p(h_{l+1}|v^(k))v^(k)
                delta_W = probability_zero - probability_k #get derivatives of all parameters
                return delta_W
            #If you don't want to train all of your weight parameters
            else:
                w_mask = args[0]
                probability_zero = torch.outer(self.forward_probability(v_0,l),v_0) #p(h_{l+1}|v^(0))v^(0)
                probability_k = torch.outer(self.forward_probability(v_k,l),v_k) #p(h_{l+1}|v^(k))v^(k)
                delta_W = probability_zero - probability_k #get derivatives of all parameters
                delta_W = torch.where(w_mask==1,delta_W,w_mask) #set the non trainable weight derivatives to zero
                return delta_W

    def v_bias_gradient(self,v_0,v_k,*args,v_bias_mask=False):
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
            if v_bias_mask == False:
                delta_v_bias = v_0 - v_k #get derivatives of all parameters
                return delta_v_bias
            #If you don't want to train all of your weight parameters
            else:
                v_mask = args[0]
                delta_v_bias = v_0 - v_k #get derivatives of all parameters
                delta_v_bias = torch.where(v_mask==1,delta_v_bias,v_mask) #set the non trainable bias derivatives to zero
                return delta_v_bias

    def h_bias_gradient(self,v_0,v_k,l,*args,h_bias_mask=False):
        """
        Given two tensors, v_0 and v_k, representing states of the units in layer l-1,
        calculate an estimate of the gradient to the log likelihood wrt the hidden
        bias weight parameters of layer l using Contrastive Divergence.
        Include the possibility that you don't want to update certain parameters,
        specified in the last argument of this method. The default is that all pararmeters
        are trained.

        Args:
            v_0 (tensor): layer l-1 units at start of CD Markov Chain
            v_k (tensor): layer l-1 units at end of CD Markov Chain
            h_bias_mask (tensor): binary tensor of 0 and 1. A value of 1 means that the corresponding hidden bias parameter will be trained.
            l (int) : The layer for which you want the bias gradient
        Returns:
            tensor of shape (self.architecture[l])
        """
        with torch.no_grad():
            #If you want to train all of your weight parameters
            if h_bias_mask == False:
                probability_zero = self.forward_probability(v_0,l-1) #p(h_{l}|v^(0))
                probability_k = self.forward_probability(v_k,l-1) #p(h_{l}|v^(k))
                delta_h_bias = probability_zero - probability_k #get derivatives of all parameters
                return delta_h_bias
            #If you don't want to train all of your weight parameters
            else:
                h_mask = args[0]
                probability_zero = self.forward_probability(v_0,l-1) #p(h_{l}|v^(0))
                probability_k = self.forward_probability(v_k,l-1) #p(h_{l}|v^(k))
                delta_h_bias = probability_zero - probability_k #get derivatives of all parameters
                delta_h_bias = torch.where(h_mask==1,delta_h_bias,h_mask) #set the non trainable bias derivatives to zero
                return delta_h_bias

    def batch_gradient_update(self,mini_batch,k,*args,option=False):
        """
        Calculate an estimate to the gradient of log likelihood wrt the weight,
        visible bias and hidden bias parameters of each layer over one mini-batch.

        Args:
            mini_batch (tensor): a tensor of training mini batch data. The size of the
                                    tensor is (batch_size,nv)

            k (int): The number of Gibbs sampling steps

            *args (tensor): Binary masks for the parameters to specify which parameters get trained.
                            weight_mask = args[0] : a list of weight masks for each layer
                            bias_mask = args[1] : a list of bias masks for each layer

        Returns:
            None. Updates self.parameter.grad
        """
        with torch.no_grad():
            batch_size = len(mini_batch)

            #initialise the gradients of all parameters to zero
            weight_gradients = []
            bias_gradients = [torch.zeros(self.nv)]
            for layer in range(self.numlayers-1):
                weight_gradients.append(torch.zeros(self.architecture[layer+1],self.architecture[layer]))
                bias_gradients.append(torch.zeros(self.architecture[layer+1]))


            #for each training vector in your batch, make an update to all parameters
            for visible in mini_batch:
                dream_list = self.alternating_update(visible,k)
                #update visible
                v0 = visible
                vk = dream_list[0]
                delta_W = self.weight_gradient(v0,vk,0)
                delta_bias = self.v_bias_gradient(v0,vk) #update visible bias gradient
                weight_gradients[0] += delta_W
                bias_gradients[0] += delta_bias
                delta_bias_first_hidden = self.h_bias_gradient(v0,vk,1)
                bias_gradients[1] += delta_bias_first_hidden
                #update hidden
                h0 = self.forward_units(v0,0)
                for layer in range(self.numlayers-2):
                    hk = dream_list[layer+1]
                    delta_W = self.weight_gradient(h0,hk,layer+1)
                    delta_bias = self.h_bias_gradient(h0,hk,layer+2) #update hidden bias gradient

                    weight_gradients[layer+1] += delta_W
                    bias_gradients[layer+2] += delta_bias

                    h0 = self.forward_units(h0,layer+1) #get h_{layer+2}^{0}

            if option == False:
                for layer in range(self.numlayers-1):
                    self.weights[layer].grad = -weight_gradients[layer]
                    self.bias[layer].grad = -bias_gradients[layer]
                self.bias[-1].grad = -bias_gradients[-1]


            else:
                weight_mask = args[0]
                bias_mask = args[1]

                for layer in range(self.numlayers-1):
                    # print(weight_gradients[layer].size())
                    # print(weight_mask[layer].size())
                    weight_gradients[layer] = torch.where(weight_mask[layer]==1,weight_gradients[layer],weight_mask[layer]) #set the non trainable weight derivatives to zero
                    weight_gradients[layer] = torch.where(weight_mask[layer]==2,torch.zeros(weight_mask[layer].size()),weight_gradients[layer]) #set the block connections to 1000
                    bias_gradients[layer] = torch.where(bias_mask[layer]==1,bias_gradients[layer],bias_mask[layer]) #set the non trainable visible bias derivatives to zero
                    bias_gradients[layer] = torch.where(bias_mask[layer]==-1,torch.zeros(bias_mask[layer].size()),bias_gradients[layer]) #set the non trainable visible bias derivatives to zero
                    self.weights[layer].grad = -weight_gradients[layer]
                    self.bias[layer].grad = -bias_gradients[layer]


                #1 more bias layer than weight layer
                bias_gradients[-1] = torch.where(bias_mask[-1]==1,bias_gradients[-1],bias_mask[-1])
                bias_gradients[-1] = torch.where(bias_mask[-1]==-1,torch.zeros(bias_mask[-1].size()),bias_gradients[-1])
                self.bias[-1].grad = -bias_gradients[-1]


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
                            weight_mask = args[0] : a list of weight masks for each layer
                            bias_mask = args[1] : a list of bias masks for each layer


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
                bias_mask = args[1]
                self.batch_gradient_update(mini_batch,k,weight_mask,bias_mask,option=True) #update the trainable gradients
                optimiser_class.step() #update the trainable parameters

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
                sampled_list = self.alternating_update(visible,1)
                v1 = sampled_list[0]
                mse = F.mse_loss(visible,v1)
                batch_error+=float(mse)

            batch_size = len(mini_batch)
            mean_batch_error = batch_error/batch_size

        return mean_batch_error
