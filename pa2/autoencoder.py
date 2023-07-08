# -*- coding: utf-8 -*-

#   *** Do not import any library except already imported libraries ***
import numpy as np
import math
import random
#   *** Do not import any library except already imported libraries ***

class AutoEncoder:
    def __init__(self, input_size: int, hidden_size: int, latent_size: int, output_size: int, learning_rate: float):
        '''
        Refer to mlp.py
        '''
        ############################################## EDIT HERE ###########################################
        
        # ininiiate weights and biases with He initialization
        def He(size):
            limit = np.sqrt(2 / np.prod(size[:-1]))
            return np.random.uniform(low=-limit, high=limit, size=size)
        
        # input
        self.W_i1 = He((hidden_size, input_size))
        self.B_i1 = np.zeros((hidden_size, 1))
        self.W_i2 = He((latent_size, hidden_size))
        self.B_i2 = np.zeros((latent_size, 1))

        # output
        self.W_o1 = He((hidden_size, latent_size))
        self.B_o1 = np.zeros((hidden_size, 1))
        self.W_o2 = He((output_size, hidden_size))
        self.B_o2 = np.zeros((output_size, 1))

        # input gradients
        self.grad_W_i1 = np.zeros((hidden_size, input_size))
        self.grad_B_i1 = np.zeros((hidden_size, 1))
        self.grad_W_i2 = np.zeros((latent_size, hidden_size))
        self.grad_B_i2 = np.zeros((latent_size, 1))
        
        # output gradients
        self.grad_W_o1 = np.zeros((hidden_size, latent_size))
        self.grad_B_o1 = np.zeros((hidden_size, 1))
        self.grad_W_o2 = np.zeros((output_size, hidden_size))
        self.grad_B_o2 = np.zeros((output_size, 1))

        '''
        Define any additional variables you need
        '''

        self.input = None
        self.hidden1 = None
        self.latent = None
        self.hidden2 = None
        self.output = None

        self.ReLU_mask = None
        self.ReLU_mask1 = None
        self.ReLU_mask2 = None
        self.ReLU_mask3 = None
        self.ReLU_mask4 = None

        self.lr = learning_rate

        ################################################# END ##############################################

    def forward(self, x):
        ############################################## EDIT HERE ###########################################
        
        self.input = np.array(x).reshape((len(x), 1)) # reshape input to numpy array (input_size, 1)
        
        # hidden 1
        hidden1 = np.dot(self.W_i1, self.input) + self.B_i1
        self.hidden1 = self.ReLU(hidden1)
        self.ReLU_mask1 = self.ReLU_mask

        # latent
        latent = np.dot(self.W_i2, self.hidden1) + self.B_i2
        self.latent = self.ReLU(latent)
        self.ReLU_mask2 = self.ReLU_mask
        
        # hidden 2
        hidden2 = np.dot(self.W_o1, self.latent) + self.B_o1
        self.hidden2 = self.ReLU(hidden2)
        self.ReLU_mask3 = self.ReLU_mask
        
        # output
        output = np.dot(self.W_o2, self.hidden2) + self.B_o2
        self.output = self.ReLU(output)
        self.ReLU_mask4 = self.ReLU_mask

        return self.output
        ################################################# END ##############################################

    def backward(self):
        ############################################## EDIT HERE ###########################################
            
            dL_dO = self.output - self.input

            # Calculate gradients using MSE loss

            # output
            dOutput_dReLU4 = self.ReLU_mask4
            dL_dReLU4 = dL_dO * dOutput_dReLU4
            # Update gradients output
            self.grad_W_o2 = np.dot(dL_dReLU4, self.hidden2.T)
            self.grad_B_o2 = dL_dReLU4
            
            # hidden 2
            dL_dHidden2 = np.dot(self.W_o2.T, dL_dReLU4)
            dH2_dReLU3 = self.ReLU_mask3
            dL_dReLU3 = dL_dHidden2 * dH2_dReLU3
            # Update gradients hidden 2
            self.grad_W_o1 = np.dot(dL_dReLU3, self.latent.T)
            self.grad_B_o1 = dL_dReLU3

            # latent
            dL_dHidden1 = np.dot(self.W_o1.T, dL_dReLU3)
            dH1_dReLU2 = self.ReLU_mask2
            dL_dReLU2 = dL_dHidden1 * dH1_dReLU2
            # Update gradients latent
            self.grad_W_i2 = np.dot(dL_dReLU2, self.hidden1.T)
            self.grad_B_i2 = dL_dReLU2
            
            # hidden 1
            dL_dLatent = np.dot(self.W_i2.T, dL_dReLU2)
            dLatent_dReLU1 = self.ReLU_mask1
            dL_dReLU1 = dL_dLatent * dLatent_dReLU1
            # Update gradients hidden 1
            self.grad_W_i1 = np.dot(dL_dReLU1, self.input.T)
            self.grad_B_i1 = dL_dReLU1

        ################################################# END ##############################################

    def step(self):
        self.W_i1 -= self.lr * self.grad_W_i1
        self.B_i1 -= self.lr * self.grad_B_i1
        self.W_i2 -= self.lr * self.grad_W_i2
        self.B_i2 -= self.lr * self.grad_B_i2
        
        self.W_o1 -= self.lr * self.grad_W_o1
        self.B_o1 -= self.lr * self.grad_B_o1
        self.W_o2 -= self.lr * self.grad_W_o2
        self.B_o2 -= self.lr * self.grad_B_o2

    def ReLU(self, x):
        self.ReLU_mask = np.zeros(x.shape)
        self.ReLU_mask[x >= 0] = 1.0

        return np.multiply(self.ReLU_mask, x)
