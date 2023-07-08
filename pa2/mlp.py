# -*- coding: utf-8 -*-
# *** Do not modify the code ***

#   *** Do not import any library except already imported libraries ***
import numpy as np
import math
import random
#   *** Do not import any library except already imported libraries ***

class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        self.W1 = np.random.uniform(low = -0.1, high = 0.1, size = (hidden_size, input_size))
        self.B1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.uniform(low = -0.1, high = 0.1, size = (output_size, hidden_size))
        self.B2 = np.zeros((output_size, 1))
        
        self.grad_W1 = np.zeros(self.W1.shape)
        self.grad_B1 = np.zeros(self.B1.shape)
        self.grad_W2 = np.zeros(self.W2.shape)
        self.grad_B2 = np.zeros(self.B2.shape)

        self.input = None
        self.hidden = None
        self.output = None
        self.label = None
        self.ReLU_mask = None

        self.lr = learning_rate

    def forward(self, x):
        self.input = np.array(x).reshape((len(x), 1)) # reshape input to numpy array (input_size, 1)

        hidden = np.dot(self.W1, self.input) + self.B1
        self.hidden = self.ReLU(hidden)

        output = np.dot(self.W2, self.hidden) + self.B2
        self.output = self.softmax(output)

        return self.output

    def backward(self):
        dL_dO = self.output - self.label
        
        self.grad_W2 = np.dot(dL_dO, self.hidden.T)
        self.grad_B2 = dL_dO
        
        dL_dReLU = np.dot(self.W2.T, dL_dO)
        dL_dH = np.multiply(self.ReLU_mask, dL_dReLU)

        self.grad_W1 = np.dot(dL_dH, self.input.T)
        self.grad_B1 = dL_dH

    def step(self):
        self.W1 -= self.lr * self.grad_W1
        self.W2 -= self.lr * self.grad_W2
        self.B1 -= self.lr * self.grad_B1
        self.B2 -= self.lr * self.grad_B2

    def loss(self, label, logits):
        # make label(int) to numpy array (num_label, 1)
        label_array = np.zeros(logits.shape)
        label_array[label] = 1.0

        self.label = label_array

        return -np.sum(self.label * np.log(logits)) # Cross Entropy Loss

    def softmax(self, x):
        if np.sum(np.exp(x)) == 0:
            import pdb; pdb.set_trace()
        return np.exp(x) / np.sum(np.exp(x))

    def ReLU(self, x):
        self.ReLU_mask = np.zeros(x.shape)
        self.ReLU_mask[x >= 0] = 1.0

        return np.multiply(self.ReLU_mask, x)

# *** Do not modify the code ***