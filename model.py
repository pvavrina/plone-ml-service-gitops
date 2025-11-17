# model.py: Simple PyTorch Neural Network Definition

import torch.nn as nn

# Define a simple linear model class
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        # Define one linear layer
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Pass the input through the linear layer
        return self.fc1(x)
