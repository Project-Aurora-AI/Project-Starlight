import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the model architecture.
        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layers.
            output_size (int): The size of the output (e.g., number of classes).
        """
        super(CustomModel, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Define an activation function (ReLU in this case)
        self.relu = nn.ReLU()
        
        # If you're using batch normalization or dropout, you can add those here
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output of the network after forward pass.
        """
        # Pass through the first layer and apply activation function
        x = self.fc1(x)
        x = self.relu(x)
        
        # Pass through the second layer and apply activation function
        x = self.fc2(x)
        x = self.relu(x)
        
        # Apply dropout (helps with regularization)
        x = self.dropout(x)
        
        # Pass through the third layer (output layer)
        x = self.fc3(x)
        
        # Optionally, you can apply softmax or another activation on the output layer
        return x

    def summary(self):
        """
        Prints the summary of the model architecture.
        """
        print(self)

def build_model(config):
    # initialize your model here using config
    model = CustomModel(config)
    return model

# Example usage:
if __name__ == "__main__":
    # Example input size, hidden size, and output size (for a classification task)
    input_size = 784  # e.g., for MNIST, 28x28 images flattened into a vector
    hidden_size = 128
    output_size = 10  # e.g., 10 classes for MNIST

    # Initialize the model
    model = CustomModel(input_size, hidden_size, output_size)
    
    # Print model architecture summary
    model.summary()
