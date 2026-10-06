import copy
import torch
import torch.nn as nn

#Uses GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    """
    Fully connected neural network with two hidden layers and ReLu activations.
    Is used as base for all merging experiments

    """

    def __init__(self, hidden_size=512, input_size=28 * 28, output_size=10):
        """
        Args:
            hidden_size:    number of neurons in each hidden layer (default 512)
            input_size:     number of input features (default 784)
            output_size:    number of output classes (default 10 for MNIST/FashionMNIST)
        """
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(model, dataloader, epochs, model_name, lr=1e-3):
    """
    Trains a neural network using the Adam optimizer and cross-entropy loss.

    Args:
        model:          neural network to train
        dataloader:     DataLoader with training data
        epochs:         number of training epochs
        model_name:     name shown in training progress output
        lr:             learning rate for Adam optimizer (default 1e-3)

    Returns:
        None 
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    size = len(dataloader.dataset)
    model.train()

    for epoch in range(epochs):

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 500 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"Model: {model_name} | "
                      f"Epoch {epoch:3d} | "
                      f"Loss {loss:.4f} | "
                      f"Samples {current:5d}/{size:5d}")

def evaluate(model, dataloader, model_name=None):
    """
    Evaluates a neural network on a test dataset
    
    Args:
        model:          neural network to evaluate
        dataloader:     DataLoader with test data
        model_name:     name shown during evaluation log

    Returns:
        accuracy as a float between 0 and 1
    """
    
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss = 0
    accuracy = 0

    with torch.no_grad():

        for X, y in dataloader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy /= size

    print(
        f"Model: {model_name} | "
        f"Accuracy: {100 * accuracy:.1f}% | "
        f"Avg loss: {test_loss:.4f}")
    return accuracy

def clone_model(model):
    """
    Creates a deep copy of a model 

    Args:
        model:  neural network to copy

    Returns:
        independent copy of the model 
    """
    
    return copy.deepcopy(model)


# Predefined model sizes for experiments
# All share the same layout with two hidden layers, 
# only difference is hidden layer width

class NeuralNetwork512(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=512)


class NeuralNetwork256(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=256)

class NeuralNetwork128(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=128)


class NeuralNetwork64(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=64)


class NeuralNetwork32(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=32)

class NeuralNetwork16(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=16)


class NeuralNetwork8(NeuralNetwork):
    def __init__(self):
        super().__init__(hidden_size=8)    
        


