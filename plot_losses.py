import matplotlib.pyplot as plt
import numpy as np
import sys

# Plot the training and validation loss values of "log.txt" file
def plot(path):
    # Read the log file
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Extract the loss and validation loss values
    losses = [float(line.split('\t')[0]) for line in lines[1:]]
    vlosses = [float(line.split('\t')[1]) for line in lines[1:]]

    # Get the number of epochs
    EPOCHS = len(losses)

    # Generate a sequence of integers to represent the epoch numbers
    epochs = range(1, EPOCHS + 1)
    
    # Plot and label the training and validation loss values
    plt.plot(epochs, losses, label='Training Loss')
    plt.plot(epochs, vlosses, label='Validation Loss')
    
    # Add in a title and axes labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Set the tick locations
    plt.xticks(np.arange(0, EPOCHS + 1, 2))
    
    # Display the plot
    plt.legend(loc='best')
    plt.show()

path = sys.argv[1]
plot(path)