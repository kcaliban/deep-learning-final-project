from model import *
from data import *
import sys
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)

MAX_CENTIPAWN_VALUE = 10000

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inverse(x):
    return np.log(x / (1 - x))

# Load dataset
dataset = loadDataSet(sys.argv[1])
dataset_size = len(dataset)

# Load model state
model_path = sys.argv[2]
model = CustomNet(device=device).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Get predictions and actual values
def get_predictions_actual(model, dataset, dataset_size):
    predictions = []
    actual = []
    for i in range(10000):  # the whole dataset takes too long
        print(f"Processing {i}/{dataset_size}")
        data = dataset[i]
        output = model(torch.reshape(data["fen"], (-1, 13, 8, 8))).detach().cpu()
        eval = data["eval"].detach().cpu()
        predictions.append(output)
        actual.append(eval)
    return (predictions, actual)

def to_centipawns(x):
    res = sigmoid_inverse(x)
    if res == np.inf:
        return MAX_CENTIPAWN_VALUE
    return 100 * res

# Calculate MSE on the dataset
def centipawn_MSE(predictions, actual):
    mse = 0
    for i in range(len(predictions)):
        print(predictions[i][0], actual[i])
        val = (to_centipawns(predictions[i][0]) - to_centipawns(actual[i])) ** 2
        print(val)
        mse += val
    print(mse)
    mse /= len(predictions)
    return mse

# Plot actual on x-axis and predictions on y-axis as a scatter plot
def plot(predictions, actual):
    # Add in a title and axes labels
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    plt.scatter(actual, predictions)

    # Add line of equal values
    plt.plot([0, 1], [0, 1], color='red')

    plt.show()

# Plot actual on x-axis and predictions on y-axis as a scatter plot in centipawns
def plot_centipawns(predictions, actual):
    # Add in a title and axes labels
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predictions')
    centipawn_actual = list(map(lambda x: to_centipawns(x), actual))
    centipawn_predictions = list(map(lambda x: to_centipawns(x), predictions))
    plt.scatter(centipawn_actual, centipawn_predictions)

    # Add line of equal values
    plt.plot([min(centipawn_actual), max(centipawn_actual)], [min(centipawn_actual), max(centipawn_actual)], color='red')

    plt.show()

# Plot sigmoid function for reference
def plot_sigmoid():
    x = np.linspace(-2000, 2000, 1000)
    y = sigmoid(x/100)
    plt.plot(x, y)
    plt.plot(x, sigmoid(x/1000))
    plt.show()

predictions, actual = get_predictions_actual(model, dataset, dataset_size)
# print(centipawn_MSE(predictions, actual))
plot_centipawns(predictions, actual)
# plot(predictions, actual)
# plot_sigmoid()
