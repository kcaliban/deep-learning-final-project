import sys
from model import *
from data import *
from data_preparation.prep import *
import numpy as np
import torch

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

def to_centipawns(x):
    res = sigmoid_inverse(x)
    if res == np.inf:
        return MAX_CENTIPAWN_VALUE
    return 1000 * res

# Load model state
model = CustomNetV3(device=device).to(device)
model.load_state_dict(torch.load("./model/final"))
model.eval()

# Get FEN
fen_string = sys.argv[1]

# Transform to tensor
tensor = mat_to_input_tensor(torch.Tensor(fen_to_matrix(fen_string)), fen_string).to(device=device)

# Get evaluation
eval = model(torch.reshape(tensor, (-1, 13, 8, 8))).detach().cpu().item()

# Convert to centipawns
centipawns = to_centipawns(eval)

# Print result
if abs(MAX_CENTIPAWN_VALUE - centipawns) < 100:
    if centipawns < 0:
        print("Mate for black in unknown amount of moves")
    else:
        print("Mate for white in unknown amount of moves")
else:
    print(f"Estimated evaluation: {centipawns/100:.3f} pawns")