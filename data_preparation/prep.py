import torch
from datetime import datetime
import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
import io
import struct
from torch.autograd import Variable

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)

# Global variables 
pieces = ['-', # empty
          'r', 'n', 'b', 'q', 'k', 'p', 
          'R', 'N', 'B', 'Q', 'K', 'P']
piece_values = [0,
                -5 ,-3, -3, -9, -2 ,-1,
                5 ,3, 3, 9, 2 ,1,]

def fen_to_matrix(fen: str):
  """ Convert FEN board notation to a matrix
  """
  result = []
  for row in fen.split("/"):
    new_row = []
    for char in row:
      if char in pieces:
        new_row.append(pieces.index(char))
      else:
        for i in range(int(char)):
          new_row.append(0)
    result.append(new_row)
  return result

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_inverse(x):
    return np.log(x / (1 - x))

# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23?page=2
def to_one_hot(y, n_dims=13):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)  #flattened it
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def add_piece_value(y_one_hot, piece_values=piece_values):
  """Add the values of pieces to the 8x8x13 tensor
  
  Instead of one-hot for every layer, the value of the pieces are included
  
  Args:
    y_one_hot: the tensor right after going through the function to_one_hot()
    piece_values: already defined pawn-equivalent values of the different pieces"""
  y_tensor = y_one_hot.data if isinstance(y_one_hot, Variable) else y_one_hot
  for i in range(len(piece_values)):
    y_tensor[:,:,i] = y_tensor[:,:,i] * piece_values[i]
  return Variable(y_one_hot) if isinstance(y_one_hot, Variable) else y_one_hot

def mat_to_input_tensor(board_mat, fen_string):
  """Turn position matrix to the desired input tensor.

  Take the input of a position matrix and go through a number of transformations.
  First one if the one-hot encoding and produce the 8x8x13 skeleton. Then add piece 
  values to each layer. Another step if to include the legal moves of the pieces
  which is a proxy for the controled area of either side. 
  """
  # Preparation for generating legal moves
  algebraic_to_index = {f"{file}{rank}": (8 - rank, ord(file) - ord('a')) for rank in range(8, 0, -1) for file in 'abcdefgh'}
  moves_by_piece_type = {'P': [], 'N': [], 'B': [], 'R': [], 'Q': [], 'K': [], 'p': [], 'n': [], 'b': [], 'r': [], 'q': [], 'k': []}
  board = chess.Board(fen_string)

  for turn in [True, False]:
    board.turn = turn
    for move in board.legal_moves:
      piece = board.piece_at(move.from_square)
      if piece:
        moves_by_piece_type[str(piece)].append(move.uci()[2:])
  
  # Transformations of the board matrix
  y = to_one_hot(board_mat)
  y = add_piece_value(y)

  # include legal moves in the input tensor
  for piece, move_list in moves_by_piece_type.items():  # Iterate through each pieces and their moves
    i = pieces.index(piece) # Find out which layer of tensor is this piece
    c = 1 if piece.isupper() else -1  # 1 for white nad -1 for black
    for square in move_list:
      rank, file = algebraic_to_index[square[:2]] # moves that lead to promotion have 3 letters
      y[:,:,i][rank][file] = c
  
  return Variable(y) if isinstance(y, Variable) else y

class StockFishDataPreparation():
  def __init__(self, csv_file, percentage):
    self.data = pd.read_csv(csv_file, encoding='utf-8-sig')
    self.data = self.data.sample(frac=percentage, ignore_index=True)
    self.data['Evaluation'] = self.data['Evaluation'].apply(lambda x: x if x[0] != '#' else x[1] + "10000")
    self.data['Evaluation'] = self.data['Evaluation'].apply(lambda x: x if (not x.startswith('\ufeff')) else x[1:])
    self.data['Evaluation'] = self.data['Evaluation'].astype(float)
    self.data['Evaluation'] = self.data['Evaluation'].apply(lambda x: sigmoid(x/1000))

  def prepareAndSaveToDisk(self, output_tensors, output_evals):
    with open(output_tensors, 'ab') as f_tensors:
      with open(output_evals, 'ab') as f_evals:
        # batch process 100 at a time
        for chunk in np.array_split(self.data, 100):
          for index, row in chunk.iterrows():
            # Write tensor
            fen_string = row['FEN'].split(' ')[0]
            buffer = io.BytesIO()
            tensor = mat_to_input_tensor(torch.Tensor(fen_to_matrix(fen_string)), fen_string)
            torch.save(tensor, buffer)
            byts = buffer.getvalue()
            f_tensors.write(byts)
            # Write evals
            eval = row['Evaluation']
            s = struct.pack('d', eval)
            f_evals.write(s)
    # Write length of dataset to file
    with open("len", "w") as f:
      f.write(str(len(self.data)))

if __name__ == "__main__":
  # Prepare 5% of the data (takes ~25 min)
  # Save to files "tensors","eval", "len"
  data_prep = StockFishDataPreparation("./chessData.csv", 0.05)
  data_prep.prepareAndSaveToDisk("tensors", "eval")