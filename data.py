import torch
import io
import os
import struct

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

device = torch.device(dev)

class StockfishEvaluationsDatasetFromDisk(torch.utils.data.Dataset):
    def __init__(self, fen, eval, number_of_elements):
        self.fen = open(fen, 'rb')
        self.eval = open(eval, 'rb')
        self.len = number_of_elements

        self.tensor_size_in_bytes = 4075
        self.eval_size_in_bytes = 8 # double precision float has 8 bytes

        # Read all evals, we can keep them in memory
        self.evals_struct = struct.unpack("d" * self.len, self.eval.read(self.eval_size_in_bytes * self.len))
        self.evals = torch.tensor(self.evals_struct)
        self.eval.close()

    def __del__(self):
        self.fen.close()

    def __len__(self):
        return self.len
  
    def readTensorsFromDisk(self, idx):
        self.fen.seek(self.tensor_size_in_bytes * idx)
        tensor_bytes = self.fen.read(self.tensor_size_in_bytes)
        buffer = io.BytesIO(tensor_bytes)
        return torch.load(buffer)

    def readEvalFromDisk(self, idx):
        self.eval.seek(self.eval_size_in_bytes * idx)
        eval_bytes = self.eval.read(self.eval_size_in_bytes)
        return struct.unpack('d', eval_bytes)

    def __getitem__(self, idx):
        return { "fen": self.readTensorsFromDisk(idx).to(device),
                 "eval": self.evals[idx].to(device) }
    

def loadDataSet(path):
    dataset_path = path
    dataset_size = 0
    with open(os.path.join(dataset_path, "len")) as f:
        dataset_size = int(f.readlines()[0].strip())
        dataset = StockfishEvaluationsDatasetFromDisk(os.path.join(dataset_path, "tensors"), os.path.join(dataset_path, "eval"), dataset_size)
    return dataset
