import torch
from torch import Tensor
from ...tree import Forest

def subtensor(tensor: Tensor, start: Tensor, length: Tensor):
    start = start.reshape(-1, 1)
    length = length.reshape(-1, 1)
    end = start + length
    indices = torch.arange(tensor.shape[1], device="cuda") + start
    indices = torch.clamp(indices, max=tensor.shape[1] - 1)
    shifted_tensor = tensor.gather(1, indices)
    return torch.where(indices < end, shifted_tensor, 0)

def vmap_subtree(forest: Forest, pos):
    pos = pos.reshape(-1, 1)
    length = forest.batch_subtree_size.gather(1, pos)
    return Forest(
        forest.input_len,
        forest.output_len,
        subtensor(forest.batch_node_value, pos, length),
        subtensor(forest.batch_node_type, pos, length),
        subtensor(forest.batch_subtree_size, pos, length),
    )