# import torch

# # Create a tensor
# tensor = torch.tensor([[0, 1, 2, 0, 0],
#                        [3, 4, 5, 6, 0],
#                        [7, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0]])

# # Iterate over each row of the tensor
# for row in tensor:
#     # Reverse the row and find the index of the first non-zero element
#     non_zero_index = (row != 0).nonzero(as_tuple=False).flip(dims=(0,))[0, 0]
#     # Compute the length by adding 1 to the index
#     length = non_zero_index + 1
#     print("Length:", length)

import torch

# Example shapes
batch_size = 2
sequence_length = 3
vocab_size = 4

# Example tensors
logits = torch.randn(batch_size, sequence_length, vocab_size)
labels = torch.tensor([[0, 1, 2], [3, 2, 1]])

# Compute per_token_logps
per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

# Print the shape of per_token_logps
print(per_token_logps.shape)