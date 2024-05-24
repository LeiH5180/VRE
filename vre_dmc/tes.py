import torch
A = torch.tensor([1,2,3],dtype=torch.float32)
B = torch.tensor([2,4,5],dtype=torch.float32)
C = torch.mean(A)
print(C)