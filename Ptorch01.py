import torch

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([2, 3, 4, 5])

print(x + y)

z = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(z)