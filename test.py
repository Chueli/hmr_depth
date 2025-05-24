import torch

f = torch.nn.Conv2d(3, 8, 3, device="cuda")
X = torch.randn(2, 3, 4, 4, device="cuda")

Y = X @ X

print("matrix multiply works")

Y = f(X)

print("Conv2d works")

device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device_str)