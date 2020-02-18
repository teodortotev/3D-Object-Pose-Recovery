import torch.nn as nn
import torch

class CNNET(nn.Module):
    def __init__(self):
        super(CNNET, self).__init__()
        self.features = nn.Linear(32, 64)

    def forward(self, x):
        print("Here noe")
        return self.features(x)

input = torch.zeros(32)
model = CNNET()
print(input)
print(model)
print("Before output")
output = model(input)
print("After output")
print(output)