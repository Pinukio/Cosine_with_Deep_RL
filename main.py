# 바닥부터 배우는 강화 학습 P.184 Deep RL을 이용한 Cosine Function Fitting 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Input이 1개, Output이 128개
        self.fc1 = nn.Linear(1, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        # 출력층
        self.fc4 = nn.Linear(128, 1, bias=False) 

    # Forward Propagation
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 연쇄적으로 연산해줌
        x = F.relu(self.fc3(x)) 
        # 출력층의 Activation Function은 Identity Function을 사용
        x = self.fc4(x) 
        return x
    
# 실제 Target이 되는 Cosine Function.
def cosine_function(X):
    # -0.2 ~ 0.2 사이의 랜덤값을 Noise로 줌
    # mini-batch를 고려하여 개수를 맞춰줌
    noise = np.random.randn(X.shape[0]) * 0.2
    return np.cos(1.5 * np.pi * X) + X + noise
        
def plot_results(model):
    x = np.linspace(0, 5, 100)
    input_x = torch.from_numpy(x).float().unsqueeze(1)
    plt.plot(x, cosine_function(x), label="Cosine")
    plt.plot(x, model(input_x).detach().numpy(), label="Prediction")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim((0,5))
    plt.ylim((-1,5))
    plt.grid()
    plt.show()

def main():
    # 0~5 사이 숫자 1만 개를 Sampling
    data_x = np.random.rand(10000) * 5

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for step in range(10000):
        # mini-batch로 32개의 데이터를 사용
        batch_x = np.random.choice(data_x, 32)
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
        # 순전파 연산
        pred = model.forward(batch_x_tensor)
        
        batch_y = cosine_function(batch_x)
        truth = torch.from_numpy(batch_y).float().unsqueeze(1)
        # MSE 계산
        loss = F.mse_loss(pred, truth)

        # 누적 Gradient 초기화, AutoGrad 기능의 Cash영역을 날리는 걸까?
        optimizer.zero_grad()
        # Tensor.backward(), Back propagation을 통한 Gradient 계산
        loss.mean().backward()
        # Gradient를 이용해 실제로 Parameter를 업데이트함
        optimizer.step()

    plot_results(model)

main()