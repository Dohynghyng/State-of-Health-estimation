import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import network
import utils
import data_preprocessing
import matplotlib.pyplot as plt


# setting #
battery = 'B0005' # B0005, B0006, B0007, B0018
epoch = 100
learning_rate = 0.001
batch_size = 2
hidden_dim = 256
layer_dim = 3
train_ratio = 0.7
scaler = StandardScaler()


X,y,total_volume = data_preprocessing.load_data(battery)
X, y, fx, TEM, Time = data_preprocessing.preprocessing(X,y,scaler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device

linear_X_train, linear_X_test,linear_fx_train,linear_fx_test, linear_y_train, linear_y_test, linear_train_idxs, linear_test_idxs = data_preprocessing.rand_sampling(X,fx,y,False,train_ratio=train_ratio)
criterion = nn.MSELoss()

linear_model = network.my_lstm(hidden_dim, layer_dim).to(device)
linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)

linear_losses = []
rand_losses = []

min_rand_test_loss = 100
min_linear_test_loss = 100
beat_linear_train_loss = 100

for i in range(1,epoch+1):
  # 1부터 100까지의 값을 순회하는 반복문
  # 100개의 에폭을 나타냄
  if i % 100 == 0:
    linear_optimizer.param_groups[0]['lr'] *= 0.5
    # 에폭이 100의 배수일 때 옵티마이저의 학습률을 0.5 배로 감소시킴
  linear_loss = utils.train(linear_model,device,criterion,linear_optimizer,linear_X_train,linear_fx_train,linear_y_train,batch_size)
  # 모델 학습 손실 저장
  # 모델을 한 번의 에폭 동안 학습

  # 현재 에폭의 번호, 손실 값을 출력
  print(f'epoch : {i} / linear_loss : {round(linear_loss,5)}')
  linear_test_loss = utils.test(linear_model,device,linear_X_test,linear_fx_test, linear_y_test)
  linear_train_loss = utils.test(linear_model,device,linear_X_train,linear_fx_train,linear_y_train)

  if min_linear_test_loss > linear_test_loss:
    min_linear_test_loss = linear_test_loss
    torch.save(linear_model, './linear_model.pt')

  if beat_linear_train_loss > linear_train_loss:
    beat_linear_train_loss = linear_train_loss
    torch.save(linear_model, './best_train_linear_model.pt')

tensor = torch.FloatTensor(X).to(device)
tensor_fx = torch.FloatTensor(fx).to(device)

linear_model = torch.load('linear_model.pt')
best_linear_model = torch.load('best_train_linear_model.pt')

linear_pred = linear_model(tensor,tensor_fx)
best_linear_pred = best_linear_model(tensor,tensor_fx)

plt.plot(linear_pred.cpu().detach().numpy()/total_volume *100) # 첫번째 값으로
plt.plot(y/total_volume *100)
for i in linear_test_idxs:
  plt.axvline (i,c='r',linestyle='dashed',alpha=0.4,linewidth=1)
plt.ylabel('SOH (%)')
plt.xlabel('Cycle')
plt.title('Proposed Method')
plt.legend(['pred','target','test set'])
plt.show()
print(f'Error : {min_linear_test_loss/total_volume * 100} %')

plt.plot(best_linear_pred.cpu().detach().numpy()/total_volume *100) # 첫번째 값으로
plt.plot(y/total_volume *100)
for i in linear_test_idxs:
  plt.axvline (i,c='r',linestyle='dashed',alpha=0.4,linewidth=1)
plt.ylabel('SOH (%)')
plt.xlabel('Cycle')
plt.title('Traditional Sampling performance')
plt.legend(['pred','target','test set'])
plt.show()
linear_test_loss = utils.test(best_linear_model,device,linear_X_test,linear_fx_test,linear_y_test)
print(f'Error : {linear_test_loss/total_volume * 100} %')