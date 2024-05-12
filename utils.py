import numpy as np
import torch

def get_batch(X,fx,y,device,batch_size):
  idxs = np.arange(len(X))
  np.random.shuffle(idxs)
  rand_idxs = idxs[:batch_size]
  X_batch, fx_batch, y_batch = [],[],[]
  for idx in rand_idxs:
    X_batch.append(X[idx])
    fx_batch.append(fx[idx])
    y_batch.append(y[idx])
  return torch.FloatTensor(X_batch).to(device),torch.FloatTensor(fx_batch).to(device),torch.FloatTensor(y_batch).to(device)

def train(model,device,criterion,optimizer,X_train,fx_train,y_train,batch_size):
  model.train()
  # 모델을 학습 모드로 변경
  for i in range(int(len(X_train)/batch_size)+1): #학습 반복
    optimizer.zero_grad()
    # optimizer 변화도 초기화
    X_batch,fx_batch, y_batch = get_batch(X_train,fx_train,y_train,device,batch_size)
    # get_batch 함수를 사용하여 입력 데이터와 대상 데이터를 배치 단위로 가져옴
    pred = model(X_batch,fx_batch)
    # X_batch 예측값 계산
    loss = torch.sqrt(criterion(pred, y_batch))
    # 손실 계산
    loss.backward()
    # 역전파 수행
    optimizer.step()
    # 모델 파라미터 업데이트
    # 위 과정을 배치 개수 + 1 만큼 반복
  return loss.item()
  # 마지막 배치 오차 반환

def test(model,device,X,fx,y):
  tensor = torch.FloatTensor(X).to(device)
  model.eval()
  with torch.no_grad():
    X,fx,y = torch.FloatTensor(X).to(device),torch.FloatTensor(fx).to(device),torch.FloatTensor(y).to(device)
    pred = model(X,fx)
  acc = torch.mean(abs(pred - y),dim=0).item()
  return acc