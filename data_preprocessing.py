import zipfile
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data):
    # 압축 해제
    z_file = zipfile.ZipFile('1. BatteryAgingARC-FY08Q4.zip')
    z_file.extractall('./DB/')
    z_file.close()
    os.remove('./DB/README.txt')

    # 데이터 처리
    X = {}
    y = {}
    path = './DB/'
    f_names = ['B0005', 'B0006', 'B0007', 'B0018']
    for f_name in f_names:
        mat_file = scipy.io.loadmat(path + f_name)
        is_charge = 0
        X_temp = []
        y_temp = []
        for i in range(len(mat_file[f_name][0][0][0][0])):
            if mat_file[f_name][0][0][0][0][i][0] == 'charge':
                if is_charge != 1:
                    X_temp.append(mat_file[f_name][0][0][0][0][i][3][0][0])
                    is_charge = 1
            elif mat_file[f_name][0][0][0][0][i][0] == 'discharge':
                if is_charge != 0:
                    y_temp.append(mat_file[f_name][0][0][0][0][i][3][-1][-1][-1][0])
                    is_charge = 0

        X[f_name] = X_temp
        y[f_name] = y_temp

    # 마지막 charge로 끝나는 데이터 제거
    if data =='B0018':
      X = X[data][1:]
    else:
      X = X[data][1:-1]

    # 전체 용량
    total_volume = y[data][0]

    # 방전 용량 첫값 제거
    y = y[data][1:]
    return X,y,total_volume


# uniform sampling, del outlier, normalization
def preprocessing(data, y, scaler=0, min_mA=0.01, num=30):
    # point time
    point_time = []
    for i in range(len(data)):
        temp_point_time = []
        for k in range(2, len(data[i][1][0])):
            if data[i][0][0][k] >= 4.2:
                temp_point_time.append(data[i][5][0][k])
                break
        point_time.append(temp_point_time)

    for k in range(2, len(data[0][0][0])):
        if data[0][0][0][k] >= 4.2:
            point_time1 = data[0][5][0][k - 1]
            break

    CHV = []
    CHC = []
    TIME = []
    TEM = []
    Time = []

    for i in range(len(data)):
        temp_CHV = []
        temp_CHC = []
        temp_TIME = []
        temp_TEM = []
        temp_Time = []
        for k in range(0, len(data[i][1][0])):
            temp_TEM.append(data[i][2][0][k])
            temp_Time.append(data[i][5][0][k])
        TEM.append(temp_TEM)
        Time.append(temp_Time)
        for k in range(2, len(data[i][0][0])):
            if data[i][5][0][k] < point_time1:
                temp_CHV.append(data[i][0][0][k])
                temp_TIME.append(data[i][5][0][k])
            else:
                break

        for k in range(2, len(data[i][1][0])):
            if data[i][5][0][k] >= point_time1 and data[i][1][0][k] >= min_mA:
                temp_CHC.append(data[i][1][0][k])
        # uniform sampling
        lines = np.linspace(2, len(temp_CHV) - 1, num=num, dtype=np.uint64, endpoint=True)
        uniform_CHV = []
        uniform_TIME = []
        for line in lines:
            uniform_CHV.append(temp_CHV[line])
            uniform_TIME.append(temp_TIME[line])
        CHV.append(uniform_CHV)
        TIME.append(uniform_TIME)

        lines = np.linspace(2, len(temp_CHC) - 1, num=num, dtype=np.uint64, endpoint=True)
        uniform_CHC = []
        for line in lines:
            uniform_CHC.append(temp_CHC[line])
        CHC.append(uniform_CHC)

    # normalization
    CHV_norm = scaler.fit_transform(np.asarray(CHV).reshape(-1, 1))
    CHC_norm = scaler.fit_transform(np.asarray(CHC).reshape(-1, 1))
    TIME_norm = scaler.fit_transform(np.asarray(TIME).reshape(-1, 1))
    point_time = scaler.fit_transform(np.asarray(point_time).reshape(-1, 1))
    # ['충전 전압', '충전 전류', '온도', '방전 용량']
    CHV = CHV_norm.reshape(-1, num, 1)
    CHC = CHC_norm.reshape(-1, num, 1)
    TIME = TIME_norm.reshape(-1, num, 1)
    return np.concatenate((CHV, CHC, TIME), axis=2), np.asarray(y).reshape(-1, 1), point_time, TEM, Time

def rand_sampling(X,fx,y,shuffle,train_ratio):
  indices = np.arange(len(X))
  X_train, X_test,fx_train,fx_test, y_train, y_test, train_idxs, test_idxs = train_test_split(X,fx,y,indices,train_size=train_ratio, shuffle = shuffle)
  return X_train, X_test,fx_train,fx_test, y_train, y_test, train_idxs, test_idxs