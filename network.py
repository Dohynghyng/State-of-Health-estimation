import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class LSTMCell(nn.Module) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) :
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters() :
            w.data.uniform_(-std, std)

    def forward(self, x, hidden) :
        hx, cx = hidden
        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)
        # 현재 입력과 이전 은닉 상태에 대한 선형 변환 결과(출력)을 더함
        # self.x2h(x)와 self.h2h(hx)의 결과를 더하면 현재 입력과 이전 은닉 상태에 대한 선형 변환 결과가 결합된 게이트의 값이 계산됨
        # 이 게이트의 값은 LSTM 셀에서 중요한 역할을 하는 입력 게이트, 망각 게이트, 셀 게이트, 출력 게이트에 대한 정보를 포함
        gates = gates.squeeze()
        # 텐서의 크기가 1인 차원을 제거
        # 계산 상 편의를 위해 텐서의 크기를 조정
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate) # 입력 게이트에 시그모이드 적용
        forgetgate = F.sigmoid(forgetgate) # 망각 게이트에 시그모이드 적용
        cellgate = F.tanh(cellgate) # 셀 게이트에 탄젠트 적용
        outgate = F.sigmoid(outgate) # 출력 게이트에 시그모이드 적용

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))

        return (hy, cy)

class my_lstm(nn.Module) :
   def __init__(self,  hidden_dim, layer_dim, input_dim=3, output_dim=1, dropout_prob=0.085, bias=True) :
      # __init__: init 메서드, 컨스트럭터
      # init 메서드의 역할은 주로 클래스의 인스턴스 변수를 초기화하고, 모델의 구조를 설정하는 등의 작업을 수행
      # 1. 입력 차원, 은닉 상태 차원, 레이어 개수 등을 인자로 받아 해당 값을 인스턴스 변수에 저장

       super(my_lstm, self).__init__()
       # 부모 클래스의 __init__ 메서드를 호출
       self.hidden_dim = hidden_dim
       self.layer_dim = layer_dim
       self.dropout = nn.Dropout(dropout_prob)
       self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
       self.fc = nn.Linear(hidden_dim+1, output_dim)
       # 2. LSTM 셀과 선형 레이어를 초기화하여 클래스의 인스턴스를 생성할 때 사용될 수 있도록 준비

       # 인스턴스: 클래스(템플릿)을 사용하여 실제 메모리에 생성된 객체
       # 인스턴스는 클래스에 정의된 속성이나 메서드에 접근할 수 있음
       # 인스턴스 변수: 인스턴스 변수는 해당 인스턴스에 속하는 변수로 인스턴스의 속성을 저장
       # 속성: 속성(Attribute)은 객체(object)가 가지는 데이터나 상태를 나타내는 변수
       # 속성 값을 인스턴스 변수에 저장

   def forward(self, x, fx) :
       if torch.cuda.is_available() :
        # GPU가 사용 가능한지 확인, 초기 은닉 상태와 초기 셀 상태를 GPU에 할당
           h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
           # Variable 클래스를 사용하여 데이터와 그래디언트를 저장할 수 있는 텐서를 생성
           # Pytorch 0.40 부터 텐서 자체가 그래디언트를 추적할 수 있게 됨
           # 3차원 텐서 생성
           # 첫 번째 차원: LSTM 레이어의 개수
           # 두 번째 차원: 한 번에 처리할 시퀀스(데이터)의 개수 = 입력 데이터의 첫 번째 차원
           # 세 번째 차원: LSTM 레이어의 은닉 상태(hidden state)의 차원
       else :
           h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

       if torch.cuda.is_available() :
           c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
       else :
           c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

       outs = []
       cn =  c0[0,:,:]
       # c0의 첫 번째 차원 인덱스 0에 해당하는 2차원 텐서를 cn에 할당
       hn = h0[0,:,:]

       for seq in range(x.size(1)) :
           hn, cn = self.lstm(x[:, seq, :], (hn, cn))
           # 모든 배치에 대해 특정 시퀀스 인덱스의 모든 데이터를 추출
           outs.append(hn)

       out = outs[-1].squeeze()

       out = torch.cat([out, fx], dim = 1)
       out = self.fc(out)
       return out