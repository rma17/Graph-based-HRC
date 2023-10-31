# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:05:37 2023

@author: Ruidong

Part of the fine-tune for 2D motion classification 
model used :LSTM-FCN
8 motion labels



"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, num_features,
                 num_lstm_out=128, num_lstm_layers=1, 
                 conv1_nf=128, conv2_nf=256, conv3_nf=128,
                 lstm_drop_p=0.8, fc_drop_p=0.3):
        super(MLSTMfcn, self).__init__()

        self.num_classes = num_classes
        
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

       

        self.relu = nn.ReLU()
       
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
       
        x1, (ht,ct) = self.lstm(x)
        # x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                  # padding_value=0.0)
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        # # x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        # x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2) #global average pooling
        
        x_all = torch.cat((x1,x2),dim=1)
        # print(x2.shape)
        x_out = self.fc(x_all)
        # x_out = F.log_softmax(x_out, dim=1)

        return x_out
    
class timeNet(nn.Module):
    '''
    model for timeseries classification
    '''
    def __init__(self, num_layers, input_size, hidden_size, num_classes):
        super(timeNet, self).__init__()
        self.lstm= nn.LSTM(input_size,hidden_size,num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)

    def forward(self,batch_input):
        out,_ = self.lstm(batch_input)
        out = self.linear(out[:,-1, :])  #Extract outout of lasttime step
        return out

X = torch.load('data.pt')
y = torch.load('label.pt')
seq_lens_train = torch.load('length.pt')

import random
index=random.sample(range(0, len(X)), 170)
X_train=X[index]
y_train=y[index]
seq_lens_train=seq_lens_train[index]

train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
model = MLSTMfcn(num_classes=9, 
                                 
                                  num_features=2)
# model=timeNet(1,2,256,8)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,shuffle=True)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.CrossEntropyLoss()
for e in range(500):
        

        model.train()
        for inputs, labels, seq_lens in train_loader:
            

            inputs = inputs.float()
            inputs, labels = inputs,labels
            
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        output=model.forward(X.float())
        pre=torch.argmax(output,dim=1)
        correct1=int((pre==y).sum())  
        
        print(correct1/len(y),e)
        # if correct1/len(y_train)==1.0:
        #     break    
torch.save(model.state_dict(), 'motion.pt')