require '../StackedAutoEncoder'
require '../AutoEncoder'
require 'nn'
require 'torch'
require 'rnn'
require 'optim'
require 'cunn'
require 'cutorch'
torch.setdefaulttensortype('torch.FloatTensor')
input = torch.zeros(1000,10):float():split(10)
encoder = nn.Sequential()
encoder:add(nn.Sequencer(nn.LSTM(10,100)))

decoder = nn.Sequencer(nn.LSTM(100,10))
decoder:add(nn.Sequencer(nn.Sigmoid()))

ae = AutoEncoder(encoder,decoder)


encoder2 = nn.Sequential()
encoder2:add(nn.Sequencer(nn.LSTM(100,100)))

decoder2 = nn.Sequencer(nn.LSTM(100,10))
decoder2:add(nn.Sequencer(nn.Sigmoid()))

ae2 = AutoEncoder(encoder2,decoder2)


encoder3 = nn.Sequential()
encoder3:add(nn.Sequencer(nn.LSTM(100,100)))

decoder3 = nn.Sequencer(nn.LSTM(100,10))
decoder3:add(nn.Sequencer(nn.Sigmoid()))

ae3 = AutoEncoder(encoder3,decoder3)


model = StackedAutoEncoder()

model:AddLayer(ae)
model:AddLayer(ae2)
model:AddLayer(ae3)

criterion = nn.SequencerCriterion(nn.BCECriterion())
model:setCriterion(criterion)

model:initParameters()
model:cudaify('torch.FloatTensor')

output = model:forward(1,input)
output2 = model:forward(2,input)
output3 = model:forward(3,input)

print(output[1]:max())
print(output2[1]:max())
print(output3[1]:max())
model:backward(1,input,output,input)

