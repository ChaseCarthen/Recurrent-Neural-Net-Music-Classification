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

model = AutoEncoder(encoder,decoder)

criterion = nn.SequencerCriterion(nn.BCECriterion())
model:setCriterion(criterion)

model:initParameters()
model:cudaify('torch.FloatTensor')

output = model:forward(input)
output2 = model:OutputHidden(input)
print(output)
print(output2)
print(input)
print (model.model)
model:backward(input,output,input)


output2 = model:OutputHidden(input)
print(output2)