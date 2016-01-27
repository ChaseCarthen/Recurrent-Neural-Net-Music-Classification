require 'rnn'
require 'nn'


input = torch.randn(10,10)
inputsplit = torch.randn(10,10):clone():split(2)
target = torch.randn(10,10)
targetsplit = target:split(2)
a = nn.Sequential():add(nn.LSTM(10,10))

aclone = a:clone()

aclone = nn.Sequencer(a)

criterion = nn.MSECriterion()

seqcriterion = nn.SequencerCriterion(nn.MSECriterion())


output = a:forward(input):clone()

outputseq = aclone:forward(inputsplit)
--outputseq = aclone:forward(inputsplit)
print(input)
print(output)
print(target)
print(criterion:forward(output,target))

print(seqcriterion:forward(outputseq,targetsplit))

--print(a:getParameters())

local df_do = criterion:backward(output,target):clone()
local df_do2 = seqcriterion:backward(outputseq,targetsplit)

--print((df_do - df_do2):sum())
print(input)
print(df_do)
--a:backward(input,df_do)
print(df_do2)
aclone:backward(inputsplit,df_do2)

print( (a:getParameters() - aclone:getParameters()):sum() )

