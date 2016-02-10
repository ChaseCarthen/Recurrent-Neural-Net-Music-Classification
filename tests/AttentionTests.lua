require 'rnn'
require 'dpnn'
require 'nn'
require 'BinaryClassReward'

action = nn.Sequential()
action:add(nn.Linear(10,3)) -- the 10 is based off the hidden size below in recurrentattention
action:add(nn.ReinforceBernoulli(true))

locationSensor = nn.Sequential()
locationSensor:add(nn.SelectTable(2)) -- use the action module 
locationSensor:add(nn.LSTM(3,10),5)
locationSensor:add(nn.Sigmoid())

attention = nn.Sequential()
ab = nn.RecurrentAttention(locationSensor,action,1,{10})
attention:add( ab)
attention:add(nn.SelectTable(-1))


reward = nn.Sequential()
reward:add(nn.Constant(1,1))


concat = nn.ConcatTable():add(nn.Identity()):add(reward)
concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)
attention:add(concat2)


out = {}

input = torch.ones(6,10):split(3)
out[1] = attention:forward(input[1])
print(ab.rnn.step)
--out[1][1]:clone()
--out[1][2][2]:clone()
out[2] = attention:forward(input[2])
print(ab.rnn.step)

attention = nn.Sequencer(attention)

attention:forward(input)
print(ab.rnn.step)
--out[1][1]:clone()
--out[2][2][2]:clone()
--print(out)

--criterion = nn.ModuleCriterion(nn.VRClassReward(attention))

criterion = nn.ParallelCriterion(true)
      :add( nn.ModuleCriterion(nn.BCECriterion(), nil, nn.Convert()) )  -- BACKPROP
      :add(nn.ModuleCriterion(nn.BinaryClassReward(attention), nil, nn.Convert())) -- REINFORCE

criterion = criterion



--print("step value: " .. attention.step)

for i = #out,1,-1 do

	criterion:forward(out[i],input[i])
	local df_do = criterion:backward(out[i],input[i]) -- clone

	print("output: ......")
	print(ab.output)
	print(ab.actions)
	print(ab.gradInput)
	print(ab.action.gradInput)
	print(ab.action.step)
	print(ab.action.updateGradInputStep)
	print("----------------")
	attention:backward(input[i],df_do)



end

print("He shoots he scores!!!! GOAL!!!!!")
--attention:backward(torch.ones(1,5),criterion:backward({out[1],torch.Tensor({1})},out[1]))
