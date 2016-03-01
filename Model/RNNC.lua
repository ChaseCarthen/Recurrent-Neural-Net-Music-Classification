require 'Model/model'
require 'rnn'
require 'nn'
require 'Model/BinaryClassReward'
local RNNC = torch.class ("RNNC","model")

function RNNC:__init()
  model.__init(self)
end

function RNNC:setCriterion(criterion)
  self.criterion = criterion
end

function RNNC:addlayer(layer)
  if self.model == nil then
    self.model = nn.Sequential()
  end
  self.model:add(layer)
end

function RNNC:cudaify(string)
  self.model:cuda()
  self.prevmodel = self.model
  local model = nn.Sequential()
  model:add(nn.Sequencer(nn.Copy(string, 'torch.CudaTensor')))
  model:add(self.model)
  model:add(nn.Sequencer(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
  self.model = model
end

function RNNC:setModel(model)
  self.model = model
end

function RNNC:initParameters()
  if self.model then
    self.parameters,self.gradParameters = self.model:getParameters()
    --self.parameters:zero()
    print("PARAMETERS")
    print(self.parameters:max())
    print(self.parameters:min())
  end
end

function RNNC:getGradParameters()
  --print ("grad parameters")
  --print(self.gradParameters)
  --print(self.parameters)
  --print(self.model)
  return self.gradParameters
end

function RNNC:getParameters()
  --print (self.parameters)
  --print("parameters")
  return self.parameters
end

function RNNC:addLSTM(input,output)
  self:addlayer(nn.Sequencer(nn.LSTM(input,output)))
end

--- Need to figure out something better for this
function RNNC:addRNN(rnn)
  self.addlayer(rnn)
end

function RNNC:printmodel()
  print(self.model)
end

function RNNC:setup()
  self.gradParameters:zero()
end

function RNNC:forward(input)
  if self.mode == "train" then
    --self.model:training()
    --print ("training")
  elseif self.mode == "test" then
    --self.model:evaluate()
  end
  return self.model:forward(input) -- return some kind of output here
end

function RNNC:backward(input,output,targets)

  local err = self.criterion:forward(output, targets) / input[1]:size(1)
  if self.mode ~= "test" then
    local df_do = self.criterion:backward(output, targets) 
    for i=1,#df_do do 
      df_do[i] = df_do[i] / input[1]:size(1)
    end   
    self.model:backward(input, df_do)
  end
  return err
end
