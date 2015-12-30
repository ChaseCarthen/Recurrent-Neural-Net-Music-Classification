require 'rnn'
require 'model'
-- A AutoEncoder Class
local AutoEncoder = torch.class("AutoEncoder","model")

function AutoEncoder:__init(encoder,decoder)
	self.encoder = encoder
	self.decoder = decoder
	self.model = nn.Sequential()
	self.model:add(self.encoder)
	self.model:add(self.decoder)
end

function AutoEncoder:setup()

end

function AutoEncoder:setCriterion(criterion)
  self.criterion = criterion
end

function AutoEncoder:initParameters()
  if self.model then
    self.parameters,self.gradParameters = self.model:getParameters()
  end
end

function AutoEncoder:getGradParameters()
  --print ("grad parameters")
  return self.gradParameters
end

function AutoEncoder:getParameters()
  --print (self.parameters)
  --print("parameters")
  return self.parameters
end

function AutoEncoder:cudaify(string)
  self.model:cuda()
  self.prevmodel = self.model
  local model = nn.Sequential()
  model:add(nn.Sequencer(nn.Copy(string, 'torch.CudaTensor')))
  model:add(self.model)
  model:add(nn.Sequencer(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
  self.model = model

  encoder = nn.Sequential()
  encoder:add(nn.Sequencer(nn.Copy(string,'torch.CudaTensor')))
  encoder:add(self.encoder)
  encoder:add(nn.Sequencer(nn.Copy('torch.CudaTensor','torch.FloatTensor')))
  self.encoder = encoder

  decoder = nn.Sequential()
  decoder:add(nn.Sequencer(nn.Copy(string,'torch.CudaTensor')))
  decoder:add(self.decoder)
  decoder:add(nn.Sequencer(nn.Copy('torch.CudaTensor','torch.FloatTensor')))
  self.decoder = decoder
end

function AutoEncoder:forward(input)
  return self.model:forward(input) -- return some kind of output here
end

function AutoEncoder:backward(input,output,targets)
  --print (output)
  local err = self.criterion:forward(output, targets) 
  if self.mode ~= "test" then    
    local df_do = self.criterion:backward(output, targets)        
    self.model:backward(input, df_do)
  end
  return err
end

function AutoEncoder:OutputHidden(input)
	return self.encoder:forward(input)
end

function AutoEncoder:OutputActual(hidden)
	return self.decoder:forward(hidden)
end
