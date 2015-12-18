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

function AutoEncoder:forward(input)
  return self.model:forward(input) -- return some kind of output here
end

function AutoEncoder:backward(input,output,targets)
  local err = self.criterion:forward(output, targets)
  local df_do = self.criterion:backward(output, targets)
  self.model:backward(input, df_do)
  return err
end

function AutoEncoder:OutputHidden(input)
	return self.encoder:forward(input)
end

function AutoEncoder:OutputActual(hidden)
	return self.decoder:forward(hidden)
end
