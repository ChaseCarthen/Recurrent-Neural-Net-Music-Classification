require 'model'
require 'rnn'
require 'nn'
-- Train Auto Encoder
local StackedAutoEncoder = torch.class('StackedAutoEncoder')

function StackedAutoEncoder:__init(args)
	self.layer = {}
	self.layerCount = 0
end

function StackedAutoEncoder:AddLayer(autoencoder)
	self.layerCount = self.layerCount + 1
	self.layer[self.layerCount] = autoencoder
end

function StackedAutoEncoder:forward(count,data,hidden)
		output = data
	for i = 1,count - 1 do
		output = self.layer[i]:OutputHidden(output)
	end
	if hidden then
		output = self.layer[count]:OutputHidden(output)
	else
		output = self.layer[count]:forward(output)
	end
	return output
end

function StackedAutoEncoder:backward(count,input,output,targets)
	return self.layer[count]:backward(input,output,targets)
end