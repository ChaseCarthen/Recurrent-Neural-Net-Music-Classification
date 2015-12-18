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

function StackedAutoEncoder:forward(count,data)
	return self.layer[count]:OutputHidden(data)
end

function StackedAutoEncoder:backward(count,input,output,targets)
	return self.layer[count]:backward(input,output,targets)
end

function StackedAutoEncoder:layerForward(count,data)
	output = data
	for i = 1,count do
		output = self.layer[count]:OutputHidden(output)
	end
end