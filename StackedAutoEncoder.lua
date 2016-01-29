require 'model'
require 'rnn'
require 'nn'
-- Train Auto Encoder
local StackedAutoEncoder = torch.class('StackedAutoEncoder')

function StackedAutoEncoder:__init(args)
	self.layer = {}
	self.layerCount = 0
end

function StackedAutoEncoder:train()
   self.mode = "train"
  for i=1,self.layerCount do
  	self.layer[i]:train()
  end
end

function StackedAutoEncoder:test()
  self.mode = "test"
  for i=1,self.layerCount do
  	self.layer[i]:test()
  end
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

function StackedAutoEncoder:OutputHidden(count,data)
	return self.layer[count]:OutputHidden(data)
end

function StackedAutoEncoder:OutputActual(count,data)
	return self.layer[count]:OutputActual(data)
end

function StackedAutoEncoder:layerForward(count,data)
	return self.layer[count]:forward(data)
end

function StackedAutoEncoder:setCriterion(criterion)
  for i=1,self.layerCount do
  	self.layer[i]:setCriterion(criterion)
  end
end


function StackedAutoEncoder:backward(count,input,output,targets)        
  return self.layer[count]:backward(input,output,targets)
end


function StackedAutoEncoder:initParameters()
  for i= 1,#self.layer do 
  	self.layer[i]:initParameters()
  end
end

function StackedAutoEncoder:getGradParameters(count)
  return self.layer[count]:getGradParameters()
end

function StackedAutoEncoder:getParameters(count)
  return self.layer[count]:getParameters()
end

function StackedAutoEncoder:getLayerCount()
	return #self.layer
end

function StackedAutoEncoder:cudaify(string)
  for i=1,self.layerCount do
  	self.layer[i]:cudaify(string)
  end
end