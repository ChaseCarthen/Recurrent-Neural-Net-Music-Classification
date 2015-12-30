require 'rnn'
local model = torch.class('model')
function model:__init()
self.mode = "train"
self.model = nil
end

function model:train()
self.mode = "train"
if self.model ~= nil then
	self.model:training()
end
end

function model:test()
self.mode = "test"
if self.model ~= nil then
	self.model:evaluate()
end
end

-- The only first comment here -- my assumption is that :cuda() is defined.
function model:cuda()
	if self.model ~= nil then
		self.model:cuda()
	end
end

function model:clear()
	print ("I do nothing just a stub overwrite me")
end

function model:setCriterion(criterion)
  self.criterion = criterion
end

function model:setup()
	print ("I do nothing just a stub overwrite me")
end

function model:forward()
	print (self.mode)
	print ("I do nothing just a stub overwrite me")
end

function model:backward()
	print ("I do nothing just a stub overwrite me.")
end

function model:addlayer()
	print ("I do nothing just a stub overwrite me.")
end

function model:save(filename)
	if self.model ~= nil then
		torch.save(filename,self.model)
	end
end

function model:load(filename)
	if filename ~= nil then
		self.model = torch.load(filename)
	end
end

function model:floattize()
	self.model = self.prevmodel
	self.model = self.model:float()
end
