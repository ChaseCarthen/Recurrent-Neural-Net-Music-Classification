require 'model'
local testmodel = torch.class ('testmodel','model')

function testmodel:__init()
	model.__init(self)
end

function testmodel:forward()
	print("foward")
end

function testmodel:setup()
	print("setup")
end

function testmodel:backward()
	print ("backward")
end