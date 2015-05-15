torch = require 'torch'
require 'cutorch'
require 'nn'
SOM = {}
SOM.__index = SOM
---http://chem-eng.utoronto.ca/~datamining/Presentations/SOM.pdf
function SOM.create(weights,numClusters)
   local W = {}
   setmetatable(W,SOM)
   W.weights = torch.randn(numClusters,weights):uniform()
   W.numClusters = numClusters
   W.lattice = 30000
   W.timeconstant = 10000
   W.iteration = 1
   W.learningrate = .01
   return W
end

function SOM:forward(input)
   local min = 1000000000000000000000
   local class = 1
   local inp = input:cuda()
   for i =1,self.numClusters 
   do
         --print("WEIGHTS" .. i)
         --print(self.weights[i]:size())
         --print (input:size())
         --print(i)
  	 local distance = self.weights[i]:dist(inp)
         --print(distance)
 	 if distance < min then
             min = distance  
	     class = i 
   	end 

   end
   --print("asdf")
   return class,min
end

function SOM:learningRate()
	return self.learningrate*torch.exp(-self.iteration/self.timeconstant)
end

function SOM:latticeAtTimeT()
	return self.lattice*torch.exp(-self.iteration/self.timeconstant)	
end

function SOM:neighborhood(dist)
   return torch.exp(-dist*dist/(2*self:latticeAtTimeT()*self:latticeAtTimeT()) )
end

function SOM:backward(input,class)
local inp = input
-- time to adjust or inputs
local weights = self.weights:float() 
for c=1,self.numClusters  
 do
    --print(c)
    local d = weights[c]:dist(weights[class])
    if d < self.lattice then
    self.weights[c] = (weights[c] + (inp-weights[c]) * self:learningRate() * self:neighborhood(d) ):cuda()
    end

 end
 return (self.weights:float() - weights):norm()
 --print "backward"
end

function SOM:update()
self.iteration = self.iteration + 1
end

function SOM:cuda()
self.weights = self.weights:cuda()
end

function SOM:save(filename)
torch.save(filename,self.weights:double())
end

function SOM:load(filename)
self.weights = torch.load(filename)
end

--b = SOM.create(10,4)
--b:cuda()
--print(b:forward(torch.randn(10):cuda()))
--b:backward(torch.randn(10),1)
--print(b.weights)
--b:save("l.txt")
--b:load("l.txt")
--print(b.weights)
