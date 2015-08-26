torch = require 'torch'
require 'cutorch'
require 'nn'
require 'math'
SOM = {}
SOM.__index = SOM
---http://chem-eng.utoronto.ca/~datamining/Presentations/SOM.pdf
function SOM.create(weightsw,weightsh,Width,Height,epoch)
   local W = {}
   setmetatable(W,SOM)
   
   W.weights = torch.randn(Width,Height,weightsw,weightsh):uniform()
   W.width = Width
   W.height = Height
   W.lattice = math.max(Width,Height)
   W.timeconstant = epoch/math.log(W.lattice)--1000
   W.iteration = 1
   W.learningrate = .1
   W.besti = 0
   W.bestj = 0
   return W
end

function SOM:forward(input)
   self.besti = 0
   self.bestj = 0
   local min = 1000000000000000000000
   local class = 1
   local inp = input:cuda()
   for i =1,self.width 
   do
     for j=1,self.height
         do
         --print("WEIGHTS" .. i)
         --print(self.weights[i]:size())
         --print (input:size())
         --print(i)
  	 local distance = self.weights[i][j]:dist(inp)
      distance = distance * distance
         --print(distance)
 	 if distance < min then
             min = distance  
	     class = i
             self.besti = i
             self.bestj = j 
   	end
      end 

   end
   --print("asdf")

   return (self.besti-1)*self.height + self.bestj-1,min
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
for c=1,self.width  
 do
   for d=1,self.height do
    --print(c)
    local d2 = math.sqrt( (self.besti - c)*(self.besti - c) + (self.bestj - d) * (self.bestj - d)   )

    --print(c .. " " .. d)
    --print("Distance: " .. d2)
    --print(self.besti)
    --print(self.bestj)
    if d2 < self.lattice then
    self.weights[c][d] = (weights[c][d] + (inp-weights[c][d]) * self:learningRate() * self:neighborhood(d2) ):cuda()
    end

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
