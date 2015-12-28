-- Ash catch them all
-- A class that will train,test, and validate models
-- using optim
-- This class assumes that it will be given a audiodataset
require 'xlua'
require 'optim'
local trainer = torch.class("trainer")


function trainer:__init(args)
	if args.epochLimit ~= nil then
		self.epochLimit = args.epochLimit
	else
		self.epochLimit = 100
	end

	if args.model == nil then
		-- throw an error here
	else
		self.model = args.model
	end

	if args.datasetLoader == nil then 
		-- throw an error here
	else
		self.datasetLoader = args.datasetLoader
	end

	if args.optimState == nil then
		-- throw an error here
	else
		self.optimState = args.optimState
	end

	if args.optimModule == nil then
		-- throw an error here
	else
		self.optimModule = args.optimModule
	end

	self.dataSplit = args.dataSplit or 10000

	-- What is our target and input..
	self.target = args.target
	self.input = args.input

	-- hyperparameter watching to be added -- 
	-- Watch error rate --
	-- Watch norm of weights --
	-- enable graphing --
	self.graphing = args.graphing or false
	self.epoch = 1
end

function trainer:splitData(data)
	local input = nil
	local target = nil
	--print(data.midi:size())
	--print(data.audio:size())
	if self.target == "midi" then
		target = data.midi:float():split(self.dataSplit)	
	else
		target = data.audio:float():split(self.dataSplit)
	end

	if self.input == "midi" then
		input = data.midi:float():split(self.dataSplit)	
	else
		input = data.audio:float():split(self.dataSplit)
	end

	return input,target
end

function trainer:train()

   -- epoch tracker
   self.epoch = self.epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   self.model:train()
   self.datasetLoader:loadTraining()
   numTrain = self.datasetLoader:numberOfTrainingSamples() 
   shuffle = torch.randperm(numTrain)
   done = false
   loss = 0
   count = 0
   while not done do
    data = self.datasetLoader:loadNextSet()
    collectgarbage();
   	done = data.done

              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)

                           -- get new parameters
                           if x ~= self.model:getParameters() then
                              print (x)
                              self.model:getParameters():copy(x)
                           end
                           -- reset gradients
                           self.model:getGradParameters():zero()

                           -- f is the average of all criterions
                           local f = 0

                           -- evaluate function for complete mini batch
                           for i = 1,#data do
                            xlua.progress(i, #data)

                          --[[  inputs = data[i].data:float():split(rhobatch)

                            inputs[#inputs] = nil
                           if self.target == "midi" then
                          		target = data[i].binVector:t():float():split(rhobatch)
                      	  else
                      	  	target = data[i].binVector:t():float():split(rhobatch)
                      	  end--]]
                      	   inputs,target = self:splitData(data[i])
                           local out = {}
                           for testl = 1,#inputs do
                            input = {inputs[testl]}
                           local output = self.model:forward(input)

                           out[testl] = output[1]:clone()
                          local err = self.model:backward(input,output,{target[testl]})--inputs)
                           f = f + err
                          
                          count = count + 1
      
                         end

                        if self.epoch % 5 == 0 and count % 4 == 0 then
                            torch.save("test" .. count .. "epoch" .. self.epoch .. ".dat",out)
                            
                        end  
                            
                        out = nil
                           end

                           -- normalize gradients and f(X)
                           self.model:getGradParameters():div(count)--#data)
                            f = f/count--#data

                           return f,self.model:getGradParameters()
                end
                _,fs2 = self.optimModule(feval, self.model:getParameters(), self.optimState)
                loss = loss + fs2[1]

   end -- End of while loop

          print(loss/count)
           print(confusion)

           -- next epoch
           confusion:zero()
           self.epoch = self.epoch + 1
end

function trainer:done()
	return self.epoch == self.epochLimit
end

function trainer:test()

end


function trainer:validate()

end


-- This is where our for loops will occur --
function trainer:evaluate()

end

