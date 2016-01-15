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

	self.dataSplit = args.dataSplit or 20000
	self.sequenceSplit = args.sequenceSplit or 5000
	print(self.dataSplit)
	print(self.sequenceSplit)
	-- What is our target and input..
	self.target = args.target
	self.input = args.input

	-- hyperparameter watching to be added -- 
	-- Watch error rate --
	-- Watch norm of weights --
	-- enable graphing --
	self.graphing = args.graphing or false
	self.epoch = 1
	self.join = nn.JoinTable(1)
	print (args.serialize)
	self.serialize = false or args.serialize
	self.frequency = args.frequency or 10
	self.epochrecord = args.epochrecord or 50
	self.modelfile = args.modelfile or "train.model"
end

function trainer:splitData(data)
	local input = nil
	local target = nil

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
                           --print (self.model)
                           -- reset gradients
                           self.model:getGradParameters():zero()

                           -- f is the average of all criterions
                           local f = 0

                           -- evaluate function for complete mini batch
                           for i = 1,#data do
                            xlua.progress(i, #data)

                      	   inputs,target = self:splitData(data[i])
                           local out = {}
                           for testl = 1,#inputs do
                           	if testl % 10 == 0 then
                           		--self.model:forget()
                           	end
                            input = inputs[testl]:split(self.sequenceSplit)
                            t = target[testl]:split(self.sequenceSplit)

                            -- Making sure the last split has the proper size for passing into a sequencer element.
                            if testl == #inputs then
                            	if t[#t]:size(1) ~= self.sequenceSplit then
                            		t[#t] = torch.cat(t[#t], torch.zeros(self.sequenceSplit - t[#t]:size(1), t[#t]:size(2) ),1 )
                            		input[#input] = torch.cat(input[#input], torch.zeros(self.sequenceSplit - input[#input]:size(1), input[#input]:size(2) ), 1 )
                            		
                            	end
                            end

                           print(type(input))
                           local output = self.model:forward(input)

                           for os = 1,#output do
                              --output[os] = output[os][1]
                              if type(output[os]) == 'table' then
                           		   output[os][1]:round()
                              else
                                 output[os]:round()
                              end
                           end
                           if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                           	out[testl] = self.join:forward(output):clone()
                       	   end
                           --print(testl)
                          local err = self.model:backward(input,output,t)--inputs)
                           f = f + err
                           
                          
      
                         end
                         count = count + 1
                         
                        if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                            torch.save("train" .. count .. "epoch" .. self.epoch .. ".dat",out)
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
   --print(confusion)

   -- next epoch
   --confusion:zero()
   self.epoch = self.epoch + 1

   return (loss/count)
end

function trainer:done()
	return self.epoch > self.epochLimit
end

function trainer:test()
   -- epoch tracker
   self.epoch = self.epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   self.model:test()
   self.datasetLoader:loadTesting()
   numTest = self.datasetLoader:numberOfTestSamples()
   shuffle = torch.randperm(numTest)
   done = false
   local loss = 0
   count = 0
   while not done do
    data = self.datasetLoader:loadNextSet()
    collectgarbage();
   	done = data.done


	-- evaluate function for complete mini batch
	for i = 1,#data do
		xlua.progress(i, #data)

		inputs,target = self:splitData(data[i])

        local out = {}
        for testl = 1,#inputs do
        	input = inputs[testl]:split(self.sequenceSplit)
            t = target[testl]:split(self.sequenceSplit)

            if testl == #inputs then
            	if t[#t]:size(1) ~= self.sequenceSplit then
                	t[#t] = torch.cat(t[#t], torch.zeros(self.sequenceSplit - t[#t]:size(1), t[#t]:size(2) ),1 )
                    input[#input] = torch.cat(input[#input], torch.zeros(self.sequenceSplit - input[#input]:size(1), input[#input]:size(2) ), 1 )
                            		
                end
            end

                            
			local output = self.model:forward(input)
			if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
        		out[testl] = self.join:forward(output)
        	end
        	local err = self.model:backward(input,output,t)--inputs)
        	loss = loss + err
                          
        	count = count + 1
      
        end

        if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
        	torch.save("test" .. count .. "epoch" .. self.epoch .. ".dat",out)                    
        end
    end  
                            
	end -- End of while loop

	print(loss/count)
	--print(confusion)
	return loss/count

end


function trainer:validate()
	torch.save("test.model",self.model)
   -- epoch tracker
   self.epoch = self.epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   self.model:test()
   self.datasetLoader:loadValidation()
   numValidation = self.datasetLoader:numberOfValidSamples()
   shuffle = torch.randperm(numValidation)
   done = false
   local loss = 0
   count = 0
   while not done do
    data = self.datasetLoader:loadNextSet()
    collectgarbage();
   	done = data.done


	-- evaluate function for complete mini batch
	for i = 1,#data do
		xlua.progress(i, #data)

		inputs,target = self:splitData(data[i])

        local out = {}
        for testl = 1,#inputs do
        	input = inputs[testl]:split(self.sequenceSplit)
            t = target[testl]:split(self.sequenceSplit)

            if testl == #inputs then
            	if t[#t]:size(1) ~= self.sequenceSplit then
                	t[#t] = torch.cat(t[#t], torch.zeros(self.sequenceSplit - t[#t]:size(1), t[#t]:size(2) ),1 )
                    input[#input] = torch.cat(input[#input], torch.zeros(self.sequenceSplit - input[#input]:size(1), input[#input]:size(2) ), 1 )
                            		
                end
            end

                            
			local output = self.model:forward(input)
			if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
        		out[testl] = self.join:forward(output)
        	end
        	local err = self.model:backward(input,output,t)--inputs)
        	loss = loss + err
                          
        	count = count + 1
      
        end

        if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
        	torch.save("test" .. count .. "epoch" .. self.epoch .. ".dat",out)                    
        end
    end  
                            
	end -- End of while loop

	print(loss/count)
	--print(confusion)s
	return loss/count
end


-- This is where our for loops will occur --
function trainer:evaluate()

end

function trainer:saveModel()
	torch.save(self.modelfile,self.model)
end