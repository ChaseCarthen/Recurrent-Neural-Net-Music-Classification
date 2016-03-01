-- Ash catch them all
-- A class that will train,test, and validate models
-- using optim
-- This class assumes that it will be given a audiodataset
require 'xlua'
require 'optim'
require 'nn'
require 'image'

 function TemporalSplit(tensor,windowidth,stepsize)
  local Tensor = {}
  local counter = 1
  local step = 1
  local End = tensor:size(1)
  -- (nInputFrame - kW) / dW + 1
  steps = ( End - windowidth ) / stepsize + 1
  for i = 1,steps do
      Tensor[counter] = tensor:sub(step,step + windowidth - 1 ):float():sum(1):ge(1):clone()
      counter = counter + 1
      step = step + stepsize  
  end

  join = nn.JoinTable(1)
  return {join:forward(Tensor)}
end



local AutoEncoderTrainer = torch.class("AutoEncoderTrainer")

function deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[deepcopy(orig_key)] = deepcopy(orig_value)
        end
        setmetatable(copy, deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function AutoEncoderTrainer:__init(args)
	if args.epochLimit ~= nil then
		self.epochLimit = args.epochLimit
	else
		self.epochLimit = 200
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
    self.startState = deepcopy(args.optimState)
	end

	if args.optimModule == nil then
		-- throw an error here
	else
		self.optimModule = args.optimModule
	end

	self.dataSplit = args.dataSplit or 20000
	self.sequenceSplit = args.sequenceSplit or 5000

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

	self.serialize = false or args.serialize
	self.frequency = args.frequency or 10
	self.epochrecord = args.epochrecord or 50
  self.predict = args.predict
	self.modelfile = args.modelfile or "train.model"
  self.autofile = args.autofile or "auto.model"
  self.AutoEncoder = args.AutoEncoder
  self.layer = args.layer -- the current layer 6....
  self.TrainAuto = args.TrainAuto

  self.layerCount = args.layerCount
  self.temporalconv = args.temporalconv
  self.stepsize = args.stepsize or 1
  self.windowidth = args.windowidth or 1000

  self.fn = 0
  self.fp = 0
  self.tn = 0
  self.tp = 0

  -- applying normalization to inputs
  self.normalize = args.normalize or false
end

function AutoEncoderTrainer:splitData(data)
	local input = nil
	local target = nil

	if self.target == "midi" then
		target = data.midi:float()
	else
		target = data.audio:float()
	end

	if self.input == "midi" then
		input = data.midi:float()
	else
		input = data.audio:float()
	end
  if self.predict then
    input = input:sub(1,40000-1)
    target = target:sub(2,40000)
  end
  if self.normalize then
    input = image.minmax{tensor=input}
    target = image.minmax{tensor=target}
  end

  input = input:split(self.dataSplit)
  target = target:split(self.dataSplit)
	return input,target
end

-- Expecting output and target to be tables
function AutoEncoderTrainer:UpdateAccuracy(output,target)
  for i = 1, #output do
    local out = output[i]:clone():round()
    local tar = target[i]:clone():round()
    -- Calculate false negatives
    self.fn = self.fn + (tar - out):eq(1):sum()

    -- Calculate false positivess
    self.fp = self.fp + (tar - out):eq(-1):sum()

    -- calculate true positives
    self.tp = self.tp + (tar + out):eq(2):sum()

    -- calculate true negatives
    self.tn  = self.tn + (tar + out):eq(0):sum()
  end
  --print(self.fn)
  --print(self.tn)
  --print(self.fp)
  --print(self.tp)
end

function AutoEncoderTrainer:train()
  self.fn = 0
  self.fp = 0
  self.tn = 0
  self.tp = 0

   -- epoch tracker
   self.epoch = self.epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   if self.training then 
   self.datasetLoader:loadTraining()
   numTrain = self.datasetLoader:numberOfTrainingSamples() 
   elseif self.validate then
    self.datasetLoader:loadValidation()
    numTrain = self.datasetLoader:numberOfTestSamples()
   else
    self.datasetLoader:loadTesting()

    numTrain = self.datasetLoader:numberOfValidSamples()
   end

   
   shuffle = torch.randperm(numTrain)
   done = false
   local loss = 0
   local count = 0

   while not done do
    data = self.datasetLoader:loadNextSet()
    
   	done = data.done
    local prevout = nil
              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)
                          count = 0
                           -- get new parameters
                           if self.training then
                            if not self.AutoEncoder then
                              if x ~= self.model:getParameters() then
                                --print (x)
                                self.model:getParameters():copy(x)
                              end
                              --print (self.model)
                              -- reset gradients
                              self.model:getGradParameters():zero()
                              else
                                if x ~= self.model:getParameters(self.layer) then
                                  --print (x)
                                  self.AutoEncoder:getParameters(self.layer):copy(x)
                                end
                                --print (self.model)
                                -- reset gradients
                                self.AutoEncoder:getGradParameters(self.layer):zero()
                            end -- auto encoder end
                          end -- training end

                           -- f is the average of all criterions
                           local f = 0
                           -- evaluate function for complete mini batch
                           for i = 1,#data do

                            xlua.progress(i, #data)

                      	   inputs,target = self:splitData(data[i])
                           local out = {}
                           for testl = 1,#inputs do

                            if testl == 1 then
                              prevout = nil
                            end

                            if not self.temporalconv then
                              input = inputs[testl]:split(self.sequenceSplit)
                            else
                              input = TemporalSplit(inputs[testl], self.windowidth, self.stepsize)
                            end

                            if self.predict and prevout ~= nil then
                              input = prevout
                            end
                            if not self.temporalconv then
                              t = target[testl]:split(self.sequenceSplit)
                            else
                              t = TemporalSplit(target[testl], self.windowidth, self.stepsize)
                            end

                            -- Making sure the last split has the proper size for passing into a sequencer element.
                            if testl == #inputs then
                            	if t[#t]:size(1) ~= self.sequenceSplit then
                            		t[#t] = torch.cat(t[#t], torch.zeros(self.sequenceSplit - t[#t]:size(1), t[#t]:size(2) ),1 )
                                if self.predict == false or prevout == nil then
                            		  input[#input] = torch.cat(input[#input], torch.zeros(self.sequenceSplit - input[#input]:size(1), input[#input]:size(2) ), 1 )
                            		end
                            	end
                            end
                            if self.predict and #input ~= #t then
                              for ij = #input,#t,-1 do
                                input[ij] = nil
                              end
                            end

                           --print(t)
                           if not self.TrainAuto then
                            tempinput = self.AutoEncoder:forward(1,input,false)

                            for i=1,#input do
                              input[i] = input[i] - tempinput[i]
                            end
                            --print(input)
                            output = self.model:forward(input)
                            --print(output[1][1]:max())
                            --print(output)
                           else
                            if self.layer > 1 then
                              input = self.AutoEncoder:forward(self.layer - 1,input,true)
                            end
                            t = input
                            output = self.AutoEncoder:layerForward(self.layer,input)
                           end
                           

                           for os = 1,#output do
                              --output[os] = output[os]:clone()
                              if type(output[os]) == 'table' then
                           		   --output[os][1]:round()
                              else
                                 --output[os]:round()
                              end
                           end
                           prevout = output
                           if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                           	out[testl] = self.join:forward(output):clone()
                       	   end
                           local err = 0
                           if not self.TrainAuto then
                            --print(self.layer)
                            --print(input)
                            err = self.model:backward(input,output,t)--inputs)
                            self:UpdateAccuracy(output,t)
                           else
                            err = self.AutoEncoder:backward(self.layer,input,output,t)
                           end
                            f = f + err --/ input[1]:size(1) --/ (#input*input[1]:size(1))

                           
                          
                        collectgarbage();
                         end
                         count = count + 1
                         
                        if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                            torch.save("train" .. count .. "epoch" .. self.epoch .. ".dat",out)
                        end  
                            
                        out = nil
                           end

                           -- normalize gradients and f(X)
                           if self.training and not self.TrainAuto then
                            self.model:getGradParameters():div(count)--#data)
                           elseif self.training and self.TrainAuto then
                            self.AutoEncoder:getGradParameters(self.layer):div(count)
                           end
                            f = f/count--#data
                            --print("ORIG: " .. f)
                            --print(count)
                            if not self.TrainAuto then
                              return f,self.model:getGradParameters()
                            else
                              return f,self.AutoEncoder:getGradParameters(self.layer)
                            end

                end
                if self.training and not self.TrainAuto then
                  _,fs2 = self.optimModule(feval, self.model:getParameters(), self.optimState)
                  loss = loss + fs2[1]
                elseif self.training and self.TrainAuto then
                  _,fs2 = self.optimModule(feval, self.AutoEncoder:getParameters(self.layer), self.optimState)
                  loss = loss + fs2[1]
                else
                  fs2,_ = feval(nil)
                  loss = loss + fs2
                end
                

   end -- End of while loop

   print("LOSS: " .. loss)--/count)
   --print(confusion)


   if not self.TrainAuto then
    acc = self.tp / (self.tp + self.fn + self.fp)
    pre = self.tp / (self.tp + self.fp)
    rec = self.tp / (self.tp + self.fn)
    fmeasure = (2 * pre * rec) / (pre + rec)
    print("Accuracy: " .. acc)
    print("Precision: " .. pre)
    print("Recall: " .. rec)
    print("F-Measure: " .. fmeasure)
   end
   -- next epoch
   --confusion:zero()

   --self.epoch = self.epoch + 1

   return (loss)--/count)
end

function AutoEncoderTrainer:setLayer(layer)
  self.layer = layer
end

function AutoEncoderTrainer:done()
  print(self.layer)
  print(self.layerCount)
  if not (self.layer > self.layerCount) and self.TrainAuto and self.epoch > self.epochLimit then
    self.epoch = 1
    self.layer = self.layer + 1
    self.optimState = self.startState
    self.startState = deepcopy(self.startState)
    self.AutoEncoder:setCriterion(nn.SequencerCriterion(nn.MSECriterion()))
  end

	return self.epoch > self.epochLimit or (self.TrainAuto and self.layer > self.layerCount)
end


-- This is where our for loops will occur --
function AutoEncoderTrainer:tester()
  self.test = true
  self.training = false
  self.validate = false
  self.model:test()
  return self:train()
end

function AutoEncoderTrainer:trainer()
  self.training = true
  self.test = false
  self.validate = false
  self.model:train()
  return self:train()
end

function AutoEncoderTrainer:validater()
  self.model:test()
  self.test = false
  self.training = false
  self.validate = true
  return self:train()
end

function AutoEncoderTrainer:saveModel()
	torch.save(self.modelfile,self.model)
end
