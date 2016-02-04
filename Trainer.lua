-- Ash catch them all
-- A class that will train,test, and validate models
-- using optim
-- This class assumes that it will be given a audiodataset
require 'xlua'
require 'optim'
local Trainer = torch.class("Trainer")


function Trainer:__init(args)
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
  self.predict = args.predict
  self.modelfile = args.modelfile or "train.model"

  self.temporalconv = args.temporalconv
  self.stepsize = args.stepsize or 1
  self.windowidth = args.windowidth or 1000
end

function Trainer:splitData(data)
  local input = nil
  local target = nil
  --print(data.filename)
  --print(data.audio:size())
  --print(data.midi:size())
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
  input = input:split(self.dataSplit)
  target = target:split(self.dataSplit)   
  return input,target
end

function Trainer:train()

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
   loss = 0
   count = 0
   while not done do
    data = self.datasetLoader:loadNextSet()
    
    done = data.done
    local prevout = nil
              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)
                           -- get new parameters
                           if self.training then
                            if x ~= self.model:getParameters() then
                              print (x)
                              self.model:getParameters():copy(x)
                            end
                            --print (self.model)
                            -- reset gradients
                            self.model:getGradParameters():zero()

                          end

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

                            input = inputs[testl]:split(self.sequenceSplit)

                            if self.predict and prevout ~= nil then
                              input = prevout
                            end
                              t = target[testl]:split(self.sequenceSplit)
                              --print("windowidth" .. self.windowidth)
                              --print("stepsize" .. self.stepsize)
                              

                            --print(input)
                            --print(t)
                            --print(self.temporalconv)
                            --print("=====================================================================================================")
                            -- Making sure the last split has the proper size for passing into a sequencer element.
                            if testl == #inputs  then
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

                            if self.temporalconv then
                              for i=1,#t do
                                t[i] = TemporalSplit(t[i], self.windowidth, self.stepsize)[1]
                              end
                            end
                           --print(t)
                           local output = self.model:forward(input)

                           --print(t)
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
                            out[testl] = self.join:forward(output):clone():round()
                           end

                          local err = self.model:backward(input,output,t)--inputs)
                           f = f + err
                           
                          
                        collectgarbage();
                         end
                         count = count + 1
                         
                        if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                            torch.save("train" .. count .. "epoch" .. self.epoch .. ".dat",out)
                        end  
                            
                        out = nil
                           end

                           -- normalize gradients and f(X)
                           if self.training then
                            self.model:getGradParameters():div(count)--#data)
                           end
                            f = f/count--#data
                            --print(count)
                            return f,self.model:getGradParameters()

                end
                if self.training then
                  _,fs2 = self.optimModule(feval, self.model:getParameters(), self.optimState)
                  loss = loss + fs2[1]
                else
                  fs2,_ = feval(nil)
                  loss = loss + fs2
                end
                

   end -- End of while loop

   print(loss/count)
   --print(confusion)

   -- next epoch
   --confusion:zero()
   self.epoch = self.epoch + 1

   return (loss/count)
end

function Trainer:done()
  return self.epoch > self.epochLimit
end


-- This is where our for loops will occur --
function Trainer:tester()
  self.test = true
  self.training = false
  self.validate = false
  self.model:test()
  return self:train()
end

function Trainer:trainer()
  self.training = true
  self.test = false
  self.validate = false
  self.model:train()
  return self:train()
end

function Trainer:validater()
  self.model:test()
  self.test = false
  self.training = false
  self.validate = true
  return self:train()
end

function Trainer:saveModel()
  torch.save(self.modelfile,self.model)
end