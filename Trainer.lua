-- Ash catch them all
-- A class that will train,test, and validate models
-- using optim
-- This class assumes that it will be given a audiodataset
require 'xlua'
require 'optim'
require 'image'
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
  self.normalize = args.normalize
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

  if self.normalize then
    input = image.minmax{tensor=input}
  end
  --input = input - input*input:mean()/input:std()
  input = (input - input:mean())/input:std() --join:forward(output)}
  --input = image.minmax{tensor=input - input*1.0/input:mean()}
  --input = image.minmax{tensor=input - input*1.0/input:mean()}

  input = input:split(self.dataSplit)
  target = target:split(self.dataSplit) 
  --print(input)  
  return input,target
end

-- Expecting output and target to be tables
function Trainer:UpdateAccuracy(output,target)
  for i = 1, #output do
    --print(output[i]:max())
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

function Trainer:train()
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
   loss = 0
   count = 0

   while not done do
    data = self.datasetLoader:loadNextSet()
    collectgarbage();
    done = data.done
    local prevout = nil


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

              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)
                          -- f is the average of all criterions
                          local f = 0

                           -- get new parameters
                           if self.training then
                            if x ~= self.model:getParameters() then
                              print (x)
                              self.model:getParameters():copy(x)
                            end
                            --print("CLEAR")
                            --print (self.model)
                            -- reset gradients
                            self.model:getGradParameters():zero()

                          end

                           --print(t)
                           local output = self.model:forward(input)

                           --print(t)

                           prevout = output
                           if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                            out[testl] = self.join:forward(output):clone():round()
                           end 

                          local err = self.model:backward(input,output,t)--inputs)
                          self:UpdateAccuracy(output,t)
                          --err = err / input[1]:size(1)
                           f = f + err
                           
                          


                           -- normalize gradients and f(X)
                           if self.training then
                            self.model:getGradParameters():div(input[1]:size(1))--#data)
                           end

                            --print(count)
                            return f,self.model:getGradParameters()

                end -- end function

                if self.training then
                  _,fs2 = self.optimModule(feval, self.model:getParameters(), self.optimState)
                  loss = loss + fs2[1]
                else
                  fs2,_ = feval(nil)
                  loss = loss + fs2
                end
                

              collectgarbage();
               end  -- end inner for loop
               count = count + 1
               
              if self.epoch % self.epochrecord == 0 and count % self.frequency == 0 and self.serialize then
                  torch.save("train" .. count .. "epoch" .. self.epoch .. ".dat",out)
              end  
                  
              out = nil
                 end -- end top for loop

   end -- End of while loop

   print(loss)--/input[1]:size(1))
   --print(confusion)

   -- next epoch
   --confusion:zero()
   --self.epoch = self.epoch + 1
    acc = self.tp / (self.tp + self.fn + self.fp)
    pre = self.tp / (self.tp + self.fp)
    rec = self.tp / (self.tp + self.fn)
    fmeasure = (2 * pre * rec) / (pre + rec)
    print("Accuracy: " .. acc)
    print("Precision: " .. pre)
    print("Recall: " .. rec)
    print("F-Measure: " .. fmeasure)
   return loss
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