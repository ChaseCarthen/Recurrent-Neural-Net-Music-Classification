


local torch = require 'torch'
local nn = require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'




--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels

trainData, testData, classes = GetTrainAndTestData("./music", .6)
print("AFTER")
--print (table.getn(obj.Songs))
--print(classes)
--print(trainData)




--print (trainData.Songs[1]:size())



--Step 2: Create the model
inp=128;  -- dimensionality of one sequence element 
outp=128; -- number of derived features for one sequence element
kw=128;   -- kernel only operates on one sequence element at once
dw=128;   -- we step once and go on to the next sequence element
--spl =150 -- split constant
--print(nn)
mlp=nn.Sequential()
mlp:add(nn.TemporalConvolution(inp,outp,kw,dw))
mlp:add(nn.Reshape(outp))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(outp,32))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(32,4))
mlp:add(nn.LogSoftMax())
mlp:add(nn.Sum(1))
--mlp:add(nn.parallel)
--mlp:add(nn.Sum(1))
--mlp:add(nn.Mean(2))
--mlp2 = nn.TemporalMaxPooling(3) --nn.TemporalSubSampling(128,3)
x=torch.rand(256,inp) -- a sequence of 7 elements
--x=nil
--y=torch.rand(8,1)
--print(y)
--print(x)
--print(mlp2:forward(x))

print(mlp:forward(x))
model = mlp
--print(x)
--print (trainData)







--mlp = nn.Sequential()
--mlp:add( nn.Linear(10, 25) ) -- 10 input, 25 hidden units
--mlp:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
--mlp:add( nn.Linear(25, 1) ) -- 1 output

--print(mlp:forward(torch.randn(10,10)))
--print(model)


--[[



ninputs = 128
noutputs = 1
nhidden = 300
--MLP
model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhidden))

model:add(nn.ReLU())
model:add(nn.Linear(nhidden, noutputs))
model:add(nn.LogSoftMax())
--]]




--Step 3: Defne Our Loss Function
criterion = nn.ClassNLLCriterion()




-- classes
--classes = {'Classical','Jazz'}
--Obtained from GetTrainAndTestData

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
print(confusion)
-- Log results to files
trainLogger = optim.Logger(paths.concat('.', 'train.log'))
testLogger = optim.Logger(paths.concat('.', 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end




optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = 0.01,
    learningRateDecay = 5e-7
}
optimMethod = optim.sgd
--print(torch.randperm(11))




epoch = 1
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()
   --print(#trainData)
   -- shuffle at each epoch
   shuffle = torch.randperm(trainData:size())
   --print(shuffle:size(1) - #trainData)
   -- do one epoch
   --print('==> doing epoch on training data:')
   --print("==> online epoch # " .. epoch .. ' [batchSize = ' .. 64 .. ']')
   --print(trainData:size())
    
   for t = 1, trainData:size() do
        --break
      --print ("HERE" .. shuffle[t])
      -- disp progress
      --xlua.progress(t, trainData:size())
      --for ts = 1,trainData[shuffle[t]]:size(2),64 do
      -- create mini batch
      
        
        
      --local inputs = {}
      --local targets = {}

      --for i = t,math.min(t+64-1,trainData:size()) do
         -- load new sample
         --local input = trainData.Songs[shuffle[i]]
         --local target = trainData.Labels[shuffle[i]]
         --input = input:double()
         --table.insert(inputs, input)
         --table.insert(targets, target)
      --end
    
        
        
        
        
        
       
      local inputs = {}
      inputs[1] = trainData.Songs[shuffle[t]]
      --print(inputs[0]:size(2))
      local targets = {}
      targets[1] = trainData.Labels[shuffle[t]]
        
        
        
        
    --print(inputs)
    -- print(targets)

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0
                       local spl_counter = 0
                       --print("Evaluating mini-batch")
                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          --print(i)
                          -- estimate f   
                          --print("Calculating output via mode:forward(inputs[i])")
                          --local splitted = inputs[i]:split(spl,1)
                           --print(splitted[1]:size())
                          --for j = 1,#splitted do
                          
                          --if splitted[j]:size(1) * splitted[j]:size(2) ~= 128*spl then
                          -- break
                        --end
                          spl_counter  = spl_counter+1
                          --print(inputs[i]:size())
                          --print(splitted[j])
                          local output = model:forward(inputs[i])
                          --output = torch.reshape(output, #classes)
                          --Maybe have to reshape outpit
                          
                          --print("Calculating error")
                          --print(output:size())
                          --print(targets[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          
                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                        --if (j % 3) then
                          confusion:add(output, targets[i])
                        --end
                       --end
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(spl_counter)
                       f = f/spl_counter

                       --print("Returning from feval")
                       -- return f and df/dX
                       return f,gradParameters
                    end

                   config = {learningRate = 0.003, weightDecay = 0.01, 
      momentum = 0.01, learningRateDecay = 5e-7}
        --print("Before optim.sgd")
        optim.sgd(feval, parameters, config)
        --print("After optim.sgd")
   end

    --print("Before taking time")
    
   -- time taken
   time = sys.clock() - time
   time = time / #trainData
   --print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if true then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat('.', 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
  end
--train()




getClass = function(preds,target,confusion)
local c = {}
c[1] = 0
c[2] = 0
c[3] = 0
for i = 1,#preds
    do 
        local m = preds[i][1]
        local current = 1
        for j=2,3
        do
            if(preds[i][j] > m)
            then
                current = j
            end
        end
        c[current] = c[current] + 1
    end
--print(c)
--print(target)
end




function test()
   -- local vars
   local time = sys.clock()

   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
  print(testData:size())
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.Songs[t]
      --input = input:double()
      local target = testData.Labels[t]
      --local sum = torch.Tensor(3)
      --preds = {}
      -- test sample
      --local splitted = input:split(spl,1)
      ---                          for j = 1,#splitted do
         --                 if splitted[j]:size(1) * splitted[j]:size(2) ~= 128*spl then
           --                break
             --           end
      local pred = model:forward(input)
      pred = torch.reshape(pred, #classes)
      --preds[j] = pred
      --print("prediction")
      --print(pred)
      --print("target")
      --print(target)
      --sum = sum + pred
      confusion:add(pred, target)
      --print (confusion)
      end
      --getClass(preds,target,confusion)
      --confusion:add(sum,target)
  -- end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if true then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
--test()




for i = 1, 40 do
    train()
    test()    
end


















