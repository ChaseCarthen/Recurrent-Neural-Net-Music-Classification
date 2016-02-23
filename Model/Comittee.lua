local torch = require 'torch'
nn = require "nn"
require "nn"
require "optim"
require 'DatasetGenerator'
require 'lfs'
local math = require 'math'

require 'cunn'

----------------------------------GATHER DATA----------------------------------
--Get Data
trainData, testData, classes = GetTrainAndTestData("./music", .5)


print (testData)
--Add a Results parameter to trainData and testData
trainData.Results = {}
testData.Results = {}




---------------------------------GATHER MODELS---------------------------------
Models = {}

--Get Single Neural Network Data
local filename = paths.concat('.', 'SNNModel.net')
SingleModels = torch.load(filename)
for i=1, #classes do
	table.insert(Models, SingleModels[i])
end


--Add Next Model
--local filename = paths.concat('.', 'NextModel.net')
--NextModel = torch.load(filename)
--table.insert(Models, NextModel)


print("Number of Models: ", #Models)
print("Finished Gathering Models")


---------------------------------Concatinate All Outputs of Models---------------------------------
for songIndex=1, trainData:size() do
	classOutput = Models[1]:forward(trainData.Songs[songIndex])
	trainData.Results[songIndex] = classOutput
	--print("Result Size", #trainData.Results[songIndex])

	for i=2, #Models do
		classOutput = Models[i]:forward(trainData.Songs[songIndex])
		--print("classOutput Result Size", #classOutput)
		trainData.Results[songIndex] = torch.cat(trainData.Results[songIndex], classOutput)
		--print("Result Size1", #trainData.Results[songIndex])
	end
end

--print("Output size", #trainData.Results[2])
OutputSize = trainData.Results[1]:size(1)



for songIndex=1, testData:size() do

	classOutput = Models[1]:forward(testData.Songs[songIndex])
	testData.Results[songIndex] = classOutput
	for i=2, #Models do
		classOutput = Models[i]:forward(testData.Songs[songIndex])
		testData.Results[songIndex] = torch.cat(testData.Results[songIndex], classOutput)	
	end
end

print("Finished Gathering Data")
------------------------------END OF GATHER DATA---------------------------------


model = nn.Sequential()
model:add(nn.Linear(OutputSize, 2500))
model:add(nn.Dropout(0.2))
model:add(nn.Tanh())
model:add(nn.Linear(2500,#classes))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

confusion = optim.ConfusionMatrix(classes)

if model then
   parameters,gradParameters = model:getParameters()
end




optimState = {
    learningRate = 0.001,
    weightDecay = 0.01,
    momentum = 0.01,
    learningRateDecay = 5e-7
}
optimMethod = optim.sgd
--print(torch.randperm(11))




epoch = 1
batch_size = 32
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

    
   for t = 1, trainData:size(),batch_size do
   
      xlua.progress(t, trainData:size())  
      local inputs = {}
      
      --print(inputs[0]:size(2))
      local targets = {}
      for s=0,batch_size
      do
      if t+s > trainData:size() then
      break
      end
      inputs[s] = trainData.Results[shuffle[t+s]]
      targets[s] = trainData.Labels[shuffle[t+s]]
      end  
        
        
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
                     
                          spl_counter  = spl_counter+1
                          --print(inputs[i]:size())
                          --print(splitted[j])
                          local output = model:forward(inputs[i])
                      
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
                       --print(spl_counter)
                       --print("Returning from feval")
                       -- return f and df/dX
                       return f,gradParameters
                    end

                   --config = {learningRate = 0.003, weightDecay = 0.01, 
      ---momentum = 0.01, learningRateDecay = 5e-7}
        --print("Before optim.sgd")
        optim.sgd(feval, parameters, optimState)
        --print("After optim.sgd")
   end

    --print("Before taking time")
    
   -- time taken
   time = sys.clock() - time
   time = time / #trainData
   --print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)


   -- save/log current net
   local filename = paths.concat('.', 'ComModel.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   --torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
train()



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
      local input = testData.Results[t]
      local target = testData.Labels[t]
      local pred = model:forward(input)
      pred = torch.reshape(pred, #classes)
      confusion:add(pred, target)
      end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)


   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end



epochs = 100
for i=1, epochs do
    train()
    test()
end
