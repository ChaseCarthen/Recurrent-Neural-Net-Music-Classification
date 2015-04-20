local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
require 'nnx'
require 'writeMidi'


--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData, testData, classes = GetTrainAndTestData("./music", .8)


--Step 2: Create the model
inp = 128;  -- dimensionality of one sequence element 
outp = 32; -- number of derived features for one sequence element
kw = 8;   -- kernel only operates on one sequence element at once
dw = 8;   -- we step once and go on to the next sequence element
spl = 64 -- split constant
--print(nn)
mlp=nn.Sequential()
--mlp:add(nn.TemporalConvolution(inp,128,kw,dw))
--mlp:add(nn.Reshape(inp))
--mlp:add(nn.Tanh())
--mlp:add(nn.TemporalMaxPooling(8))

--mlp:add(nn.TemporalConvolution(inp,128,4,4))
--mlp:add(nn.Reshape(inp))
--mlp:add(nn.Tanh())
--mlp:add(nn.TemporalMaxPooling(2))

mlp:add(nn.Linear(128,64))
--mlp:add(nn.Dropout(.2))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(64,32))

--mlp:add(nn.Square())
--mlp:add(nn.Sum())
mlp:add(nn.ReLU())
--mlp:add(nn.Linear(16,#classes))
--mlp:add(nn.LogSoftMax())

--r = mlp

inpMod = nn.Linear(16,16)
rho = 10
r = nn.Recurrent(
   32, mlp, 
   nn.Linear(32, 32), nn.ReLU(), 
   rho
)

model = nn.Sequential()
--model:add(mlp)
model:add(r)
--model:add(nn.Reshape(1,40))
model:add(nn.Linear(32,#classes))
model:add(nn.Sum())
model:add(nn.LogSoftMax())
--model:add(nn.MulConstant(127))
--smodel:add(nn.Abs())

--ab = torch.randn(100,128)l

--print(model:forward(ab))
--print(model:forward(a))
--Step 3: Defne Our Loss Function
criterion = nn.MultiMarginCriterion()
--criterion = nn.ClassNLLCriterion()
--criterion = nn.AbsCriterion()
--criterion = nn.MSECriterion()
--criterion = nn.DistKLDivCriterion()
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
    momentum = .01,
    learningRateDecay = 1e-7
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

   loss = 0
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
      inputs[s] = trainData.Songs[shuffle[t+s]]
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
                       local songs = {}
                       --print("Evaluating mini-batch")
                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          local is = inputs[i]:split(100)
                          for j=1,#is
                          do
                          if(is[j]:size(1) ~= 100)
                          then
                          break
                          end
                          spl_counter  = spl_counter+1
                          --print(is[j])
                          --print(inputs[i]:size())l
                          --print(splitted[j])
                          --print(inputs[i])
                          --local output = model:forward(inputs[i])
                          local output = model:forward(is[j])
                          output = output:float()
                          --print("Calculating error")
                          --print(output:size())
                          --print(targets[i])
                          --local err = criterion:forward(output, inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          
                          -- estimate df/dW
                          --local df_do = criterion:backward(output, inputs[i])
                          --model:backward(inputs[i], df_do)
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(is[j], df_do)
			  local reshaper = nn.Reshape(1,#classes)
                    
                          songs[j] = reshaper:forward(output)
                          if(j % rho ==0)
                          then
                          r:updateParameters(optimState.learningRate)
                           end 
                          end
                          local combine = nn.Sequential()
                          combine:add(nn.JoinTable(1))
			  combine:add(nn.Sum())
                          --print(songs)
                          combine = combine:forward(songs)
                          r:forget()
--print(combine)
                          --combine = combine:int()
                          --print (combine)
                          
                          -- update confusion
                        --if (j % 3) then
                          confusion:add(combine, targets[i])
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
        _,fs2 = optim.sgd(feval, parameters, optimState)
        loss = loss + fs2[1]
        --print("After optim.sgd")
   end

    --print("Before taking time")
    --print(loss/trainData:size())
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
      --trainLogger:plot()
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
      --testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end




for i = 1, 400 do
    print("Epoch: " .. i)
    train()
    --test()
end

