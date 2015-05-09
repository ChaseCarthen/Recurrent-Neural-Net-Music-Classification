print("STARTING")
local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
require 'nnx'
require 'midibinning'
--require 'cunnx'
require 'cunn'
local file = require 'file'

local files = getFiles("./midibins",'.dat')
print(files)
-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp= nn.Sequential(); 
p1_mlp:add(nn.Linear(128,2))
p1_mlp:add(nn.Sum())

-- But we want to push examples towards or away from each other
-- so we make another copy of it called p2_mlp
-- this *shares* the same weights via the set command, but has its own set of temporary gradient storage
-- that's why we create it again (so that the gradients of the pair don't wipe each other)
p2_mlp= nn.Sequential(); 
p2_mlp:add(nn.Linear(128,2))
p2_mlp:add(nn.Sum())
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

-- we make a parallel table that takes a pair of examples as input. they both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)



-- now we define our top level network that takes this parallel table and computes the pairwise distance betweem
-- the pair of outputs
mlp2= nn.Sequential()
mlp2:add(prl)
mlp2:add(nn.PairwiseDistance(3))

-- and a criterion for pushing together or pulling apart pairs
crit=nn.HingeEmbeddingCriterion(1)


criterion = nn.ClassNLLCriterion()

--- Richards model
model = nn.Sequential()

--model:add(nn.Reshape(1,128,512))
--model:add(nn.SpatialContrastiveNormalization(1,image.gaussian1D(5)))
model:add(nn.SpatialConvolution(1, 6, 5, 5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(6, 16, 5,5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(16 * 125 * 29))
model:add(nn.Linear(16 * 125 * 29, 256))
model:add(nn.Tanh())
model:add(nn.Dropout(0.2))
--model:add(nn.Linear(256, #classes))
model:add(nn.Linear(256,128*512))
model:add(nn.ReLU())
model:add(nn.Linear(128*512,10))
--model:add(nn.Tanh())
model:add(nn.LogSoftMax())
model2 = model
model2:cuda()

print("MODEL LOADED")
model = nn.Sequential()
model:add(nn.Reshape(1,128,512))
model:add(nn.SpatialContrastiveNormalization(1,image.gaussian1D(5)))
model:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
model:add(model2)
model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))

local numofw = 11
--distance2 = nn.Euclidean(128*512,128*512)


-- GetClass
function getClass(input,weights,numberOfClasses,inputWidth)
  --print("GETCLASS")
  
  local max = 0
  local class = 1 
  --print("GetClass")
  --print(numberOfClasses)
  --print(weights[1]:size())
  --print(input:size())
  a = nn.View(512*128)
  local inp = a:forward(input)
  --print(inp[1]:size())
  for i=1,numberOfClasses do
    --print(inp)
    --print(weights[i])
    if torch.dist(inp,weights[i]) > max then
      class = i
      max = torch.dist(inp,weights[i])
    end


  end
  --print(class)
  return class
end



-- Use a typical generic gradient update function
function gradUpdate(x, y, criterion, learningRate)
--  print(#x)
--print(x[1])
local pred = mlp2:forward(x)
print(pred)
--print("pred")
--print(pred)
if pred[1] > 5000
then
y = 1
else
y = -1
end 
--print (y)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)

--mlp2:zeroGradParameters()
mlp2:backward(x, gradCriterion)
--mlp2:updateParameters(learningRate)
return err
end





--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
--trainData, testData, classes = GetTrainAndTestData("./music", .8)
--print(classes)
classes = 10

--print(model:forward(ab))
--print(model:forward(a))
--Step 3: Defne Our Loss Function
--criterion = nn.MultiMarginCriterion()
--criterion = nn.ClassNLLCriterion()
--criterion = nn.AbsCriterion()
--criterion = nn.MSECriterion()
--criterion = nn.DistKLDivCriterion()
-- classes
--classes = {'Classical','Jazz'}
--Obtained from GetTrainAndTestData

-- This matrix records the current confusion across classes
cla = {}
for i = 1,classes do
  cla[i] = i
end
confusion = optim.ConfusionMatrix(cla)
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
batch_size = 10
function train()

   -- epoch tracker
   epoch = epoch or 1

   local counter = 1
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()
   --print(#trainData)
   -- shuffle at each epoch
   --print(files)
   shuffle = torch.randperm(#files)

   loss = 0
   for t = 1, #files do
      xlua.progress(t, #files ) 
      
      local inputs = torch.load(files[shuffle[t]]).data
      for l = 1, #inputs do
      xlua.progress(l, #inputs ) 
      --trainData.Songs[t]
      --print(files[t])
      --print(inputs[1]:size())

      

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

                       --for i = 1,1 do
                          --print(1)
                          --print(classes)
                          spl_counter  = spl_counter+1
                          counter = counter+1

                          local output = model:forward(inputs[l])
                          --local output2 = model:forward(targets[i])
                          --print("Labels: " .. l[i] .. " " .. trainData.Labels[t])
                          --print(model:get(12))
                          local W = nn.Copy('torch.CudaTensor', 'torch.FloatTensor'):forward(model2:get(numofw).weight)
                          local cla = getClass(inputs[l],W,classes,128*512)
                          local err = criterion:forward(output,cla)--gradUpdate({inputs[1],targets[i]},1,crit,optim.learningRate)--criterion:forward(output, inputs[1])
                          f = f + err

                          
                          -- estimate df/dW
                          --local df_do = criterion:backward(output, inputs[1])
                          --model:backward(inputs[i], df_do)
                          local df_do = criterion:backward(output, cla)
                          --print(df_do)
                          model:backward(inputs[1], df_do)
                          --songs[j] = (output)
                          --confusion:add(output, cla)
                          

                       --end

                       -- normalize gradients and f(X)
                       gradParameters:div(spl_counter)
                       f = f/spl_counter
                       --print(spl_counter)
                       --print("Returning from feval")
                       -- return f and df/dX
                       return f,gradParameters
                    end

        _,fs2 = optim.sgd(feval, parameters, optimState)
        --print(fs2)
        loss = loss + fs2[1]
        end

        --print("After optim.sgd")
   end

    --print("Before taking time")
    print(loss/#files)
    --print(trainData:size())
   -- time taken
   time = sys.clock() - time
   time = time / #files
   --print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if true then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      --trainLogger:plot()
   end

   -- save/log current net
   if epoch % 1 == 0 then
   local filename = paths.concat('.', 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   end

   -- next epoch
   --confusion:zero()
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
   --model:evaluate()
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


clusterfile = "cluster"

for i = 1, 40 do
    print("Epoch: " .. i)
        if i%1 == 0 then
      for j = 1,classes do
        file.write(clusterfile .. j .. ".txt","")
      end
      print("Writing to Clusters")

      local W = nn.Copy('torch.CudaTensor', 'torch.FloatTensor'):forward(model2:get(numofw).weight)
      --print(trainData)
      for k =1,#files do
        local DETA = torch.load(files[k])
        xlua.progress(k, #files ) 
        --if DETA ~= nil then
        for detas = 1,#DETA.files do
        xlua.progress(detas, #DETA.files ) 
        output = model:forward(DETA.data[detas])
        local group = getClass(DETA.data[detas],W,classes,128*512)
        file.write(clusterfile .. group .. ".txt",DETA.files[detas] .. "\n","a")
        --end
        end
      end


    end
    train()


    --test()
end

