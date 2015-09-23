local torch = require 'torch'
require "rnn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
--require 'nnx'
require 'writeMidi'
require 'cunn'
require 'cutorch'

cutorch.setDevice(1)
--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData, testData, classes = GetTrainAndTestData("./miniMusic", .8)


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

mlp:add(nn.Linear(128,128))
--mlp:add(nn.Dropout(.2))
--mlp:add(nn.Tanh())
--mlp:add(nn.Linear(64,32))

--mlp:add(nn.Square())
--mlp:add(nn.Sum())
mlp:add(nn.Tanh())
--mlp:add(nn.Linear(16,#classes))
--mlp:add(nn.LogSoftMax())

--r = mlp

inpMod = nn.Linear(16,16)
rhobatch = 100
rho = 50
r2 = nn.Recurrent(
   128, mlp, 
   nn.Linear(128, 128), nn.Sigmoid(), 
   rho
)
r = nn.LSTM(128,128)

Cudaify = function (mlp)
  mlp:cuda()
  local model = nn.Sequential()
  --model:add(nn.Reshape(128))
  --model:add(nn.Sequencer(nn.Copy('torch.FloatTensor', 'torch.CudaTensor')))
  --model:add(mlp)
  --model:add(nn.Sequencer(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
  --model:cuda()
  return mlp
end


model = nn.Sequential()
model:add(nn.Sequencer(r))
model:add(nn.Sequencer(nn.Sigmoid()))
--model:add(mlp)
--model:add(r)
--model:add(nn.Tanh())
--model:add(r2)
--model:add(nn.ReLU())

--model:add(nn.Reshape(1,40))
--model:add(nn.Linear(2 ,128))
--model:add(nn.Sum())
--model:add(nn.Tanh())
--model:add(nn.LogSoftMax())
--model:add(nn.MulConstant(127))
--model:add(nn.Abs())
--ab = torch.randn(100,128)

--print(model:forward(ab))
--print(model:forward(a))
--Step 3: Defne Our Loss Function
--criterion = nn.MultiMarginCriterion()
--criterion = nn.ClassNLLCriterion()
--criterion = nn.AbsCriterion()
--criterion = nn.MSECriterion()
model = Cudaify(model)
 criterion = nn.SequencerCriterion(nn.BCECriterion())
 criterion:cuda()
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
                          local testcounter = 0 
                          local is = inputs[i]:split(rhobatch)
                          --print(is)
                          for j=1,#is--,rhobatch
                          do
                          if(is[j]:size(1) ~= rhobatch)
                          then
                          is[j] = nil
                          end
                          end
                          testcounter = testcounter + 1
                          spl_counter  = spl_counter+1
                          counter = counter+1
                          --print(is[j])
                          --print(inputs[i]:size())
                          --print(splitted[j])
                          --print(inputs[i])
                          --local output = model:forward(inputs[i])

                          --local tr = is[j]:split(1)
                          --for k=1,rhobatch do
                          --  table.insert(tr,is[j][k])
                          --end
                          --print(tr)
                          local output = model:forward(is)
                          --print(output)
                          --print("Calculating error")
                          --print(output:size())
                          --print(targets[i])
                          --local err = criterion:forward(output, inputs[i])
                          --print(output[1]:size())
                          --print(tr[1]:size())
                          local err = criterion:forward(output, is)
                          f = f + err

                          
                          -- estimate df/dW
                          --local df_do = criterion:backward(output, inputs[i])
                          --model:backward(inputs[i], df_do)
                          --print(tr)
                          local df_do = criterion:backward(output, is)
                          model:backward(is, df_do)
                          if epoch % 20 == 0 and i % 10 == 0 then
                          local combine = nn.JoinTable(1)
                          songs[testcounter] = (combine:forward(output))
                          end
                          --print("HERE")
--print(output)
                          --end
                          --r:updateParameters(optimState.learningRate)
                          --r:forget()
                          --r2:forget()
                          --print(songs)

                          --print (combine)
                          if epoch % 20 == 0 and i % 10 == 0
                          then
                          local combine = nn.JoinTable(1)
                          combine = combine:forward(songs)
                          
                          combine = combine:round()
                          --print(combine)
                          local val = combine:clone()
                          --val = val:float()
                          --print(val:size())
                                  --val = val:round()
                                  --print(val)
                                  val =  val * 127
                      --local ma = val:max()
                      -- local mi = val:min()
                       --val = (val - mi) / (ma - mi) * 10
                       --print(i)
                       print("Writing " .. epoch .. "song" .. t .. ".mid")
                       writeMidi(epoch .. "song" .. t .. ".mid",val, 50,10)
                          --writeMidi(epoch .. "song" .. t .. "orig.mid",inputs[i]*127,50,10)
                        end
                          -- update confusion
                        --if (j % 3) then
                          --confusion:add(output, inputs[i])
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
        _,fs2 = optim.rmsprop(feval, parameters, optimState)
        --print(fs2)
        loss = loss + fs2[1]
        --print("After optim.sgd")
   end

    --print("Before taking time")
    print(loss/trainData:size())
    --print(trainData:size())
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
   if(epoch % 20 == 0) then
   local filename = paths.concat('.', 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)
   end

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




for i = 1, 400 do
    print("Epoch: " .. i)
    train()
    --test()
end

