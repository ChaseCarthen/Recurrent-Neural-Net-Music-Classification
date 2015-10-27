local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetLoader'
require 'lfs'
require 'math'

require 'rnn'


require 'model'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A music classifier of doom.")
cmd:text()
cmd:text('Options')
cmd:option("--cuda",false,"Use cuda")
cmd:option("-data","processed","Specify the directory with data.")
cmd:text()

params = cmd:parse(arg or {})
cuda = params.cuda

if cuda
then
	print("CUNN")
	require 'cunn'
end

local math = require 'math'




--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData = {}
testData = {}
classes = {}

dl = DatasetLoader("processed","au","audio")

classes = dl.classes

--trainData, testData, classes = GetTrainAndTestData("./music", .8)
trainData,testData,validationData,classes = GetTrainAndTestData{BaseDir="./audio",Ratio=.8,Ratio2=.1}


DatasetLoader()


print(classes)
print(trainData.Labels[1])


Cudaify = function (mlp)
	mlp:cuda()
	local model = nn.Sequential()
  --model:add(nn.Copy('torch.ByteTensor', 'torch.FloatTensor'))
   
	--model:add(nn.Sequencer(nn.View(32)))
  --model:add(nn.SplitTable(1,2))
	model:add(nn.Sequencer(nn.Copy('torch.ByteTensor', 'torch.CudaTensor')))
	model:add(mlp)
	model:add(nn.Sequencer(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
	return model
end

-- This Describes the default model to be generate for classification.
DefaultModel = function(num_output)

	--Best NN so far
	mlp = nn.Sequential()
  
        if not cuda then
        print("view")
        mlp:add(nn.SplitTable(1,2))
	mlp:add(nn.Sequencer(nn.View(32)))
        end

  r2 = nn.Recurrent(
   20, nn.Linear(32, 20), 
   nn.Linear(20, 20), nn.Sigmoid(), 
   500
)      
	--mlp:add(nn.Sequencer(nn.LSTM(32,20)))
  mlp:add(nn.Sequencer(r2))
  --mlp:add(nn.Sequencer(nn.Sigmoid()))
  mlp:add(nn.Sequencer(nn.Linear(20,num_output)))
	mlp:add(nn.Sequencer(nn.LogSoftMax()))
 
  if(cuda) then
    print("CUDA")
    mlp = Cudaify(mlp)         
  end
	return mlp
end


model = DefaultModel(#classes)


--Step 3: Defne Our Loss Function
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


confusion = optim.ConfusionMatrix(classes)
trainLoggers = {}
testLoggers = {}
parameters = {}
gradParameters = {}

if model then
  parameters,gradParameters = model:getParameters()
end


optimState = {
    learningRate = 0.001,
    weightDecay = 0.1,
    momentum = 0.001,
    learningRateDecay = 5e-7
}
optimMethod = optim.sgd



epoch = 1
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   shuffle = torch.randperm(trainData:size())


       for t = 1, trainData:size() do

            --print("Label:", trainData.Labels[shuffle[t]], "ModelIndex: ", modelIndex)

            
               local inputs = {}
               table.insert(inputs, trainData.Songs[shuffle[t]])

              local targets = {}
                class = trainData.Labels[shuffle[t]]
                
               table.insert(targets, class)      

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

                           -- evaluate function for complete mini batch
                           for i = 1,#inputs do
                            inputs[i] = inputs[i]:split(100)
                            inputs[i][#inputs[i]] = nil
                              --xlua.progress(i,#inputs)
                              print("HERE")
                           local output = model:forward(inputs[i])
                           local c = targets[i]
                           targets[i] = torch.ones(#inputs[i],100)
                           targets[i]:fill(c)
                          local err = criterion:forward(output, targets[i])
                           f = f + err        
                          current_loss = current_loss + f
                           local df_do = criterion:backward(output, targets[i])        
                            model:backward(inputs[i], df_do) 
                            
                            for i2 = 1,#output do
                            for j2 = 1,100 do
                            confusion:add(output[i2][j2], targets[i][1][1])
                            end
                            end
                            collectgarbage();
                           end

                           -- normalize gradients and f(X)
                           gradParameters:div(#inputs)
                            f = f/#inputs

                           return f,gradParameters
                end
                optim.sgd(feval, parameters, optimState)                         
           end --End of for loop




            current_loss = current_loss / trainData:size()
            
           time = sys.clock() - time
           time = time / #trainData

           -- print confusion matrix
           print(confusion)

           -- next epoch
           confusion:zero()
           epoch = epoch + 1
end
current_loss = 0
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
          local input = testData.Songs[t]:split(100)
          input[#input] = nil
          --input = input:double()
          local target = testData.Labels[t]

              local pred = model:forward(input)

              local c = target
              target = torch.ones(#input,100)
              target:fill(c)
              --pred = torch.reshape(pred, 2)
              for i2 = 1,#pred do
                for j2 = 1,100 do
                  confusion:add(pred[i2][j2], target[1][1])
                end
              end
      end
       time = sys.clock() - time
       time = time / testData:size()
       print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

       -- print confusion matrix
       print(confusion)
       if average then
          -- restore parameters
          parameters:copy(cachedparams)
       end

       -- next iteration:
       confusion:zero()       
end
--test()

for i = 1, 400 do
    print("Epoch: ", i)
    train()
    if math.fmod(i,2) == 0 then
        test()
    end
end


-- save/log current net
local filename = paths.concat('.', 'SNNModel.net')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving model to '..filename)
torch.save(filename, models)




print(classes)



local RNNC = torch.class ("RNNC","model")

function RNNC:__init()
  model.__init(self)
end

function RNNC:addlayer(layer)
  if self.model == nil then
    self.model = nn.Sequential()
  end
  self.model:add(layer)
end

function RNNC:addLSTM(input,output)
  self.addlayer(nn.FastLSTM(input,output))
end

--- Need to figure out something better for this
function RNNC:addRNN(rnn)
  self.addlayer(rnn)
end

function printmodel()
  print(self.model)
end

function RNNC:setup()

end

function RNNC:forward(input)
  return self.model:forward(input) -- return some kind of output here
end

function RNNC:backward(input,output,target)
end