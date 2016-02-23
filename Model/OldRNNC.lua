local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
require 'math'

cuda = false
if arg[1] == "cuda"
then
	cuda = true
end

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
trainData, testData, classes = GetTrainAndTestData("./music", .8)




print(classes)
print(trainData.Labels[1])


Cudaify = function (mlp)
	mlp:cuda()
	local model = nn.Sequential()
	model:add(nn.Reshape(2*500*128))
	model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
	model:add(mlp)
	model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
	return model
end

-- This Describes the default model to be generate for classification.
DefaultModel = function(num_output)

	--Best NN so far
	mlp = nn.Sequential()
	--mlp:add(nn.SpatialContrastiveNormalization(2,image.gaussian1D(5)))
	--mlp:add(nn.SpatialConvolution(2, 2, 1, 5))
	--mlp:add(nn.SpatialMaxPooling(2,2,2,2))

	--mlp:add(nn.SpatialContrastiveNormalization(4,image.gaussian1D(5)))
	--mlp:add(nn.SpatialConvolution(4, 4, 5, 5))
	--mlp:add(nn.SpatialMaxPooling(2,2,2,2))
	
	--16 layers, 30x125 image
        if not cuda then
        print("view")
	mlp:add(nn.View(2*500*128))
        end
	mlp:add(nn.Linear(2*500*128, 100))
	mlp:add(nn.Dropout(.1))
	mlp:add(nn.Tanh())
	mlp:add(nn.Linear(100, 50))
	mlp:add(nn.Dropout(.1))
	mlp:add(nn.Tanh())
	mlp:add(nn.Linear(50, 2))
	mlp:add(nn.LogSoftMax())
 
        if(cuda) then
		--mlp:cuda()
        	mlp = Cudaify(mlp)         
        end
        print(mlp)
	return mlp
end



-- Generating a bag of classifiers -- of default model type
GenerateBagOfClassifiers = function(numberofclasses)
	local models = {}
	for i=1,numberofclasses
	do
		models[i] = DefaultModel(2)
	end

	return models
end

models = GenerateBagOfClassifiers(#classes)


--Step 3: Defne Our Loss Function
criterion = nn.ClassNLLCriterion()


confusions = {}
trainLoggers = {}
testLoggers = {}
parameters = {}
gradParameters = {}

for i=1,#classes
do
    confusions[i] = optim.ConfusionMatrix({classes[i], "Not "..classes[i]})
    --trainLoggers[i] = optim.Logger(paths.concat('.', 'train'..classes[i]..'.log'))
    --testLoggers[i] = optim.Logger(paths.concat('.', 'test'..classes[i]..'.log'))
    --print(confusions[i])

    if models[i] then
        parameters[i],gradParameters[i] = models[i]:getParameters()
    end

end


SkipCounter = {}
for i=1, #classes do
   SkipCounter[i] = 0 
end


optimState = {
    learningRate = 0.001,
    weightDecay = 0.1,
    momentum = 0.001,
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
    for i=1,#classes
    do
        models[i]:training()
    end

   shuffle = torch.randperm(trainData:size())

    for modelIndex = 1, #classes do
	SkipSize = math.floor((#trainData.Songs - trainData.GenreSizes[modelIndex])/trainData.GenreSizes[modelIndex])
	print(SkipSize, trainData.GenreSizes[modelIndex], classes[modelIndex])
        SkipCounter[modelIndex] = 0
       for t = 1, trainData:size() do

            --print("Label:", trainData.Labels[shuffle[t]], "ModelIndex: ", modelIndex)
            if not (trainData.Labels[shuffle[t]] == modelIndex) then
                SkipCounter[modelIndex] = SkipCounter[modelIndex] + 1
            end
            
            --if 1 == 1 then
            if math.fmod(SkipCounter[modelIndex],SkipSize) == 0 or trainData.Labels[shuffle[t]] == modelIndex then
               local inputs = {}
               table.insert(inputs, trainData.Songs[shuffle[t]])

              local targets = {}
                class = 2
                if trainData.Labels[shuffle[t]] == modelIndex then class = 1 end
                
               table.insert(targets, class)      

              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)
                           -- get new parameters
                           if x ~= parameters[modelIndex] then
                              parameters[modelIndex]:copy(x)
                           end
                           -- reset gradients
                           gradParameters[modelIndex]:zero()

                           -- f is the average of all criterions
                           local f = 0

                           -- evaluate function for complete mini batch
                           for i = 1,#inputs do

			--print("Calculating output")
			--print("Input: ", inputs[i])
                           local output = models[modelIndex]:forward(inputs[i])
			 --print(output)
                          local err = criterion:forward(output, targets[i])
                           f = f + err        
                          current_loss = current_loss + f
                           local df_do = criterion:backward(output, targets[i])        
                            models[modelIndex]:backward(inputs[i], df_do)                
                            confusions[modelIndex]:add(output, targets[i])
                           end

                           -- normalize gradients and f(X)
                           gradParameters[modelIndex]:div(#inputs)
                            f = f/#inputs

                           return f,gradParameters[modelIndex]
                end
                optim.sgd(feval, parameters[modelIndex], optimState)              
                end            
           end --End of for loop




            current_loss = current_loss / trainData:size()
            --print(current_loss)
           time = sys.clock() - time
           time = time / #trainData

           -- print confusion matrix
           print(confusions[modelIndex])

           -- update logger/plot
--[[
           trainLoggers[modelIndex]:add{['% mean class accuracy (train set)'] = confusions[modelIndex].totalValid * 100}
           if true then
              trainLoggers[modelIndex]:style{['% mean class accuracy (train set)'] = '-'}
              trainLoggers[modelIndex]:plot()
           end
--]]
           -- save/log current net
           --local filename = paths.concat('.', 'model'..modelIndex..'.net')
           --os.execute('mkdir -p ' .. sys.dirname(filename))
           --print('==> saving model to '..filename)
           --torch.save(filename, model)

           -- next epoch
           confusions[modelIndex]:zero()
           epoch = epoch + 1
           end
end
current_loss = 0
--train()


function test()
   -- local vars
   local time = sys.clock()

    for i=1, #classes do
       if average then
          cachedparams = parameters[i]:clone()
          parameters[i]:copy(average)
       end

       -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
       models[i]:evaluate()
      print(testData:size())
       -- test over test data
       print('==> testing on test set:')
       for t = 1,testData:size() do
          -- disp progress
          xlua.progress(t, testData:size())

          -- get new sample
          local input = testData.Songs[t]
          --input = input:double()
          local target = 2
          if testData.Labels[t] == i then target = 1 end
              --if target == 0 then target = 2 end

              local pred = models[i]:forward(input)
              --pred = torch.reshape(pred, 2)
              confusions[i]:add(pred, target)
      end
       time = sys.clock() - time
       time = time / testData:size()
       print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

       -- print confusion matrix
       print(confusions[i])

       -- update log/plot
       --testLoggers[i]:add{['% mean class accuracy (test set)'] = confusions[i].totalValid * 100}
       --if true then
       --   testLoggers[i]:style{['% mean class accuracy (test set)'] = '-'}
       --   testLoggers[i]:plot()
       --end

       if average then
          -- restore parameters
          parameters[i]:copy(cachedparams)
       end

       -- next iteration:
       confusions[i]:zero()   
    end     
end
--test()

for i = 1, 10 do
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


