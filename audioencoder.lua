
require 'rnn'
require 'optim'
require 'DatasetGenerator'
require 'cunn'
require 'writeMidi'
--version = 9

require 'cunn'
require 'cutorch'

cutorch.setDevice(1)

function tensorToNumber(tensor)
	local number = 0
	--print(tensor)
	for i=1,32 do
		if(tensor[i] == 1) then
			number = bit.bor(bit.lshift(1,i-1),number)
		end
	end
	return number
end

function numberToTensor(number)
	local tensor = torch.ones(32)
	for i=1,32 do
		tensor[i] = bit.rshift( bit.band( bit.lshift(1,i-1), number ), i-1 )
	end
	return tensor
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Lets generate some audio ya ya')
cmd:text('Options: ')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
local opt = cmd:parse(arg or {})


trainData, testData, classes = GetTrainAndTestData("./audio", .8)


mlp=nn.Sequential()
mlp:add(nn.Linear(1,32))
mlp:add(nn.Tanh())

rhobatch = 50
rho = 50
r2 = nn.Recurrent(
   32, mlp, 
   nn.Linear(32, 32), nn.Sigmoid(), 
   rho
)
r = nn.FastLSTM(1,32)

Cudaify = function (mlp)
  mlp:cuda()
  local model = nn.Sequential()
  --model:add(nn.Sequencer(nn.Copy('torch.FloatTensor', 'torch.CudaTensor')))
  --model:add(mlp)
  --model:add(nn.Sequencer(nn.Copy('torch.CudaTensor', 'torch.FloatTensor')))
  return mlp
end


model = nn.Sequential()
model:add(nn.Sequencer(r2))
model:add(nn.Sequencer(nn.Sigmoid()))

model = Cudaify(model)

 criterion = nn.SequencerCriterion(nn.BCECriterion())
criterion = criterion:cuda()

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
print(confusion)

-- Log results to files
trainLogger = optim.Logger(paths.concat('.', 'train.log'))

if model then
   parameters,gradParameters = model:getParameters()
end


optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = .01,
    learningRateDecay = 1e-7
}


epoch = 1
batch_size = 10
-- the train function
function train()

   -- epoch tracker
   epoch = epoch or 1

   local counter = 1
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

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
                          for j=1,#is
                          do
                           xlua.progress(j, #is)  
                          --print(j)
                          --print(inputs[i]:size())
                          --if(is[j]:size(1) ~= rhobatch)
                          --then
                          --break
                          --end
                          --testcounter = testcounter + 1
                          spl_counter  = spl_counter+1
                          counter = counter+1

                          

                          local tr = is[j]:split(1)
                          local tr2 = is[j]:split(1)
                          for tri=1,#tr do
                          	--print(tr2[i])
                          	tr2[tri] = numberToTensor(tr2[tri][1]):cuda()
                          end
                          local output = model:forward(tr)


                          local err = criterion:forward(output, tr2)
                          f = f + err

                          
                          -- estimate df/dW
                          local df_do = criterion:backward(output, tr2)
                          model:backward(tr, df_do)
                          --print(testcounter .. " " .. inputs[i]:size(1))
                          for jt=1,#output do
                          	--print(jt)
                          	testcounter = testcounter + 1
                          	songs[testcounter] = torch.Tensor({tensorToNumber(output[jt])})
                          	--print(songs[testcounter])
                      	  end
                          --print(songs[testcounter])
                          end


                          --print (combine:size())
                          if epoch % 20 == 0 and i % 1 == 0
                          then
                          local combine = nn.JoinTable(1)

                          ---print(#songs)
                          combine = combine:forward(songs)
                          
                          --combine = combine:round()

                          --local val = torch.ones(combine:size(1),1)
                                  --val =  val * 127
                       --for items=1,combine:size(1) do
                       	--print(combine[items])
                       	--val[items] = tensorToNumber(combine[items])
                       --end
                       --print(combine:size())
                       print("Writing " .. epoch .. "song" .. i .. ".mid")
                       audio.save(epoch .. "song" .. i .. ".au",torch.reshape(combine,combine:size(1),1), 44100/2)

                        end

                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(spl_counter)
                       f = f/spl_counter

                       -- return f and df/dX
                       return f,gradParameters
                    end


        _,fs2 = optim.rmsprop(feval, parameters, optimState)
        loss = loss + fs2[1]

   end

    print(loss/trainData:size())
   -- time taken
   time = sys.clock() - time
   time = time / #trainData
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')


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

for i = 1, 400 do
    print("Epoch: " .. i)
    train()
    --test()
end