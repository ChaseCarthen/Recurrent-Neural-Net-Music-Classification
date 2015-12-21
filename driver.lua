require 'RNNC'
local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetLoader'
require 'lfs'
require 'math'
require 'torch'
require 'rnn'
require 'image'
require 'nn'
require 'model'
require 'writeMidi'
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A music transcription neural network for going from midi to wav.")
cmd:text()
cmd:text('Options')
cmd:option("--cuda",false,"Use cuda")
cmd:option("-data","processed","Specify the directory with data.")
cmd:text()

params = cmd:parse(arg or {})
cuda = params.cuda


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

if cuda
then
	print("CUNN")
  require 'cutorch'
	require 'cunn'
end

local math = require 'math'




--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData = {}
testData = {}
classes = {}

dl = DatasetLoader("processed","au","audio")

classes = dl.classes


-- This Describes the default model to be generate for classification.
DefaultModel = function(num_output)

  local model = RNNC() 

  mlp=nn.Sequential()
mlp:add(nn.Linear(32,128))
mlp:add(nn.Tanh())

rhobatch = 30000
rho = 5000
r2 = nn.Recurrent(
   128, mlp, 
   nn.Linear(128, 128), nn.Sigmoid(), 
   rho
)

  mlp2=nn.Sequential()
mlp2:add(nn.Linear(16,128))
mlp2:add(nn.Tanh())

--rhobatch = 10000
--rho = 50
r3 = nn.Recurrent(
   128, mlp2, 
   nn.Linear(128, 128), nn.Sigmoid(), 
   rho
)
r2 = nn.Sequencer(r2)
r3 = nn.Sequencer(r3)
encoder = nn.Sequential()
  model:addlayer(nn.BiSequencer(nn.FastLSTM(32,128)))
  model:addlayer(nn.Sequencer(nn.Linear(256,128)))
  model:addlayer(nn.Sequencer(nn.Sigmoid()))
  if(cuda) then
  	model:cudaify('torch.FloatTensor')       
  end
  model:printmodel()
	return model
end


model = DefaultModel(#classes)


--Step 3: Defne Our Loss Function
criterion = nn.SequencerCriterion(nn.BCECriterion())
model:setCriterion(criterion)

confusion = optim.ConfusionMatrix(classes)

model:initParameters()


optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = .01,
    learningRateDecay = 1e-7
}


epoch = 1
function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:train()
   dl:loadTraining()
   numTrain = dl:numberOfTrainingSamples() 
   shuffle = torch.randperm(numTrain)
   done = false
   loss = 0
   count = 0
   while not done do
    data = dl:loadNextSet()
    collectgarbage();
   	done = data.done

              -- create closure to evaluate f(X) and df/dX
              local feval = function(x)

                           -- get new parameters
                           if x ~= model:getParameters() then
                              print (x)
                              model:getParameters():copy(x)
                           end
                           -- reset gradients
                           model:getGradParameters():zero()

                           -- f is the average of all criterions
                           local f = 0

                           -- evaluate function for complete mini batch
                           for i = 1,#data do
                            xlua.progress(i, #data)

                            inputs = data[i].data:float():split(rhobatch)

                            inputs[#inputs] = nil

                          target = data[i].binVector:t():float():split(rhobatch)
                           local out = {}
                           for testl = 1,#inputs do
                            input = {inputs[testl]}
                           local output = model:forward(input)

                           out[testl] = output[1]:clone()
                          local err = model:backward(input,output,{target[testl]})--inputs)
                           f = f + err
                          
                          count = count + 1
      
                         end

                        if epoch % 5 == 0 and count % 4 == 0 then
                            torch.save("test" .. count .. "epoch" .. epoch .. ".dat",out)
                            
                        end  
                            
                        out = nil
                           end

                           -- normalize gradients and f(X)
                           print(#data)
                           model:getGradParameters():div(count)--#data)
                            f = f/count--#data

                           return f,model:getGradParameters()
                end
                _,fs2 = optim.rmsprop(feval, model:getParameters(), optimState)
                loss = loss + fs2[1]

   end -- End of while loop

          print(loss/count)
           print(confusion)

           -- next epoch
           confusion:zero()
           epoch = epoch + 1
end -- end function
--train()


function test()
   -- local vars
   local time = sys.clock()

       -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
       model:test()
       -- test over test data
       print('==> testing on test set:')
       done = false
       while not done  do
          -- disp progress
          --xlua.progress(t, testData:size())
          dl:loadValidation()
          data = dl:loadNextSet()
          done = data.done
          for i=1,#data do 
          -- get new sample
          local input = data[i].data:split(rhobatch)
          
          --input[#input] = nil
          --input = input:double()
          input[#input] = nil
          --print (input)
          local output = model:forward(input)
          join = nn.JoinTable(1)
          output = join:forward(output)
          output:round()
          song = torch.zeros(output:size(1),1)

          end
      end     
end


for i = 1, 5 do
    print("Epoch: ", i)
    --test()
    train()
    if i % 40 == 0 then
        --test()
    end
end


-- save/log current net
local filename = paths.concat('.', 'SNNModel.net')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving model to '..filename)
torch.save(filename, models)




print(classes)
