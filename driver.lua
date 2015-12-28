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
require 'trainer'
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
  model:addlayer(nn.Sequencer(nn.FastLSTM(32,128)))
  model:addlayer(nn.Sequencer(nn.Linear(128,128)))
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

train = trainer{epochLimit = 200, model = model, datasetLoader = dl, optimModule = optim.rmsprop, optimState = optimState, target = "midi",input = "audio"}

while not train:done() do
    print("Epoch: ", train.epoch)
    --test()
    --train()
    train:train()
    if train.epoch % 40 == 0 then
        --test()
    end
end


-- save/log current net
--local filename = paths.concat('.', 'SNNModel.net')
--os.execute('mkdir -p ' .. sys.dirname(filename))
--print('==> saving model to '..filename)
--torch.save(filename, models)
