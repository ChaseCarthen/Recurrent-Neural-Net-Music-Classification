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
require 'dpnn'
require 'model'
require 'writeMidi'
require 'Trainer'
require 'AutoEncoder'
require 'BinaryClassReward'
require 'AutoEncoderTrainer'
require 'StackedAutoEncoder'
torch.setdefaulttensortype('torch.FloatTensor')
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A music transcription neural network for going from midi to wav.")
cmd:text()
cmd:text('Options')
cmd:option("--cuda",false,"Use cuda")
cmd:option("-data","processed","Specify the directory with data.")
cmd:option("--serialize",false,"Serialize outputs")
cmd:option("-epochrecord", 50, "Every nth epoch to serialize data.")
cmd:option("-frequency",10,"Every jth dataset is used to be serialized")
cmd:option("-modelfile","train.model","What you wish to save this model as!")
cmd:option("-autoencoderfile","auto.model","The autoencoder file to be used.")
cmd:option("-savemodel", 4, "Every nth epoch you save the model used by this driver.")
cmd:option("--autoencoder",false,"An option to use an autoencoder")
cmd:option("--rnnc",false,"An option to use rnnc model.")
cmd:option("-target","midi","What will be this models target.")
cmd:option("-input","audio","What will be ths models input.")
cmd:option("-dataSplit",20000,"How much data will be split up into sequences be split up.")
cmd:option("-sequenceSplit",5000,"How much a sequence will be split up.")
cmd:option("-epochLimit",200,"How many epochs to run for.")
cmd:option("--attention",false,"Use an attention model.")
cmd:option("--predict",false,"Writing a prediction model?")
cmd:option("--encoded",false,"Use a encoded model.")
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

if params.autoencoder then
  params.target = params.input
end

dl = DatasetLoader("processed",params.input,params.target)

classes = dl.classes


layer = 1

print(params.rnnc)
print(params.attention)
print(params.input)
params.AutoTrain = false
if params.rnnc and params.input == "audio" and not params.attention then
  -- This Describes the default model to be generate for classification.
  DefaultModel = function(num_output)

  local model = RNNC() 

  mlp=nn.Sequential()
  mlp:add(nn.Linear(32,128))
  mlp:add(nn.Tanh())

  rhobatch = 30000
  rho = 5000

  mlp2=nn.Sequential()
  mlp2:add(nn.Linear(256,128))
  mlp2:add(nn.RReLU())

  --rhobatch = 10000
  --rho = 50
  r3 = nn.Recurrent(
   128, mlp2, 
   nn.Linear(128, 128), nn.Sigmoid(), 
   rho
  )

  r3 = nn.Sequencer(r3)
  model:addlayer(nn.Sequencer(nn.FastLSTM(32,256)))
  --model:addlayer(nn.Sequencer(nn.GRU(1000,80)))
  model:addlayer(r3)
  --model:addlayer(nn.Sequencer(nn.Sigmoid()))
   if(cuda) then
  	model:cudaify('torch.FloatTensor')       
   end
    model:printmodel()
	 return model
  end

  model = DefaultModel(#classes)

  elseif params.rnnc and params.attention and params.input == "audio" then
    print("ATTENTION")
  params.sequenceSplit = params.dataSplit
  soft = nn.Sequential()
  soft:add(nn.Linear(200,100))
  soft:add(nn.SoftMax())

  soft2 = nn.Sequential()
  soft2:add(nn.Linear(200,100))
  soft2:add(nn.SoftMax())

  soft3 = nn.Sequential()
  soft3:add(nn.Linear(200,100))
  soft3:add(nn.SoftMax())

  allsoft = nn.Sequential()
  allsoft:add(nn.ConcatTable():add(soft):add(soft2):add(soft3))
  allsoft:add(nn.JoinTable(1,1))
  


  action = nn.Sequential()
  action:add(nn.Linear(20,20))
  action:add(nn.Sigmoid())
  action:add(nn.ReinforceBernoulli(true))

  locationSensor = nn.Sequential()
  locationSensor:add(nn.ParallelTable():add(nn.Linear(16,10)):add(nn.Linear(20,10))) -- first is the passed dataset and second is the action
  locationSensor:add(nn.JoinTable(1,1))
  locationSensor:add(nn.Tanh())
  locationSensor:add(nn.FastLSTM(20,20))
  --locationSensor:add(nn.ReLU())

  attention = nn.Sequential()
  attention:add(nn.RecurrentAttention(locationSensor,action,1,{20}) )
  attention:add(nn.SelectTable(-1))
  --attention:add(allsoft)
  --attention:add(nn.GradientReversal())
  attention:add(nn.GRU(20,128))
  attention:add(nn.Sigmoid())



  reward = nn.Sequential()
  reward:add(nn.Constant(1,1))

  concat = nn.ConcatTable():add(nn.Identity()):add(reward)
  concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

  --attention:add(concat2)
  model = RNNC()
  model:setModel(nn.Sequencer(attention))

   if(cuda) then
    model:cudaify('torch.FloatTensor')       
   end
   model:addlayer(nn.Sequencer(concat2))
  --model:remember('both')
elseif params.autoencoder and params.input == "audio" then
  params.layer = 1
  params.target = params.input
  mlp=nn.Sequential()
  mlp:add(nn.Linear(32,64))
  mlp:add(nn.Tanh())

  rhobatch = 10000
  rho = 50
  r2 = nn.Recurrent(
    64, mlp, 
    nn.Linear(64, 64), nn.Sigmoid(), 
    rho
  )

  mlp2=nn.Sequential()
  mlp2:add(nn.Linear(64,32))
  mlp2:add(nn.Tanh())

  --rhobatch = 10000
  --rho = 50
  r3 = nn.Sequential():add(nn.Dropout(.2)):add(nn.Recurrent(
  32, mlp2, 
  nn.Linear(32, 32), nn.Sigmoid(), 
  rho
  ))
  r2 = nn.Sequencer(r2)
  r3 = nn.Sequencer(r3)
  encoder = nn.Sequential()
  encoder:add(nn.Sequencer(nn.Sequential():add(nn.LSTM(32,64)):add(nn.Sigmoid())  ))
  decoder = nn.Sequential()
  --decoder:add(nn.Sequencer(nn.Dropout(.1)))
  decoder:add(r3)
  params.modelfile = params.autoencoderfile
  
  --print(encoder)

  ae = AutoEncoder(encoder,decoder)
  encoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(64,40)):add(nn.Sigmoid()))
  decoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(40,64)):add(nn.Sigmoid()))
  ae2 = AutoEncoder(encoder,decoder)

  encoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(40,30)):add(nn.Sigmoid()))
  decoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(30,40)):add(nn.Sigmoid()))
  ae3 = AutoEncoder(encoder,decoder)

  encoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(30,16)):add(nn.Sigmoid()))
  decoder = nn.Sequencer(nn.Sequential():add(nn.LSTM(16,30)):add(nn.Sigmoid()))
  ae4 = AutoEncoder(encoder,decoder)

  model = StackedAutoEncoder()
  model:AddLayer(ae)
  model:AddLayer(ae2)
  model:AddLayer(ae3)
  model:AddLayer(ae4)
  layer = model:getLayerCount()
  AutoEncoderMod = model
  params.TrainAuto = true
  if(cuda) then
    model:cudaify('torch.FloatTensor')
    print("DONE")
  end

elseif params.autoencoder and params.input == "midi" then
  params.layer = 1
  params.target = params.input
  mlp=nn.Sequential()
  mlp:add(nn.Linear(128,64))
  mlp:add(nn.Tanh())

  rhobatch = 10000
  rho = 50
  r2 = nn.Recurrent(
    64, mlp, 
    nn.Linear(64, 64), nn.Sigmoid(), 
    rho
  )

  mlp2=nn.Sequential()
  mlp2:add(nn.Linear(64,128))
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
  encoder:add(r2)
  decoder = nn.Sequential()
  decoder:add(r3)
  params.modelfile = params.autoencoderfile

  ae = AutoEncoder(encoder,decoder)
  model = StackedAutoEncoder()
  AutoEncoderMod = model
  model:AddLayer(ae)
  layer = model:getLayerCount()
  params.TrainAuto = true
  print("HERE" .. params.TrainAuto)
  if(cuda) then
    model:cudaify('torch.FloatTensor')
  end

end

--Step 3: Defne Our Loss Function
if params.autoencoder or not params.attention then
criterion = nn.BCECriterion()
else
criterion = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.BCECriterion(nil,false), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.BinaryClassReward(attention), nil, nn.Convert())) -- REINFORCE
end
criterion = nn.SequencerCriterion(criterion)


model:setCriterion(criterion)

confusion = optim.ConfusionMatrix(classes)

model:initParameters()

-- Add a datetime to this output later
trainLogger = optim.Logger('train.log')

trainLogger:setNames{'training error'}--, 'test error')

optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = .01,
    learningRateDecay = 1e-7
  }

if params.encoded then
  AutoEncoderMod = torch.load(params.autoencoderfile)
  layer = #AutoEncoderMod.layer
  params.layer = layer
end


if not params.autoencoder and not params.encoded then
  train = Trainer{dataSplit = params.dataSplit, sequenceSplit = params.sequenceSplit, epochLimit = params.epochLimit, model = model, datasetLoader = dl,
optimModule = optim.rmsprop, optimState = optimState, target = params.target,input = params.input,
serialize = params.serialize,epochrecord = params.epochrecord,
frequency = params.frequency, modelfile = params.modelfile, epochLimit = params.epochLimit, predict = params.predict}
else

  train = AutoEncoderTrainer{dataSplit = params.dataSplit, sequenceSplit = params.sequenceSplit, epochLimit = params.epochLimit, model = model, datasetLoader = dl,
optimModule = optim.rmsprop, optimState = optimState, target = params.target,input = params.input,
serialize = params.serialize,epochrecord = params.epochrecord,
frequency = params.frequency, modelfile = params.modelfile, epochLimit = params.epochLimit, predict = params.predict, TrainAuto = params.TrainAuto, 
layerCount = layer, AutoEncoder = AutoEncoderMod, layer = params.layer }
end

while not train:done() do
    print("Epoch: ", train.epoch)
    --train:tester()
    --train:validater()
    --train()
    train:saveModel()
    trainLogger:add{train:trainer()}
    trainLogger:style{'-'}
    --trainLogger:plot()
    if train.epoch % params.savemodel == 0 then
        --test()
        train:saveModel()
    end
end


-- save/log current net
--local filename = paths.concat('.', 'SNNModel.net')
--os.execute('mkdir -p ' .. sys.dirname(filename))
--print('==> saving model to '..filename)
--torch.save(filename, models)
