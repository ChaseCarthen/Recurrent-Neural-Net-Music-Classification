--- driver.lua 
--- Author: Chase Carthen
--- Description: A driver used for training the models I have attempted to build
--- 

require 'Model/RNNC'
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
require 'writeMidi'
require 'Trainer'
require 'Model/AutoEncoder'
require 'Model/BinaryClassReward'
require 'AutoEncoderTrainer'
require 'Model/StackedAutoEncoder'
require 'paths'

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
--cmd:option("-epochrecord", 10, "Every nth epoch to serialize data.")
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
cmd:option("-windowsize",1000,"Windowsize for passing in samples.")
cmd:option("-stepsize",1,"Step size to step through the samples.")
cmd:option("--temporalconv",false,"Use temporal convolution.")
cmd:option("--normalize",false,"Normalize input data into the model.")
cmd:option("-GPU",1,"The GPU number to use.")
cmd:option("-model","","The Lua file that contains your script file.")
cmd:option("-logname","out","Specifies a common name for log files.")
cmd:text()

params = cmd:parse(arg or {})
cuda = params.cuda

if params.model == "" then
  print("End of the line.")
  print("Where is your model file....")
  print("Specify -model somefile.lua")
end

require (params.model)


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
  if cutorch.getDeviceCount() > params.GPU or params.GPU < 1 then
    params.GPU = 1
  end
  cutorch.setDevice(params.GPU)
end

local math = require 'math'


paths.mkdir(paths.concat(paths.cwd(),"logs"))

--Step 1: Gather our training and testing data - trainData and testData contain a table of Songs and Labels
trainData = {}
testData = {}
classes = {}

if params.autoencoder then
  params.target = params.input
end

dl = DatasetLoader(params.data,params.input,params.target)

classes = dl.classes


--layer = 1

print(params.rnnc)
print(params.attention)
print(params.input)
params.AutoTrain = false

if params.autoencoder and params.input == "audio" then
  params.layer = 1
  params.target = params.input
  params.modelfile = params.autoencoderfile

  AutoEncoderMod = model
  params.TrainAuto = true

end

criterion = nn.SequencerCriterion(criterion)


model:setCriterion(criterion)
model:initParameters()

local dated = os.date()

-- Add a datetime to this output later
logpath = paths.concat(paths.cwd(),"logs",params.logname .. dated)
paths.mkdir(logpath)
logpath = logpath .. "/"
print('traintest' .. params.data .. dated ..  '.log')
print(params.logname)
trainLogger = optim.Logger(logpath .. 'traintest' .. params.logname .. dated ..  '.log')
trainLogger:setNames{'training error', 'test error'}

validLogger = optim.Logger(logpath .. 'valid' .. params.logname .. dated ..  '.log')
validLogger:setNames{'valid error'}

trainAccLogger = optim.Logger(logpath .. 'trainacc' .. params.logname .. dated ..  '.log')
trainAccLogger:setNames{"acc","pre","rec","fmeasure"}

testAccLogger = optim.Logger(logpath .. 'testacc' .. params.logname .. dated ..  '.log')
testAccLogger:setNames{"acc","pre","rec","fmeasure"}

validAccLogger = optim.Logger(logpath .. 'validacc' .. params.logname .. dated ..  '.log')
validAccLogger:setNames{"acc","pre","rec","fmeasure"}

if params.encoded then
  AutoEncoderMod = torch.load(params.autoencoderfile)
  layer = #AutoEncoderMod.layer
  params.layer = layer
end


if not params.autoencoder and not params.encoded then
  train = Trainer{dataSplit = params.dataSplit, sequenceSplit = params.sequenceSplit, epochLimit = params.epochLimit, model = model, datasetLoader = dl,
optimModule = optim.rmsprop, optimState = optimState, target = params.target,input = params.input,
serialize = params.serialize,epochrecord = params.savemodel,
frequency = params.frequency, modelfile = params.modelfile, epochLimit = params.epochLimit, predict = params.predict,
stepsize = params.stepsize, windowidth = params.windowsize, temporalconv = params.temporalconv, normalize = params.normalize}
else

  train = AutoEncoderTrainer{dataSplit = params.dataSplit, sequenceSplit = params.sequenceSplit, epochLimit = params.epochLimit, model = model, datasetLoader = dl,
optimModule = optim.adadelta, optimState = optimState, target = params.target,input = params.input,
serialize = params.serialize,epochrecord = params.savemodel,
frequency = params.frequency, modelfile = params.modelfile, epochLimit = params.epochLimit, predict = params.predict, TrainAuto = params.TrainAuto, 
layerCount = layer, AutoEncoder = AutoEncoderMod, layer = params.layer,
stepsize = params.stepsize, windowidth = params.windowsize, temporalconv = params.temporalconv,normalize = params.normalize }
end



while not train:done() do
    print("Epoch: ", train.epoch)

    --train:tester()
    --train:validater()
    --train()
    --train:saveModel()
    print("======= train ======")
    local trainerror,trainacc,trainpre,trainrec,trainfmeasure = train:trainer()

    print("======= test =======")
    local testerror,testacc,testpre,testrec,testfmeasure = train:tester()
    local validerror = 0
    if train.epoch % params.savemodel == 0 then
      print("====== valid ========")
      validerror,valacc,valpre,valrec,valfmeasure = train:validater()

      -- Log Accuracy -- This is important...
      validAccLogger:add{valacc,valpre,valrec,valfmeasure}
      validAccLogger:style{"-","-","-","-"}

      trainAccLogger:add{trainacc,trainpre,trainrec,trainfmeasure}
      trainAccLogger:style{"-","-","-","-"}

      testAccLogger:add{testacc,testpre,testrec,testfmeasure}
      testAccLogger:style{"-","-","-","-"}

      validLogger:add{validerror}
      validLogger:style{"-"}
      --validLogger:plot()

    end
    trainLogger:add{trainerror,testerror}
    trainLogger:style{'-','-'}
    --trainLogger:plot()
    if train.epoch % params.savemodel == 0 then
        --test()
        train:saveModel()
    end
    train.epoch = train.epoch + 1
end


-- save/log current net
--local filename = paths.concat('.', 'SNNModel.net')
--os.execute('mkdir -p ' .. sys.dirname(filename))
--print('==> saving model to '..filename)
--torch.save(filename, models)
