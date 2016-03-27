  require 'Model/StackedAutoEncoder'
  require 'Model/AutoEncoder'
  require 'rnn'

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

  params.layer = 1
  params.target = params.input

  encoder = nn.Sequential()

  encoder:add(nn.FastLSTM(1281,500))
  encoder:add(nn.HardTanh())
  encoder = nn.Sequencer(encoder)
  decoder = nn.Sequential():add(nn.Linear(500,1281))
  decoder = nn.Sequencer(decoder)
  print(decoder)
  params.modelfile = params.autoencoderfile
  
  --print(encoder)

  ae = AutoEncoder(encoder,decoder)
  model = StackedAutoEncoder()

  encoder = nn.Sequential()
  encoder:add(nn.FastLSTM(500,100))
  encoder:add(nn.ReLU())
  encoder = nn.Sequencer(encoder)
  decoder = nn.Sequential()
  decoder:add(nn.Linear(100,500))
  decoder = nn.Sequencer(decoder)
  ae2 = AutoEncoder(encoder,decoder)

  model:AddLayer(ae)
  --model:AddLayer(ae2)
  --model:AddLayer(ae3)
  --model:AddLayer(ae4)
  layer = model:getLayerCount()
  print("LAYER: " .. layer)
  AutoEncoderMod = model
  params.TrainAuto = true
  if(cuda) then
    model:cudaify('torch.FloatTensor')
    print("DONE")
  end


  criterion = nn.MSECriterion()


  optimState = {
    learningRate = 0.005,
    --weightDecay = 0.01,
    --momentum = .01,
    --learningRateDecay = 1e-7
  }