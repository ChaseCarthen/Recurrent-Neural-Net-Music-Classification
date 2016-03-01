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

  encoder:add(nn.FastLSTM(1281,256))
  encoder:add(nn.Sigmoid())
  encoder = nn.Sequencer(encoder)
  decoder = nn.Sequential():add(nn.Linear(256,1281))
  decoder = nn.Sequencer(decoder)
  params.modelfile = params.autoencoderfile
  
  --print(encoder)

  ae = AutoEncoder(encoder,decoder)
  model = StackedAutoEncoder()
  model:AddLayer(ae)
  --model:AddLayer(ae2)
  --model:AddLayer(ae3)
  --model:AddLayer(ae4)
  layer = model:getLayerCount()
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