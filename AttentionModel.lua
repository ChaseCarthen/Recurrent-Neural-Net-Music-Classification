  require 'rnn'
  require 'RNNC'
  print("ATTENTION")
  require 'BinaryClassReward'
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

  action = nn.Sequential()
  action:add(nn.Linear(20,20))
  action:add(nn.Sigmoid())
  action:add(nn.ReinforceBernoulli(true))

  locationSensor = nn.Sequential()
  locationSensor:add(nn.ParallelTable():add(nn.Linear(8,10)):add(nn.Linear(20,10))) -- first is the passed dataset and second is the action
  locationSensor:add(nn.JoinTable(1,1))
  locationSensor:add(nn.Tanh())
  locationSensor:add(nn.FastLSTM(20,20))
  --locationSensor:add(nn.ReLU())

  attention = nn.Sequential()
  attention:add(nn.RecurrentAttention(locationSensor,action,1,{20}) )
  attention:add(nn.SelectTable(-1))
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


  criterion = nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.BCECriterion(nil,false), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.BinaryClassReward(attention), nil, nn.Convert())) -- REINFORCE




optimState = {
    learningRate = 0.005,
    --weightDecay = 0.01,
    --momentum = .01,
    --learningRateDecay = 1e-7
  }