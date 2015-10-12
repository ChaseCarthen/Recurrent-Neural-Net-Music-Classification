require 'dp'
require 'rnn'
require 'optim'
model = nn.Sequential()
model:add(nn.SplitTable(1,2))
model:add(nn.Sequencer(nn.LSTM(1,1)))

if model then
   parameters,gradParameters = model:getParameters()
end

local trainInput = dp.SequenceView('bw',torch.randn(100,1))
local targetInput = dp.ClassView('b',torch.ones(100))


local dataset = dp.DataSet{inputs=trainInput,targets=targetInput,which_set='train'}
local ds = dp.DataSource{train_set=dataset}
ds:classes{'one'}
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
optimState = {
    learningRate = 0.003,
    weightDecay = 0.01,
    momentum = .01,
    learningRateDecay = 1e-7
}
-- Lets set up a optimizer
train = dp.Optimizer
{
	acc_update=false,
	loss = criterion,
	callback = function (model,report)

	local feval = function(x)
       	-- get new parameters
        if x ~= parameters then
      		parameters:copy(x)
       	end

       	-- reset gradients
       	gradParameters:zero()
       	--print(report)
       	--for key,value in pairs(model) do
    	--	print("found member " .. key);
    	--	print("value")
    	--	print( value)
		--end
		--print(report)

		local mod = nn.SplitTable(1,2)
		local is = mod:forward(model.dpnn_input)

      	model:backward(is, df_do)
      	return f/#is,gradParameters/#is
	end
	--optim.rmsprop(feval, parameters, optimState)
	--print("learningRate",12)

	end,
	--model = , passed in by experiment
	epoch_callback = function (model, report)
	--print(1)
	--print("learningRate",13)
	end,
	sampler = dp.ShuffleSampler{batch_size = 2},
	feedback = nil,
	progress = true,
	stats = true
}


xp = dp.Experiment
{
	model = model,
	optimizer = train,
	target_module = nn.SplitTable(1,1):type('torch.IntTensor'),
	observer = 
	{
		dp.FileLogger()
	},
	random_seed = os.time(),
	max_epoch = 10	
}
xp:verbose(true)
xp:run(ds)