require 'dp'
require 'rnn'
require 'optim'
model = nn.Sequential()
model:add(nn.Linear(1,1))

local trainInput = dp.SequenceView('bw',torch.randn(1,1))
local targetInput = dp.ClassView('b',torch.ones(1))


local dataset = dp.DataSet{inputs=trainInput,targets=targetInput,which_set='train'}
local ds = dp.DataSource{train_set=dataset}
ds:classes{'one'}
-- Lets set up a optimizer
train = dp.Optimizer
{
	acc_update=false,
	loss = nn.ClassNLLCriterion(),
	callback = function (model,report)
	print("learningRate",12)
	end,
	--model = , passed in by experiment
	epoch_callback = function (model, report)
	print("learningRate",13)
	end,
	sampler = dp.ShuffleSampler{batch_size = 1},
	feedback = nil,
	progress = true,
	stats = true
}


xp = dp.Experiment
{
	model = model,
	optimizer = train,
	observer = 
	{
		dp.FileLogger()
	},
	random_seed = os.time(),
	max_epoch = 10	
}
xp:verbose(true)
xp:run(ds)