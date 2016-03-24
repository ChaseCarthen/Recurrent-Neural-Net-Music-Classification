require 'Model/AutoEncoder'
require 'Model/RNNC'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'image'
require 'audio'
require 'gnuplot'
require 'Model/StackedAutoEncoder'
require 'writeMidi'
require 'audio'
require 'DatasetLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A tool to determine what notes are not in the test and validation sets given the training set.")
cmd:text()
cmd:text('Options')
cmd:option("-data","processed","Specify the directory with data.")
cmd:text()

params = cmd:parse(arg or {})

function noteOverlap()
	local set = {}
	local done = false
	while not done do
  	local data = dl:loadNextSet()
  	done = data

  		for song= 1, #data do
  			tmidi = data[song].midi:t()
  			for note=1,128 do
  				result = tmidi[note]:sum()
  				if result > 0 then
  					set[note] = 1
  				end
  			end
  		end
	end
	return set 
end

dl = DatasetLoader(params.data,'audio','midi')

dl:loadTraining()
numTrain = dl:numberOfTrainingSamples()
print(numTrain)
trainset = noteOverlap()

dl:loadValidation()
numTrain = dl:numberOfValidSamples()
validset = noteOverlap()

dl:loadTesting()
numTrain = dl:numberOfTestSamples()
testset = noteOverlap()

print("trainset")
print(trainset)
print("testset")
print(testset)
print("validset")
print(validset)

print("NON-Overlap")
-- remove items that are in trainset and testset 
for k,v in pairs(testset) do
	if trainset[k] then
		testset[k] = nil
	end
end

for k,v in pairs(validset) do
	if trainset[k] then
		validset[k] = nil
	end
end

if trainset[1000] then
print("NIL TEST FAILED")
end
print("testset" )
print( testset)
print("validset" )
print( validset)

