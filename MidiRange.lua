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
require 'TestLSTM'

dl = DatasetLoader('processed','audio','audio')

 dl:loadTraining()
 numTrain = dl:numberOfTrainingSamples() 

min = 100000000
max = -10000000

note = {}
while not done do
	data = dl:loadNextSet()
    
	done = data.done
	for song= 1, #data do
		vec = data[song].midi
		for i = 1, vec:size(1) do
			for j = 1,vec:size(2) do
				if vec[i][j] == 1 then
					note[j] = j
					if j < min then
						min = j
					end
					if j > max then
						max = j
					end
				end
			end
		end
	end
	print('data')

end

counter = 0

for i,v in pairs(note) do
	counter = counter + 1
end

print(counter)
print(min)
print(max)
print(max - min)