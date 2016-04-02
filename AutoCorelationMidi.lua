samplerate = 22050
windowsize = 2560
a = 440
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
require 'Model/model'
require 'writeMidi'
require 'Trainer'
require 'Model/AutoEncoder'
require 'Model/BinaryClassReward'
require 'AutoEncoderTrainer'
require 'Model/StackedAutoEncoder'
require 'audio'
require 'gnuplot'
require 'image'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A music transcription neural network for going from midi to wav.")
cmd:text()
cmd:text('Options')
cmd:option("-audiofile","","A wav file to processed.")
cmd:option("-autoencoderfile","auto.model","The autoencoder file to be used.")
cmd:option("-rnncfile","train.model","The translation model to be used.")
cmd:option("-windowSize",2560,"Setting the window size for the fft.")
cmd:option("-stride",1280,"Setting the stride for the window.")
cmd:option("-midifileout","test.mid","Use this parameter to specify the midi output.")
cmd:option("--corelate",true,"Use auto correlation.")
cmd:option("--model",false,"Use the lovely model.")
cmd:option("--autoencoder",true,"Use the autoencoder model.")
cmd:text()

params = cmd:parse(arg or {})

miditable = {}

for x = 0, 127 do
miditable[x] = (a / 32) * (2 ^ ((x - 9) / 12)) 
end

function frequencyToBin(notefreq,samplerate,windowsize)
	return torch.round(miditable[notefreq-1] / (samplerate/windowsize/2) )
end

if params.audiofile == "" then
	os.exit()
end

data,samplerate = audio.load(params.audiofile)


print(data:size())
spectrogram = audio.spectrogram(data,params.windowSize,'hann',params.stride)
spectrogram = spectrogram:t()

if params.model and params.autoencoder then
	print("USING Encoded MODEL")
	automodel = torch.load(params.autoencoderfile)
	trainmodel = torch.load(params.rnncfile)

	spectrogram = (spectrogram - spectrogram:mean()) / spectrogram:std()
	--print(spectrogram:size())
	--print(automodel.layer[1].encoder)


	songsplit = spectrogram:split(1000)

	join = nn.JoinTable(1)

	local output = {}
	for i=1,#songsplit do
		--print(i)
		local input = songsplit[i]:split(100)
		--print(input)

		if input[#input]:size(1) ~= 100 then
			input[#input] = torch.cat(input[#input], torch.zeros(100 - input[#input]:size(1), input[#input]:size(2) ), 1 )
		end

		local out = automodel:forward(1,input,true )
		out = trainmodel:forward(out)
		--print("-------------")
		--print(out)
		output[#output +1] = join:forward(out):clone()
		--print("-------------output")
		--print(output)
	end
	notes = join:forward(output)

-- end of model if
--print(output)

elseif params.model then
	print("USING MODEL")
	--automodel = torch.load(params.autoencoderfile)
	trainmodel = torch.load(params.rnncfile)

	spectrogram = (spectrogram - spectrogram:mean()) / spectrogram:std()
	--print(spectrogram:size())
	--print(automodel.layer[1].encoder)


	songsplit = spectrogram:split(1000)

	join = nn.JoinTable(1)

	local output = {}
	for i=1,#songsplit do
		--print(i)
		local input = songsplit[i]:split(100)
		--print(input)

		if input[#input]:size(1) ~= 100 then
			input[#input] = torch.cat(input[#input], torch.zeros(100 - input[#input]:size(1), input[#input]:size(2) ), 1 )
		end

		out = trainmodel:forward(input)
		--print("-------------")
		--print(out)
		output[#output +1] = join:forward(out):clone()
		--print("-------------output")
		--print(output)
	end
	notes = join:forward(output)

-- end of model if

elseif params.corelate then
	output = image.minmax{tensor=(spectrogram - spectrogram:mean())/spectrogram:std()} --join:forward(output)}
	output = image.minmax{tensor=output - output*1.0/output:mean()}
	output = image.minmax{tensor=output - output*1.0/output:mean()}
	--output = image.minmax{tensor=output - output*1.0/output:mean()}
	--output = image.minmax{tensor=(spectrogram - spectrogram:mean())/spectrogram:std()}
	--output = image.minmax{tensor=spectrogram - (spectrogram*1.0/spectrogram:std()) }--spectrogram:mean())/spectrogram:std()}

	print("MAX: " .. output:max())
	print("MEAN: " .. output:mean())
	print("STD: " .. output:std())
	origoutput = output:clone()
	--output = output:le(.18)
	output = output:ge(.93)
    print("SAVING IMAGE")
    image.save('out.png',image.minmax{tensor=output})
	print("=======")
	print("MAX: " .. output:max())
	--print("MEAN: " .. output:mean())
	--print("STD: " .. output:std())
	--output = output:le(output:std() /2)

	notes = torch.zeros(spectrogram:size(1),128)


		for notefreq = 0,127 do
			bin =  spectrogram:size(2) - math.floor(miditable[notefreq] / (samplerate/windowsize)) --spectrogram:size(2) -
			--print("INFO==============")
			--print(notefreq)
			--print(miditable[notefreq])
			--print(bin)
			--print(spectrogram:size())
		end


	--print(samplerate)
	print(spectrogram:size())
    print(notes:size())
	for column= 1,spectrogram:size(1) do
		for notefreq = 0,118 do
			bin =  spectrogram:size(2) - math.floor(miditable[notefreq] / (samplerate/windowsize)) --spectrogram:size(2) -
			notes[column][notefreq+1] = output[column][bin]
		end
	end

end --end correlate if

samplerate = samplerate/params.stride

image.save('outmidi.png',image.minmax{tensor=notes})
writeMidi(params.midifileout,notes:round(),samplerate ,samplerate )
