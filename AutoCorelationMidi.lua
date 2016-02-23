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
cmd:option("-rnncfile","final.model","The translation model to be used.")
cmd:option("-windowSize",2560,"Setting the window size for the fft.")
cmd:option("-stride",1280,"Setting the stride for the window.")
cmd:option("-midifileout","test.mid","Use this parameter to specify the midi output.")
cmd:text()

params = cmd:parse(arg or {})

miditable = {}

for x = 0, 127 do
miditable[x] = (a / 32) * (2 ^ ((x - 9) / 12)) 
end

function frequencyToBin(notefreq,samplerate,windowsize)
	return torch.round(miditable[notefreq-1] / (samplerate/windowsize)/2 )
end

if params.audiofile == "" then
	os.exit()
end

data,samplerate = audio.load(params.audiofile)
automodel = torch.load(params.autoencoderfile)


spectrogram = audio.spectrogram(data,params.windowSize,'hann',params.stride)



spectrogram = image.minmax{tensor=spectrogram:t()}
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

	local out = automodel:forward(1,input,false )
	for j =1,#out do
		out[j] = (input[j] - out[j]):clone()
	end
	--print("-------------")
	--print(out)
	output[#output +1] = join:forward(out):clone()
	--print("-------------output")
	--print(output)
end

--print(output)



output = join:forward(output)


output = output:le(0.08)


notes = torch.zeros(spectrogram:size(1),128)


--print(samplerate)
--print(spectrogram:size())
for column= 1,spectrogram:size(1) do
	for notefreq = 0,127 do
		bin = spectrogram:size(2) - math.floor(miditable[notefreq] / (samplerate/windowsize) )
		--print(bin)
		notes[column][notefreq+1] = 1-output[column][bin]
	end
end


image.save('testmidi.png',image.minmax{tensor=notes:round()})


samplerate = (1.0/samplerate * data:size(1)) / spectrogram:size(1)


writeMidi(params.midifileout,notes:round(),1.0/samplerate*8 ,1.0/samplerate*8 )
