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
require 'model'
require 'writeMidi'
require 'Trainer'
require 'Model/AutoEncoder'
require 'Model/BinaryClassReward'
require 'AutoEncoderTrainer'
require 'Model/StackedAutoEncoder'
require 'audio'


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
cmd:text()

params = cmd:parse(arg or {})

if params.audiofile == ""
	os.exit()
end

song = audio.load(params.audiofile)
automodel = torch.load(params.autoencoderfile)
translatemodel = torch.load(params.rnncfile)

spectrogram,samplerate = audio.spectrogram(song,params.windowSize,'hann',params.stride)

spectrogram = spectrogram:t()

songsplit = spectrogram:split(1000)

output = {}
for i=1,#songsplit do
	input = songsplit[i]:split(100)
	out = automodel:forward(1,input,false )
	for j =1,#out do
		out[j] = input[i][j] - out[j]
	end
	output[#output + 1] = translatemodel:forward(out)
end

join = nn.JoinTable()

output = join:forward(output)

samplerate = (1.0/samplerate * song:size(1)) / spectrogram:size(1)


writeMidi('test.midi',out:round(),1.0/samplerate*1000 ,1.0/samplerate*1000 )

