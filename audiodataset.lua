-- AudioDatatset class
-- Description: This will contain the data of audio
-- fields: 
-- midi
-- audio
-- sampleRate
-- types:
-- audio --
-- spectrogram --- 
-- encoded --- to be handled later
require 'dp'
require 'torchx'
require 'audio'
require 'midiToBinaryVector'
require 'image'

local signal = require 'signal'
local audiodataset = torch.class('audiodataset')

function applyToTensor(tensor)
    --print(tensor)
    local temp = torch.ones(tensor:size(1),32) 
    for i=1,tensor:size(1) do
        temp[i] = numberToTensor(tensor[i])
    end
    return temp:byte()
end


function numberToTensor(number)
    local tensor = torch.ones(32)
    for i=1,32 do
        tensor[i] = bit.rshift( bit.band( bit.lshift(1,i-1), number ), i-1 )
    end
    return tensor:byte()
end

-- make it so that args can be passed in... and handle it
-- figure out some things to make this better
function audiodataset:__init(arg)
	--print("DEFAULT CALLED")
	if type(arg) == "table" then
		-- take care of the args here
		if arg.file ~= nil and arg.classname ~= nil and (arg.type == "audio" or arg.type == "spectrogram") then
			self:setfile(arg.file,arg.classname)
		elseif arg.file ~= nil and arg.type == "midi" then
			--self:loadMidi(arg.file)
			self.file = arg.file
			self.ext = paths.extname(arg.file)
		end
	end
end

function audiodataset:setfile(file,classname)
	self.samplerate = -1
	self.file = file
	self.class = classname
	self.ext = paths.extname(file)
	self.filename = paths.basename(file,self.ext)
end

-- make this load the notes of the midi in 
-- save the target vector generation for later
function audiodataset:loadAudioMidi(filename,wavdirectory)
	if filename == nil then
		filename = self.file
		self.filename = paths.basename(self.file,self.ext)
	end
	print ("HERE")
	print(filename)
	notes = openMidi(filename)
	directory = paths.dirname(filename)
	filebase = paths.basename(filename,"mid")
	print(wavdirectory .. "/" .. filebase .. '.wav')
	generateWav(filename,wavdirectory .. "/")
	self.audio,self.midi,self.samplerate = generateMidiTargetVector(wavdirectory .. "/" .. filebase .. '.wav',notes)
	self.file = wavdirectory .. "/" .. filebase 
	self.audio = applyToTensor(self.audio:t()[1])
end

function audiodataset:loadMidiSpectrogram(filename,wavdirectory,windowSize,stride)
	if filename == nil then
		filename = self.file
		self.filename = paths.basename(self.file,self.ext)
	end
	print ("HERE")
	print(filename)
	notes = openMidi(filename)
	directory = paths.dirname(filename)
	filebase = paths.basename(filename,"mid")
	print(wavdirectory .. "/" .. filebase .. '.wav')
	generateWav(filename,wavdirectory .. "/")

	-- Generate spectrogram and set the audio field
	self.file = wavdirectory .. "/" .. filebase .. '.wav'

	self:loadIntoSpectrogram(windowSize,stride)
	self.audio,self.midi,self.samplerate = generateMidiSpectrogramVector(self.audio,self.samplerate,notes)
end


function audiodataset:loadIntoFFT()
	self.audio,self.samplerate = audio.load(self.file)
	-- Compute FFT -- assuming 1D for now
	self.audio = signal.fft(self.data)
	collectgarbage()
end

function audiodataset:loadIntoSTFT(windowSize,stride)
	self.audio,self.samplerate = audio.load(self.file)
	totaltime = self.audio:size(2) * samplerate
	self.audio = audio.stft(self.data,windowSize,'hann',stride):t()
	self.samplerate = totaltime / self.audio:size(1)
	collectgarbage()
end

function audiodataset:loadIntoSpectrogram(windowSize,stride)
	self.audio,self.samplerate = audio.load(self.file)
	totaltime = self.audio:size(1) * 1.0/self.samplerate
	self.numsamples = self.audio:size(1)
	self.audio = audio.spectrogram(self.audio,windowSize,'hann',stride)
	self.samplerate = self.audio:size(2) / totaltime

	--print(self.audio:size())
	collectgarbage()
end

function audiodataset:loadIntoRaw()
	self.audio,self.samplerate = audio.load(self.file)
	--print(self.data:size())
	collectgarbage()
end

function audiodataset:loadIntoBinaryFormat()
	self.audio,self.samplerate = audio.load(self.file)
	
	self.audio = applyToTensor(self.audio:t()[1])
	collectgarbage()
end

-- Time will be passed in as seconds
function audiodataset:getNearTime(time)
	if self.samplerate == -1 then
		return nil
	end
end

-- A function to serialize this object as a whole including its data
function audiodataset:serialize(directory)
	local container = {}
	container["samplerate"] = self.samplerate
	container["audio"] = self.audio
	container["midi"] = self.midi
	container["file"] = self.file
	container["ext"] = self.ext
	container["filename"] = self.filename
	container["class"] = self.class
	torch.save(paths.concat(directory,self.filename .. ".dat"),container)
end

function audiodataset:generateImage()
	print(self.audio:size())
	print("MAX: " .. self.audio:max())
	image.save(self.filename .. ".pgm", image.scale(image.minmax{tensor=self.audio},1000,1000))
	image.save(self.filename .. "midi.pgm", image.scale(image.minmax{tensor=self.midi},1000,1000))
end

function audiodataset:deserialize(file)
  dict = torch.load(file)
  self.audio = dict.audio
  self.samplerate = dict.samplerate
  self.file = dict.file
  self.ext = dict.extract
  self.filename = dict.filename
  self.class = dict.class
  self.midi = dict.midi
end
