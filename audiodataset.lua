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
		self.audiofile = arg.audiofile
		self.midifile = arg.midifile
		if self.audiofile ~= nil then
			self.ext = paths.extname(self.audiofile)
			self.filename = paths.basename(self.audiofile,self.ext)
		end
		if self.midifile ~= nil then
			self.ext = paths.extname(self.midifile)
			self.filename = paths.basename(self.midifile,self.ext)
		end

	end
end


-- make this load the notes of the midi in 
-- save the target vector generation for later
function audiodataset:loadAudioMidi(filename,wavdirectory)
	status,notes = pcall(openMidi,self.midifile)
	if not status then
		print("THERE WERE NO NOTES IN THIS MIDI")
		return false
	end
	directory = paths.dirname(self.midifile)
	filebase = paths.basename(self.midifile,"mid")

	if self.audiofile == nil then
		generateWav(self.midifile,wavdirectory .. "/")
		self.audiofile = wavdirectory .. "/" .. filebase .. '.wav'
	end

	self.audio,self.midi,self.samplerate = generateMidiTargetVector(self.audiofile,notes)
	self.file = wavdirectory .. "/" .. filebase 
	self.audio = applyToTensor(self.audio:t()[1])
	return true
end

function audiodataset:loadMidiSpectrogram(filename,wavdirectory,windowSize,stride)
	status,notes = pcall(openMidi,self.midifile)
	if not status  then
		print("THERE WERE NO NOTES IN THIS MIDI")
		return false
	end
	directory = paths.dirname(self.midifile)
	filebase = paths.basename(self.midifile,"mid")
	
	if self.audiofile == nil then
		print(wavdirectory .. "/" .. filebase .. '.wav')
		okay = pcall(generateWav,self.midifile,wavdirectory .. "/")
		if not okay then
			return false
		end
		self.audiofile = wavdirectory .. "/" .. filebase .. '.wav'
	end

	-- Generate spectrogram and set the audio field
	if not self:loadIntoSpectrogram(windowSize,stride) then
		return false
	end
	okay,self.audio,self.midi,self.samplerate = pcall(generateMidiSpectrogramVector,self.audio,self.samplerate,notes)
	return okay
end


function audiodataset:loadIntoFFT()
	self.audio,self.samplerate = audio.load(self.audiofile)
	-- Compute FFT -- assuming 1D for now
	self.audio = signal.fft(self.data)
	collectgarbage()
end

function audiodataset:loadIntoSTFT(windowSize,stride)
	self.audio,self.samplerate = audio.load(self.audiofile)
	totaltime = self.audio:size(2) * samplerate
	self.audio = audio.stft(self.data,windowSize,'hann',stride):t()
	self.samplerate = totaltime / self.audio:size(1)
	collectgarbage()
end

function audiodataset:loadIntoSpectrogram(windowSize,stride)
	okay,self.audio,self.samplerate = pcall(audio.load,self.audiofile)
	if not okay then
		return false
	end
	totaltime = self.audio:size(1) * 1.0/self.samplerate
	self.numsamples = self.audio:size(1)
	print("TOTALTIME: " .. totaltime)
	self.audio = audio.spectrogram(self.audio,windowSize,'hann',stride)
	self.samplerate = self.audio:size(2) / totaltime--(windowSize / self.samplerate) * (stride/windowSize)--
	--self.samplerate = 1.0 / self.samplerate
	print ("SAMPLERATE: " .. self.samplerate .. "===========================================")

	
	--print(self.audio:size())
	collectgarbage()
	return true
end

function audiodataset:loadIntoRaw()
	self.audio,self.samplerate = audio.load(self.audiofile)
	--print(self.data:size())
	collectgarbage()
end

function audiodataset:loadIntoBinaryFormat()
	self.audio,self.samplerate = audio.load(self.audiofile)
	
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
	container["ext"] = self.ext
	container["audiofile"] = self.audiofile
	container["midifile"] = self.midifile
	container["class"] = self.class

	torch.save(paths.concat(directory,self.filename .. ".dat"),container)
end

function audiodataset:generateImage()
	print(self.audio:size())
	print("MAX: " .. self.audio:max())
	image.save(self.audiofile .. ".pgm", image.scale(image.minmax{tensor=self.audio},1000,1000))
	image.save(self.midifile .. "midi.pgm", image.scale(image.minmax{tensor=self.midi},1000,1000))
end

function audiodataset:deserialize(file)
  dict = torch.load(file)
  self.audio = dict.audio
  self.samplerate = dict.samplerate
  self.ext = dict.extract
  self.audiofile = dict.audiofile
  self.midifile = dict.midifile
  self.class = dict.class
  self.midi = dict.midi
end
