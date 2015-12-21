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
		if arg.file ~= nil and arg.classname ~= nil and arg.type == "audio" then
			self:setfile(arg.file,arg.classname)
		elseif arg.file ~= nil and arg.type == "midi" then
			--self:loadMidi(arg.file)
			self.file = arg.file
			self.ext = paths.extname(arg.file)
		end
	end
end

function audiodataset:setfile(file,classname)
	--print("called")
	--print(file)
	--print(classname)
	self.samplerate = -1
	self.file = file
	self.class = classname
	self.ext = paths.extname(file)
	self.filename = paths.basename(file,self.ext)
end

-- make this load the notes of the midi in 
-- save the target vector generation for later
function audiodataset:loadMidi(filename,wavdirectory)
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
	self.data,self.binVector,self.samplerate = generateMidiTargetVector(wavdirectory .. "/" .. filebase .. '.wav',notes)
	self.file = wavdirectory .. "/" .. filebase .. '.wav'
	self.data = applyToTensor(self.data:t()[1])
end


--- here are two functions that need to be implemented
function audiodataset:generateMidiSpectrogrmTarget(windowSize,stride)

end

function audiodataset:GenerateMidiRawTarget()

end
------

function audiodataset:loadIntoFFT()
	self.data,self.samplerate = audio.load(self.file)
	-- Compute FFT -- assuming 1D for now
	self.data = signal.fft(self.data)
	collectgarbage()
end

function audiodataset:loadIntoSTFT(windowSize,stride)
	self.data,self.samplerate = audio.load(self.file)
	self.data = audio.stft(self.data,windowSize,'hann',stride)
	collectgarbage()
end

function audiodataset:loadIntoRaw()
	self.data,self.samplerate = audio.load(self.file)
	--print(self.data:size())
	collectgarbage()
end

function audiodataset:loadIntoBinaryFormat()
	self.data,self.samplerate = audio.load(self.file)
	
	self.data = applyToTensor(self.data:t()[1])
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
	--print ("FILENAME: " .. self.filename)
	--print(directory)
	local container = {}
	container["samplerate"] = self.samplerate
	container["data"] = self.data
	container["file"] = self.file
	container["ext"] = self.ext
	container["filename"] = self.filename
	container["class"] = self.class
	container["binVector"] = self.binVector
	torch.save(paths.concat(directory,self.filename .. ".dat"),container)
end

function audiodataset:generateImage()
	image.save(self.filename .. ".pgm", image.scale(self.binVector *50,1000,1000))
end

function audiodataset:deserialize(file)
	--print(file)
  dict = torch.load(file)
  self.data = dict.data
  self.samplerate = dict.samplerate
  self.file = dict.file
  self.ext = dict.extract
  self.filename = dict.filename
  self.class = dict.class
  self.binVector = dict.binVector

end