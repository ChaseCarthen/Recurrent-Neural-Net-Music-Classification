require 'dp'
require 'torchx'
require 'audio'
local signal = require 'signal'
local audiodataset = torch.class('audiodataset')

function applyToTensor(tensor)
    --print(tensor)
    local temp = torch.ones(tensor:size(1),32) 
    for i=1,tensor:size(1) do
        --print(tensor[i])
        temp[i] = numberToTensor(tensor[i])
    end
    return temp
end


function numberToTensor(number)
    local tensor = torch.ones(32)
    for i=1,32 do
        tensor[i] = bit.rshift( bit.band( bit.lshift(1,i-1), number ), i-1 )
    end
    return tensor
end


function audiodataset:__init(file,clas) 
	self.samplerate = -1
	self.file = file
	self.class = class
end

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
	collectgarbage()
end

function audiodataset:loadIntoBinaryFormat()
	self.data,self.samplerate = audio.load(self.file)
	self.data = applyToTensor(self.data):byte()
	collectgarbage()
end

-- Time will be passed in as seconds
function audiodataset:getNearTime(time)
	if self.samplerate == -1 then
		return nil
	end
end