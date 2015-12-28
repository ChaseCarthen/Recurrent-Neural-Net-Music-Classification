require 'audiodataset'
-- DatasetLoader
-- This will only load audio datasets


-- Function utils
function loadListData(type,list,start,limit,shufflelist)
	local outdata = {}
	local counter = 1
	for i=start+1,math.min(start+limit,#list) do
			if type == "audio" then 
				-- Load audio
				--print(list[i])
				temp = audiodataset()
				temp:deserialize(list[shufflelist[i]])
				outdata[counter] = temp
				counter = counter + 1
				--print(temp.data:size())
				--print(temp.class)
			end
	end

	outdata["counter"] = math.min(start+limit,#list) 
	return outdata
end

local DatasetLoader = torch.class('DatasetLoader')

function DatasetLoader:__init(datadir,format,type)
	self.datadir = datadir
	self.train = {}
	self.valid = {}
	self.test = {}
	self.shufflelist = {}
	self.traincount = 0
	self.testcount = 0
	self.validcount = 0
	self.type = 'audio'--type
	file = torch.DiskFile(paths.concat(datadir,"class.txt"))
	file:quiet()
	str = " "
	self.classes = {}
	counter = 0
	while str ~= "" do
		counter = counter + 1
		str = file:readString("*l")

		if str ~= '' then
			self.classes[counter] = str
		end
	end
	for i in paths.iterfiles(paths.concat(datadir,"train")) do
		self.traincount = self.traincount + 1
		self.train[self.traincount] = paths.concat(datadir,"train",i)
	end

	for i in paths.iterfiles(paths.concat(datadir,"test")) do
		self.validcount = self.validcount + 1
		self.valid[self.validcount] = paths.concat(datadir,"test",i)
	end

	for i in paths.iterfiles(paths.concat(datadir,"valid")) do
		self.testcount = self.testcount + 1
		self.test[self.testcount] = paths.concat(datadir,"valid",i)
	end
	
	self.loadlist = {}
	self.counter = 0
	self.mode = "none"
	self.limit = 10
end

function DatasetLoader:loadTraining(numToLoad)
	self.counter = 0
	self.limit = 10
	self.mode = "train"
	self.shufflelist = torch.randperm(self.traincount)
end

function DatasetLoader:loadTesting(numToLoad)
	self.counter = 0
	self.limit = 10
	self.mode = "test"
	self.shufflelist = torch.randperm(self.testcount)
end

function DatasetLoader:loadValidation(numToLoad)
	self.counter = 0
	self.limit = 10
	self.mode = "validation"
	self.shufflelist = torch.randperm(self.validcount)
end

function DatasetLoader:numberOfTrainingSamples()
	return self.traincount
end

function DatasetLoader:numberOfTestSamples()
	return self.testcount
end

function DatasetLoader:numberOfValidSamples()
	return self.validcount
end

function DatasetLoader:loadNextSet()
	if self.mode == "train" then
		local out = loadListData(self.type,self.train,self.counter,self.limit,self.shufflelist)
		self.counter = out.counter
		out["done"] = self.traincount == out.counter
		return out

	elseif self.mode == "test" then
		local out = loadListData(self.type,self.test,self.counter,self.limit,self.shufflelist)
		self.counter = out.counter
		out["done"] = self.testcount == out.counter
		return out

	elseif self.mode == "validation" then
		local out = loadListData(self.type,self.valid,self.counter,self.limit,self.shufflelist)
		self.counter = out.counter
		out["done"] = self.validcount == out.counter
		return out

	end
	return true -- This means everything is done.
end




