require 'audiodataset'

-- Function utils
function loadListData(type,list,start,limit)
	for i=start,math.min(start+limit,#list) do
			if type == "audio" then 
				-- Load audio
				print ("Loading " .. list[i])
			end
	end

	return math.min(start+limit,#list)
end

local DatasetLoader = torch.class('DatasetLoader')

function DatasetLoader:__init(datadir,format,type)
	self.datadir = datadir
	self.train = {}
	self.valid = {}
	self.test = {}
	self.traincount = 0
	self.testcount = 0
	self.validcount = 0
	self.type = 'audio'--type
	for i in paths.iterfiles(paths.concat(datadir,"train")) do
		self.traincount = self.traincount + 1
		self.train[self.traincount] = i
	end

	for i in paths.iterfiles(paths.concat(datadir,"test")) do
		self.validcount = self.validcount + 1
		self.valid[self.validcount] = i
	end

	for i in paths.iterfiles(paths.concat(datadir,"valid")) do
		self.testcount = self.testcount + 1
		self.test[self.testcount] = i
	end
	
	self.loadlist = {}
	self.counter = 0
	self.mode = "none"
	self.limit = 100
end

function DatasetLoader:loadTraining(numToLoad)
	self.counter = 0
	self.limit = 100
	self.mode = "train"
end

function DatasetLoader:loadTesting(numToLoad)
	self.counter = 0
	self.limit = 100
	self.mode = "test"
end

function DatasetLoader:loadValidation(numToLoad)
	self.counter = 0
	self.limit = 100
	self.mode = "validation"
end

function DatasetLoader:loadNextSet()
	if self.mode == "train" then
		self.counter = loadListData(self.type,train,self.counter,self.limit)
		return self.counter == self.traincount

	elseif self.mode == "test" then
		self.counter = loadListData(self.type,test,self.counter,self.limit)
		return self.counter == self.testcount

	elseif self.mode == "validation" then
		self.counter = loadListData(self.type,valid,self.counter,self.limit)
		return self.counter == self.validcount

	end
	return true -- This means everything is done.
end




