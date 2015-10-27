require 'audiodataset'

-- Function utils
function loadListData(type,list,start,limit)
	for i=start+1,math.min(start+limit,#list) do
			if type == "audio" then 
				-- Load audio
				--print(list[i])
				temp = audiodataset()
				temp:deserialize(list[i])
				print(temp.data:size())
				print(temp.class)
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
	file = torch.DiskFile(paths.concat(datadir,"class.txt"))
	file:quiet()
	str = " "
	self.classes = {}
	counter = 0
	while str ~= "" do
		counter = counter + 1
		str = file:readString("*l")
		print(str)
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
	self.limit = 100
end

function DatasetLoader:loadTraining(numToLoad)
	self.counter = 1
	self.limit = 100
	self.mode = "train"
end

function DatasetLoader:loadTesting(numToLoad)
	self.counter = 1
	self.limit = 100
	self.mode = "test"
end

function DatasetLoader:loadValidation(numToLoad)
	self.counter = 1
	self.limit = 100
	self.mode = "validation"
end

function DatasetLoader:loadNextSet()
	if self.mode == "train" then
		self.counter = loadListData(self.type,self.train,self.counter,self.limit)
		return self.counter == self.traincount

	elseif self.mode == "test" then
		self.counter = loadListData(self.type,self.test,self.counter,self.limit)
		return self.counter == self.testcount

	elseif self.mode == "validation" then
		self.counter = loadListData(self.type,self.valid,self.counter,self.limit)
		return self.counter == self.validcount

	end
	return true -- This means everything is done.
end




