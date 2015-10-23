List = {}
function List.new ()
  return {first = 0, last = -1}
end

function List.pushleft (list, value)
  local first = list.first - 1
  list.first = first
  list[first] = value
end

function List.pushright (list, value)
  local last = list.last + 1
  list.last = last
  list[last] = value
end

function List.popleft (list)
  local first = list.first
  if first > list.last then error("list is empty") end
  local value = list[first]
  list[first] = nil        -- to allow garbage collection
  list.first = first + 1
  return value
end

function List.popright (list)
  local last = list.last
  if list.first > last then error("list is empty") end
  local value = list[last]
  list[last] = nil         -- to allow garbage collection
  list.last = last - 1
  return value
end


local DatasetLoader = torch.class('DatasetLoader')

function DatasetLoader:__init(datadir,format)
	self.datadir = datadir
	self.train = {}
	self.valid = {}
	self.test = {}
	self.traincount = 0
	self.testcount = 0
	self.validcount = 0
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
		self.counter = self.limit
		return false
	elseif self.mode == "test" then
		self.counter = self.limit
		return false
	elseif self.mode == "validation" then
		self.counter = self.limit
		return false
	end
	return true -- This means everything is done.
end

