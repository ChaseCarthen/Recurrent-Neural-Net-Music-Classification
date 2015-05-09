require 'lfs'
require 'nn'
-- Get this guy from https://github.com/gummesson/file.lua
file = require 'file'

local midi = require 'MIDI'
mtbv = require "midiToBinaryVector"
-- step one load model from a file

-- step two test the model to see if it works

-- step three setup classification models

-- store genres destinations here..
function getFiles(BaseDir,format)

destinations = {}
destinations[1] = "test.txt"

-- Iterate through midi directory

-- A directory of unclassified midi
--local BaseDir = "./MIDI"
local testList = {}
local counter = 0
-- time to iterate through the directory
for directoryName in lfs.dir(BaseDir) 
    do 
    print(directoryName)
    combinedName = BaseDir .. "/" .. directoryName
    --print(combinedName)

    if directoryName ~= ".." and directoryName ~= "." and lfs.attributes(BaseDir.."/"..directoryName,"mode") == "directory"
    then

    for mid in lfs.dir(combinedName)
    do
    -- classify midi datasets with model
    print(mid)
    local genre = 1 
    if destinations[1] and string.find(mid, format) then 
    local content = combinedName .. "/" .. mid
    counter = counter + 1
    --if counter >  49870   then
    --print (content .."COUNTERRRRRRRRR!!!    " ..  counter)
    --local data = midiToBinaryVec(content)
    --end 
    --print(data)
    testList[counter] = content
    --print(content)
    --file.write(destinations[1],content,"a")
    -- add midi to file -- file.write(combinedName .. "/" .. mid,content,"a") -- append on file
    end
    end
	
	end
    
end
print(#testList)
return testList
end

function createTorchContainers(files)

for i=1,#files,1000 
do
local cont = {data={}, files={}}
local count = 0
print(i)
for j=i,i+1000
do

if j > #files then
break
end

--print(j)
local data = midiToBinaryVec(files[j])

if data ~= nil then

count = count + 1
cont.files[count] = files[j]
cont.data[count] = data

end

end

torch.save("midi" .. i .. ".dat", cont)

end

end

--print(createTorchContainers(getFiles('./MIDI','.mid')) )
--print(#getFiles("./MIDI"))

-- move files into the proper classification folder


-- retrain model?????????? -- get some boot strapping action going here..
