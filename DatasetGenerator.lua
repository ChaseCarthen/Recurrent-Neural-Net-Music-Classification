local torch = require 'torch'
local midi = require 'MIDI'
mtbv = require "midiToBinaryVector"
require 'lfs'



function firstToUpper(str)
    return str:gsub("^%l", string.upper)
end

--Create a table to associate genre names with numerical values.

classifier = {}
classes = {}




--Gather the midi files from the music directory. The SongGroupContainer is neccessary since we want to split
--our data into training and testing for each genre. This was re eliminate the possibility of having too 
--few of training midis compared to testing or vica-versa

function GatherMidiData(BaseDir) 
    local SongGroupContainer = {}
    directoryCounter = 0;
    for directoryName in lfs.dir(BaseDir) 
    do 
        if directoryName ~= ".." and directoryName ~= "." and lfs.attributes(BaseDir.."/"..directoryName,"mode") == "directory"
        then
            directoryCounter = directoryCounter + 1
            directoryPath = BaseDir.."/"..directoryName.."/"
            
            local obj = 
            {
                Genre = directoryName,
                Songs = {}
            }

            --classifier[directoryName] = firstToUpper(directoryName)
	    --classes[directoryCounter] = firstToUpper(directoryName)
            classifier[directoryName] = directoryCounter
	    classes[directoryCounter] = directoryCounter


	    --print(directoryName)
            --print(classifier[directoryName])
            
            fileCounter = 0
            for filename in lfs.dir(BaseDir.."/"..directoryName) 
            do FullFilePath = BaseDir.."/"..directoryName.."/"..filename
                if string.find(filename, ".mid")
                then 
                    fileCounter = fileCounter + 1 
                    data = midiToBinaryVec(FullFilePath) 
                    if data ~= nil then
                        obj.Songs[fileCounter] = data
                    end
                end
            end
            SongGroupContainer[directoryName] = obj
            --SerializeData(directoryPath..outputFileName, obj)
        end
    end
    return SongGroupContainer
end




function SplitMidiData(data, ratio)
    local trainData = {Labels={}, Songs={}}
    local testData = {Labels={}, Songs={}}
    trainData.size = function() return #trainData.Songs end
    testData.size = function() return #testData.Songs end    


    TrainingCounter = 1
    TestingCounter = 1
    for genreKey,value in pairs(data) do 
        local shuffle = torch.randperm(#data[genreKey].Songs)
        local numTrain = math.floor(shuffle:size(1) * ratio)
        local numTest = shuffle:size(1) - numTrain
            
        for i=1,numTrain do
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[shuffle[i]]:transpose(1,2):clone()
          --print("Hello bob")
          --print(Classifier[genreKey])
          trainData.Labels[TrainingCounter] = classifier[genreKey]
          TrainingCounter = TrainingCounter + 1
        end
        
        for i=numTrain+1,numTrain+numTest do
            testData.Songs[TestingCounter] = data[genreKey].Songs[shuffle[i]]:transpose(1,2):clone()
            testData.Labels[TestingCounter] = classifier[genreKey]
            TestingCounter = TestingCounter + 1
        end
    end    
    
    return trainData, testData, classes
end




function GetTrainAndTestData(BaseDir, Ratio)    
    Data = GatherMidiData(BaseDir)
    return SplitMidiData(Data, Ratio)
end


--EXAMPLE USAGE
--trainData, testData = GetTrainAndTestData("./music", .5)
--print(trainData)
--print (testData)
--print (classes)










