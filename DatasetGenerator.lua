local torch = require 'torch'
local midi = require 'MIDI'
require 'audio'
mtbv = require "midiToBinaryVector"
require 'lfs'
require 'cunn'


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
	    classes[directoryCounter] = firstToUpper(directoryName)
            classifier[directoryName] = directoryCounter
	    --classes[directoryCounter] = directoryCounter


	    --print(directoryName)
            --print(classifier[directoryName])
            
            fileCounter = 0
            for filename in lfs.dir(BaseDir.."/"..directoryName) 
            do FullFilePath = BaseDir.."/"..directoryName.."/"..filename
                if string.find(filename, ".mid")
                then 
                    
                    data = midiToBinaryVec(FullFilePath) 
                    if data ~= nil then
                        fileCounter = fileCounter + 1 
                        obj.Songs[fileCounter] = data:transpose(1,2):clone():cuda()
                       --print("DATA: ")
                        --print(data)
                        --print(data:size())
                    end
                elseif string.find(filename, ".au")
                then

                    data = audio.load(FullFilePath):t()[1]:float():cuda()
                    fileCounter = fileCounter + 1
                    --print(data:size())
                    --print(torch.ones(10):cuda())
                    obj.Songs[fileCounter] = data
                    print("HERE")
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


    TrainingCounter = 0
    TestingCounter = 0
    for genreKey,value in pairs(data) do 
        local shuffle = torch.randperm(#data[genreKey].Songs)
        local numTrain = math.floor(shuffle:size(1) * ratio)
        local numTest = shuffle:size(1) - numTrain
            
        for i=1,numTrain do
          TrainingCounter = TrainingCounter + 1
          --print(#data[genreKey].Songs)
          --print(i)
          --print(genreKey)
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
          trainData.Labels[TrainingCounter] = classifier[genreKey]
        end
        
        for i=numTrain+1,numTrain+numTest do
            TestingCounter = TestingCounter + 1
            testData.Songs[TestingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
            testData.Labels[TestingCounter] = classifier[genreKey]

        end
    end    
    




    local shuffledTrainData = {Labels={}, Songs={}}
    local shuffledTestData = {Labels={}, Songs={}}
    shuffledTrainData.size = function() return #shuffledTrainData.Songs end
    shuffledTestData.size = function() return #shuffledTestData.Songs end    
    --Shuffle all of the data around
    local shuffle = torch.randperm(TrainingCounter)
    
    for i=1, TrainingCounter do
	shuffledTrainData.Songs[i] = trainData.Songs[shuffle[i]]
    --print(shuffledTrainData.Songs[i])
    --shuffledTrainData.Songs[i] = (trainData.Songs[shuffle[i]] - trainData.Songs[shuffle[i]]:mean())/(trainData.Songs[shuffle[i]]:std())
	shuffledTrainData.Labels[i] = trainData.Labels[shuffle[i]]
    end

    local shuffle = torch.randperm(TestingCounter)
    for i=1, TestingCounter do
	shuffledTestData.Songs[i] = testData.Songs[shuffle[i]]
    --shuffledTestData.Songs[i] = (testData.Songs[shuffle[i]] - testData.Songs[shuffle[i]]:mean())/(testData.Songs[shuffle[i]]:std())
	shuffledTestData.Labels[i] = testData.Labels[shuffle[i]]
    end

    return shuffledTrainData, shuffledTestData, classes
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










