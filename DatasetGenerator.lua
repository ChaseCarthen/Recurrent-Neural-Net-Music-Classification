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

NumberGenres = 0


--Gather the midi files from the music directory. The SongGroupContainer is neccessary since we want to split
--our data into training and testing for each genre. This was re eliminate the possibility of having too 
--few of training midis compared to testing or vica-versa

function GatherMidiData(BaseDir) 
    print("Gathering Midi Data") 
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
                        obj.Songs[fileCounter] = data
                    end
                end
            end
            SongGroupContainer[directoryName] = obj
	    NumberGenres = NumberGenres + 1
        end
    end
    print("Finished gathering Midi Data") 
    return SongGroupContainer
end




function SplitMidiData(data, ratio)
    print("Splitting Midi Data") 
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
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[shuffle[i]]:transpose(1,2):clone()


	  trainData.Labels[TrainingCounter] = torch.Tensor(NumberGenres):zero()

          trainData.Labels[TrainingCounter][classifier[genreKey]] = 1
	  --print(trainData.Labels[TrainingCounter])
        end
        
        for i=numTrain+1,numTrain+numTest do
            TestingCounter = TestingCounter + 1
            testData.Songs[TestingCounter] = data[genreKey].Songs[shuffle[i]]:transpose(1,2):clone()
            --testData.Labels[TestingCounter] = classifier[genreKey]
	    testData.Labels[TestingCounter] = torch.Tensor(NumberGenres):zero()
            testData.Labels[TestingCounter][classifier[genreKey]] = 1
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
	shuffledTrainData.Labels[i] = trainData.Labels[shuffle[i]]
    end

    local shuffle = torch.randperm(TestingCounter)
    for i=1, TestingCounter do
	shuffledTestData.Songs[i] = testData.Songs[shuffle[i]]
	shuffledTestData.Labels[i] = testData.Labels[shuffle[i]]
	--print(shuffledTestData.Labels[i])
    end
    print("Finished Splitting Midi Data") 
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










