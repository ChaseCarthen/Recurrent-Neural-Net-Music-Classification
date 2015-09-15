local torch = require 'torch'
local midi = require 'MIDI'
require 'audio'
mtbv = require "midiToBinaryVector"
require 'image'
require 'lfs'
signal = require 'signal'


function firstToUpper(str)
    return str:gsub("^%l", string.upper)
end

--Create a table to associate genre names with numerical values.

classifier = {}
classes = {}




--Gather the midi files from the music directory. The SongGroupContainer is neccessary since we want to split
--our data into training and testing for each genre. This was re eliminate the possibility of having too 
--few of training midis compared to testing or vica-versa

function GatherAudioData(BaseDir,Container) 
    --Look to see if we have already saved the data.
    if Container == nil then
        SongData_file = 'SongData.t7'
    else
        SongData_file = Container
    end
    if paths.filep(SongData_file) then
    	loaded = torch.load(SongData_file)
     classes = loaded.classes
     classifier = loaded.classifier
     SongGroupContainer = loaded.SongGroupContainer
     return SongGroupContainer
 end


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
    classes[directoryCounter] = firstToUpper(directoryName)
    classifier[directoryName] = directoryCounter
    fileCounter = 0
    for filename in lfs.dir(BaseDir.."/"..directoryName) 
        do
        FullFilePath = BaseDir.."/"..directoryName.."/"..filename
        if string.find(filename, "%.mid") then 
            data = false
            data = midiToBinaryVec(FullFilePath)

            if "userdata" == type(data) and data:size(1) == 2 then
                fileCounter = fileCounter + 1 
                obj.Songs[fileCounter] = data
            end
            
            elseif string.find(filename, "%.au") then
                print("Loading " .. filename)
                fileCounter = fileCounter + 1
                data = audio.load(FullFilePath)
                if type(data) == "userdata" and data:size()[1] == 1 then
                    data = signal.stft(data[1], 8092,4096)--image.scale(,1000,128)
                end
                obj.Songs[fileCounter] = data
            end
            collectgarbage()

        end
        SongGroupContainer[directoryName] = obj
    end
end

SaveData = 
{
	SongGroupContainer = SongGroupContainer,
	classes = classes,
	classifier = classifier
	
}

torch.save(SongData_file, SaveData)

return SongGroupContainer
end




function SplitAudioData(data, ratio)
    local trainData = {Labels={}, Songs={}, GenreSizes={}}
    local testData = {Labels={}, Songs={}, GenreSizes={}}
    trainData.size = function() return #trainData.Songs end
    testData.size = function() return #testData.Songs end    


    TrainingCounter = 0
    TestingCounter = 0

    for genreKey,value in pairs(data) do 
        local shuffle = torch.randperm(#data[genreKey].Songs)
        local numTrain = math.floor(shuffle:size(1) * ratio)
        local numTest = shuffle:size(1) - numTrain

        trainData.GenreSizes[classifier[genreKey]] = numTrain
        testData.GenreSizes[classifier[genreKey]] = numTest           

        for i=1,numTrain do
          TrainingCounter = TrainingCounter + 1
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
          trainData.Labels[TrainingCounter] = classifier[genreKey]
      end

      for i=numTrain+1,numTrain+numTest do
        TestingCounter = TestingCounter + 1
            testData.Songs[TestingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
            testData.Labels[TestingCounter] = classifier[genreKey]
        end
        


    end    


    return trainData, testData, classes
end


function SplitAudioData2(data, ratio,ratio2) 
    if paths.filep('./test.t7') and paths.filep('./train.t7') and paths.filep('validation.t7') and paths.filep('./classes.t7') then
    	testData = torch.load('./test.t7')
        trainData = torch.load('./train.t7')
        validationData = torch.load('./validation.t7')
        classes = torch.load('classes.t7')
        return trainData, testData, validationData, classes
    end

    local trainData = {Labels={}, Songs={}, GenreSizes={}}
    local testData = {Labels={}, Songs={}, GenreSizes={}}
    local validationData = {Labels={},Songs={},GenreSizes={}}

    trainData.size = function() return #trainData.Songs end
    testData.size = function() return #testData.Songs end    
    validationData.size = function() return #validationData.Songs end

    TrainingCounter = 0
    TestingCounter = 0
    ValidationCounter = 0

    for genreKey,value in pairs(data) do 
        local shuffle = torch.randperm(#data[genreKey].Songs)
        local numTrain = math.floor(shuffle:size(1) * ratio)
        local numTest = math.floor(shuffle:size(1) * ratio2)
        local numValidation = shuffle:size(1) - numTrain - numTest

        trainData.GenreSizes[classifier[genreKey]] = numTrain
        testData.GenreSizes[classifier[genreKey]] = numTest           
        validationData.GenreSizes[classifier[genreKey]] = numValdation

        for i=1,numTrain do
          TrainingCounter = TrainingCounter + 1
          trainData.Songs[TrainingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
          trainData.Labels[TrainingCounter] = classifier[genreKey]
      end

      for i=numTrain+1,numTrain+numTest do
        TestingCounter = TestingCounter + 1
            testData.Songs[TestingCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
            testData.Labels[TestingCounter] = classifier[genreKey]
        end

        for i=numTrain+numTest+1,numTrain+numTest+numValidation do
            ValidationCounter = ValidationCounter + 1
            validationData.Songs[ValidationCounter] = data[genreKey].Songs[shuffle[i]]--:transpose(1,2):clone()
            validationData.Labels[ValidationCounter] = classifier[genreKey]
        end
        


    end    

   torch.save('./test.t7',testData)
   torch.save('./train.t7',trainData)
   torch.save('./validation.t7',validationData)
   torch.save('./classes.t7',classes)

    return trainData, testData, validationData, classes
end




function GetTrainAndTestData(arg)
    if paths.filep('./test.t7') and paths.filep('./train.t7') and paths.filep('validation.t7') and paths.filep('./classes.t7') then
        testData = torch.load('./test.t7')
        trainData = torch.load('./train.t7')
        validationData = torch.load('./validation.t7')
        classes = torch.load('classes.t7')
        return trainData, testData, validationData, classes
    end
    if arg == nil or arg.BaseDir == nil then
        return nil
    end    
    Data = GatherAudioData(arg.BaseDir)
    if arg.Ratio ~= nil and arg.Ratio2 == nil then
        return SplitAudioData(Data, arg.Ratio)
    else
        return SplitAudioData2(Data, arg.Ratio, arg.Ratio2)
    end
end

--EXAMPLE USAGE
--trainData, testData = GetTrainAndTestData("./music", .5)
--print(trainData)
--print (testData)
--print (classes)











