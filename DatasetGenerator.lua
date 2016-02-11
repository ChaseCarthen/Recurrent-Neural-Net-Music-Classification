<<<<<<< HEAD
require 'audiodataset'
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A dataset convertor to torch object.")
cmd:text()
cmd:text('Options')
cmd:option("--spectrogram",false,"Processing for spectrogram")
cmd:option('-m',"","Process midi data")
cmd:option('-d',"audio","Data directory to process.")
cmd:option('-o',"processed","Processed data directory.")
cmd:option("-r",.8,"Train Split Rate")
cmd:option("-r2",.1,"Validation Split Rate")
cmd:option("-r3",.1,"Test Split Rate")
cmd:option("-windowSize",8092,"Window size for spectrogram.")
cmd:option("-stride",512,"spectrogram stride size")
cmd:text()

params = cmd:parse(arg or {})
directory = params.d
print (params.o)
trainsplit = params.r
validsplit = params.r2
testsplit = params.r3
classlist = {}
classindex = 1
if directory ~= nil then
  -- Search through directory for files
  audiolist = {}
  paths.mkdir(paths.concat(paths.cwd(),params.o))
  file = torch.DiskFile(paths.concat(paths.cwd(),params.o,"class.txt"),'w')
  counter = 0
  for dir in paths.iterdirs(directory) do
    classlist[classindex] = directory
    classindex = classindex + 1
    print(dir)
    file:writeString(dir .. "\n")
    
    outpath = paths.concat(paths.cwd(),params.o,dir)
    trainpath = paths.concat(paths.cwd(),params.o,"train")
    testpath = paths.concat(paths.cwd(),params.o,"test")
    validpath = paths.concat(paths.cwd(),params.o,"valid")
    wavpath = paths.concat(paths.cwd(),params.o,"wav")

    
    --paths.mkdir(outpath)
    paths.mkdir(trainpath)
    paths.mkdir(validpath)
    paths.mkdir(testpath)
    paths.mkdir(wavpath)
    
    
    for file in paths.iterfiles(paths.concat(directory,dir)) do
        print(file)
        if not params.spectrogram then
            ad = audiodataset{file=paths.concat(directory,dir,file),classname=dir,type="audio"}
        else
            ad = audiodataset{file=paths.concat(directory,dir,file),classname=dir,type="spectrogram"}
        end

        counter = counter + 1
        audiolist[counter] = ad
    end
    total = #audiolist
    numTrain = math.floor(trainsplit * #audiolist)
    numTest = math.floor(testsplit * #audiolist)
    numValidation = total - numTrain - numTest

    traincounter = 0
    testcounter = 0
    validcounter = 0
    audiolistcount = #audiolist

    -- This needs to get randomized

  end

    print (#audiolist)
    print(numTrain)
    print(numTest)
    print(numValidation)
    while #audiolist > 0 do
        ad = audiolist[#audiolist]
        audiolist[#audiolist] = nil
        -- now we decided what to load here...
        if not params.spectrogram then
            -- Call midi function
            format = paths.extname(ad.file)
            if format == "mid" then
                ad:loadAudioMidi(nil,wavpath)
            elseif format == "wav" or format == "au" then
                ad:loadIntoBinaryFormat()
            end
            --ad:generateImage()
        else
            -- Load spectrogram representation
            format = paths.extname(ad.file)
            if format == "mid" then
                ad:loadMidiSpectrogram(nil,wavpath,params.windowSize,params.stride)
            elseif format == "wav" or format == "au" then
                ad:loadIntoSpectrogram(params.windowSize,params.stride)
		ad.audio = ad.audio:t()
            end
        end
        print(ad.file .. "DONE")
        -- decide which path to place it
        if traincounter < numTrain then 
            ad:serialize(trainpath)
            traincounter = traincounter + 1
        elseif testcounter < numTest then
            ad:serialize(testpath)
            testcounter = testcounter + 1
        else
            validcounter = validcounter + 1
            ad:serialize(validpath)
        end
        collectgarbage()
    end
  end
=======
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
    --shuffledTrainData.Songs[i] = (shuffledTrainData.Songs[i] - shuffledTrainData.Songs[i]:mean())/shuffledTrainData.Songs[i]:std()
	shuffledTrainData.Labels[i] = trainData.Labels[shuffle[i]]
    end

    local shuffle = torch.randperm(TestingCounter)
    for i=1, TestingCounter do
	shuffledTestData.Songs[i] = testData.Songs[shuffle[i]]
    --shuffledTestData.Songs[i] = (shuffledTestData.Songs[i]-shuffledTestData.Songs[i]:mean())/shuffledTestData.Songs[i]:std()
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










>>>>>>> 15fe2d1dc75bbcaa81d9c3b12fba760c982a0479
