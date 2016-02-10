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
