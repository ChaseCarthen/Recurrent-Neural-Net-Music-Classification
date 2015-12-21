require 'audiodataset'
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text("A dataset convertor to torch object.")
cmd:text()
cmd:text('Options')
cmd:option("--midi",false,"Processing for midi..")
cmd:option('-m',"","Process midi data")
cmd:option('-d',"audio","Data directory to process.")
cmd:option('-o',"processed","Processed data directory.")
cmd:option("-r",.8,"Train Split Rate")
cmd:option("-r2",.1,"Validation Split Rate")
cmd:option("-r3",.1,"Test Split Rate")
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
  paths.mkdir(paths.concat(paths.cwd(),params.o))
  file = torch.DiskFile(paths.concat(paths.cwd(),params.o,"class.txt"),'w')
  for dir in paths.iterdirs(directory) do
    classlist[classindex] = directory
    classindex = classindex + 1
    print(dir)
    file:writeString(dir .. "\n")
    
    outpath = paths.concat(pat}
hs.cwd(),params.o,dir)
    trainpath = paths.concat(paths.cwd(),params.o,"train")
    testpath = paths.concat(paths.cwd(),params.o,"test")
    validpath = paths.concat(paths.cwd(),params.o,"valid")
    wavpath = paths.concat(paths.cwd(),params.o,"wav")

    
    --paths.mkdir(outpath)
    paths.mkdir(trainpath)
    paths.mkdir(validpath)
    paths.mkdir(testpath)
    paths.mkdir(wavpath)
    audiolist = {}
    counter = 0
    for file in paths.iterfiles(paths.concat(directory,dir)) do
        print(file)
        if not params.midi then
            ad = audiodataset{file=paths.concat(directory,dir,file),classname=dir,type="audio"}
        else
            ad = audiodataset{file=paths.concat(directory,dir,file),classname=dir,type="midi"} 
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
    while #audiolist > 0 do
        ad = audiolist[#audiolist]
        audiolist[#audiolist] = nil
        -- now we decided what to load here...
        if not params.midi then
            ad:loadIntoBinaryFormat()
        elseif params.midi then
            -- Call midi function
            ad:loadMidi(nil,wavpath)
            ad:generateImage()
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
end