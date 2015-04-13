-- Importing need libraries 
midi = require "MIDI" -- http://www.pjb.com.au/comp/lua/MIDI.html
require "torch" -- http://torch.ch/
require "image"
-- Set the default tensor type to floats
torch.setdefaulttensortype('torch.FloatTensor')

-- allocate funciton to create a 2D array
allocate_array = function(row_size,col_size)
   local out = {}
   for i=1,row_size do
     out[i] = {}
     for j=1,col_size do
     out[i][j] = 0
     end
   end
   return out
end



--[[
midiToBinaryVec
input: Takes in a filename 
output: spits out a torch float tensor.
--]]

    setIntensity = function(binVector,note,i,intensity)
  binVector[1][note][i] = (binVector[1][note][i] + intensity )--/ 128)
  binVector[2][note][i] = (binVector[2][note][i] + 1 )
  --if(binVector[note])
  --print(binVector[note][i])
end
midiToBinaryVec = function(filename)
    print(filename)
    -- read the file
    local f = assert(io.open(filename, "r"))
    local t = f:read("*all")
    if t == nil then
    f:close()
    return nil
    end
    -- Set some local max and min variabes
    local min = 100000000
    local max = 0

    -- This variabe keeps track of the current notes
    local notes = {}

    -- Concert the read in midi to a score object
    m = midi.midi2score(t)

    -- get the the total ticks in a midi
    local total_ticks =  midi.score2stats(m)["nticks"]

    -- get the number of channels
    numchannels = table.getn(m)

    --iterate through the score objects channels and find all notes
    for k, v in pairs(m) 
    do 
    if type(v)=="table" then
    for k2,v2 in pairs(v)
    do
    if v2[1] == "note" 
    then
    -- Finding the minimum and maximum amount of duration
    if max < v2[3] then max = v2[3] end
    if min > v2[3] then min = v2[3] end
    notes[#notes+1] = v2
    end
    end
    end
    end
    -- determing the overall array length using total ticks / smallest furation
    array_col = total_ticks/50

    array_row = 128 -- The number of midis notes, this can be made better.
    f:close()
    -- need to allocate array to feeat everything into
    --local binVector = allocate_array(array_row,array_col)
    local binVector = torch.Tensor(2,array_row,array_col):zero()
    --print(binVector)
    --print(image.scale(binVector,128,512))
    --local binVector2 = torch.Tensor(1,array_row,array_col):zero()
    --print("NOTES: " .. #notes)
    --print(notes)
    ma = require "math"
    -- fit all notes

    for k,n in pairs(notes)
    do
    --print(k)
    --print(n) 
    local fr = ma.min((n[2])/(min) + 1,array_col)
    local to = ma.min((n[2]+n[3])/(min)+1,array_col)
    local note = ma.min(ma.max(n[5],0),128)
    local intensity = n[6]
    --print(binVector[note])
    for i=fr,to do
    ok,err = pcall(setIntensity,binVector, note, i, intensity)
    if( not ok)

   then
      print ("ERROR: ")
    print(err)
    print(ok)
   return nil
  else
   --print ("Okay: ")
   break 
  end
    --if(intensity/100~=.96) then print(intensity/128) end
    end
    end 
    --binVector2 = torch.Tensor(binVector)
    
    return image.scale(binVector,128,512)
end


-- A simple test print function for printing out the table representation
printBinaryVector = function(binVec)
   local s = ""
   for k,v in pairs(binVec)
   do
      for k2,v2 in pairs(v) do
         s = s .. v2
      end
      print(s)
      s = ""
   end
end

