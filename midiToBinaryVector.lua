-- Importing need libraries 
midi = require "MIDI" -- http://www.pjb.com.au/comp/lua/MIDI.html
require "torch" -- http://torch.ch/
require "image"
math = require 'math'
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
binVector[1][note][i] = 1--(binVector[note][i] + intensity )--/ 128)
--binVector[2][note][i] = (binVector[2][note][i] + 1 )
--if(binVector[note])
--print(binVector[note][i])
end
midiToBinaryVec = function(filename)
print(filename)
local MaxTicks = 15000
-- read the file
local f = assert(io.open(filename, "r"))
local t = f:read("*all")
if t == nil then
  f:close()
  return nil
end
-- Set some local max and min variabes
local min = 100000000
local max = 10

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
if min > v2[3] and min ~= 0 then min = v2[3] end
notes[#notes+1] = v2
end
end
end
end
-- determing the overall array length using total ticks / smallest furation
print(min)
--if(min~= 0) then
--array_col = total_ticks/50 --25 50
--else
--array_col = total_ticks/50
--end


array_row = 128 -- The number of midis notes, this can be made better.
array_col = total_ticks
print(total_ticks)
MaxTicks = array_col
if total_ticks > MaxTicks and false then
  array_col = MaxTicks
end
f:close()
-- need to allocate array to feeat everything into
local binVector = torch.Tensor(1,array_row,array_col):zero()

ma = require "math"
-- fit all notes
min = 1
for k,n in pairs(notes)
  do
  if n[3] <= MaxTicks then  
    local fr = ma.min((n[2])/(min) + 1,array_col)
    local to = ma.min((n[2]+n[3])/(min)+1,array_col)
    local note = ma.min(ma.max(n[5],0),128)
    local intensity = n[6]

    for i=fr,to do
      ok,err = pcall(setIntensity,binVector, note, i, intensity)
      if( not ok)

        then
        print ("ERROR: ")
        print(err)
        print(ok)
        return nil
      else
--break 
end

end
end 
end

return image.scale(binVector,500,128):byte()
--return binVector:byte()
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

