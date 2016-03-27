-- Importing need libraries 
midi = require "MIDI" -- http://www.pjb.com.au/comp/lua/MIDI.html
require "torch" -- http://torch.ch/
require "image"
require 'sys'
require 'audio'
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

function SortPartition(notes)
  o = 1
  counter = 1
  print(#notes)
  --print (notes)
  sorted = {}
  if #notes <= 1 then
    return notes
  end
  for i=1,#notes,2 do
    if i + 1 <= #notes then
      s = {}
      o = 1
      k = 1
      l = 1
      set = false
      for j=1,#notes[i]+#notes[i+1] do
        if #notes[i]+1 ~= k and #notes[i+1]+1 ~= l and notes[i][k].NoteBegin <= notes[i+1][l].NoteBegin then
          s[o] = notes[i][k]
          o = o + 1 
          k = k + 1
        elseif #notes[i]+1 ~= k and #notes[i+1]+1 ~= l and notes[i+1][l].NoteBegin <= notes[i][k].NoteBegin then
          s[o] = notes[i+1][l]
          l = l + 1
          o = o +1
        else 
          if #notes[i] + 1 == k then
            s[o] = notes[i+1][l]
            o = o + 1
            l = l + 1
          elseif #notes[i+1] + 1 == l then
            s[o] = notes[i][k]
            o = o + 1
            k = k + 1
          end 
        end
      end
      else
        s = {}
        for j=1,#notes[i] do
          s[j] = notes[i][j] 
        end
    end
    sorted[counter] = s
    counter = counter + 1 
  end
  --print (sorted)
  return SortPartition(sorted)
end


function NoteMergeSort(notes)
  unsorted = {}
  for i=1,#notes do
    unsorted[i] = {notes[i]}
  end
  sorted = SortPartition(unsorted)
  --print(sorted)
  return sorted
end


--[[
midiToBinaryVec
input: Takes in a filename 
output: spits out a torch float tensor.
--]]

setIntensity = function(binVector,note,i,intensity)
binVector[1][note][i] = 1
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


-- A file opening function -- for getting the correct
openMidi = function(filename)
foundNote = false
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
iteration = 0
--print(midi.midi2opus(t))
-- Concert the read in midi to a score object
m = midi.midi2ms_score(t)--midi.opus2score(midi.to_millisecs(midi.midi2opus(t)))
--print(m)
for k, v in pairs(m) 
do 
  if type(v)=="table" then
    for k2,v2 in pairs(v)
      do
      if v2[1] == "note" then
        iteration = iteration + 1
        notes[iteration] = {}
        notes[iteration]["NoteBegin"] = v2[2]
        notes[iteration]["NoteDuration"] = v2[3]
        notes[iteration]["Note"] = v2[5] + 1
        --print ("Note Begin: " .. v2[2])
        --print ("Note Duration: " .. v2[3])

        foundNote = true
      end

    end

  end


end
--print(m)
--print("MERGE SORT")

out = NoteMergeSort(notes)[1]
--print(notes)
return out,foundNote
end 

-- A function for generating a target vector in two forms
function generateMidiTargetVector(filename,notes)
  --print "generating Target Vector"
  data,samplerate = audio.load(filename)
  --print(data:size())
  endtime = notes[#notes].NoteBegin + notes[#notes].NoteDuration
  starttime = notes[1].NoteBegin
  totalduration = (endtime - starttime) / 1000
  sampletime = (1/samplerate)
  arraylength = totalduration / sampletime

  currenttime = 0
  local binVector = torch.ByteTensor(128,data:size(1)):zero()
  --print(data:size())
  --print(binVector:size())

  for i=1,#notes do
    --print(notes[i].NoteBegin)
    from = math.floor(notes[i].NoteBegin/1000.0 * samplerate)+1
    to = math.min ( math.floor((notes[i].NoteBegin/1000.0 + notes[i].NoteDuration/1000.0) * samplerate)+1, data:size(1))
        --print(from)
      --print(to)
    for j =from,to do
      --print("set")
      binVector[notes[i].Note][j] = 1
    end
    --print(i)
  end
  return data,binVector:t(),samplerate
end


-- A function for generating a target vector in two forms
function generateMidiSpectrogramVector(data,samplerate,striderate,notes)
  --print "generating Target Vector"
  --data,samplerate = audio.load(filename)
  --print(data:size())
  data = data:t()
  endtime = notes[#notes].NoteBegin + notes[#notes].NoteDuration
  starttime = notes[1].NoteBegin
  totalduration = (endtime - starttime) / 1000
  print("==========================")
  print(data:size())
  print("Total Duration: " .. totalduration)
  print("Sample Rate Duration: " .. ( (1.0/samplerate) + (data:size(1)) * 1/striderate) ) 
  print(samplerate)
  print(data:size(1))
  print("==========================")

  if totalduration > 1.0/samplerate*data:size(1) then
    --error("Error midi is greater than actual duration.")
  end
  currenttime = 0
  local binVector = torch.ByteTensor(128,data:size(1)):zero()
  --print(data:size())
  --print(binVector:size())
  local sum = 0

  for i=1,#notes do
    --print(notes[i].NoteBegin)
    from = math.max( torch.floor( math.max(notes[i].NoteBegin/1000.0, 0) * striderate)+1, 1)
    to = math.min ( torch.round( math.max(notes[i].NoteBegin/1000.0 + notes[i].NoteDuration/1000.0, 0) * striderate)+1, data:size(1))
    --print("NOTE: " .. i)
   -- print("Duration: " .. notes[i].NoteDuration/1000)
    --print("Begin: " .. notes[i].NoteBegin/1000.0)
    --print(1/striderate)
      -- print(from)
      --print(to)
     -- print(notes[i].Note)
      sum = sum + to - from + 1
      --print("========")
    for j =from,to do
      --print("set")
      --print(notes[i].Note)
      binVector[notes[i].Note][j] = 1
    end
    --print(i)
  end
  --print (sum)
  --print (binVector:sum())
  return data,binVector:t(),samplerate
end


function generateWav(filename,directory)

  filebase = paths.basename(filename,"mid")
  
  if not paths.filep('"' .. directory .. filebase .. ".wav" .. '"') then
    --print('timidity ' .. '"' .. filename .. '"' .. " --output-mono -s 22k -Ow -o " .. '"' .. directory .. filebase .. ".wav" .. '"')
    sys.execute('timidity ' .. '"' .. filename .. '"' .. " --output-mono -s 22050 -Ow -o " .. '"' .. directory .. filebase .. ".wav" .. '"')
  end

end

--filename = "./music/latin/todotodo.mid"
--notes = openMidi(filename)
--generateWav(filename)
--filebase = paths.basename(filename,"mid")
--generateMidiTargetVector(filebase .. '.wav',notes)

