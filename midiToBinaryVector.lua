local midi = require "midi"

--print(midi)

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

midiToBinaryVec = function(filename)
    local f = assert(io.open(filename, "r"))
    local t = f:read("*all")
    local min = 100000000
    local max = 0
    local notes = {}
    m = midi.midi2score(t)
    local total_ticks =  midi.score2stats(m)["nticks"]
    print(total_ticks)
    --return
    numchannels = table.getn(m)
    --print(m)
    for k, v in pairs(m) 
    do 
    print (k,type(v))
    if type(v)=="table" then
    for k2,v2 in pairs(v)
    do
    if v2[1] == "note" 
    then
    if max < v2[3] then max = v2[3] end
    if min > v2[3] then min = v2[3] end
    --print (v2[1])
    notes[#notes+1] = v2
    end
    end
    end
    end
    --
    array_col = total_ticks/min
    array_row = 128 -- The number of midis notes, this can be made better.
    local binVector = allocate_array(array_row,array_col)
    --print(binVector)
    for k,n in pairs(notes)
    do 
    --print(n)
    local fr = (n[2])/(min) + 1
    local to = (n[2]+n[3])/(min) + 1
    local note = n[5]
    print(fr,to,note)
    for i=fr,to do
    print(i)
    binVector[note][i] = 1
    end
    end 

    print(max,min)
    print(#notes)
    printBinaryVector(binVector)
    f:close()
end

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
midiToBinaryVec("outf200.mid") 
