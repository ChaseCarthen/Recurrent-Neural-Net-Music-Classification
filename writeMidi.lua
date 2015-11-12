 local MIDI = require 'MIDI'


 function writeMidi(fileName,pianoRoll,sampleRate,noteLength)
print ("write Midi ")
 local my_score = {
    96,  -- ticks per beat
    {    -- first track
          {'patch_change', 0, 1, 6},
        --{'patch_change', 0, 1, 6},
        --{'note', 5, 500, 1, 25, 98},
        --{'note', 101, 500, 1, 29, 98},
        --{'note',100,500,1,29,98}
    },  -- end of first track
 }
 -- Going through a score within a Lua program...
 channels = {[2]=true, [3]=true, [5]=true, [8]=true, [13]=true}
 for itrack = 2,#my_score do  -- skip 1st element, which is ticks
    for k,event in ipairs(my_score[itrack]) do
       if event[1] == 'note' then
          -- for example, do something to all notes
       end
       -- to work on events in only particular channels...
       channelindex = MIDI.Event2channelindex[event[1]]
       if channelindex and channels[event[channelindex]] then
          -- do something to channels 2,3,5,8 and 13
       end
    end
 end
 local note_on = false
 local to = 0
 local from = 0
 local counter = 2
 for j=1,128 do 
 for i=1,pianoRoll:size(1) do

  if(pianoRoll[i][j] > 0 and not note_on)
  then
  from = i 
  note_on = true
  --print("on")
  end
  --print(pianoRoll[from][j])
  if ((i == pianoRoll:size(1)) and note_on) or (note_on and pianoRoll[i][j] <= 0)
  then
  --print("off")
  pianoRoll[i][j] = 0
  --print(i-from)
  to = i
  my_score[2][counter]={'note',(from)*sampleRate,(to-from+1)*noteLength,1,j,pianoRoll[from][j]}
  counter = counter + 1
  note_on = false
  end 
end
 end
 local midifile = assert(io.open(fileName,'w'))
 --print(my_score)
 midifile:write(MIDI.score2midi(my_score))
 midifile:close()

end

--song = torch.randn(2000,128)
--song = song:bernoulli(.2)*100
--print(song)
--writeMidi("f.mid",song,40,100)