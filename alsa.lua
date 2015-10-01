local ALSA = require 'midialsa'
local midi = require 'MIDI'
 ALSA.client( 'Lua client', 1, 1, true)
-- ALSA.connectfrom( 0, 14, 0 )    --  input port is lower (0)
ALSA.connectto( 0, 129,0 ) -- output port is higher (1)

print(ALSA.noteevent(1,40,100,0,.001))
--ALSA.output(ALSA.noteevent(1,40,100,0,.001))
filename = '100song1.mid'
 local f = assert(io.open(filename, "r"))
 local t = f:read("*all")

m = midi.midi2score(t)

--play_score(m)
--local alsaevent = ALSA.input()
print(m[1])
print(ALSA.listconnectedto())
for i=2,#m do
   for j = 1,#m[i] do
     event = ALSA.scoreevent2alsa(m[i][j])
     print(m[i][j][1])
     if m[i][j][1] == 'note'and event ~= nil then
        print(event)
        print(m[i][j]) 
        ALSA.output(event)
        break
     end
   end
end
ALSA.start()
--ALSA.syncoutput()
--ALSA.start()
--ALSA.input()
--while(1) do
--end
