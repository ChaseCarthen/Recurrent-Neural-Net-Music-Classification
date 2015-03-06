local torch = require 'torch'
local midi = require 'MIDI'
mtbv = require "midiToBinaryVector"
require 'lfs'

obj = 
{
    Genre = "classical",
    Songs = {}
}

counter = 0.
for filename in lfs.dir("./music/jazz") 
    do print ("./music/jazz/"..filename) 
    if string.find(filename, ".mid")
    then obj.Songs[filename] = midiToBinaryVec("./music/jazz/"..filename)  
    end
end
