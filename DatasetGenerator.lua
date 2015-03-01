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
for filename in lfs.dir("./music/classical") 
    do print ("./music/classical/"..filename) 
    if string.find(filename, ".mid")
    then obj.Songs[filename] = midiToBinaryVec("./music/classical/"..filename)  
    end
end
