require 'sndfile'

function writeWav(filename,data)
--data = data:short() 
f = sndfile.SndFile(filename, 'w', {samplerate=22050, channels=1, format="WAV", subformat="PCM16"})
f:string('title', 'dude it is awesome') -- write some title in there
f:writeShort(data)
f:close()
end