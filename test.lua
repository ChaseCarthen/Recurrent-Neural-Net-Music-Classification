require 'rbm2'
require 'writeMidi'
require 'nn'
local midi = require 'MIDI'
mtbv = require "midiToBinaryVector"
require 'lfs'
a = nn.rbm(128,128)
b = nn.Linear(128,128)
--vals = {}
--for i=1,100 do
--val = torch.ones(15,5)*.5
--val = val:bernoulli(.5)
--vals[i] = val:double()
--end
print(val)

fileCounter = 0
BaseDir = "./miniMusic"
vals = {}
for directoryName in lfs.dir(BaseDir) do
        if directoryName ~= ".." and directoryName ~= "." and lfs.attributes(BaseDir.."/"..directoryName,"mode") == "directory"
        then
for filename in lfs.dir(BaseDir.."/"..directoryName) 
do FullFilePath = BaseDir.."/"..directoryName.."/"..filename
                if string.find(filename, ".mid")
                then 
                   
                    data = midiToBinaryVec2(FullFilePath) 
                    if data ~= nil then
                        fileCounter = fileCounter + 1 
                        vals[fileCounter] = data:t()
                       print("DATA: ")
                       print(data:t():size())
                       print(FullFilePath)
                       --break
                        --print(data)
                        --print(data:size())
                    end
                end
            end
            --break
        end

end
--output = b:forward(val)
--b:backward(val,output)


for j=1,100 do
for i=1,#vals do
	--print("OUTPUT")
	print(vals[i]:size())
output = a:forward(vals[i])
--print(output:size())
print("OUTPUT __________________________________________________________" .. i .. " " .. j)
print(a:sample_v(output):size())
if(j % 5 == 0)
then
print(a:sample_v(output))
writeMidi("test" .. i .. ".mid",a:sample_v(output),10,10)
end

--print(vals[i])
a:backward(vals[i]:float(),output)
--a:updateParameters(-.01)
end
end
--print(val)
--print(output)
