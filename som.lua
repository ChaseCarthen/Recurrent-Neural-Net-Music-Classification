

print("STARTING")
local torch = require 'torch'
require "nn"
local midi = require 'MIDI'
require "optim"
require "midiToBinaryVector"
require 'DatasetGenerator'
require 'lfs'
require 'nnx'
require 'midibinning'
require 'SOM'
require 'writeMidi'
require 'image'

--require 'cunnx'
--require 'cunn'
local file = require 'file'

local files = getFiles("./midibins",'.dat')
print(files)
print(#files)
classes = 10
model = SOM.create(128,512,4,4,200)
model:cuda()

epoch = 1
batch_size = 10

function train()

   -- epoch tracker
   epoch = epoch or 1
   local maxmindist = 1000000000000
   local err = 0
   local counter = 1
   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
  -- model:training()
   --print(#trainData)
   -- shuffle at each epoch
   print(#files)
   shuffle = torch.randperm(#files)

   loss = 0
   for t = 1, #files do

    collectgarbage()
    xlua.progress(t, #files ) 

    local inputs = torch.load(files[shuffle[t]]).data
    for m=1,#inputs do
      xlua.progress(m,#inputs)
      print(inputs[m]:size(1))
      if inputs[m]:size(1) == 128 then
      --print(inputs[m]:size())
      
      local class,distance = model:forward(inputs[m])

      if maxmindist > distance then
        maxmindist = distance
      end
      
      --print("Class" .. class)
      err = model:backward(inputs[m],class) + err

end
collectgarbage()
end
end
print("Error: " .. err)
print("Max Minimum Distance" .. maxmindist)

print("Saving the model")
model:save("model.net")
epoch = epoch + 1

end





clusterfile = "cluster"

for i = 1, 200 do
  print("Epoch: " .. i)
  print("Learning Rate: " .. model:learningRate())
  print("Lattice" .. model:latticeAtTimeT())
    --print(model:update())
    --print( model:learningRate())
    if i%4 == 0 then
      for l=1,4 do
      for m=1,4 do
        local val = model.weights[l][m]:double()
        val = val:round()
        local ma = val:max()
        local mi = val:min()
        val = (val - mi) / (ma - mi) * 255
        writeMidi("f".. l .. "," .. m .. "epoch".. i .. ".mid",val,40,100)
        image.save("f".. l .. "," .. m .. "epoch".. i ..".png",val)
      end
      end
      os.execute('rm cluster*.txt')
      print("Writing to Clusters")

      --local W = nn.Copy('torch.CudaTensor', 'torch.FloatTensor'):forward(model2:get(numofw).weight)
      --print(trainData)
      for k =1,#files do
        local DETA = torch.load(files[k])
        xlua.progress(k, #files ) 
        --if DETA ~= nil then
        for detas = 1,#DETA.files do
--print(DETA.data)
if DETA.data[detas]:size(1) == 128
  then
  xlua.progress(detas, #DETA.files ) 
        --output = model:forward(DETA.data[detas])
        --print(DETA.data[detas])
        local group = model:forward(DETA.data[detas])
        --print(detas .. "File: " .. DETA.files[detas] )
        file.write(clusterfile .. group .. ".txt",DETA.files[detas] .. "\n","a")
        --end
      end
      collectgarbage()
    end
  end


end
print("TRAIN")
train()
model:update()


    --test()
  end

