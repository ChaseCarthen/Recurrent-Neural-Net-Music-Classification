require 'Model/AutoEncoder'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'Model/StackedAutoEncoder'
require 'image'
require 'audio'

function tensorToNumber(tensor)
  local number = 0
  --print(tensor)
  for i=1,32 do
    if(tensor[i] == 1) then
      number = bit.bor(bit.lshift(1,i-1),number)
    end
  end
  return number
end

torch.setdefaulttensortype('torch.FloatTensor')
data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/MiniProcessed/train/ty_januarMINp_align.dat')
print(data.samplerate)
join = nn.JoinTable(1)

model = torch.load('./auto.model')

data2 = ((data.audio:float() -data.audio:mean())/data.audio:float():std()):split(100)
--data3 = data.midi:float():split(10000)
out2 = {}

out3 = {}

data2[#data2] = nil

print(data2)

out2 = model:forward(model.layerCount,data2,true)

for i = 1,#data2 do
	print(data2[i]:size())
	print(out2[i]:max())
	print(out2[i]:min())
	print(out2[i]:mean())
	--print(out2[i][1])
	print("=================")
	--image.save('test'.. i ..'.pgm',image.scale(image.minmax{tensor=out2[i]},1000,1000))
	if i > 1 and i < #data2 then
		--print("SUM: " .. (out2[i] - out2[i-1]):sum() )
		--print("Original SUM: " .. (data2[i] - data2[i-1]):sum() )

	end
	--out3[i] = out2[i] - data2[i]
	--print(out2[i])
	--image.save('testor'.. i ..'.pgm',image.scale(image.minmax{tensor=data2[i]},1000,1000))
	--image.save('testma'.. i ..'.pgm',image.scale(image.minmax{tensor=data3[i]},1000,1000))

end

out2 = join:forward(out2):clone()
out3 = join:forward(out3):clone()
data2 = join:forward(data2):clone()

print(out2:size())
image.save('autospectrogram.pgm',image.minmax{tensor=out2} )
--print("done test")
image.save('spectrogram.pgm',  image.minmax{tensor=data2} )
--image.save('compare.pgm',image.minmax{tensor=out3})
--image.save('midi.pgm',image.minmax{tensor=data.midi})
--image.save('midi.pgm',image.scale(image.minmax{tensor=data.midi},1000,1000))
test = torch.zeros(2,2)

test[1][1] = 1
test[2][2] = 1

--print("done test2")
--image.save('test3.pgm',image.scale(image.minmax{tensor=out2},1000,1000))
--print("done test3")

--song = torch.zeros(out:size(1),1)
--for i=1,out:size(1) do
--	song[i][1] = tensorToNumber(out[i])
--end
print("done")
--print(out3:min())
print(data2:min())


--audio.save('test.wav',song,data.samplerate+50)
