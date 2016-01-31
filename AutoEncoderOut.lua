require 'AutoEncoder'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'StackedAutoEncoder'
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
data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/processed/train/027600b_.dat')
print(data.samplerate)
join = nn.JoinTable(1)

model = torch.load('./auto.model')

data2 = data.audio:float():split(10000)
out2 = {}


for i = 1,#data2 do
	
	out2[i] = model:forward(2,{data2[i]},true)[1]:clone():round()
	print(out2[i]:size())
	print(out2[i]:max())
	print(out2[i]:min())
	print(out2[i]:mean())
	--print(out2[i][1])
	print("=================")
	image.save('test'.. i ..'.pgm',image.scale(image.minmax{tensor=out2[i]},1000,1000))
	image.save('testor'.. i ..'.pgm',image.scale(image.minmax{tensor=data2[i]},1000,1000))

end

out2 = join:forward(out2):clone()

--image.save('test.pgm',image.scale(image.minmax{tensor=out},1000,1000))
--print("done test")
--image.save('test2.pgm',image.scale(image.minmax{tensor=data.audio},1000,1000))
--print("done test2")
--image.save('test3.pgm',image.scale(image.minmax{tensor=out2},1000,1000))
--print("done test3")

--song = torch.zeros(out:size(1),1)
--for i=1,out:size(1) do
--	song[i][1] = tensorToNumber(out[i])
--end
print("done")
--audio.save('test.wav',song,data.samplerate+50)