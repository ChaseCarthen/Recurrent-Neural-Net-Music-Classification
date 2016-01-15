require 'AutoEncoder'
require 'RNNC'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'image'
require 'audio'
require 'gnuplot'
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
data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/processed/train/027700b_.dat')
print(data.samplerate)
join = nn.JoinTable(1)

model = torch.load('./train.model')

data2 = data.audio:float():split(10000)
out = {}

for i = 1,#data2 do
	print(data2[i]:size())
	--model:forget()
	out[i] = model:forward({data2[i]})[1]:clone()
	print(out[i]:max())
	print(out[i]:size())

       image.save('test' .. i .. '.pgm',image.scale(image.minmax{tensor=out[i]:round()},1000,1000))
end



--out = join:forward(out):clone()
--gnuplot.hist(out)
--print( (out:ge(.3) ):sum())
--print(data.midi:sum())
--out = out:round()
--print(out:max())

--print("done test")
--image.save('test2.pgm',image.scale(image.minmax{tensor=data.audio},1000,1000))
--print("done test2")
--image.save('test3.pgm',image.scale(image.minmax{tensor=out2},1000,1000))
--print("done test3")
