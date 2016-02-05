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
data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/processed3/train/ashover_simple_chords_9.dat')
print(data.samplerate)
join = nn.JoinTable(1)

model = torch.load('./train.model')

data2 = data.audio:float():split(20000)
data3 = data.midi:float():split(20000)
out = {}

for i = 1,#data2 do
  --print("=================")
	--print(data2[i]:mean())
  --print(model:forward({data2[i]}))
	--model:forget()
  d = data2[i]:split(100)
  d[#d] = nil
	out[i] = model:forward(d)
  --print(out)
	--print(model:forward({data2[i]})[1]:mean())
  --print(out[i]:mean())
  --print(data2[i]:mean())
  --print(out[i]:size())
  --print("=====================")
  for j = 1,#out[i] do
    print(out[i][j]:mean())
    print(out[i][j]:size())
    print(data2[i]:mean())
    --print(out[i][j])
    image.save('tests' .. i .. 'i' .. j .. 'j.pgm',image.scale(image.minmax{tensor=out[i][j]},1000,1000))
  end
    --image.save('test' .. i .. '.pgm',image.scale(image.minmax{tensor=out[i]},1000,1000))
    --image.save('testor' .. i .. '.pgm',image.scale(image.minmax{tensor=data3[i]},1000,1000) )
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
