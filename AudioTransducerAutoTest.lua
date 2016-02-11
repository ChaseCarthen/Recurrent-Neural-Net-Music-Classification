require 'AutoEncoder'
require 'RNNC'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'image'
require 'audio'
require 'gnuplot'
require 'StackedAutoEncoder'
require 'writeMidi'

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
data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/processed/test/hpps_simple_chords_46.dat')
print(data.samplerate)
join = nn.JoinTable(1)
print(data.audio:sum())
model = torch.load('./train.model')
auto = torch.load('./auto.model')

data2 = image.minmax{tensor=data.audio}:float():split(100)
out = {}


print(data2)

--print(model:forward({data2[i]}))
--model:forget()
data2[#data2] = nil
input = auto:forward(1,data2,false )
for j = 1,#input do
  input[j] = data2[j] - input[j]
end

out = model:forward(input)

print(#out)
print(#data2)

for i = 1,#out do
  print(out[i]:mean())
	--print(out[i]:size())
  print(out[i]:max())
end
       --image.save('test' .. i .. '.pgm',image.scale(image.minmax{tensor=out[i]},1000,1000))
       --image.save('testor' .. i .. '.pgm',image.scale(image.minmax{tensor=data3[i]},1000,1000) )



out = join:forward(out):clone()

image.save('test.pgm',image.scale(out:round(),1000,1000 ) )


writeMidi('test.midi',out,10,100)

if data.midi ~= nil then
  image.save('midi.pgm',image.scale(image.minmax{tensor=data.midi},1000,1000))
end

