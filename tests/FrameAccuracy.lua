require 'Model/AutoEncoder'
require 'Model/RNNC'
require 'cunn'
require 'rnn'
require 'audiodataset'
require 'image'
require 'audio'
require 'gnuplot'
require 'Model/StackedAutoEncoder'
require 'writeMidi'
require 'audio'
require 'DatasetLoader'
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

join = nn.JoinTable(1)

model = torch.load('./train.model')
auto = torch.load('./auto.model')


out = {}

fn = 0
fp = 0
tn = 0
tp = 0
split = 50

function calculateAccuracy()
  local fn2 = 0
  local fp2 = 0
  local tn2 = 0
  local tp2 = 0
  local acc = 0
  local done = false
while not done do
  data = dl:loadNextSet()
  
  done = data.done
  for song= 1, #data do
    --print(data[song].samplerate)
    data2 = image.minmax{tensor=data[song].audio}:float()
    data2 = data2:split(split)
    data3 = data[song].midi:float():split(split)
  if data2[#data2]:size(1) ~= split then
    data2[#data2] = torch.cat(data2[#data2], torch.zeros(split - data2[#data2]:size(1), data2[#data2]:size(2) ), 1 )
    data3[#data2] = torch.cat(data3[#data3], torch.zeros(split - data3[#data3]:size(1), data3[#data3]:size(2) ), 1 )
  end
  input = auto:forward(1,data2,false )
  for j = 1,#input do
    input[j] = data2[j] - input[j]
  end

  out = model:forward(input)


  out = join:forward(out):clone()
  data3 = join:forward(data3):clone()
  out:round()

  -- Calculate false negatives
  fn = fn + (data3 - out):eq(1):sum()

  -- Calculate false positivess
  fp = fp + (data3 - out):eq(-1):sum()

  -- calculate true positives
  tp = tp + (data3 + out):eq(2):sum()

  -- calculate true negatives
  tn  = tn + (data3 + out):eq(0):sum()

  -- Calculate false negatives
  fn2 = fn2 + (data3 - out):eq(1):sum()

  -- Calculate false positivess
  fp2 = fp2 + (data3 - out):eq(-1):sum()

  -- calculate true positives
  tp2 = tp2 + (data3 + out):eq(2):sum()

  -- calculate true negatives
  tn2  = tn2 + (data3 + out):eq(0):sum()


  end
end
  
acc = tp2 / (tp2+fn2+fp2)
pre = tp2 / (tp2 + fp2)
rec = tp2 / (tp2 + fn2)
fmeasure = (2 * pre * rec) / (pre + rec)
print("accuracy: " .. acc)
print('pre: ' .. pre)
print('rec: ' .. rec)
print ('fmeasure: ' .. fmeasure)
end


dl = DatasetLoader('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/NottinghamProcessed','audio','midi')

print('train')
 dl:loadTraining()
 numTrain = dl:numberOfTrainingSamples() 
 calculateAccuracy()


print("valid")
dl:loadValidation()
 numTrain = dl:numberOfValidSamples()
 calculateAccuracy() 



print("test")
dl:loadTesting()
 numTrain = dl:numberOfTestSamples() 
 calculateAccuracy()

print('final')

accuracy = tp / (tp+fn+fp)
pre = tp / (tp + fp)
rec = tp / (tp + fn)
fmeasure = (2 * pre * rec) / (pre + rec)

print("accuracy: " .. accuracy)
print('pre: ' .. pre)
print('rec: ' .. rec)
print ('fmeasure: ' .. fmeasure)




