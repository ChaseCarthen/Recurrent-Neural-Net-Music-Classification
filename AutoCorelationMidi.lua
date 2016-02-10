samplerate = 22050
windowsize = 8192
require 'gnuplot'
require 'image'
a = 440

miditable = {}

for x = 0, 127 do
miditable[x] = (a / 32) * (2 ^ ((x - 9) / 12)) 
end

function frequencyToBin(notefreq,samplerate,windowsize)
	return torch.round(miditable[notefreq-1] / (samplerate/windowsize)/2 )
end


data = torch.load('/home/ace/Documents/Recurrent-Neural-Net-Music-Classification/processed3/train/ashover_simple_chords_12.dat')
mid = data.midi
data = data.audio
max = 0
for column= 1,data:size(1) do
	for notefreq = 0,127 do
		bin = data:size(2) - math.floor(miditable[notefreq] / (samplerate/windowsize)/2 )
		val = data[column][bin]
		if val > max then
			max = val
			print(data[column][bin])
			--print(bin)
			notefreq = notefreq + 1
			print(notefreq)
			for jj = 1, mid:size(1) do
				if mid[jj][notefreq] > 0 then
					print("Here")
				end
			end
		end
	end
end

print("=====================================================")
print("=====================================================")
print("=====================================================")
print("=====================================================")
print("=====================================================")


notes = {}
for column = 1, mid:size(1) do
	for row = 1, mid:size(2) do
		if mid[column][row] > 0 then
			bin = data:size(2) - torch.round(miditable[row-1] / (samplerate/windowsize)/2 )
			print("row " .. row .. " note: " .. data[column][bin])
			print(data[column][bin-1])
			print(data[column][bin])
			print(data[column][bin+1])
			print(data[column][bin+2])
			notes[row] = row
		end
	end
end


for key,value in pairs(notes) do
	print(value)
	bin = data:size(2) - frequencyToBin(value,samplerate,windowsize)
	ten = torch.ones(data:size(1),1)
	ten2 = torch.ones(data:size(1),1)
	for column = 1, data:size(1) do
		ten[column] = data[column][bin]
		ten2[column] = 300*mid[column][value]
	end
	gnuplot.pngfigure('test' .. value .. '.png')
	gnuplot.plot({tostring(value),ten}, {'notes',ten2})
	gnuplot.plotflush()
end


image.save('testmidi.png',image.minmax{tensor=mid})