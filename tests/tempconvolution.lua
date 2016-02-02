 function TemporalSplit(tensor,windowidth,stepsize)
	Tensor = {}
	counter = 1
	
	End = tensor:size(1)
	while step <= End do
		if step + windowidth <= End then
			Tensor[counter] = tensor:sub(step,step + windowidth )
			counter = counter + 1 
		end
		step = step + stepsize
	end
	return Tensor
end