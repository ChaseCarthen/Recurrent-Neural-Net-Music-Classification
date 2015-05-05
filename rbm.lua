require 'nn'
local Rbm, parent = torch.class('nn.Rbm', 'nn.Module')

function Rbm:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)

   self:reset()
end

function Rbm:gibbs_step(output,input_size)
if output then
--print(output)
end

local h = torch.Tensor(output:size())
for i=1,output:size(1)
   do
   for j=1,output:size(2)
      do
      h[i][j] = torch.normal(output[i][j])
      end
   end
   --self.output:zero():addr(1, self.addBuffer, self.bias)
   print(h:size())
   print(self.weight:size())
   
   local h2 = torch.zeros(output:size(1),self.weight:size(2))
   h2:addmm(1, h,self.weight)
   for i=1,h2:size(1)
   do
   for j=1,h2:size(2)
      do
      h2[i][j] = torch.normal(h2[i][j])
      end
   end
   return h2
end

function Rbm:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function Rbm:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)
      self.output:resize(nframe, nunit)
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      if nunit == 1 then
         -- Special case to fix output size of 1 bug:

         self.output:copy(self.bias:view(1,nunit):expand(#self.output))
         self.output:select(2,1):addmv(1, input, self.weight:select(1,1))
      else
         --print(self.output)
         self.output:zero():addr(1, self.addBuffer, self.bias)
         --print(self.output)
         self.output:addmm(1, input, self.weight:t())
      end
   else
      error('input must be vector or matrix')
   end
      print(self.output:size())
   print(self.weight:size())
   --print(self.output)
   return self:gibbs_step(self.output)
end

function Rbm:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Rbm:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      local nunit = self.bias:size(1)

      if nunit == 1 then
         -- Special case to fix output size of 1 bug:
         self.gradWeight:select(1,1):addmv(scale, input:t(), gradOutput:select(2,1))
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      else
         self.gradWeight:addmm(scale, gradOutput:t(), input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end

end

-- we do not need to accumulate parameters when sharing
Rbm.sharedAccUpdateGradParameters = Rbm.accUpdateGradParameters


function Rbm:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
