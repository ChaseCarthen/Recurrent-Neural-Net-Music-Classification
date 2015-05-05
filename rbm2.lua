require 'nn'
local rbm, _ = torch.class('nn.rbm', 'nn.Container')

function rbm:__init(visible,hidden)
    --parent.__init(self)
    self.modules = {}
    self.linear = nn.Linear(visible,hidden)
    self.linear2 = self.linear:clone()
    self.linearw = self.linear:clone()
    self.linearw.weight = self.linear.weight
    self.linearw.bias = self.linear.bias
    self.linear2.weight = self.linear.weight:t()
    self.linear2.bias = torch.randn(visible)
    self.linearw2 = self.linear2:clone()
    self.linearw2.weight = self.linear2.weight
    self.linearw2.bias = self.linear2.bias
    self:add(self.linear)
    self:add(nn.Sigmoid())
    self.Lin2 = nn.Sequential()
    self.Lin2:add(self.linear2)
    self.Lin2:add(nn.Sigmoid())
    self.Lin2w =nn.Sequential()
    self.Lin2w:add(self.linearw2)
    self.Lin2w:add(nn.Sigmoid())

    --print(self.linear2)
    --self.linear2:add(nn.ReLU())
end

function rbm:propup(input)
   --print(input)
      local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end 
   return currentOutput
end

function rbm:propdown(input)
   --print(input)
   return self.Lin2:forward(input)
end

function rbm:sample_h(input)
   h_mean = self:propup(input)
   local h1_sample = torch.Tensor(h_mean:size())
   for i=1,h1_sample:size(1)
   do
      for j=1,h1_sample:size(2)
      do
         h1_sample[i][j] = torch.bernoulli(h_mean[i][j])
      end
   end
   return h1_sample 
end

function rbm:sample_v(input)
   v_mean = self:propdown(input)
   local v1_sample = torch.Tensor(v_mean:size())
   for i=1,v1_sample:size(1)
   do
      for j=1,v1_sample:size(2)
      do
         v1_sample[i][j] = torch.bernoulli(v_mean[i][j])
      end
   end
   return v1_sample 
end

function rbm:free_energy(input)
   --input = self:sample_v(inputs)
   --print(self.linear2.bias:size())
   --print(inputs:size())
   --print(torch.randn(5))
   --print(self.linear.bias:size())
   --print(self.linear2.bias:size())
   wx_b = self.linearw:forward(input)
   --print(wx_b)
   vbias_term = torch.Tensor(input:size(1))
   --print(self.linear2.bias:double())
   --print(input)
   vbias_term:addmv(1,input, self.linear2.bias)
   vbias_term = vbias_term
   --vbias_term = self.linearw2:forward(inputs)
   hidden_term = wx_b:exp():add(1):log():sum(2)
   --print("HIDDEN_TERM")
   --print(hidden_term)
   --print("VISIBLE_TERM")
   --print(vbias_term)
   --print(self.linear2.weight:t())
   --print(self.linear.weight)
   --hidden_term = nn.Reshape(128):forward(hidden_term)
   --print((-hidden_term - vbias_term))
   --print(input)
   
   --hidden_term = hidden_term:resize(15)
   return (-hidden_term - vbias_term)
end

function rbm:__len()
   return #self.modules
end

function rbm:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function rbm:insert(module, index)
   index = index or (#self.modules + 1)
   if index > (#self.modules + 1) then
      error"index should be contiguous to existing modules"
   end
   table.insert(self.modules, index, module)
   self.output = self.modules[#self.modules].output
   self.gradInput = self.modules[1].gradInput
end

function rbm:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end 
   currentOutput = self:sample_h(input)
   self.output = currentOutput
   self.weight = self.linear.weight
   return currentOutput
end

function rbm:gibbs_step(output,input_size)
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
   --print(h:size())
   --print(self.weight:size())
   
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

function rbm:updateGradInput(input, gradOutput)
   --gradOutput = self:free_energy(gradOutput)
   self.orig = gradOutput
   --print("ORIGNAL")
   --print(gradOutput:size())
   gradOutput = self:sample_v(gradOutput)
   self.sampled = gradOutput
   --print("HERE")
   --print(self:free_energy(input):mean() )
  -- print("HERE: " .. self:free_energy(gradOutput):mean())
   --print(self:free_energy(input):mean() - self:free_energy(gradOutput):mean())
   scalar = (self:free_energy(input):mean() - self:free_energy(gradOutput):mean())
   self.scalar = scalar 
   print(scalar)
   gradOutput = self.orig*-scalar

   --print(gradOutput)
   --gradOutput = gradOutput:t()
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      --print(previousModule.output:size())
      --print(currentGradOutput:size())
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function rbm:accGradParameters(input, gradOutput, scale)
   
   scale = scale or 1
   self.orig = gradOutput
   gradOutput = self.sampled --self:sample_v(gradOutput)
   --scalar = (self:free_energy(input):mean() - self:free_energy(gradOutput):mean()) 
   --print("HERE")
   --print("HERE: " .. self:free_energy(input):mean() )
   --print("HERE: " .. self:free_energy(gradOutput):mean())
   --print(self:free_energy(input):mean() - self:free_energy(gradOutput):mean())
   gradOutput = self.orig*-self.scalar --(self:free_energy(input):mean() - self:free_energy(gradOutput):mean())
   --print(gradOutput)
   --gradOutput = gradOutput:t()
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accGradParameters(input, currentGradOutput, scale)
   --self.linear2.bias = self.linear2.bias - (self.linear2.bias*scalar)
end

function rbm:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)

end


function rbm:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.rbm'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
