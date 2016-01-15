------------------------------------------------------------------------
--[[ VRClassReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
require 'nn'
require 'rnn'
require 'dpnn'
local BinaryClassReward, parent = torch.class("nn.BinaryClassReward", "nn.Criterion")

function BinaryClassReward:__init(module, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

function BinaryClassReward:updateOutput(input, target)
   assert(torch.type(input) == 'table')
   local input = self:toBatch(input[1], 1)
   local target = self:toBatch(target, 1)

   assert(input:size():size() == 2)
   self._maxVal = self._maxVal or input.new()
   self._maxIdx = self._maxIdx or torch.type(input) == 'torch.CudaTensor' and input.new() or torch.LongTensor()
   
   input = input:clone()
   input = input:round()
   self._reward = input.new():resize(input:size(1),1)

   for i=1,input:size(1) do
      self._reward[i] = 1.0 - math.abs(( (target[i] - input[i]):sum()) / (target[i]:size(1)))
   end

   self._target = target
   --self._reward = torch.Tensor({self._reward})
   -- reward = scale when correctly classified
   self.reward = self._reward:clone()--self.reward or input.new()
   --self.reward:resize(1):copy(self._reward)
   self.reward:mul(self.scale)

   -- loss = -sum(reward)
   self.output = -self.reward
   if self.sizeAverage then
      self.output = self.output/input:size(2)
   end
   return self.output:sum()
end

function BinaryClassReward:updateGradInput(inputTable, target)
   local input = self:toBatch(inputTable[1], 1)
   local baseline = self:toBatch(inputTable[2], 1)

   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(2))
   end

   -- broadcast reward to modules
   for i = 1, self.vrReward:size(1) do
      self.module:reinforce(self.vrReward[i])
   end  
   


   -- zero gradInput (this criterion has no gradInput for class pred)
   --print("BAKA")
   --print(self.gradInput)
   self.gradInput[1]:resizeAs(input):zero()
   self.gradInput[1] = self:fromBatch(self.gradInput[1], 1)
   
   -- learn the baseline reward
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   self.gradInput[2] = self:fromBatch(self.gradInput[2], 1)


   return self.gradInput
end

function BinaryClassReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end