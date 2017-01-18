require 'nn'
require 'nngraph'

--require 'Rectifier'

--local CNN = require 'CNN'
local model_utils = require '../utils/model_utils'

local NCNet = torch.class('tracking.NCNet')
--local utils = detection.GeneralUtils()
function NCNet:__init(model_opt)
    -- rnn opts
    self.gpu = model_opt.gpu
    self.maxm     = model_opt.maxm
    
    self.dnn = nn.Sequential()
    self.dnn:add(nn.Linear(self.maxm,self.maxm * 2))  -- output actions maxm * (maxm + 1)
    self.dnn:add(nn.Linear(self.maxm*2,2))
    --self.lnet = LNET.lnet(model_opt)

    -- set to gpu
    if self.gpu >= 0 then
        self.dnn:cuda()

    end

    self.parameters, self.gradParameters = model_utils.combine_all_parameters(self.dnn)

    self.gradParameters:zero()
        
    -- clone for hist_len times
end


function NCNet:clone()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(self)
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    mem:close()
    return clone
end


function NCNet:getParameters()
	return self.parameters, self.gradParameters
end


function NCNet:training()
	self.dnn:training()
end

function NCNet:evaluate()
	self.dnn:evaluate()
end


function NCNet:forward(inputs)
    --local input = inputs:transpose(1,2):cuda()
    self.predictions = self.dnn:forward(inputs)
end


function NCNet:backward(inputs,targets)
    self.gradParameters:zero()
    --local input = inputs:transpose(1,2):cuda()
    local dinput = self.dnn:backward(inputs, targets:cuda())
    self.gradParameters:clamp(-5,5)
end

