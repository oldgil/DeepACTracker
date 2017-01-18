require 'nn'
require 'nngraph'

local model_utils = require '../utils/model_utils'

local AlexNet = torch.class('tracking.AlexNet')
--local utils = detection.GeneralUtils()
function AlexNet:__init(model_opt)
    -- rnn opts
    self.gpu = model_opt.gpu
    self._n_rois = model_opt._n_rois
	self.feat_size = model_opt.feat_size
	self.model_path = model_opt.model_path
	self.weight_file_path = model_opt.weight_file_path
	--self.model_opt = model_opt
	
    if self.model_path == nil then
		error 'The first argument can not be nil!'
	end

	-- build CNN model with ROIPooling, LSTM, output layer
	self.cnn,self.cnn_name = dofile(self.model_path)(model_opt)
	print('build cnn model '..self.cnn_name)
	
    self.classifier = nn.Linear(4096, 31)
    self.classifier.name = 'classifier'

	self.model = {}
	self.model.cnn = self.cnn
	self.model.classifier = self.classifier
    -- set to gpu
    if self.gpu >= 0 then
        self.cnn:cuda()
        self.classifier:cuda()
    end

        -- set parallel
	--if config.nGPU > 1 and not model_opt.test then
	--	self:_makeParallel()
	--end

    --self.cnnw,self.cnndw = model_utils.combine_all_parameters(self.cnn,self.classifier)
	self.cnnw,self.cnndw = model_utils.combine_all_parameters(self.classifier)
	self.cnndw:zero()

        -- load weight_file for cnn
	if self.weight_file_path~=nil then
		self:load_weight()
	end
end

function AlexNet:clone()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(self)
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    mem:close()
    return clone
end


--[[function DDRLNet:_makeParallel()
	local gpus = torch.range(1, config.nGPU):totable()
	local fastest, benchmark = cudnn.fastest, cudnn.benchmark
	local parallel_model = nn.DataParallelTable(1,true,true):cuda():add(self.model,gpus)
	:threads(function()
            local cudnn = require 'cudnn'
            local inn = require 'inn'
            local detection = require 'detection'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      parallel_model.gradInput = nil

     self.model = parallel_model:cuda()
end]]

function AlexNet:load_weight(weight_file_path)

	if weight_file_path~=nil then
		self.weight_file_path = weight_file_path
	end
	local loaded_model = torch.load(self.weight_file_path)

	print('load pre_trained models.......')
	-- See if the loaded model supports our naming method for loading the weights
	local has_name = false
	local loaded_modules = loaded_model:listModules()
	for i=1,#loaded_modules do
		if loaded_modules[i].name then
			has_name = true
			break
		end
	end
	if has_name then
		-- Load the weights based on names
		local model_modules = self.cnn:listModules()
		for i=1,#loaded_modules do
			local copy_name = loaded_modules[i].name
			if copy_name then
				for j=1,#model_modules do
					local my_name = model_modules[j].name
					if my_name and my_name == copy_name then
						print('Copying weights from ' .. my_name .. ' layer!')
						model_modules[j].weight:copy(loaded_modules[i].weight)
						model_modules[j].bias:copy(loaded_modules[i].bias)
					end
				end  
			end
		end
	else
		--[[if self.model_opt.fine_tunning then
			-- romove the last layer then
			loaded_model:remove(#loaded_model.modules)
		end
		-- Loading parameters
		params = loaded_model:getParameters()
		-- Copying parameters

	 	self.parameters[{{1,params:size(1)}}]:copy(params)]]
	    print('load pre_trained models failed!')
	 end
	print('load pre_trained models completed!')
 end

function AlexNet:getCNNParameters()
    return self.cnnw, self.cnndw
end

function AlexNet:training()
	self.cnn:training()
    self.classifier:training()
    --self.fusionnet:training()
end

function AlexNet:evaluate()
	self.cnn:evaluate()
    self.classifier:evaluate()
end

function AlexNet:save(save_path)
	-- First sanitize the net
    --local network = self:clone()
	self:_sanitize(self)
	--[[local isParallel = (torch.type(self.model) == 'nn.DataParallelTable')
	if isParallel then
		self.model = self.model:get(1)
	end]]
	collectgarbage()
	-- Save the snapshot
	torch.save(save_path,self)
	 --[[if isParallel then
	 	self:_makeParallel()
	 end]]
end

function AlexNet:load_from_caffe(proto_path,caffemodel_path,save_path,model_name)
	caffeModelLoader = tracking.CaffeModelConverter(self.cnn,proto_path,caffemodel_path,model_name,save_path)
	caffeModelLoader:convert()
end

function AlexNet:get_model()
	return self.model
end


function AlexNet:cnnforward(inputs)
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    self.observation = self.cnn:forward(inputs)
	--print(self.observation:size())
    self.scores = self.classifier:forward(self.observation)
    return self.observation ,self.scores
end

function AlexNet:cnnbackward(inputs, targets)
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    self.cnndw:zero()
    local cnntargets = self.classifier:backward(self.observation,targets)
    --self.cnn:backward(inputs,cnntargets)
    self.cnndw:clamp(-5, 5)
end


function AlexNet:_sanitize(network)
  for name, net in pairs(network.model) do
    local list = net:listModules()
    for _,val in ipairs(list) do
        for name,field in pairs(val) do
            if torch.type(field) == 'cdata' then val[name] = nil end
            if name == 'homeGradBuffers' then val[name] = nil end
            if name == 'input_gpu' then val['input_gpu'] = {} end
            if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
            if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
            if (name == 'output' or name == 'gradInput') then
      	    if type(field) ~= 'table' then
        	   val[name] = field.new()
            end
        end
        end
    end
  end
end
