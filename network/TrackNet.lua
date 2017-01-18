require 'nn'
require 'nngraph'

--require 'Rectifier'

--local CNN = require 'CNN'
local LNET = require '../models/Linker/LNET'
local LSTM = require '../models/RNNs/LSTM'
local FusionNet = require '../models/Linker/FusionNet'
local model_utils = require '../utils/model_utils'

local TrackNet = torch.class('tracking.TrackNet')
--local utils = detection.GeneralUtils()
function TrackNet:__init(model_opt)
    -- rnn opts
    self.rnn_size = model_opt.rnn_size
    self.input_size = model_opt.input_size
    self.minibatch_size = model_opt.maxm*(model_opt.maxm+1)
    self.gpu = model_opt.gpu
    self._n_rois = model_opt._n_rois
    self.hist_len = model_opt.hist_len
    self.maxm     = model_opt.maxm
    self.feat_size= model_opt.feat_size

	self.model_path = model_opt.model_path
	self.weight_file_path = model_opt.weight_file_path
	--self.model_opt = model_opt  
	
    if self.model_path == nil then
		error 'The first argument can not be nil!'
	end

    self.filter = torch.Tensor(1,self.feat_size):fill(0):cuda()
	self.model = {}
    self.rnn = {}

	-- build CNN model with ROIPooling, LSTM, output layer
	self.cnn,self.cnn_name = dofile(self.model_path)(self.model_opt)
	print('build cnn model '..self.cnn_name)
	--[[for i,module in ipairs(self.cnn:listModules()) do
		print(module)	
	end]]
    self.classifier = nn.Linear(self.feat_size, 2)
    self.classifier.name = 'classifier'

    self.fusionnet = FusionNet.fusionnet(model_opt)
    self.rnn.lstm = LSTM.lstm(model_opt)
    self.rnn.dqn = nn.Sequential()
    self.rnn.dqn:add(nn.Linear(self.rnn_size,1))  -- output actions maxm * (maxm + 1)
    self.rnn.dqn:add(nn.Reshape(self.maxm,self.maxm+1))
    --self.lnet = LNET.lnet(model_opt)

    self.model.cnn = self.cnn
    self.model.classifier = self.classifier
    self.model.lstm = self.rnn.lstm
    self.model.dqn = self.rnn.dqn
    self.model.fusionnet = self.fusionnet

    -- set to gpu
    if self.gpu >= 0 then
        self.cnn:cuda()
        self.classifier:cuda()
        self.rnn.lstm:cuda()
        self.rnn.dqn:cuda()
        self.fusionnet:cuda()
    end

        -- set parallel
	--if config.nGPU > 1 and not model_opt.test then
	--	self:_makeParallel()
	--end
        
        
    -- index parameters and gradParameters
    self.parameters, self.gradParameters = model_utils.combine_all_parameters(self.rnn.lstm, self.rnn.dqn, self.fusionnet)

    self.cnnw,self.cnndw                  = model_utils.combine_all_parameters(self.cnn,self.classifier)
    self.cnndw:zero()
	--self.parameters, self.gradParameters = model_utils.combine_all_parameters		   	(self.rnn.lstm, self.rnn.dqn, self.lnet)
    self.gradParameters:zero()
        
    -- clone for hist_len times
    self.clones = {}
    for name,proto in pairs(self.rnn) do
	    print('cloning '..name)
	    self.clones[name] = model_utils.clone_many_times(proto, self.hist_len, not proto.parameters)
	end

	-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	self.initstate_c = torch.zeros(self.minibatch_size, self.rnn_size)
	self.initstate_h = self.initstate_c:clone()
	if self.gpu >=0 then
		self.initstate_c = self.initstate_c:cuda()
		self.initstate_h = self.initstate_h:cuda()
	end

	-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
	self.dfinalstate_c = self.initstate_c:clone()

	-- LSTM initial state for prediction, note that we're using minibatches OF SIZE ONE here
	self.prev_c = torch.zeros(1, self.rnn_size)
	self.prev_h = self.prev_c:clone()
	if self.gpu >=0 then
		self.prev_c = self.prev_c:cuda()
		self.prev_h = self.prev_h:cuda()
	end

	--self.parameters, self.gradParameters = self.model:getParameters()
        -- load weight_file for cnn
	if self.weight_file_path~=nil then
		self:load_weight()
	end
end


function TrackNet:clone()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(self)
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    mem:close()
    return clone
end


function TrackNet:_makeParallel()
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
end

--function Net:_prepare_regressor(means,stds)
--	stds[{{1,4}}] = 1.0
--	self.regressor.weight = self.regressor.weight:cdiv(stds:expand(self.regressor.weight:size()))
--	self.regressor.bias = self.regressor.bias - means:view(-1)
--	self.regressor.bias = self.regressor.bias:cdiv(stds:view(-1))
--end

--function Net:initialize_for_training()
	-- initialize classifier and regressor with appropriate random numbers 
--	self.classifier.weight:normal(0,0.01)
--	self.classifier.bias:fill(0)
--	self.regressor.weight:normal(0,0.001)
--	self.regressor.bias:fill(0)
--end

function TrackNet:load_weight(weight_file_path)

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

function TrackNet:getParameters()
	return self.parameters, self.gradParameters
end

function TrackNet:getCNNParameters()
    return self.cnnw, self.cnndw
end

function TrackNet:training()
	self.model.cnn:training()
    self.model.classifier:training()
	self.model.lstm:training()
	self.model.dqn:training()
    self.model.fusionnet:training()
end

function TrackNet:evaluate()
	self.model.cnn:evaluate()
    self.model.classifier:evaluate()
	self.model.lstm:evaluate()
	self.model.dqn:evaluate()
    self.model.fusionnet:evaluate()
end

function TrackNet:save(save_path)
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

function TrackNet:load_from_caffe(proto_path,caffemodel_path,save_path,model_name)
	caffeModelLoader = tracking.CaffeModelConverter(self.cnn,proto_path,caffemodel_path,model_name,save_path)
	caffeModelLoader:convert()
end

function TrackNet:get_model()
	return self.model
end


function TrackNet:cnnforward(inputs)
    -- body
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    --self.gradParameters:zero()
    
    self.observation = self.cnn:forward(inputs)
    --local features = self.observation:clone()
    self.scores = self.classifier:forward(self.observation)
    return self.observation ,self.scores
end

function TrackNet:cnnbackward(inputs, targets)
    -- body
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}

    self.cnndw:zero()
    local cnntargets = self.classifier:backward(self.observation,targets)

    self.cnn:backward(inputs,cnntargets)
    self.cnndw:clamp(-5, 5)

end

function TrackNet:dqnforward(inputs)
    -- inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    -- features maxm x hist_len x n
    local observation = inputs.observation

    ------------------- forward pass -------------------
    self.lstm_c = {[0]=self.initstate_c} -- internal cell states of LSTM
    self.lstm_h = {[0]=self.initstate_h} -- output values of LSTM
    self.features = {}

    --print(self.props:size())
    for t=1,self.hist_len do
        self.features[t],self.props = unpack(self.fusionnet:forward{inputs.features[{{},{t},{}}]:squeeze():cuda(),observation:cuda(),self.filter})

        self.lstm_c[t], self.lstm_h[t] = unpack(self.clones.lstm[t]:forward{self.features[t]:cuda(),self.props:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]})
            --self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
    end
    self.predictions = self.clones.dqn[self.hist_len]:forward(self.lstm_h[self.hist_len])
end

function TrackNet:dqnbackward(inputs,targets)
    
    -- target maxm * (maxm + 1)
    -- zero gradients of parameters
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    self.gradParameters:zero()
    --self.cnndw:zero()
    local dobservation = {}  
    local dfeatures     = {} 
    local dfeature    = {}    
    local dprops    = {} 
    local dfilter   = {}                                -- d loss / d input observation
    local dlstm_c = {[self.hist_len]=self.dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                      -- output values of LSTM
    --local targets = {}
    --local targets, dmask = unpack(self.lnet:backward({self.predictions,cnntargets},{target,dactions}))

    local observation = inputs.observation
    --self.features = inputs.features:clone():transpose(1,2)

    for t=self.hist_len,1,-1 do

        -- backprop through loss/target, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.dqn[t]:backward(self.lstm_h[t], targets:cuda())
        --else
        --    dlstm_h[t]:add(self.clones.dqn[t]:backward(self.lstm_h[t], targets[t]))
        end

        -- backprop through LSTM timestep
        dfeatures[t],dprops[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
        {self.features[t]:cuda(),self.props:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]},
        {dlstm_c[t], dlstm_h[t]}))
        --print(dfeatures:size())
        dfeature[t], dobservation[t],dfilter[t] = unpack(self.fusionnet:backward({inputs.features[{{},{t},{}}]:squeeze():cuda(),observation,self.filter},{dfeatures[t],dprops[t]}))
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    self.initstate_c:copy(self.lstm_c[#self.lstm_c])
    self.initstate_h:copy(self.lstm_h[#self.lstm_h])

    -- clip gradient element-wise
    self.gradParameters:clamp(-5, 5)
end



function TrackNet:forward(inputs)
    -- inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    -- features maxm x hist_len x n
        local img = inputs.img
        local rois = inputs.rois
        local targets = inputs.targets

        ------------------- forward pass -------------------
        self.lstm_c = {[0]=self.initstate_c} -- internal cell states of LSTM
        self.lstm_h = {[0]=self.initstate_h} -- output values of LSTM
    
        --self.observation = {}         -- input observation
        --self.predictions = {}         -- dqn outputs
        self.observation = self.cnn:forward{img:cuda(),rois:cuda()}
        --print('cnn forward compelecated!')
        --print('feature:')
        --print(self.observation)
        self.scores      = self.classifier:forward(self.observation)
        -- observation maxm x n
        
        self.features = {}
        
        --print(self.features[{{},{1},{}}]:size())
        --print(self.minibatch_size)

        --print(self.props:size())
        for t=1,self.hist_len do
            self.features[t],self.props = unpack(self.fusionnet:forward{inputs.features[{{},{t},{}}]:squeeze():cuda(),self.observation,self.filter})

            self.lstm_c[t], self.lstm_h[t] = unpack(self.clones.lstm[t]:forward{self.features[t]:cuda(),self.props:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]})
            --self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
        end
        self.predictions = self.clones.dqn[self.hist_len]:forward(self.lstm_h[self.hist_len])
end


function TrackNet:backward(inputs,targets)
    
    -- target maxm * (maxm + 1)
    -- zero gradients of parameters
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    self.gradParameters:zero()
    self.cnndw:zero()
    local img = inputs.img
    local rois = inputs.rois
    local cnntargets = inputs.targets:cuda()
  ----------------- backward pass -------------------
    -- complete reverse order of the above
    local dobservation = {}  
    local dobservationAll
    local dfeatures     = {} 
    local dfeature    = {}    
    local dprops    = {} 
    local dfilter   = {}                          		-- d loss / d input observation
    local dlstm_c = {[self.hist_len]=self.dfinalstate_c}   	-- internal cell states of LSTM
    local dlstm_h = {}                                		-- output values of LSTM
    --local targets = {}
    local dactions = torch.Tensor(self.maxm,self.maxm+1):zero():cuda()
    --local targets, dmask = unpack(self.lnet:backward({self.predictions,cnntargets},{target,dactions}))


    --self.features = inputs.features:clone():transpose(1,2)

    for t=self.hist_len,1,-1 do

        -- backprop through loss/target, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.dqn[t]:backward(self.lstm_h[t], targets:cuda())
        --else
        --    dlstm_h[t]:add(self.clones.dqn[t]:backward(self.lstm_h[t], targets[t]))
        end

        -- backprop through LSTM timestep
           --local feature = torch.Tensor(self.maxm+1,self.feat_size):zero()
        
            dfeatures[t],dprops[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
            {self.features[t]:cuda(),self.props:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}))
            --print(dfeatures:size())
            dfeature[t], dobservation[t],dfilter[t] = unpack(self.fusionnet:backward({inputs.features[{{},{t},{}}]:squeeze():cuda(),self.observation,self.filter},{dfeatures[t],dprops[t]}))
        if t == self.hist_len then
            dobservationAll = dobservation[t]:clone() 
        else
            dobservationAll:add(dobservation[t])
        end
    end
    -- train cnn or not
    dobservationAll:mul(1/self.hist_len)
    self.cnn:backward({img:cuda(),rois:cuda()},dobservationAll:cuda())
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    self.initstate_c:copy(self.lstm_c[#self.lstm_c])
    self.initstate_h:copy(self.lstm_h[#self.lstm_h])

    -- clip gradient element-wise
    self.gradParameters:clamp(-5, 5)
    self.cnndw:clamp(-5,5)
end


-- borrowed from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
function TrackNet:_sanitize(network)
  --net = self.model
  --[[self.initstate_c = nil
  self.initstate_h = nil
  self.observation = nil
  self.lstm_h = nil
  self.lstm_c = nil
  self.scores = nil
  self.features = nil
  self.props    = nil
  self.predictions = nil
  self.cnndw       = nil
  self.gradParameters = nil]]
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



