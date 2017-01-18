require 'nn'
require 'nngraph'

--require 'Rectifier'

--local CNN = require 'CNN'
--local LNET = require '../models/Linker/LNET'
local LSTM = require '../models/RNNs/LSTM'
local LSTMOne = require '../models/RNNs/LSTMOne'
--local FusionNet = require '../models/Linker/FusionNet'
local model_utils = require '../utils/model_utils'

local DDRLNet = torch.class('tracking.DDRLNet')
--local utils = detection.GeneralUtils()
function DDRLNet:__init(model_opt)
    -- rnn opts
    self.rnn_size = model_opt.rnn_size
    self.input_size = model_opt.input_size
    self.minibatch_size = model_opt.maxm*(model_opt.maxm+1)  --%%
    self.gpu = model_opt.gpu
    self._n_rois = model_opt._n_rois
    self.hist_len = model_opt.hist_len
    self.maxm     = model_opt.maxm
    self.feat_size= model_opt.feat_size
	--self.model_opt = model_opt  
	
    --if self.model_path == nil then
	--	error 'The first argument can not be nil!'
	--end

    self.filter = torch.Tensor(1,self.feat_size):fill(0):cuda()
    self.rnn = {}
    self.rnn.lstm = LSTM.lstm(model_opt)
	self.rnn.lstmone = LSTMOne.lstm(model_opt)
	self.rnn.dqn = nn.Linear(self.rnn_size,2)    -- pi(a|s,theta)  policy
    self.softmax = cudnn.SoftMax()
	self.rnn.val = nn.Linear(self.rnn_size,1)    -- V(s) state-value


    -- set to gpu
    if self.gpu >= 0 then
        self.rnn.lstm:cuda()
		self.rnn.lstmone:cuda()
        self.rnn.dqn:cuda()
        self.softmax:cuda()
		self.rnn.val:cuda()
        --self.fusionnet:cuda()
    end

        -- set parallel
	--if config.nGPU > 1 and not model_opt.test then
	--	self:_makeParallel()
	--end
        
        
    -- index parameters and gradParameters
    self.dqnw, self.dqndw = model_utils.combine_all_parameters(self.rnn.lstm, self.rnn.dqn)
	self.valw, self.valdw = model_utils.combine_all_parameters(self.rnn.lstmone,self.rnn.val)
	self.dqndw:zero()
	self.valdw:zero()
	--self.parameters, self.gradParameters = model_utils.combine_all_parameters		   	(self.rnn.lstm, self.rnn.dqn, self.lnet)

        
    -- clone for hist_len times
    self.clones = {}
    for name,proto in pairs(self.rnn) do
	    print('cloning '..name)
	    self.clones[name] = model_utils.clone_many_times(proto, self.hist_len, not proto.parameters)
	end

	-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	--[[self.initstate_c = torch.zeros(self.minibatch_size, self.rnn_size)
	self.initstate_h = self.initstate_c:clone()
	if self.gpu >=0 then
		self.initstate_c = self.initstate_c:cuda()
		self.initstate_h = self.initstate_h:cuda()
	end

	-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
	self.dfinalstate_c = self.initstate_c:clone()
	]]

	
	
	-- LSTM initial state for prediction, note that we're using minibatches OF SIZE ONE here
	self.prev_c = torch.zeros(1, self.rnn_size)
	self.prev_h = self.prev_c:clone()
	if self.gpu >=0 then
		self.prev_c = self.prev_c:cuda()
		self.prev_h = self.prev_h:cuda()
	end

	--self.parameters, self.gradParameters = self.model:getParameters()
        -- load weight_file for cnn
end

function DDRLNet:clone()
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

function DDRLNet:getDQNParameters()
	return self.dqnw, self.dqndw
end

function DDRLNet:getValParameters()
	return self.valw, self.valdw
end

function DDRLNet:training()
	self.rnn.lstm:training()
	self.rnn.dqn:training()
    self.softmax:training()
	self.rnn.lstmone:training()
	self.rnn.val:training()
    --self.fusionnet:training()
end

function DDRLNet:evaluate()
	self.rnn.lstm:evaluate()
	self.rnn.dqn:evaluate()
    self.softmax:evaluate()
	self.rnn.lstmone:evaluate()
	self.rnn.val:evaluate()
    --self.fusionnet:evaluate()
end

--[[function DDRLNet:save(save_path)
	-- First sanitize the net
    --local network = self:clone()
	self:_sanitize(self)
	local isParallel = (torch.type(self.model) == 'nn.DataParallelTable')
	if isParallel then
		self.model = self.model:get(1)
	end
	collectgarbage()
	-- Save the snapshot
	torch.save(save_path,self)
	 if isParallel then
	 	self:_makeParallel()
	 end
end]]

function DDRLNet:load_from_caffe(proto_path,caffemodel_path,save_path,model_name)
	caffeModelLoader = tracking.CaffeModelConverter(self.cnn,proto_path,caffemodel_path,model_name,save_path)
	caffeModelLoader:convert()
end

function DDRLNet:get_model()
	return self.rnn
end

function DDRLNet:dqnforward(inputs)
    -- inputs[1] n times trajectory 
    -- inputs[2] n candicates 
	
	local traj = inputs[1]
	local cand = inputs[2]
	
	local minibatch_size = traj:size(1)
	
	-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	
	self.initstate_c = torch.zeros(minibatch_size, self.rnn_size)
	self.initstate_h = self.initstate_c:clone()
	if self.gpu >=0 then
		self.initstate_c = self.initstate_c:cuda()
		self.initstate_h = self.initstate_h:cuda()
	end

	-- LSTMO final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
	self.dfinalstate_c = self.initstate_c:clone()

    ------------------- forward pass -------------------
    self.lstm_c = {[0]=self.initstate_c} -- internal cell states of LSTM
    self.lstm_h = {[0]=self.initstate_h} -- output values of LSTM

    --print(traj:size())
    --print(cand:size())
    for t=1,self.hist_len do
        self.lstm_c[t], self.lstm_h[t] = unpack(self.clones.lstm[t]:forward{traj[{{},{t},{}}]:squeeze():cuda(),cand:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]})
		
            --self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
    end
    self.predictions = self.clones.dqn[self.hist_len]:forward(self.lstm_h[self.hist_len])
    self.pi = self.softmax:forward(self.predictions)
    return self.pi
end

function DDRLNet:dqnbackward(inputs,targets)
    
    -- target maxm * (maxm + 1)
    -- zero gradients of parameters
    --inputs[t] {features, cnnInput {pre_img,rois = {id = boxinfo{box,gt}}}
    self.dqndw:zero()
	
	
	local traj = inputs[1]
	local cand = inputs[2]
    --self.cnndw:zero()                            -- d loss / d input observation
	local dtraj     = {} 
    local dcand    = {} 
    local dlstm_c = {[self.hist_len]=self.dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                      -- output values of LSTM
    local dsoftmax = self.softmax:backward(self.predictions,targets:cuda())
    for t=self.hist_len,1,-1 do

        -- backprop through loss/target, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = self.clones.dqn[t]:backward(self.lstm_h[t], dsoftmax)
        --else
        --    dlstm_h[t]:add(self.clones.dqn[t]:backward(self.lstm_h[t], targets[t]))
        end

        -- backprop through LSTM timestep
        dtraj[t],dcand[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(self.clones.lstm[t]:backward(
        {traj[{{},{t},{}}]:squeeze():cuda(),cand:cuda(), self.lstm_c[t-1], self.lstm_h[t-1]},
        {dlstm_c[t], dlstm_h[t]}))
        --print(dfeatures:size())
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    --self.initstate_c:copy(self.lstm_c[#self.lstm_c])
    --self.initstate_h:copy(self.lstm_h[#self.lstm_h])

    -- clip gradient element-wise
    self.dqndw:clamp(-5, 5)
end


function DDRLNet:valforward(inputs)
	-- LSTMOne initial state (zero initially, but final state gets sent to initial state when we do BPTT)
	
    local minibatch_size = inputs:size(1)
    self.initstate_co = torch.zeros(minibatch_size, self.rnn_size)
	self.initstate_ho = self.initstate_co:clone()
	if self.gpu >=0 then
		self.initstate_co = self.initstate_co:cuda()
		self.initstate_ho = self.initstate_ho:cuda()
	end
	self.dfinalstate_co = self.initstate_co:clone()
    ------------------- forward pass -------------------
	self.lstm_co = {[0]=self.initstate_co} -- internal cell states of LSTM
    self.lstm_ho = {[0]=self.initstate_ho} -- output values of LSTM 
    --print(self.props:size())

    for t=1,self.hist_len do
		self.lstm_co[t], self.lstm_ho[t] = unpack(self.clones.lstmone[t]:forward{inputs[{{},{t},{}}]:squeeze():cuda(), self.lstm_co[t-1], self.lstm_ho[t-1]})
            --self.predictions[t] = self.clones.dqn[t]:forward(self.lstm_h[t])
    end
	self.value = self.clones.val[self.hist_len]:forward(self.lstm_ho[self.hist_len])
    return self.value
end

function DDRLNet:valbackward(inputs,targets)
    -- inputs trajectory

    self.valdw:zero()
    local dinputs     = {}                              -- d loss / d input observation
    local dlstm_co = {[self.hist_len]=self.dfinalstate_co}    -- internal cell states of LSTM
    local dlstm_ho = {}                                      -- output values of LSTM

    for t=self.hist_len,1,-1 do
        -- backprop through loss/target, and DQN/linear
        -- Two cases for dloss/dh_t: 
        --   1. h_T is only used once, sent to the DQN (but not to the next LSTM timestep).
        --   2. h_t is used twice, for the DQN and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == self.hist_len then
            assert(dlstm_ho[t] == nil)
            dlstm_ho[t] = self.clones.val[t]:backward(self.lstm_ho[t], targets:cuda())
        end

        -- backprop through LSTM timestep
        dinputs[t], dlstm_co[t-1], dlstm_ho[t-1] = unpack(self.clones.lstmone[t]:backward(
        {inputs[{{},{t},{}}]:squeeze():cuda(), self.lstm_co[t-1], self.lstm_ho[t-1]},
        {dlstm_co[t], dlstm_ho[t]}))
    end
    
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    --self.initstate_co:copy(self.lstm_co[#self.lstm_co])
    --self.initstate_ho:copy(self.lstm_ho[#self.lstm_ho])

    -- clip gradient element-wise
    self.valdw:clamp(-5, 5)
end
