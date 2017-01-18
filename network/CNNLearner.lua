--require 'nn'
--require 'nngraph'

local CNNLearner = torch.class('tracking.CNNLearner')

-- initial trainer 
function CNNLearner:__init(args)
    -- state contains three parts
    self.maxm    = args.maxm
	
    --- learing rate annealing
    self.lr_start       = args.lr or 0.01
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 100000
    self.wc             = args.wc or 0  -- L2 weight cost

    -- in this work, the minibatch size is seted as 1
    self.minibatch_size = args.minibatch_size  or 1      
    --self.valid_size     = self.minibatch_size

    --- Reinforcement learning parameters
    self.discount       = args.discount or 0.99 -- what is this?

    -- Number of steps after which learning starts
    self.learn_start    = args.learn_start or 0

    self.clip_delta     = args.clip_delta
    self.gpu            = args.gpu

    self.cnn        = args.cnn

    self._softmax = cudnn.SoftMax():cuda()
    self._criterion = nn.CrossEntropyCriterion():cuda()
    self.train      = args.train
    --
    
    --
    self.t = args.t or 10
    self.numSteps = 1

    self.cnnw,self.cnndw = self.cnn:getCNNParameters()
    self.cnndw:zero()

    self.dwc = self.cnndw:clone():zero()
    self.cnndeltas = self.cnndw:clone():fill(0)
    self.cnng      = self.cnndw:clone():fill(0)
    self.cnntmp    = self.cnndw:clone():fill(0)
    self.cnng2     = self.cnndw:clone():fill(0)

    print('Number of all parameters: '..self.cnnw:nElement())

end


function CNNLearner:reset(state)
    
end


function CNNLearner:setData(dataloader)

    --self.dataloader = {}
    self.dataloader = dataloader
    self.cur_state,self.term,self._cur_image = self.dataloader:getNextState()
end


function CNNLearner:updateCNN()
    local t = math.max(0,self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr,self.lr_end)
    
    -- SGD

    self.cnng:mul(0.99):add(0.01, self.dwc)
    --self.deltas:mul(0):addcdiv(self.lr, self.dwc, self.tmp)
    self.cnndeltas:mul(0):add(self.lr,self.cnng)
    self.cnnw:add(-self.cnndeltas)
    self.dwc:zero()
    -- body

    --[[ RMSProp
    self.cnntmp:cmul(self.dwc, self.dwc)
    self.cnng2:mul(0.95):add(0.05, self.cnntmp)

    self.cnntmp:zero():add(self.cnng2)
    self.cnntmp:add(1e-8)
    self.cnntmp:sqrt()

    self.cnndeltas:mul(0):addcdiv(self.lr, self.dwc, self.cnntmp)
    self.cnnw:add(self.cnndeltas)
    self.dwc:zero()]]
end

function CNNLearner:preData(state)
    local img = state.img:clone()
    local maxm = state._rois
    local rois = torch.FloatTensor(maxm,5)
    local targets = torch.Tensor(maxm,1)
    local rois_id_map = {}
    for id,boxInfo in pairs(state.rois) do
        rois[{#rois_id_map+1,1}] = 1
        rois[{{#rois_id_map+1},{2,5}}] = boxInfo.box[{{1,4}}]
        targets[#rois_id_map+1][1] = boxInfo.gt

        table.insert(rois_id_map,#rois_id_map+1,id)
    end

    return img,rois,targets,rois_id_map


end


function CNNLearner:oneStepClassifier()

    --local features = self.featurepool:getFeatures()
    
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)

    local features,scores = self.cnn:cnnforward{img:cuda(),rois:cuda()}

    -- add weight cost to gradient  self.dw = self.dw - self.wc*self.w
    local _rois = rois:size(1)
    self.c_results = self._softmax:forward(scores)
    local y = torch.Tensor(_rois):zero()
    y = targets[{{},{1}}]
    --y = y:mul(-1):add(2)
    local loss = self._criterion:forward(self.c_results,y:cuda())
    local tmp,prediction = self.c_results:max(2)
    local count = 0
    
    for i = 1, _rois do
       if prediction[i][1] == y[i][1] then count = count + 1 end
    end
    --print(count)
    if self.numSteps%20 == 0 then
        local logInfo2 = string.format('Step: %d ==> Acc is %f, loss %f, lr %f',self.numSteps,count/prediction:size(1),loss,self.lr)
        print(logInfo2)
        --cmd:log('./log/CLS_201611102229.log',{logInfo})
    end
    local dresults = self._criterion:backward(self.c_results,y:cuda())
    local dscores = self._softmax:backward(scores,dresults)
    
    self.cnn:cnnbackward({img:cuda(),rois:cuda()},dscores)
    self.cnndw:add(-self.wc, self.cnnw)
    self.dwc:add(self.cnndw:mul(1.0))
	
	if self.numSteps%self.t == 0 then
		self:updateCNN()
	end
	self.numSteps = self.numSteps + 1
	local state2, term2, fr2 = self.dataloader:getNextState()
	
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2
	return term2
end
