--require 'nn'
--require 'nngraph'
require 'utils.KM'

local DDRLLearner = torch.class('tracking.DDRLLearner')

-- initial trainer 
function DDRLLearner:__init(args)
    -- state contains three parts
    -- 1, tfeature, is the cnn feature of the target
    -- 2, single frame raw image
    -- 3, rois of this frame
    self.feat_size = args.feat_size
    self.input_size = args.input_size
    self._n_rois    = args._n_rois
    self.maxm       = args.maxm 
    self.hist_len   = args.hist_len
    -- number of actions is defined as the number of rois
    -- the action is selecting one of the proposels 
    --self.n_actions    = #args.actions
    
    self.best         = args.best or 0

    --- epsilon annealing
    self.ep_start     = args.ep or 0.99
    self.ep           = self.ep_start -- Exploration probability.
    self.ep_end       = args.ep_end or 0.00
    self.ep_endt      = args.ep_endt or 3000
	
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

    self.hist_len       = args.hist_len or 1

    self.clip_delta     = args.clip_delta
    self.target_q       = true
    self.bestq          = 0
	
    self.gpu            = args.gpu

    self.ddlnet        = args.ddlnet
    self.cnn           = args.cnn

    self.AlexNet      = args.AlexNet or nil
    self.rnn_size       = args.rnn_size
    
    self._softmax = cudnn.SoftMax():cuda()
    self._criterion = nn.CrossEntropyCriterion():cuda()
    self.train      = args.train
    --

    self.ratio     = 0.5   -- distance vs R
    self.maxV      = 8
    self.minV      = 0.125  -- volume change ratio
    self.nX        = 0.5 -- greedy probability
    --
    self.t = args.t or 10
    self.T = args.T or 25
    self.numSteps = 1

    self.q_max = 1
    self.r_max = 1
    self.dqn = {}
    self.val = {}
    self.dqn.w, self.dqn.dw = self.ddlnet:getDQNParameters()
    self.dqn.dw:zero()
    self.dqn.dwt    = self.dqn.dw:clone():zero()
    self.dqn.deltas = self.dqn.dw:clone():fill(0)
    self.dqn.tmp    = self.dqn.dw:clone():fill(0)
    self.dqn.g      = self.dqn.dw:clone():fill(0)
    self.dqn.g2     = self.dqn.dw:clone():fill(0)

    self.val.w, self.val.dw = self.ddlnet:getValParameters()
    self.val.dw:zero()
    self.val.dwt    = self.val.dw:clone():zero()
    self.val.deltas = self.val.dw:clone():fill(0)
    self.val.tmp    = self.val.dw:clone():fill(0)
    self.val.g      = self.val.dw:clone():fill(0)
    self.val.g2     = self.val.dw:clone():fill(0)

      
    --self.cnnw,self.cnndw = self.network:getCNNParameters()
	
    print('Number of all parameters of policy network: '..self.dqn.w:nElement())
    print('Number of all parameters of value network: '..self.val.w:nElement())
    if self.target_q then
        self.t_ddlnet = self.ddlnet:clone()
        self.t_dqn = {}
        self.t_val = {}
        self.t_dqn.w,self.t_dqn.dw = self.t_ddlnet:getDQNParameters()
        self.t_val.w,self.t_val.dw = self.t_ddlnet:getValParameters()
        --self.tcnnw,self.tcnndw = self.target_network:getCNNParameters() 
    end
end


function DDRLLearner:reset(state)
    
end


function DDRLLearner:setData(memoryPool)
    self.memoryPool = memoryPool
    self.dataloader = memoryPool.dataloader 
    --self.network:training()
    self.cur_state,self.term,self._cur_image = self.dataloader:getNextState()
    self.total_reward = 0
    self.last_reward = 0
    self.value_loss  = 0
    self.name = self.memoryPool.name
end

function DDRLLearner:RMSProp(args,opts)
    local a = opts.a or 0.95
    --local b = opts.b or 0.95

    args.tmp:cmul(args.dwt, args.dwt)
    args.g2:mul(a):add(1-a, args.tmp)
    args.tmp:zero():add(args.g2)
    args.tmp:add(1e-8)
    args.tmp:sqrt()

    args.deltas:mul(0):addcdiv(opts.lr, args.dwt, args.tmp)
    args.w:add(args.deltas)
    args.dwt:zero()
end

function DDRLLearner:Adam(args,opts)
    local a = opts.a or 0.95
    local b = opts.b or 0.95

    -- use gradients
    args.g:mul(a):add(1-a, args.dwt)
    args.tmp:cmul(args.dwt, args.dwt)
    args.g2:mul(b):add(1-b, args.tmp)
    args.tmp:cmul(args.g, args.g)
    args.tmp:mul(-1)
    args.tmp:add(args.g2)
    args.tmp:add(1e-8)
    args.tmp:sqrt()
    
    -- accumulate update
    -- self.deltas = self.deltas + self.lr*self.dw/self.tmp
    args.deltas:mul(0):addcdiv(opts.lr, args.dwt, args.tmp)
    args.w:add(args.deltas)
    args.dwt:zero()
end

function DDRLLearner:SGD(args,opts)
    local a = opts.a or 0.9
    args.g:mul(a):add(1-a, args.dwt)
    args.deltas:mul(0):add(opts.lr,args.g)
    args.w:add(args.deltas)
    args.dwt:zero()
end

function DDRLLearner:updateNet(args,method)
    local t = math.max(0,self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr,self.lr_end)
    
    --[[]]
    local opts = {}
    opts.lr = self.lr
    opts.a = 0.95
    opts.b = 0.95

    if method == 'SGD' then
        self:SGD(args,opts)
    elseif method == 'ADAM' then
        self:Adam(args,opts)
    else
        self:RMSProp(args,opts)
    end
    --
end

function DDRLLearner:updateTargetNet(a)
    local a = a or 0.99 
    self.t_dqn.w:mul(a):add(1-a,self.dqn.w)
    self.t_val.w:mul(a):add(1-a,self.val.w)
end

function DDRLLearner:getProps(trajBoxes,lt,rois)
    -- return candidates with a distance threshold r
    local box = trajBoxes[self.hist_len]
    local threshold = (box[3] - box[1])*(box[3]-box[1])+(box[4] - box[2])*(box[4]-box[2])
    
    threshold = threshold*(lt/20+1)/self.ratio
    local centor_x = box[1] + (box[3] - box[1])/2
    local centor_y = box[2] + (box[4] - box[2])/2

    local volume1  = (box[3] - box[1])*(box[4] - box[2])
    local candIDs = {}
    local candNum = 0
    for id, cbox in pairs(rois) do
        local tmp_x = cbox[1] + (cbox[3] - cbox[1])/2
        local tmp_y = cbox[2] + (cbox[4] - cbox[2])/2
        local distance = (tmp_x - centor_x)*(tmp_x - centor_x) + (tmp_y - centor_y)*(tmp_y - centor_y)

        local volume2 = (cbox[3] - cbox[1])*(cbox[4] - cbox[2])
        local volumeRate = volume1/volume2
        if distance < threshold and volumeRate < (lt/20+1)*self.maxV and volumeRate > self.minV/(lt/20+1) then
            candIDs[#candIDs+1] = id
            --table.insert(candIDs,#candIDs+1,id)
            candNum = candNum + 1
        end
    end
    
    return candIDs,candNum
end

function DDRLLearner:trainBatch(fr,trajsFeatures,candIDs,totalCands,candFeatures)
    local trajs = torch.Tensor(#trajsFeatures,self.hist_len,self.feat_size):zero():cuda()

    for i = 1, #trajsFeatures do
        trajs[{{i},{},{}}] = trajsFeatures[i]
    end

    -- build weight matrix for KM match
    
    local indexIdMap = {}
    local idIndexMap = {}
    local indexId = 1
    for candId in pairs(candFeatures) do
        indexIdMap[indexId] = candId
        idIndexMap[candId]  = indexId
        indexId = indexId + 1
    end

    local N = math.max(#trajsFeatures,indexId-1)
    local love = torch.Tensor(N,N):zero()
    -- test KM
    --testKM()
    -------

    --print(trajs:size())
    local vals = self.ddlnet:valforward(trajs)

    local candTrajs = torch.Tensor(totalCands,self.hist_len,self.feat_size):zero():cuda()
    local cands = torch.Tensor(totalCands,self.feat_size):zero():cuda()
    local candIdMap = torch.Tensor(totalCands,2):zero()
    local index = 1
    for trajId,ids in pairs(candIDs) do
        for i,id in pairs(ids) do

            candTrajs[{{index},{},{}}] = trajsFeatures[trajId]
            cands[{{index},{}}] = candFeatures[id]
            candIdMap[index][1] = trajId
            candIdMap[index][2] = id
            index = index + 1
        end
    end

    local t_pis = self.t_ddlnet:dqnforward({candTrajs,cands})
    local pis = self.ddlnet:dqnforward({candTrajs,cands})


    local m = 1
    if type(t_pis[1]) == 'number' then 
        m = 1
    else
        m = pis:size(1)
    end
    --print('pis size and candIdMap size',pis:size(),candIdMap)
    for i = 1, m do
        local trajId = candIdMap[i][1]
        local indexId = idIndexMap[candIdMap[i][2]]
        local predict = 0
        if type(t_pis[i]) == 'number' then
             predict = t_pis[1]*1000000
        else
             predict = t_pis[i][1]*1000000
        end
	self.ep_threshold = self.ep_end + math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt - math.max(0, self.numSteps - self.learn_start))/self.ep_endt)
        local noise  = torch.random(0,math.floor(1000000*self.ep_threshold))
	local noise = 0
        love[trajId][indexId] = math.floor(predict+noise)
    end

    --local t1 = os.clock()
    local res, match = KM(love)
    --local t2 = os.clock() - t1
    --print(string.format('For %d x　%d match, cost %f secs,got res %f',love:size(1),love:size(1),t2,res))
    local selecteds = torch.Tensor(#trajsFeatures):zero()

    for i = 1, match:size(1) do
        if love[match[i]][i] > 0 then
            selecteds[match[i]] = indexIdMap[i]
        end
    end

    local rewards   = torch.FloatTensor(#trajsFeatures,1):zero():cuda()
    local candRewards = torch.FloatTensor(totalCands,2):zero():cuda()
    local mask = torch.Tensor(totalCands,2):zero():cuda()
    local bias = 1
    local n = 0
    for trajId,ids in pairs(candIDs) do
        n = #ids
        local actions = {}
        --print(n)
        if n ~= 0 then
            for i,id in pairs(ids) do
                if id ~= selecteds[trajId] then
                    actions[id] = -1
                else
                    actions[id] = 1
                end
            end
        end
    
        local count,reward = self.memoryPool:execActions(trajId,fr,actions,selecteds[trajId])
        
        local re = 0
        local tmpi = 1
        for i,id in pairs(ids) do
            local ind = bias+tmpi-1
            local Sid = candIdMap[ind][2]
            assert(reward[Sid]~=nil)
            if Sid ~= selecteds[trajId] then               
                --re = re + reward[Sid]/count
                candRewards[ind][1] = reward[Sid]
                candRewards[ind][2] = reward[Sid]
		--print(reward[Sid])
                mask[ind][1] = 0
                mask[ind][2] = 0.1/count
                if reward[Sid] < 0 then
                   mask[ind][2] = 10
                end
            else
                re = re + reward[Sid]
                candRewards[ind][1] = reward[Sid]
                candRewards[ind][2] = reward[Sid]
                mask[ind][1] = 1
                mask[ind][2] = 0
            end     
            tmpi = tmpi + 1  
        end
        rewards[trajId][1] = re
        self.total_reward = self.total_reward + re
        bias = bias + n
    end

    -- train state-value net
    local _trajs = trajs:clone()
    for i = 1, #trajsFeatures do
        if selecteds[i] ~= 0 then
            _trajs[{{i},{1,self.hist_len-1},{}}] = _trajs[{{i},{2,self.hist_len},{}}]
            _trajs[{{i},{self.hist_len},{}}] = candFeatures[selecteds[i]]:clone()
        end
    end

    local t_vals = self.t_ddlnet:valforward(_trajs)
    local t_vals2 = self.t_ddlnet:valforward(trajs)
    local dvals = t_vals:clone()
    dvals:mul(self.discount):add(rewards)
    local _dvals = vals:clone()
    _dvals:mul(-1):add(dvals)
    
    if self.clip_delta then
        _dvals:clamp(-self.clip_delta, self.clip_delta)
    end
    local tmp = _dvals:clone()
    tmp:cmul(_dvals,_dvals)
    self.value_loss = self.value_loss+tmp:sum()
    self.ddlnet:valbackward(trajs,_dvals)
    self.val.dwt:add(self.val.dw)

    -- train policy network

    local A = torch.Tensor(totalCands,2):zero():cuda()
    local B = torch.Tensor(totalCands,2):zero():cuda()
    for i=1,totalCands do
	    if candIdMap[i][2] == selecteds[candIdMap[i][1]] then
	        A[i][1] = t_vals[candIdMap[i][1]][1]
            A[i][2] = t_vals[candIdMap[i][1]][1]
        else
            A[i][1] = vals[candIdMap[i][1]][1]/self.discount
            A[i][2] = vals[candIdMap[i][1]][1]/self.discount
	    end
        B[i][1] = vals[candIdMap[i][1]][1]
        B[i][2] = vals[candIdMap[i][1]][1]
    end
    A:mul(self.discount):add(candRewards)
    A:mul(-1):add(B)
    A:mul(-1)

    local dpis = pis:clone()
    local _dpis = pis:clone():add(1e-8)
    dpis:mul(0):addcdiv(1,A,_dpis)
    dpis:cmul(mask)
    if self.clip_delta then
        dpis:clamp(-self.clip_delta, self.clip_delta)
    end
    self.ddlnet:dqnbackward({candTrajs,cands},dpis)
    self.dqn.dwt:add(self.dqn.dw)
    
    -----------------
    return selecteds
end
---------------------------------------------------------------------------------------
function DDRLLearner:testBatch(fr,trajsFeatures,candIDs,totalCands,candFeatures,candRois)
    local trajs = torch.Tensor(#trajsFeatures,self.hist_len,self.feat_size):zero():cuda()
    local selecteds = torch.Tensor(#trajsFeatures):zero()
    if totalCands == 0 then return selecteds end
    
    for i = 1, #trajsFeatures do
        trajs[{{i},{},{}}] = trajsFeatures[i]
    end

    -- build weight matrix for KM match
    
    local indexIdMap = {}
    local idIndexMap = {}
    local indexId = 1
    for candId in pairs(candFeatures) do
        indexIdMap[indexId] = candId
        idIndexMap[candId]  = indexId
        indexId = indexId + 1
    end

    local N = math.max(#trajsFeatures,indexId-1)
    local love = torch.Tensor(N,N):zero()

    local candTrajs = torch.Tensor(totalCands,self.hist_len,self.feat_size):zero():cuda()
    local cands = torch.Tensor(totalCands,self.feat_size):zero():cuda()
    local candIdMap = torch.Tensor(totalCands,2):zero()
    local index = 1
    for trajId,ids in pairs(candIDs) do
        for i,id in pairs(ids) do

            candTrajs[{{index},{},{}}] = trajsFeatures[trajId]
            cands[{{index},{}}] = candFeatures[id]
            candIdMap[index][1] = trajId
            candIdMap[index][2] = id
            index = index + 1
        end
    end
    local pis = self.ddlnet:dqnforward({candTrajs,cands})
    local m = 1
    if type(pis[1]) == 'number' then 
        m = 1
    else
        m = pis:size(1)
    end
    --print('pis size and candIdMap size',pis:size(),candIdMap)
    for i = 1, m do
        local trajId = candIdMap[i][1]
        local indexId = idIndexMap[candIdMap[i][2]]
        local predict = 0
        if type(pis[i]) == 'number' then
             predict = pis[1]*1000000
        else
             predict = pis[i][1]*1000000
        end
        love[trajId][indexId] = math.floor(predict)
    end

    local t1 = os.clock()
    local res, match = KM(love)
    local t2 = os.clock() - t1
    print(string.format('For %d x　%d match, cost %f secs,got res %f',love:size(1),love:size(1),t2,res))
    local selecteds = torch.Tensor(#trajsFeatures):zero()

    for i = 1, match:size(1) do
        if love[match[i]][i] > 0 then
            selecteds[match[i]] = indexIdMap[i]
        end
    end

    local rewards   = torch.FloatTensor(#trajsFeatures,1):zero():cuda()
    local candRewards = torch.FloatTensor(totalCands,2):zero():cuda()
    local mask = torch.Tensor(totalCands,2):zero():cuda()
    local n = 0
    for trajId,ids in pairs(candIDs) do
        n = #ids
        local actions = {}
        --print(n)
        if n ~= 0 then
            for i,id in pairs(ids) do
                if id ~= selecteds[trajId] then
                    actions[id] = -1
                else
                    actions[id] = 1
                end
            end
        end
    
        self.memoryPool:execActionsTest(trajId,fr,actions,selecteds[trajId],candRois)
    end
    -----------------
    return selecteds
end
---------------------------------------------------------------------------------------

function DDRLLearner:preData(state)
    local img = state.img:clone()
    self.maxm = state._rois
    local rois = torch.FloatTensor(self.maxm,5)
    local targets = torch.Tensor(self.maxm,1)
    local rois_id_map = {}
    for id,boxInfo in pairs(state.rois) do
        rois[{#rois_id_map+1,1}] = 1
        rois[{{#rois_id_map+1},{2,5}}] = boxInfo.box[{{1,4}}]
        targets[#rois_id_map+1][1] = boxInfo.gt

        --table.insert(rois_id_map,#rois_id_map+1,id)
        rois_id_map[#rois_id_map+1] = id
    end

    return img,rois,targets,rois_id_map

end

function DDRLLearner:oneStep()
    --local t1 = os.clock()
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)
    for i = 1,rois:size(1) do
        local re = torch.uniform()
        if re < self.nX then
            --print(img:size())
            local H = img:size(3)
            local W = img:size(4)
            local roiW = rois[i][4] - rois[i][2]
            local roiH = rois[i][5] - rois[i][3]

            --print(string.format('roiW %f,W %f, roiH %f, H %f, old (x1,x2)=(%f,%f),(y1,y2)=(%f,%f)',roiW,W,roiH,H,rois[i][2],rois[i][4],rois[i][3],rois[i][5]))
            local Cx   = rois[i][2] + roiW/2
            local Cy   = rois[i][3] + roiH/2
            local noiseCx = (torch.uniform()*2 - 1)/20 -- [-0.05,0.05]
            local noiseCy = (torch.uniform()*2 - 1)/20 -- [-0.05,0.05]
            local noiseW = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
            local noiseH = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
            roiW  = roiW*(1+noiseW)
            roiH  = roiH*(1+noiseH)
            Cx  = Cx+noiseCx*roiW
            Cy  = Cy+noiseCy*roiH
            
            rois[i][2] = Cx - roiW/2
            if rois[i][2] < 1 then rois[i][2] = 1 end
            rois[i][3] = Cy - roiH/2
            if rois[i][3] < 1 then rois[i][3] = 1 end
            rois[i][4] = Cx + roiW/2
            if rois[i][4] > W then rois[i][4] = W end
            rois[i][5] = Cy + roiH/2
            if rois[i][5] > H then rois[i][5] = H end
            --print(string.format('at frame %d,id %d, noiseCx %f, noiseCy %f, noiseW %f, noiseH %f,  new (x1,x2)=(%f,%f),(y1,y2)=(%f,%f)',self._cur_image,rois_id_map[i],noiseCx,noiseCy,noiseW,noiseH, rois[i][2],rois[i][4],rois[i][3],rois[i][5]))
            
        end
    end
    -- cnn forward extract features and clssify the candidates 1 human 2 unhuman
    local cand_features,scores = self.cnn:cnnforward{img:cuda(),rois:cuda()}
    local c_results = self._softmax:forward(scores)
    local tmp, pre  = c_results:max(2)
    local candFeat = {}
    local candRois = {}
    
    local allNum = 0
    for i = 1, pre:size(1) do
        --if pre[i][1] == 1 and cand_features[i]~= nil and targets[i][1] == 1 then
        --if targets[i][1] == 1 then
            allNum = allNum + 1
            local id = rois_id_map[i]
            candFeat[id] = cand_features[i]
            --candRois[id] = self.cur_state.rois[id].box:clone()
            candRois[id] = rois[i][{{2,5}}]:clone()
        --end
    end
    --print(string.format('Found %d humen in frame %d with time cost %.2f\n !!!',allNum,self._cur_image,t2-t1))

    local frame = {}
    frame.img = self.cur_state.img:clone()
    frame.bboxes = {}

    for id,box in pairs(candRois) do
        local boxInfo = {}
        boxInfo.box = box:clone()
        boxInfo.feature = candFeat[id]:clone()
        boxInfo.lifetime = 0
        frame.bboxes[id] = boxInfo
    end

    local features,bboxes,lts = self.memoryPool:getFeatures()
    local maxId = 0
    for id in pairs(features) do
        if maxId < id then maxId = id end
    end
    -- got features and bboxes
    local n = #features
    assert(n==maxId)
    local candIDs = {}
    local totalCands = 0
    --local cand_feats = {}
    for trajId, trajFeat in pairs(features) do
        local t3 = os.clock()
        local candID,candNum = self:getProps(bboxes[trajId],lts[trajId],candRois)
        candIDs[trajId] = candID
        totalCands = totalCands + candNum
    end

    local selecteds = self:trainBatch(self._cur_image,features,candIDs,totalCands,candFeat)
    for trajId in pairs(features) do
        local id = selecteds[trajId]
        if id ~= 0 then
            frame.bboxes[id].lifetime = frame.bboxes[id].lifetime + 1
        end
    end
    self.memoryPool:updateNewFrame(self._cur_image,frame)
    
    if self.numSteps %self.t == 0 then
        self:updateNet(self.val,'ADAM')
        self:updateNet(self.dqn,'ADAM')
    end

    if self.numSteps %self.T == 0 then
        self:updateTargetNet(0.99)
    end

    if self.numSteps %20 == 0 then
        print(string.format("<%s>, frame <%d>, got reward <%f>, lr = <%f>,trajs <%d>,loss <%f>",self.name,self._cur_image,self.total_reward-self.last_reward,self.lr,self.memoryPool.curm,self.value_loss/20))
        self.value_loss = 0
        self.last_reward = self.total_reward
    end
    self.numSteps = self.numSteps + 1
    local state2, term2, fr2 = self.dataloader:getNextState()
    if term2 == 1 then
        print(string.format("<%s>, frame <%d>, reward <%f>, lr = <%f>,trajs <%d>",self.name,self._cur_image,self.total_reward,self.lr,self.memoryPool.curm))
        return term2
    end
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2
    return term2
end

function DDRLLearner:IoU(box1,box2)
    --boxIoU caculate IoU of two bounding boxes
    local ax1 = box1[1]; local ax2 = box1[3];
    local ay1 = box1[2]; local ay2 = box1[4];

    local bx1 = box2[1]; local bx2 = box2[3];
    local by1 = box2[2]; local by2 = box2[4];
    
    local hor = math.min(ax2,bx2) - math.max(ax1,bx1)

    local ver = 0
    if hor > 0 then
        ver = math.min(ay2,by2) - math.max(ay1,by1)
    end
    if ver < 0 then ver = 0 end
    local isect = hor*ver

    local aw = ax2 - ax1
    local ah = ay2 - ay1
    local bw = bx2 - bx1
    local bh = by2 - by1
    local union = aw*ah + bw*bh - isect
    if union <= 0 then return 0 end

    return isect / union
end

function DDRLLearner:findHighestScoreId(scores)
    local maxScore = 0
    local maxId    = 0
    for id, score in pairs(scores) do
        if score > maxScore then
            maxId = id
            maxScore = score
        end
    end
    return maxId
end

function DDRLLearner:NMS1(candRois,scores)
    local count = 0
    for id, score in pairs(scores) do
        count = count + 1
    end
    while count ~= 0 do
        local hi = self:findHighestScoreId(scores)
        self:NMS2(hi,candRois,scores)
        count = 0
        for id, score in pairs(scores) do
            count = count + 1
        end
    end
end

function DDRLLearner:NMS2(hi,candRois,scores)
    local hbox = candRois[hi]
    scores[hi] = nil
    for id,box in pairs(candRois) do
        if id ~= hi then
            local iou = self:IoU(hbox,box)
            if iou > 0.3 then
                candRois[id] = nil
                scores[id] = nil
            end
        end
    end
end

function DDRLLearner:smoothBoxes(img,rois,n)
    local H = img:size(3)
    local W = img:size(4)
    local roiW = rois[4] - rois[2]
    local roiH = rois[5] - rois[3]
    local Cx   = rois[2] + roiW/2
    local Cy   = rois[3] + roiH/2
    local boxes = {}
    local box = torch.Tensor(5):zero()
    for i = 1, n do

        local noiseCx = (torch.uniform()*2 - 1)/20 -- [-0.05,0.05]
        local noiseCy = (torch.uniform()*2 - 1)/20 -- [-0.05,0.05]
        local noiseW = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
        local noiseH = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
        local nroiW  = roiW*(1+noiseW)
        local nroiH  = roiH*(1+noiseH)
        nCx  = Cx+noiseCx*nroiW
        nCy  = Cy+noiseCy*nroiH
        
        box[1] = 1
        box[2] = nCx - nroiW/2
        if box[2] < 1 then box[2] = 1 end
        box[3] = nCy - nroiH/2
        if box[3] < 1 then box[3] = 1 end
        box[4] = nCx + nroiW/2
        if box[4] > W then box[4] = W end
        box[5] = nCy + nroiH/2
        if box[5] > H then box[5] = H end 

        boxes[i] = box:clone()
    end
    return boxes
end
function DDRLLearner:smoothBoxes1(img,rois,n)
    local H = img:size(3)
    local W = img:size(4)
    local roiW = rois[3] - rois[1]
    local roiH = rois[4] - rois[2]
    local Cx   = rois[1] + roiW/2
    local Cy   = rois[2] + roiH/2
    local boxes = {}
    local box = torch.Tensor(5):zero()
    for i = 1, n do

        local noiseCx = (torch.uniform()*2 - 1)/10 -- [-0.05,0.05]
        local noiseCy = (torch.uniform()*2 - 1)/10 -- [-0.05,0.05]
        local noiseW = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
        local noiseH = (torch.uniform()*2 - 1)/10 -- [-0.1,0.1]
        local nroiW  = roiW*(1+noiseW)
        local nroiH  = roiH*(1+noiseH)
        nCx  = Cx+noiseCx*nroiW
        nCy  = Cy+noiseCy*nroiH
        
        box[1] = 1
        box[2] = nCx - nroiW/2
        if box[2] < 1 then box[2] = 1 end
        box[3] = nCy - nroiH/2
        if box[3] < 1 then box[3] = 1 end
        box[4] = nCx + nroiW/2
        if box[4] > W then box[4] = W end
        box[5] = nCy + nroiH/2
        if box[5] > H then box[5] = H end 

        boxes[i] = box:clone()
    end
    return boxes
end
function DDRLLearner:TestOneStep()
  --local t1 = os.clock


    self.ratio     = 16   -- distance vs R
    self.maxV      = 4
    self.minV      = 0.25  -- volume change ratio
    self.nX        = 0.25 -- greedy probability



    local sn = 5
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)
    local maxMapId = 0
    local features,bboxes,lts = self.memoryPool:getFeatures()
    local n = #features
    local n_rois = rois:size(1)

    local nrois = rois
    --[[for mapI,mapId in pairs(rois_id_map) do
        if mapId > maxMapId then maxMapId = mapId end
    end
    
    
    local roisNum = rois:size(1)
    local numWithSmooth = n_rois*sn + roisNum
    local nrois = torch.Tensor(numWithSmooth,5)
    
    nrois[{{1,roisNum},{}}] = rois
    
    local sI = 1
    local tbefore = os.clock()
    for i = 1,n_rois do
        local smoothBboxes = self:smoothBoxes(img,rois[i],sn)
        for id,smoothRois in pairs(smoothBboxes) do
            nrois[{{roisNum+sI},{}}] = smoothRois
            rois_id_map[#rois_id_map+1] = maxMapId + sI
            sI = sI + 1
        end
    end
    --print(sI)
    local tafter = os.clock()
    print(string.format('Have %d trajectories and %d rois, Smooth bounding boxes costed %f secs--------->',n,rois:size(1),tafter-tbefore))
    -- cnn forward extract features and clssify the candidates 1 human 2 unhuman]]

     --[[for mapI,mapId in pairs(rois_id_map) do
        if mapId > maxMapId then maxMapId = mapId end
    end
    
    
    local roisNum = rois:size(1)
    local numWithSmooth = n*sn + roisNum
    local nrois = torch.Tensor(numWithSmooth,5)
    
    nrois[{{1,roisNum},{}}] = rois
    
    local sI = 1
    local tbefore = os.clock()
    for trajId,tranBox in pairs(bboxes) do
        local smoothBboxes = self:smoothBoxes1(img,tranBox[self.hist_len],sn)
        for id,smoothRois in pairs(smoothBboxes) do
            nrois[{{roisNum+sI},{}}] = smoothRois
            rois_id_map[#rois_id_map+1] = maxMapId + sI
            sI = sI + 1
        end
    end
    --print(sI)
    local tafter = os.clock()
    print(string.format('Have %d trajectories and %d rois, Smooth bounding boxes costed %f secs--------->',n,rois:size(1),tafter-tbefore))
    -- cnn forward extract features and clssify the candidates 1 human 2 unhuman]]


    tbefore = os.clock()
    local cand_features,scores = self.cnn:cnnforward{img:cuda(),nrois:cuda()}
    local tmp_features,Ascores = self.AlexNet:cnnforward{img:cuda(),nrois:cuda()}
    local c_results = self._softmax:forward(Ascores)
    
    tafter = os.clock()

    local tmp, pre  = c_results:max(2)
    --print(c_results[1])
    --print(pre[1])
    local candFeat = {}
    local candRois = {}
    local scores   = {}
    local allNum = 0
    for i = 1, pre:size(1) do
        --print(pre[i])
        --if pre[i][1] == 1 and cand_features[i]~= nil then
        if c_results[i][1] > 0.3 and cand_features[i]~= nil then
        --print('got1')
            local id = rois_id_map[i]
            local width = nrois[i][4] - nrois[i][2]
            local height = nrois[i][5] - nrois[i][3]
            if height/width < 4 and height/width > 1 then 
                allNum = allNum + 1
                candFeat[id] = cand_features[i]
                candRois[id] =  nrois[i][{{2,5}}]:clone()
                scores[id]   = c_results[i][1]
            end
        end
    end
 
    self:NMS1(candRois,scores)
    local afterNMS = 0
    for lastI,lastId in pairs(candRois) do
        afterNMS = afterNMS + 1
    end
    print(string.format('After NMS has %d/%d cands, cnn costed %f secs========>',afterNMS,allNum,tafter-tbefore))
    --print(string.format('Found %d humen in frame %d with time cost %.2f\n !!!',allNum,self._cur_image,t2-t1))

    local frame = {}
    frame.img = self.cur_state.img:clone()
    frame.bboxes = {}

    for id,box in pairs(candRois) do
        local boxInfo = {}
        boxInfo.box = box:clone()
        boxInfo.feature = candFeat[id]:clone()
        boxInfo.lifetime = 0
        frame.bboxes[id] = boxInfo
    end

    
    local maxId = 0
    for id in pairs(features) do
        if maxId < id then maxId = id end
    end
    assert(n==maxId)
    -- got features and bboxes

    local candIDs = {}
    local totalCands = 0
    for trajId, trajFeat in pairs(features) do
        local t3 = os.clock()
        local candID,candNum = self:getProps(bboxes[trajId],lts[trajId],candRois)
        candIDs[trajId] = candID
        totalCands = totalCands + candNum
    end

    local selecteds = self:testBatch(self._cur_image,features,candIDs,totalCands,candFeat,candRois)
    for trajId in pairs(features) do
        local id = selecteds[trajId]
        if id ~= 0 then
            frame.bboxes[id].lifetime = frame.bboxes[id].lifetime + 1
        end
    end
    self.memoryPool:updateNewFrame(self._cur_image,frame)
    
    local state2, term2, fr2 = self.dataloader:getNextState()
    if term2 == 1 then
        return term2
    end
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2
    return term2
end

function DDRLLearner:eGreedy(predictions)

    self.ep_threshold = testing_ep or (self.ep_end + math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt - math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    
    -- epsilon = ep_end + max(0,(ep_start - ep_end)*(ep_endt - max(0, numsteps - learn_start))/ep_endt))
    local n = predictions:size(1)
    local ep = torch.uniform()
    local maxpi,maxid = predictions:max(1) 
    if ep < self.ep_threshold then
        maxid[1][1] = torch.random(1,n)
        maxpi[1][1] = predictions[maxid[1][1]][1]
    end
    return maxpi, maxid
end
