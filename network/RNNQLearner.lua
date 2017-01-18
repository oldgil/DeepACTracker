--require 'nn'
--require 'nngraph'

local RNNQLearner = torch.class('tracking.RNNQLearner')

-- initial trainer 
function RNNQLearner:__init(args)
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
    self.ep_start     = args.ep or 0.4
    self.ep           = self.ep_start -- Exploration probability.
    self.ep_end       = args.ep_end or 0.0001
    self.ep_endt      = args.ep_endt or 2000

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

    self.network        = args.network
    self.ncnet          = args.ncnet

    self.rnn_size       = args.rnn_size
    
    self.samples        = {}
    self.i_sample       = 1
    self.batchSize      = args.batchSize or 10
    self.samplesIndex   =  nil
    self.max_sample     = args.max_sample or 100


    self._softmax = cudnn.SoftMax():cuda()
    self._criterion = nn.CrossEntropyCriterion():cuda()
    self.train      = args.train
    --
    
    --
    self.t = args.t or 10
    self.T = args.T or 25
    self.numSteps = 0

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.cnnw,self.cnndw = self.network:getCNNParameters()

      self.dw:zero()
      self.cnndw:zero()


    self.dwt = self.dw:clone():zero()
    self.dwc = self.cnndw:clone():zero()
    self.cnndeltas = self.cnndw:clone():fill(0)
    self.cnng      = self.cnndw:clone():fill(0)
    self.cnntmp    = self.cnndw:clone():fill(0)
    self.cnng2     = self.cnndw:clone():fill(0)


    self.deltas = self.dw:clone():fill(0)
    self.tmp    = self.dw:clone():fill(0)
    self.g      = self.dw:clone():fill(0)
    self.g2     = self.dw:clone():fill(0)

    self.ncw, self.ncdw = self.ncnet:getParameters()
    self.ncdw:zero()
    self.ncdwc = self.ncdw:clone():zero()
    self.ncdeltas = self.ncdw:clone():fill(0)
    self.ncg      = self.ncdw:clone():fill(0)
    self.nctmp    = self.ncdw:clone():fill(0)
    self.ncg2     = self.ncdw:clone():fill(0)


    print('Number of all parameters: '..self.w:nElement())

    if self.target_q then
        self.target_network = self.network:clone()
        self.tw,self.tdw = self.target_network:getParameters()
        self.tcnnw,self.tcnndw = self.target_network:getCNNParameters() 
    end

    self.ncpool = {}
    self.pcpool = {}
end


function RNNQLearner:reset(state)
    
end


function RNNQLearner:setData(featurepool)
    self.featurepool = {}
    self.dataloader = {}
    self.featurepool = featurepool
    self.dataloader = featurepool.dataloader
    --self.network:training()
    self.cur_state,self.term,self._cur_image = self.dataloader:getNextState()
end


function RNNQLearner:updateNet()
    local t = math.max(0,self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr,self.lr_end)
    
    --[[ use gradients
    self.g:mul(0.95):add(0.05, self.dwt)
    self.tmp:cmul(self.dwt, self.dwt)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()
    
    -- accumulate update
    -- self.deltas = self.deltas + self.lr*self.dw/self.tmp
    self.deltas:mul(0):addcdiv(self.lr, self.dwt, self.tmp)
    self.w:add(-self.deltas)
    self.dwt:zero()]]

    --
    self.tmp:cmul(self.dwt, self.dwt)
    self.g2:mul(0.95):add(0.05, self.tmp)

    self.tmp:zero():add(self.g2)
    self.tmp:add(1e-8)
    self.tmp:sqrt()

    self.deltas:mul(0):addcdiv(self.lr, self.dwt, self.tmp)
    self.w:add(-self.deltas)
    self.dwt:zero()
    --

    --[[self.g2:mul(0.95):add(0.05, self.dwt)
    --self.deltas:mul(0):addcdiv(self.lr, self.dwc, self.tmp)
    self.deltas:mul(0):add(self.lr,self.g2)
    self.w:add(-self.deltas)
    self.dwt:zero()]]
end

function RNNQLearner:updateCNN()
    local t = math.max(0,self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
    self.lr = math.max(self.lr,self.lr_end)
    
    -- SGD

    self.cnng:mul(0.95):add(0.05, self.dwc)
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

function RNNQLearner:updateTargetNet()
    self.tw:mul(0.99):add(0.01,self.w)
    self.tcnnw:mul(0.99):add(0.01,self.cnnw)
end


function RNNQLearner:preData(state)
    local img = state.img:clone()
    local rois = torch.FloatTensor(self.maxm,5)
    local targets = torch.Tensor(self.maxm,1)
    local rois_id_map = {}
    for id,boxInfo in pairs(state.rois) do
        rois[{#rois_id_map+1,1}] = 1
        rois[{{#rois_id_map+1},{2,5}}] = boxInfo.box[{{1,4}}]
        targets[#rois_id_map+1][1] = boxInfo.gt

        table.insert(rois_id_map,#rois_id_map+1,id)
    end

    return img,rois,targets,rois_id_map


end

function RNNQLearner:getQA(prediction,count,rois_id_map)

    --local prediction = self.network.predictions:clone()

    local action = torch.Tensor(count):fill(self.maxm+1)
    
    local a      = torch.Tensor(self.maxm,self.maxm+1):zero()
    local q      = torch.Tensor(self.maxm):zero()
    --local tmpPre = prediction[{{1,self.featurepool.curm},{1,count}}]
    local tmpPre = prediction
    ---local _mat   = math.min(self.featurepool.curm,count)

    for i = 1,self.maxm do
        local maxq,row = tmpPre:max(1)
        local maxy,col = maxq:max(2)


        local x = col[1][1]
        local y = row[1][x]
        if y > self.featurepool.curm then
            tmpPre[{{y},{}}] = - Infi
        else
            if x < count + 1 then
                action[x] = y
                a[y][x] = 1
                q[y] = maxy[1][1]
                tmpPre[{{},{x}}] = - Infi
                tmpPre[{{y},{}}] = - Infi
            else
				a[y][x] = 1
                q[y] = maxy[1][1]
                tmpPre[{{y},{}}] = - Infi
                --a[y][x] = 1
            end
        end
    end

    --local features = self.featurepool:getFeatures()
    
    
   

    --[[if self.train then 
        local y = torch.Tensor(count):fill(2)
        for i,id in pairs(rois_id_map) do
            if i < count +1 then
                if self.featurepool:isOldId(id) == false then
                    y[i] = 1
                    action[i] = self.maxm + 1
                    a[{{},{i}}] = 0
                    --print(string.format('id %d is not old',id))
                end
            end
        end

        for i = 1 ,count do
            local sample = {}
            sample.feature = prediction[{{},{i}}]:clone():squeeze():float()
            sample.y = 2
            if y[i] == 1 then
                sample.y = 1
                table.insert(self.ncpool,sample)
                if #self.ncpool > 10000 then 
                    table.remove(self.ncpool,1)
                end
            else
                table.insert(self.pcpool,sample)
                if #self.pcpool > 10000 then 
                    table.remove(self.pcpool,1)
                end
            end
        end
        if #self.ncpool == 0 or #self.pcpool==0 then 
             return action, a,q
        end
        for m = 1, 20 do
        local batch = torch.FloatTensor(128,self.maxm):zero()
        local y1    = torch.Tensor(128,1):zero()
        for i = 1, 64 do
            local a = torch.random(1,#self.ncpool)
            local b = torch.random(1,#self.pcpool)
            batch[2*i] = self.ncpool[a].feature:clone()
            y1[2*i][1]         = self.ncpool[a].y
            batch[2*i-1] = self.pcpool[b].feature:clone()
            y1[2*i-1][1]         = self.pcpool[b].y
        end

         self.ncnet:forward(batch:cuda())
         self.nc_out = self._softmax:forward(self.ncnet.predictions)
         local tmp,pre = self.nc_out:max(2)

        local loss = self._criterion:forward(self.nc_out,y1:cuda())

        local c = 0
        for i = 1, 128 do
            --print(string.format('pre[ %d ] is %d and y is %d',i,pre[i][1],y[i]))
            if pre[i][1] == y1[i][1] then c = c + 1 end
        end

        --if self.numSteps%20 == 0 then
            local logInfo2 = string.format('<ncnet>, <Acc is %f, loss %f, lr %f>',c/128,loss,self.lr)
            print(logInfo2)
        --cmd:log('./log/CLS_201611102229.log',{logInfo})
        --end
        local dresults = self._criterion:backward(self.nc_out,y1:cuda())
        local dscores = self._softmax:backward(self.ncnet.predictions,dresults)
    
        self.ncnet:backward(batch:cuda(),dscores)

        self.ncdwc:add(self.ncdw)
    
    --[[ SGD
        self.ncdwc:add(self.ncdw)
        --if self.numSteps%10 == 0 then
            local t = math.max(0,self.numSteps - self.learn_start)
            self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
            self.lr = math.max(self.lr,self.lr_end)
            self.ncg:mul(0.9):add(0.1, self.ncdwc)
            self.ncdeltas:mul(0):add(self.lr,self.ncg)
            self.ncw:add(-self.ncdeltas)
            self.ncdwc:zero()]]
        --end
       --[[self.nctmp:cmul(self.ncdwc, self.ncdwc)
    self.ncg2:mul(0.95):add(0.05, self.nctmp)

    self.nctmp:zero():add(self.ncg2)
    self.nctmp:add(1e-8)
    self.nctmp:sqrt()

    self.ncdeltas:mul(0):addcdiv(self.lr, self.ncdwc, self.nctmp)
    self.ncw:add(-self.ncdeltas)
    self.ncdwc:zero()
    end
    else
         self.ncnet:forward(prediction[{{},{1,count}}])
         self.nc_out = self._softmax:forward(self.ncnet.predictions)
         local tmp,pre = self.nc_out:max(2)
        for i,id in pairs(rois_id_map) do
            if i < count +1 then
                if pre[i][1] == 1 then
                    action[i] = self.maxm + 1
                    a[{{},{i}}] = 0
                end
            end
        end
    end]]

    

    --print(count)
    


    return action, a,q
end

--- calculate the gradOutput based on output Q and target Q
-- the q dimension is is rois x (rois + 1) this 1 is terminal
-- the r dimension is rois x 1
-- the a dimension is rois x 1
-- the term dimension is rois x 1 (0|1)
function RNNQLearner:getQUpdate(args)
    local target = torch.Tensor(self.maxm,self.maxm+1):zero()
    local q = args.qt       -- maxm 
    local a = args.action   -- maxm x (maxm + 1) 
    local r = args.rt       -- maxm
    local f_map = args.f_map  -- maxm
    local q2 = args.qt1     -- maxm
    local term = args.term  -- maxm
    
    -- calculate dLoss/dOutput
    -- Loss = (y - Q(s,a))^2
    -- y = r + (1-terminal) * gamma*max_aQ(s2,a)
    -- delta = y - Q(s,a)
    -- 
    
    for i = 1,self.maxm do
        local delta = r[i] + (1 - term[i])*self.discount*q2[f_map[i]] - q[i]
        target[i] = delta
    end

    target:cmul(a)
    if self.clip_delta then
        target[target:ge(self.clip_delta)] = self.clip_delta
        target[target:le(-self.clip_delta)] = -self.clip_delta
    end
    local tmp = target:clone()
    tmp:cmul(target,target)
    target:mul(-1)   
    if self.numSteps%20 == 0 then
        logInfo = string.format('Steps %d, loss is %f, reward %f, %d trajectories',self.numSteps,tmp:sum(),self.featurepool.totalReward,self.featurepool.curm)
        --local cmd = torch.CmdLine()
        --cmd:addTime('TrackNet','%F %T')
        --cmd:log('./log/DQN_201611102229.log',{logInfo})
        print(logInfo)
    end
    return target
end

function RNNQLearner:addSample(sample)
    table.insert(self.samples,sample)
    if #self.samples > self.max_sample then
        table.remove(self.samples,1)
    end
end

function RNNQLearner:random_samples(n)
    for i = 1, n/2 do
        local a = torch.random(1,n)
        local b = torch.random(1,n)
        local tmp = self.samplesIndex[a]
        self.samplesIndex[a] = self.samplesIndex[b]
        self.samplesIndex[b] = tmp
    end
end

function RNNQLearner:geBatch()
    local _n_samples = #self.samples
    local batch = {}
    self.samplesIndex = torch.Tensor(_n_samples)

    if self.i_sample + self.batchSize -1 > _n_samples then 
        self.i_sample = 1
    end 
    --if self.i_sample == 1 then
        for i = 1, _n_samples do
            self.samplesIndex[i] = i
        end
        self:random_samples(_n_samples) 
    --end

    for i = 1, self.batchSize do
        table.insert(batch,self.samples[self.samplesIndex[self.i_sample+i-1]])
    end
    self.i_sample = self.i_sample + self.batchSize
    return batch
end

function RNNQLearner:trainBatch()
    --print(string.format('------* There are %d memory replies *--------',#self.samples))
    if #self.samples > self.batchSize then
        local batch = self:geBatch()
        for i,v in pairs(batch) do
            self.network:dqnforward(v.inputs)
            self.network:dqnbackward(v.inputs,v.target)
            self.dw:add(-self.wc, self.w)
            self.dwt:add(self.dw)
        end
        self.dwt:mul(1/self.batchSize)
        self:updateNet()
    end
end

function RNNQLearner:oneStepBatch()
 
    local features = self.featurepool:getFeatures()
    
    Infi = 1000000
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)
    local s = {}
    s.features = features
    s.img = img
    s.rois = rois
    s.targets = targets

    self.network:forward(s)
    ----------excu actions and update featurepool---------------------
    local frame    = {}
    frame.img = self.cur_state.img:clone()
    frame.bboxes = {}

    local count = 0
    for i in pairs(rois_id_map) do 
        if targets[i][1] == 1 then count = count + 1 end
        --print(targets[i][1])
    end

    --local action,a,q = self:eGreedy(self.network.predictions,count)
    local action,a,q = self:getQA(self.network.predictions,count)
    --print(targets)
    for i, id in pairs(rois_id_map) do
        
        if targets[i][1] == 1 then
            local boxInfo = {}
            boxInfo.action = action[i]
            boxInfo.box    = self.cur_state.rois[id].box:clone()
            boxInfo.feature = self.network.observation[i]:float()
            table.insert(frame.bboxes,id,boxInfo)        
        end
    end

    --print(string.format('------------------>the current frame has %d bounding boxes!<--------------',table.maxn(frame.bboxes)))
    -- update featurepool and get reward,excute actions
    local r,termS,frame_map = self.featurepool:updateNewFrame(self._cur_image,frame)
    ----------------------------------------------------------------

    ----------- state2
    local state2, term2, fr2 = self.dataloader:getNextState()
    if term2== 1 then
        logInfo = string.format('Steps %d, loss is %f, reward %f, %d trajectories',self.numSteps,0,self.featurepool.totalReward,self.featurepool.curm)
        print(logInfo)
        return term2, termS
    end
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2

    local features2 = self.featurepool:getFeatures()
    local img2,rois2,targets2,rois_id_map2 = self:preData(self.cur_state)
    local s2 = {}
    s2.features = features2
    s2.img = img2
    s2.rois = rois2
    s2.targets = targets2
    self.target_network:forward(s2)

    ----------- get q2
    --local q2 = self.target_network.fusions[1]

    
    count = 0
    for i in pairs(rois_id_map2) do 
        if targets2[i][1] == 1 then count = count + 1 end
    end

    local action2,a2,q2 = self:getQA(self.target_network.predictions,count)
    
    local target    -- targets are gradOutput -> dLoss/dOutput 

    
    target = self:getQUpdate{qt = q,rt = r,action = a,qt1 = q2,term = termS,f_map = frame_map}

    -- get new gradient
    -- targets  t x n x n
    local sample = {}
    sample.inputs = {}
    sample.inputs.features = features:clone()
    sample.inputs.observation = self.network.observation:clone()
    sample.target = target:clone()

    self:addSample(sample)

    self.numSteps = self.numSteps + 1
    self:trainBatch()
    if self.numSteps < 100 then 
        self:oneStepClassifier()
        if self.numSteps%self.t == 0 or term2 == 1 then
         
        --self:updateNet()
       
            self:updateCNN()
            self:updateTargetNet()
            --self:updateTargetNet()
            --self:updateTargetNet()
            self.featurepool:updateFeatures()
        end
    end
    if self.numSteps%self.T == 0 or term2 == 1 then
        self:updateTargetNet()
    end
    
    return term2,termS
end

function RNNQLearner:oneStep()
 
    local features = self.featurepool:getFeatures()
    Infi = 1000000
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)
    local s = {}
    s.features = features
    s.img = img
    s.rois = rois
    s.targets = targets

    self.network:forward(s)
    ----------excu actions and update featurepool---------------------
    local frame    = {}
    frame.img = self.cur_state.img:clone()
    frame.bboxes = {}

    local count = 0
    for i in pairs(rois_id_map) do 
        if targets[i][1] == 1 then count = count + 1 end
        --print(targets[i][1])
    end

    --local action,a,q = self:eGreedy(self.network.predictions,count)
    local action,a,q = self:getQA(self.network.predictions,count,rois_id_map)
    --print(targets)
    for i, id in pairs(rois_id_map) do
        
        if targets[i][1] == 1 then
            local boxInfo = {}
            boxInfo.action = action[i]
            boxInfo.box    = self.cur_state.rois[id].box:clone()
            boxInfo.feature = self.network.observation[i]:float()
            table.insert(frame.bboxes,id,boxInfo)        
        end
    end

    --print(string.format('------------------>the current frame has %d bounding boxes!<--------------',table.maxn(frame.bboxes)))
    -- update featurepool and get reward,excute actions
    local r,termS,frame_map = self.featurepool:updateNewFrame(self._cur_image,frame)
    ----------------------------------------------------------------

    ----------- state2
    local state2, term2, fr2 = self.dataloader:getNextState()
    if term2== 1 then
        logInfo = string.format('Steps %d, loss is %f, reward %f, %d trajectories',self.numSteps,0,self.featurepool.totalReward,self.featurepool.curm)
        print(logInfo)
        return term2, termS
    end
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2

    local features2 = self.featurepool:getFeatures()



    local img2,rois2,targets2,rois_id_map2 = self:preData(self.cur_state)
    local s2 = {}
    s2.features = features2
    s2.img = img2
    s2.rois = rois2
    s2.targets = targets2
    self.target_network:forward(s2)

    ----------- get q2
    --local q2 = self.target_network.fusions[1]

    
    count = 0
    for i in pairs(rois_id_map2) do 
        if targets2[i][1] == 1 then count = count + 1 end
    end

    --local action2,a2,q2 = self:getQA(self.target_network.predictions,count)
    local q2 = self.target_network.predictions:max(2):squeeze()

    local target    -- targets are gradOutput -> dLoss/dOutput 

    
    target = self:getQUpdate{qt = q,rt = r,action = a,qt1 = q2,term = termS,f_map = frame_map}

    -- get new gradient
    -- targets  t x n x n
    self.network:backward(s,target)

    -- add weight cost to gradient  self.dw = self.dw - self.wc*self.w
    self.dw:add(-self.wc, self.w)
    --self.cnndw:add(-self.cnnwc,self.)
    -- update dwt  dwt = dwt + everydw
    self.dwt:add(self.dw)
    self.dwc:add(self.cnndw:mul(1))
    self.numSteps = self.numSteps + 1

    self:oneStepClassifier()
    if self.numSteps%self.t == 0 or term2 == 1 then
         
        self:updateNet()
       
        self:updateCNN()
        --self:updateTargetNet()
         --self:updateTargetNet()
        self.featurepool:updateFeatures()
    end
    if self.numSteps%self.T == 0 or term2 == 1 then
        self:updateTargetNet()
    end
    
    return term2,termS
end


function RNNQLearner:TestOneStep()
 
    local features = self.featurepool:getFeatures()
    Infi = 1000000
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)
    local s = {}
    s.features = features
    s.img = img
    s.rois = rois
    s.targets = targets

    self.network:forward(s)
    ----------excu actions and update featurepool---------------------
    local frame    = {}
    frame.img = self.cur_state.img:clone()
    frame.bboxes = {}

    local count = 0
    for i in pairs(rois_id_map) do 
        if targets[i][1] == 1 then count = count + 1 end
        --print(targets[i][1])
    end

    --local action,a,q = self:eGreedy(self.network.predictions,count)
    local action,a,q = self:getQA(self.network.predictions,count)
    --print(targets)
    for i, id in pairs(rois_id_map) do
        
        if targets[i][1] == 1 then
            local boxInfo = {}
            boxInfo.action = action[i]
            boxInfo.box    = self.cur_state.rois[id].box:clone()
            boxInfo.feature = self.network.observation[i]:float()
            table.insert(frame.bboxes,id,boxInfo)        
        end
    end

    --print(string.format('------------------>the current frame has %d bounding boxes!<--------------',table.maxn(frame.bboxes)))
    -- update featurepool and get reward,excute actions
    local r,termS,frame_map = self.featurepool:updateNewFrame(self._cur_image,frame)
    ----------------------------------------------------------------

    ----------- state2
    local state2, term2, fr2 = self.dataloader:getNextState()
    if term2== 1 then
        logInfo = string.format('Steps %d, loss is %f, reward %f, %d trajectories',self.numSteps,0,self.featurepool.totalReward,self.featurepool.curm)
        print(logInfo)
        return term2, termS
    end
    self.cur_state = state2
    self.term      = term2
    self._cur_image= fr2
    
    return term2,termS
end


function RNNQLearner:oneStepClassifier()

    --local features = self.featurepool:getFeatures()
    
    local img,rois,targets,rois_id_map = self:preData(self.cur_state)

    local features,scores = self.network:cnnforward{img:cuda(),rois:cuda()}

    -- add weight cost to gradient  self.dw = self.dw - self.wc*self.w
    
    self.c_results = self._softmax:forward(scores)
    local y = torch.Tensor(self.maxm):zero()
    y = targets[{{},{1}}]
    y = y:mul(-1):add(2)
    local loss = self._criterion:forward(self.c_results,y:cuda())
    local tmp,prediction = self.c_results:max(2)
    local count = 0
    
    for i = 1, self.maxm do
       if prediction[i][1] == y[i][1] then count = count + 1 end
    end
    --print(count)
    if self.numSteps%20 == 0 then
        local logInfo2 = string.format('<%d frames, %d rois>, <Acc is %f, loss %f, lr %f>',self.featurepool.n_frames,self.featurepool.n_rois,count/self.maxm,loss,self.lr)
        print(logInfo2)
        --cmd:log('./log/CLS_201611102229.log',{logInfo})
    end
    local dresults = self._criterion:backward(self.c_results,y:cuda())
    local dscores = self._softmax:backward(scores,dresults)
    
    self.network:cnnbackward({img:cuda(),rois:cuda()},dscores)
    self.cnndw:add(-self.wc, self.cnnw)
    self.dwc:add(self.cnndw:mul(0.0))
    --self.numSteps = self.numSteps + 1
    --[[if self.numSteps%11 == 0  then 
        self:updateCNN()
        self.featurepool:updateFeatures()
    end]]
    -- body
end



function RNNQLearner:eGreedy(predictions,count)

    self.ep_threshold = testing_ep or (self.ep_end + math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt - math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    
    -- epsilon = ep_end + max(0,(ep_start - ep_end)*(ep_endt - max(0, numsteps - learn_start))/ep_endt))

    local ep = torch.uniform()
    if ep < self.ep_threshold then

        local action = torch.Tensor(count):fill(self.maxm+1)
    
        local a      = torch.Tensor(self.maxm,self.maxm+1):zero()
        local q      = torch.Tensor(self.maxm):zero()
        local tmpPre = predictions

        local index = count
        local mp = math.min(self.featurepool.curm+10,self.maxm)
        mp = math.min(mp,count + 1)
        local fmap = torch.Tensor(mp):fill(0)
        for i = 1,mp do
            fmap[i] = i
        end

        for i = 1,count do
            local m = torch.random(1,mp-i+1)
            action[i] = fmap[m]

            if fmap[m] < self.featurepool.curm + 1 then
                a[fmap[m]][i] = 1
                q[fmap[m]] = tmpPre[fmap[m]][i]
            end

            fmap[m] = fmap[mp-i+1]
        end

        return action, a,q
        --return torch.random(1,self.n_actions)
    else
        return self:getQA(predictions,count)
    end
    
end
--[[
function Trainer:greedy(frame)
    local q = self.network:predict(frame):float():squeeze()
    
    local maxq = q[1]
    local besta = {1}

    for a = 2,self.n_actions do
        if q[a] > maxq then
            besta = {a}
            maxq = q[a]
        elseif q[a] = maxq then
            besta = [#besta + 1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1,#besta)
    return besta[r]
end

function Trainer:report()
    local cnn = self.network.protos.cnn
    print('CNN')
    print(get_weight_norms(cnn))
    print(get_grad_norms(cnn))
    print('DQN')
    print(get_weight_norms(self.network.protos.dqn))
    print(get_grad_norms(self.network.protos.dqn))
end
--]]
