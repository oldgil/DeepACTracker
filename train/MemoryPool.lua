
local MemoryPool = torch.class('tracking.MemoryPool')

-- require 'mot_data'
-- contains m x hist_len x n dims
-- __init 
-- setStart
-- getFeatures
-- updateFeatures
-- updateNewFrame

--------------------------------------* __init *---------------------------------------
function MemoryPool:__init(args)
    self.name = args.filename
    self.train = args.train
    -- scale
    
    self.im_scale = 1
    self.scale    = args.scale
    self.max_size = args.max_size

    -- image mean
    self.pixel_means = args.pixel_means

    -- maxm capacity
    self.maxm     = args.maxm
    self.curm     = 0

    -- frame info
    self.n_frames = 0
    self.n_rois   = 0

    -- hist_len
    self.hist_len = args.hist_len

    -- feature dim
    self.feat_size = args.feat_size

    -- feature maxtrix
    -- self.features = torch.FloatTensor(self.maxm,self.hist_len,self.feat_size):zero()

    -- frames store the information. one frame contains image, bboxes{ box1 ={feature,bbox,lifetime = l} }
    self.frames = {}

    -- tracjectories losttime
    self.lostTimeThr = args.lostTimeThr or 10

    -- trajectories raw information
    self.trajectories = {}  -- id, frameId, bboxId
    self.frameId_map = {}

    self.features = {}
    -- network and dataloader used for updating feature
    self.cnn = args.cnn
    self.dataloader = args.dataloader

    self.totalReward = 0
    --
    --self.idstable = {}
    self.results = {}
    self._cur_id = 0
    self.id_map  = {}
end


--------------------------------------* setStart *---------------------------------------
function MemoryPool:setStart()
  -- get 1st frame
  -- all feature repeat length times
  self.trajectories = {}  -- frameId, bboxId
  self.frames = {}
  self.curm     = 0

  local fr = 1
  while self.dataloader.rois[fr] == nil do
    fr = fr + 1
  end

  self.n_frames = 1
  if self.frames[fr] == nil then self.frames[fr] = {} end
  
  local im = self.dataloader.imgs[fr]:clone()
  -- Process the image
  --local im = self:_process_image(orig_im)
  self.frames[fr].img = im
  self.frames[fr].bboxes = {}
  self.imSize = im:size()
  local rois_table = self.dataloader.rois[fr]
  local m = 0
  for id, bbox in pairs(rois_table) do
    m = m + 1
    self._cur_id = self._cur_id + 1
    local boxInfo = {}
    local box = bbox:clone()

    boxInfo.box = box
    boxInfo.feature = torch.FloatTensor(self.feat_size):zero():cuda()
    boxInfo.lifetime = self.hist_len
    --table.insert(self.frames[fr].bboxes,id,boxInfo)
    self.frames[fr].bboxes[id] = boxInfo

    -- init trajectories
    local traj = torch.Tensor(self.hist_len,2):zero()
    --traj[{{1,self.hist_len-1},1}] = 0; traj[{{1,self.hist_len-1},2}] = 0
    --traj[{{self.hist_len},1}] = fr; traj[{{self.hist_len},2}] = id
    traj[{{1,self.hist_len},1}] = fr; traj[{{1,self.hist_len},2}] = id
    local trajInfo = {}
    trajInfo.tr = traj
    trajInfo.lt = 0
    trajInfo.gId = self._cur_id
    table.insert(self.trajectories,trajInfo)
     
    -- add to results
    local rBox = {}
    rBox.box = bbox:clone()
    rBox.id = trajInfo.gId
    rBox.fr = fr
    table.insert(self.results,rBox)
    --self.n_rois = self.n_rois + 1
    self.curm = self.curm + 1
    if self.curm == self.maxm then break end
  end

  self:updateFeatures()
  print('featurepool set up!!')
end


--------------------------------------* _process_image *---------------------------------------
-- borrowed from fast-rcnn-torch
function MemoryPool:_process_image(im)
  local im = im:clone():float()
  -- Correcting dimension
  if im:dim() == 2 then
    im = im:view(1,im:size(1) , im:size(2))
  end
  if im:size(1) == 1 then
    im = im:expand(3,im:size(2),im:size(3))
  end
  -- Scale to 255
  im:mul(255.0)
  -- Swap channels
  local out_im = torch.FloatTensor(im:size())
  local swap_order = {3,2,1}

  for i=1,im:size(1) do
     out_im[i] = im[swap_order[i]]
  end
  -- Subtracting mean from pixels
  for i=1,3 do
     out_im[i]:add(-self.pixel_means[i])
  end
  return out_im
end

--------------------------------------* getFeatures *---------------------------------------
function MemoryPool:getFeatures()
  local features = {}
  local boBoxes  = {}
  local lostTimes = {}
  for id,traj in pairs(self.trajectories) do
    local feature = torch.Tensor(self.hist_len,self.feat_size):zero():cuda()
    local trajBox = torch.Tensor(self.hist_len,4):zero()
    for t = 1,self.hist_len do
      if traj.tr[t][1] ~= 0 then
        feature[t] = self.frames[traj.tr[t][1]].bboxes[traj.tr[t][2]].feature:clone()
        trajBox[t] = self.frames[traj.tr[t][1]].bboxes[traj.tr[t][2]].box[{{1,4}}] 
      end
    end
    features[id] = feature
    boBoxes[id] = trajBox
    lostTimes[id] = traj.lt
    --table.insert(features,id,feature)
    --table.insert(boBoxes,id,trajBox)
    --stable.insert(lostTimes,id,traj.lt)
  end
  return features,boBoxes,lostTimes
end


function MemoryPool:getRoisN()
   local count = 0
   local nframe = 0 
   for fr,frame in pairs(self.frames) do
       nframe = nframe + 1
       for id,boxInfo in pairs(frame.bboxes) do
           count = count + 1
       end
   end
   return count, nframe
end
--------------------------------------* updateFeatures *----------------------------------------- 
function MemoryPool:updateFeatures()
  -- update all features
  -- read self.trajectories and use dataloader and network update features indexed by time t
  self.n_rois,self.n_frames = self:getRoisN()
  --print(string.format('updating features with %d rois and %d images with image size %d x %d x %d',self.n_rois,self.n_frames,self.imSize[1],self.imSize[2],self.imSize[3]))
   
  local imgs = torch.FloatTensor(1,self.imSize[1],self.imSize[2],self.imSize[3]):zero()

  for fr, frame in pairs(self.frames) do
    imgs[{{1},{},{1,self.imSize[2]},{1,self.imSize[3]}}] = frame.img
    
    local _rois = 0
    for id in pairs(frame.bboxes) do
      _rois = _rois + 1
    end

    local rois = torch.FloatTensor(_rois,5):zero()
    local f_map = torch.Tensor(_rois):zero()
    local idx = 1
    for id,boxInfo in pairs(frame.bboxes) do
      f_map[idx] = id
      rois[{idx,1}] = 1
      rois[{idx,{2,5}}] = boxInfo.box[{{1,4}}]
      idx = idx + 1
    end

    local features = self.cnn:cnnforward({imgs:cuda(),rois:cuda()})
    for id = 1,features:size(1) do
      self.frames[fr].bboxes[f_map[id]].feature[{{1,self.feat_size}}] = features[id]
    end
  end

end
 
--------------------------------------* deleteNode *----------------------------------------- 
function MemoryPool:delNode(idx)
  local fr = idx[1]
  local id = idx[2]
  self.frames[fr].bboxes[id].lifetime = self.frames[fr].bboxes[id].lifetime - 1
  if self.frames[fr].bboxes[id].lifetime == 0 then
    --table.remove(self.frames[fr].bboxes,id)  -- lua5.1 remove operation have bugs
    self.frames[fr].bboxes[id] = nil
    if table.maxn(self.frames[fr].bboxes) == 0 then
      self.frames[fr] = nil
    end
  end
end

--------------------------------------* getMostLostTraj *----------------------------------------- 
function MemoryPool:getMostLostTraj()
  local maxLt = -1
  local maxId = 1
  for id, traj in pairs(self.trajectories) do
    if traj.lt > maxLt then
      maxLt = traj.lt
      maxId = id
    end
  end
  return maxId
end


-- executate and return rewards
--- A fixed trajectory it's value should be a constant 
--- or else the state-value network cannot convergent  
function MemoryPool:execActions(trajId,fr,actions,selId)
  local rewards = {}
  local count = 0
  if self.train then
  for id,flag in pairs(actions) do
    rewards[id] = 0
    for t = 1, self.hist_len do
      if self.trajectories[trajId].tr[t][2] ~= 0 then
        if self.trajectories[trajId].tr[t][2] == id then
            rewards[id] = rewards[id] + flag
        else
            rewards[id] = rewards[id] - flag
        end
      end
    end
    rewards[id] = rewards[id]/self.hist_len
    --table.insert(rewards,id,reward)
    count = count + 1
  end
  end
  
  if selId == 0 then 
    -- none of any bounding box
    self.trajectories[trajId].lt = self.trajectories[trajId].lt + 1
  else
    self.trajectories[trajId].lt = 0
    if self.trajectories[trajId].tr[1][1] ~= 0 then
      self:delNode(self.trajectories[trajId].tr[1])
    end
    self.trajectories[trajId].tr[{{1,self.hist_len-1},{}}] = self.trajectories[trajId].tr[{{2,self.hist_len},{}}]
    self.trajectories[trajId].tr[{self.hist_len,1}] = fr
    self.trajectories[trajId].tr[{self.hist_len,2}] = selId
  end
  return count,rewards
end

-- executate and return rewards
--- A fixed trajectory it's value should be a constant 
--- or else the state-value network cannot convergent  
function MemoryPool:execActionsTest(trajId,fr,actions,selId,candRois)
  if selId == 0 then 
    -- none of any bounding box
    self.trajectories[trajId].lt = self.trajectories[trajId].lt + 1
  else
    self.trajectories[trajId].lt = 0
    if self.trajectories[trajId].tr[1][1] ~= 0 then
      self:delNode(self.trajectories[trajId].tr[1])
    end
    
    self.trajectories[trajId].tr[{{1,self.hist_len-1},{}}] = self.trajectories[trajId].tr[{{2,self.hist_len},{}}]
    self.trajectories[trajId].tr[{self.hist_len,1}] = fr
    self.trajectories[trajId].tr[{self.hist_len,2}] = selId
    local rBox = {}
    rBox.box = candRois[selId]:clone()
    rBox.fr = fr
    rBox.id = self.trajectories[trajId].gId
    table.insert(self.results,rBox)
  end
end
--------------------------------------* updateNewFrame *----------------------------------------- 
function MemoryPool:updateNewFrame(fr,frame)
  -- frame {img,boxes}
  --table.insert(self.frames,fr,frame)



  self.frames[fr] = frame

  if self.train == false then
    for id,boxInfo in pairs(self.frames[fr].bboxes) do
    local rBox = {}
    rBox.box = boxInfo.box:clone()
    rBox.fr = fr
    rBox.id = 0
    if boxInfo.lifetime == 0 then   -- new trajectory
      self.frames[fr].bboxes[id].lifetime = 1
      
      if self.curm == self.maxm then   -- del most lost one
        local delId = self:getMostLostTraj()
        local traj = self.trajectories[delId].tr
        for t = 1, self.hist_len do
          if traj[t][1] ~= 0 then
            self:delNode(traj[t])
          end
        end
        table.remove(self.trajectories,delId)
        self.curm = self.curm - 1 
      end
      
      -- add new trajectory
      local traj = torch.Tensor(self.hist_len,2):zero()
      traj[{{1,self.hist_len-1},1}] = 0; traj[{{1,self.hist_len-1},2}] = 0
      traj[{{self.hist_len},1}] = fr; traj[{{self.hist_len},2}] = id
      local trajInfo = {}
      trajInfo.tr = traj
      trajInfo.lt = 0
      self._cur_id = self._cur_id + 1
      trajInfo.gId = self._cur_id
      rBox.id = trajInfo.gId
      table.insert(self.trajectories,trajInfo)
      self.curm = self.curm + 1 
      table.insert(self.results,rBox)
    end
  end

  for id,traj in pairs(self.trajectories) do
        if traj.lt >= self.lostTimeThr then
          for t = 1, self.hist_len do
            if traj.tr[t][1] ~= 0 then
              self:delNode(traj.tr[t])
            end
          end
          table.remove(self.trajectories,id)
          self.curm = self.curm - 1
    end
  end
  
    return 1
  end

  for id,boxInfo in pairs(self.frames[fr].bboxes) do
    if boxInfo.lifetime == 0 then   -- new trajectory
      self.frames[fr].bboxes[id].lifetime = 1
      
      if self.curm == self.maxm then   -- del most lost one
        local delId = self:getMostLostTraj()
        local traj = self.trajectories[delId].tr
        for t = 1, self.hist_len do
          if traj[t][1] ~= 0 then
            self:delNode(traj[t])
          end
        end
        table.remove(self.trajectories,delId)
        self.curm = self.curm - 1 
      end
      
      -- add new trajectory
      local traj = torch.Tensor(self.hist_len,2):zero()
      traj[{{1,self.hist_len-1},1}] = 0; traj[{{1,self.hist_len-1},2}] = 0
      traj[{{self.hist_len},1}] = fr; traj[{{self.hist_len},2}] = id
      local trajInfo = {}
      trajInfo.tr = traj
      trajInfo.lt = 0
      table.insert(self.trajectories,trajInfo)
      self.curm = self.curm + 1 
    end
  end
  for id,traj in pairs(self.trajectories) do
        if traj.lt > self.lostTimeThr then
          for t = 1, self.hist_len do
            if traj.tr[t][1] ~= 0 then
              self:delNode(traj.tr[t])
            end
          end
          table.remove(self.trajectories,id)
          self.curm = self.curm - 1
    end
  end
end


--------------------------------------* updateNewFrame *----------------------------------------- 
function MemoryPool:filter()
   local counter = {}
   for id,rbox in pairs(self.results) do
      local gid = rbox.id
      if counter[gid] == nil then
        counter[gid] = 1
      else
        counter[gid] = counter[gid] + 1
      end
   end

   self.filtedRes = {}
   for id, rbox in pairs(self.results) do
      local gid = rbox.id
      if counter[gid] > 5 then
          table.insert(self.filtedRes,rbox)
      end
   end
end
