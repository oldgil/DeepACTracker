-- require 'utils.mot_data'
local Featurepool = torch.class('tracking.Featurepool')

-- require 'mot_data'
-- contains m x hist_len x n dims
-- __init 
-- setStart
-- getFeatures
-- updateFeatures
-- updateNewFrame

--------------------------------------* __init *---------------------------------------
function Featurepool:__init(args)
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
    self.frames   = {}
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
    self.lostTimes = torch.Tensor(self.maxm):zero()
    self.lostTimeThr = args.lostTimeThr or 5

    -- trajectories raw information
    self.trajectories = torch.Tensor(self.maxm,self.hist_len,2):zero()  -- frameId, bboxId
    self.frameId_map = {}

    -- network and dataloader used for updating feature
    self.network = args.network
    self.dataloader = args.dataloader

    self.totalReward = 0
    --
    --self.idstable = {}
    self.results = {}
    self._cur_id = 1
    self.id_map  = torch.Tensor(self.maxm):fill(0)
end

--------------------------------------* setStart *---------------------------------------
function Featurepool:setStart()
  -- get 1st frame
  -- all feature repeat length times
  self.trajectories:fill(0)  -- frameId, bboxId
  self.frames = {}
  self.lostTimes:fill(0)
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
  --local self.curm = math.min(self.maxm,#rois_table)
  --local rois = torch.Tensor(self.curm,5):zero()

  local m = 0

  for id, bbox in pairs(rois_table) do
    m = m + 1
    --table.insert(self.idstable,id,1)
    self.id_map[m] = self._cur_id
    self._cur_id = self._cur_id + 1
    local boxInfo = {}
    local box = bbox:clone()
    --box[{{1,4}}]:add(-1):mul(self.im_scale):add(1)

    boxInfo.box = box
    boxInfo.feature = torch.FloatTensor(self.feat_size):zero()
    boxInfo.lifetime = self.hist_len
    table.insert(self.frames[fr].bboxes,id,boxInfo)

    -- add to results
    local rBox = {}
    rBox.box = bbox:clone()
    rBox.id = self.id_map[m]
    rBox.fr = fr
    table.insert(self.results,rBox)

    -- init trajectories
    for t = 1,self.hist_len do
      self.trajectories[m][t][1] = fr
      self.trajectories[m][t][2] = id
    end
    --self.n_rois = self.n_rois + 1
    self.curm = self.curm + 1
    if self.curm == self.maxm then break end

  end 

  self:updateFeatures()
  print('featurepool set up!!')
end


--------------------------------------* _process_image *---------------------------------------
-- borrowed from fast-rcnn-torch
function Featurepool:_process_image(im)
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
function Featurepool:getFeatures()
  local features = torch.FloatTensor(self.maxm,self.hist_len,self.feat_size):uniform(-1,1)

  for m = 1,self.curm do
    for t = 1, self.hist_len do
      features[m][t] = self.frames[self.trajectories[m][t][1]].bboxes[self.trajectories[m][t][2]].feature:clone()
    end
  end

  return features
end


function Featurepool:getRoisN()
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
function Featurepool:updateFeatures()
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

    local features = self.network:cnnforward({imgs:cuda(),rois:cuda()}):float()
    for id = 1,features:size(1) do
      self.frames[fr].bboxes[f_map[id]].feature[{{1,self.feat_size}}] = features[id]
    end
  end

end

--------------------------------------* isodlId ----------------
function Featurepool:isOldId(id)
  --local flag = false
  for m = 1,self.curm do
    for t =1,self.hist_len do
      if self.trajectories[m][t][2] == id then
        return true
      end
    end
  end
  return false
end
--------------------------------------* deleteNode *----------------------------------------- 
function Featurepool:delNode(idx)
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
function Featurepool:getMostLostTraj()
  local maxLostTime, index = self.lostTimes:max(1)
  return index[1]
end

--------------------------------------* updateNewFrame *----------------------------------------- 
function Featurepool:updateNewFrame(fr,frame)
  -- frame is a table, frame = {img = img, bboxes = {[id]={action,bbox,feature}}}
  -- rois_table, contains [id] = {action = a, box = rois}
  -- add one new frame information
  -- features_tmp, a table of features indexed by chosen id
  -- rois_tmp, a table of roises indexed by chosen id
  local newId = {}
  local frame_map = torch.Tensor(self.maxm):fill(0)
  for i = 1,self.maxm do
    frame_map[i] = i
  end
  local term = torch.Tensor(self.maxm):fill(0)
  local reward = torch.Tensor(self.maxm):fill(0)
  self.lostTimes[{{1,self.curm}}]:add(1)

  table.insert(self.frames,fr,frame)
  local count = 0   
  for id in pairs(frame.bboxes) do
    count = count + 1
  end

  --print(string.format('now tracking %d trajectories, the frame %d has %d bounding boxes!',self.curm,fr,count))
  --local reward = 0
  for id,boxInfo in pairs(frame.bboxes) do

    local rBox = {}
    rBox.box = boxInfo.box:clone()
    rBox.fr = fr
    rBox.id = 0

    --print(boxInfo.action)
    --print(self.curm)
    local newidx = 0
    if boxInfo.action > self.curm then
      rBox.id = self._cur_id
      self._cur_id = self._cur_id + 1

      self.frames[fr].bboxes[id].lifetime = self.hist_len
      -- add new bucket
		--print(string.format('frame %d box %d lifitime %d',fr,id,self.frames[fr].bboxes[id].lifetime))
      if self.curm == self.maxm then
        -- already max m trajectories 
        -- delete most lost one ,add new on, reward + 0
        local del_m = self:getMostLostTraj()
        for t = 1,self.hist_len do
          self:delNode(self.trajectories[del_m][t])
          -- add new
          self.trajectories[del_m][t][1] = fr
          self.trajectories[del_m][t][2] = id
        end
        newidx = del_m
        self.lostTimes[del_m] = 0
        self.id_map[del_m] = rBox.id
        term[del_m] = 1
        
        
	      if self.train then
        --print('calc reward*********_______________?')
            for m = 1, self.curm do
              if m ~= del_m then
	              for t = 1,self.hist_len do
                  if self.trajectories[m][t][2] == id then reward[m] = reward[m] - 1 end
	              end
              end
            end
	        end
        
      else
        self.curm = self.curm + 1
        for t = 1,self.hist_len do
          self.trajectories[self.curm][t][1] = fr
          self.trajectories[self.curm][t][2] = id
        end
        self.id_map[self.curm] = rBox.id
        newidx = self.curm
        self.lostTimes[self.curm] = 0
      end
	    
      if self.train then
        --print('calc reward*********_______________?')
        for m = 1, self.curm-1 do
	       for t = 1,self.hist_len do
            if self.trajectories[m][t][2] == id then reward[m] = reward[m] - 1 end
	       end
        end
	    end
      
	
    else
      -- add to old trajectory

      -- caculate reward if train
      local m = boxInfo.action
      rBox.id = self.id_map[m]
      if self.train then
        --print('calc reward*********_______________?')
        for t = 1, self.hist_len do
          if self.trajectories[m][t][2] == id then
            reward[m] = reward[m] + 1
          else
            reward[m] = reward[m] - 1 
          end
        end
      end

      self.lostTimes[boxInfo.action] = 0
      self.frames[fr].bboxes[id].lifetime = 1
      self:delNode(self.trajectories[boxInfo.action][1])
      self.trajectories[{{boxInfo.action},{1,self.hist_len-1},{}}] = self.trajectories[{{boxInfo.action},{2,self.hist_len},{}}]
      self.trajectories[boxInfo.action][self.hist_len][1] = fr
      self.trajectories[boxInfo.action][self.hist_len][2] = id

      newidx = m
    end

    newId[boxInfo.action] = newidx

    table.insert(self.results,rBox)
    
  end

  

  for m = 1,self.curm do
    if self.lostTimes[m] > self.lostTimeThr then term[m] = 1 end
  end

  for m = 1,self.curm do
    if self.lostTimes[m] > self.lostTimeThr then
      -- TODO
      -- remove the losted trajectory
      for t = 1, self.hist_len do
        self:delNode(self.trajectories[m][t])
      end
      self.id_map[m] = self.id_map[self.curm]
      self.id_map[self.curm] = 0
      self.trajectories[{{m},{1,self.hist_len},{}}] = self.trajectories[{{self.curm},{1,self.hist_len},{}}]
      self.trajectories[{{self.curm},{1,self.hist_len},{}}] = 0
      
      frame_map[self.curm] = m
      self.lostTimes[m] = self.lostTimes[self.curm]
      self.lostTimes[self.curm] = 0
      self.curm = self.curm - 1
      m = m - 1
    end
  end
  reward = reward:mul(0.1)
  self.totalReward = self.totalReward + reward:sum()
  return reward,term, frame_map
end
  