require 'utils.mot_data'
local DataLoader = torch.class('tracking.DataLoader')

--require 'mot_data'

function DataLoader:__init(args)
    self.filename = args.filename
    self._n_rois  = args.maxm
    self.hist_len = args.hist_len
    self.dataPath = args.dataPath..self.filename
    self.det_file = self.dataPath..'/det/det.txt'
    self.gt_file  = self.dataPath..'/gt/gt.txt'
    self.img_path = self.dataPath..'/img1/'
    self.data_cache = args.data_cache
    self._cur_image  = 1
    self.train      = args.train
    self.loadStored = args.loadStored
    self.det_data    = readTXT(self.det_file,2)
print(string.format('has dets %d',#self.det_data))
    self.Det = {}
    -- training data has gt.txt
    if args.train then 
      self.gt_data     = readTXT(self.gt_file,1)
    else
      self.gt_data     = readTXT(self.det_file,2)
    end

    self.Gt = {}
    self.maxn_rois = 0
    --[[for fr,v in pairs(self.gt_data) do
      local count =0 
      for id,box in pairs(v) do
         count = count + 1
      end 
       print(string.format('frame %d has %d boxes',fr,count))
    end]]
    
    self._imgs       = countImages(self.img_path)
    --self.imgs        = loadImages(self.img_path)
    self.imgs = {}
    self.pixel_means  = args.pixel_means
    --self.gt          = torch.Tensor(#self.imgs,self._n_rois):fill(self._n_rois+1) -- init with term state

    
    self._n_images = self._imgs
    self.cur_state = {}
    --self.cur_state.gt = torch.Tensor(self.hist_len,self._n_rois,1):zero()
    --self.cur_state.rois = torch.Tensor(self.hist_len,self._n_rois,4):zero()

        -- scale
    self.im_scale = 1
    self.scale    = args.scale
    self.max_size = args.max_size
    self.rois     = {}
    -- term
    local imgFileName = self.data_cache..self.filename..'_imgs.t7'
    local detFileName = self.data_cache..self.filename..'_det.t7'
    local gtFileName = self.data_cache..self.filename..'_gt.t7'
    local file = io.open(imgFileName)
    if file then file:close() end
    local flag = (file ~= nil)
    self.term = 0
    if flag then
        print('file exists, loading from stored data..')
	      self.imgs = torch.load(imgFileName)
        local img = readImage(self.img_path,1)
        self.imageSize = self.imgs[1]:size()
        self.im_scale = self.imageSize[2]/img:size(2)
        print('images loaded..')
        self.Det = torch.load(detFileName)
        self.Gt = torch.load(gtFileName)
        for fr,v in pairs(self.Det) do
          local count = 0
          for i in pairs(v) do
            count = count + 1
          end
          if count > self.maxn_rois then self.maxn_rois = count end
        end
        for fr,v in pairs(self.Gt) do
          local count = 0
          for i in pairs(v) do
            count = count + 1
          end
          if count > self.maxn_rois then self.maxn_rois = count end
        end
        print('dets and gts loaded..')
    else
        
        local img = readImage(self.img_path,1)
    	print('file not exists,process data..')
        image.save('./data/tmp/original_img.jpg',img)
	self:imageProcess(1,img)
        image.save('./data/tmp/processed_img.jpg',self.imgs[1])
        --print(img)
        for fr=2,self._imgs do
            img = readImage(self.img_path,fr)
            self:imageProcess(fr,img)
        end
        print( 'save data..')
        torch.save(imgFileName,self.imgs)
	print('imgs saved..')
        torch.save(detFileName,self.Det)
	print('dets saved..')
        torch.save(gtFileName,self.Gt)
	print('gts saved..')

    end
    if self.train then
       self.rois = self.Gt
    else	 
       self.rois = self.Det
    end
    print(string.format('most %d rois in one frame',self.maxn_rois))
end


function DataLoader:_restartLoader()
	self._cur_image = 1
  self.term = 1
	--self._rand_perm = torch.randperm(self._n_images)
end


-------
function DataLoader:imageProcess(fr,img)
  local im = self:_process_image(img)
  -- Process the image
  --local im = self:_process_image(orig_im)
  local im_size = {im:size(2), im:size(3)}
  -- Scale the im and bboxes
  local im_size_min = math.min(im_size[1],im_size[2])
  local im_size_max = math.max(im_size[1],im_size[2])
  self.im_scale = self.scale/im_size_min
  if torch.round(self.im_scale*im_size_max) > self.max_size then
    self.im_scale = self.max_size/im_size_max
  end
  local new_size = {torch.round(im_size[1]*self.im_scale),torch.round(im_size[2]*self.im_scale)}
  local out_im = image.scale(im,new_size[2],new_size[1],'bicubic')
  --print(string.format('for image %d ,new size is %d x %d x %d',fr,out_im:size(1),out_im:size(2),out_im:size(3)))
  self.imageSize = out_im:size()
  table.insert(self.imgs,fr,out_im)
  if self.Gt[fr] ==nil then self.Gt[fr] = {} end
  if self.Det[fr] ==nil then self.Det[fr] = {} end
  if(self.det_data[fr]) == nil then 
    print(string.format('got nil det at frame %d',fr)) 
  else
    local count = 0
    for i in pairs(self.det_data[fr]) do count = count + 1 end
    if count>self.maxn_rois then self.maxn_rois = count end

  end
  if(self.gt_data[fr]) == nil then
     print(string.format('got nil gt at frame %d',fr)) 
  else
    local count = 0
    for i in pairs(self.gt_data[fr]) do count = count + 1 end
    if count>self.maxn_rois then self.maxn_rois = count end
   end
  self.Det[fr] = self:processBboxes(self.det_data[fr])
  self.Gt[fr] = self:processBboxes(self.gt_data[fr])
  
end


------------
function DataLoader:processBboxes(bboxes)
  -- body
  if bboxes == nil then return nil end
  local tar = {}
  --for fr,v in pairs(bboxes) do
    --if tar[fr] == nil then tar[fr] = {} end
  for id,box in pairs(bboxes) do
    local tarBox = box:squeeze()
    --print(self.im_scale)
    --print('before')
    --print(tarBox)
    tarBox[{{1,4}}]:add(-1):mul(self.im_scale):add(1)
    --print('after')
     --print(tarBox)
    if tarBox[1] < 1 then tarBox[1] = 1 end
    if tarBox[3] > self.imageSize[3] then tarBox[3] = self.imageSize[3] end
    if tarBox[2] < 1 then tarBox[2] = 1 end
    if tarBox[4] > self.imageSize[2] then tarBox[4] = self.imageSize[2] end
    table.insert(tar,id,tarBox)
  end
  --end
  return tar
end


--------------------------------------* _process_image *---------------------------------------
-- borrowed from fast-rcnn-torch
function DataLoader:_process_image(im)
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
  local swap_order = {1,2,3}

  for i=1,im:size(1) do
     out_im[i] = im[swap_order[i]]
  end
  -- Subtracting mean from pixels
  for i=1,3 do
     out_im[i]:add(-self.pixel_means[i])
  end
  return out_im
end

---------------------------------------------------------------- state contains {pre_img,cur_img,rois,gt}
function DataLoader:getNextState()
  --local term = 0
  self._cur_image = self._cur_image + 1
  --if self._cur_image == self._n_images then self.term = 1 end
  if self._cur_image > self._n_images then
    self:_restartLoader()
  end
  
  while self.Gt[self._cur_image] == nil do
    self._cur_image = self._cur_image + 1
    --if self._cur_image == self._n_images then self.term = 1 end
    if self._cur_image > self._n_images then
      self:_restartLoader()
    end
  end
  local imSize = self.imgs[self._cur_image]:size()
  if self.cur_state.img == nil then
    
    self.cur_state.img = torch.Tensor(1,3,imSize[2],imSize[3])
  end
  self.cur_state.img[{{1},{1,3},{1,imSize[2]},{1,imSize[3]}}] = self.imgs[self._cur_image]
  self.cur_state.rois = {}

  local maxId = 1
  local count = 1
  for id,box in pairs(self.Gt[self._cur_image]) do
    if count > self._n_rois then break end
    local boxInfo = {}
    boxInfo.box = box:clone()
    boxInfo.gt  = 1
    table.insert(self.cur_state.rois,id,boxInfo)
    if id > maxId then maxId = id end
    count = count + 1
  end
  --print(count-1)
  -- add nagetive samples
  if count <= self._n_rois then
    local imSize = self.imgs[1]:size()
    --print(string.format('random %d negative bounding boxes..',self._n_rois-count+1))
  
    local randBoxes = randomBbox(self._n_rois-count+1,imSize,self.Gt[self._cur_image])
    for id ,box in pairs(randBoxes) do
      local boxInfo = {}
      boxInfo.box = box:clone()
      boxInfo.gt  = 0
      table.insert(self.cur_state.rois,maxId+id,boxInfo)
    end
  end
  return self.cur_state, self.term, self._cur_image
end

-----------------useless-------------------------------
function DataLoader:_visualize_batch(im,boxes)
  local ok = pcall(require,'qt')
  if not ok then
    error('You need to run visualize_detections using qlua   ')
  end
  require 'qttorch'
  require 'qtwidget'

  local num_boxes = boxes:size(1)
  local widths  = boxes[{{},3}] - boxes[{{},1}]
  local heights = boxes[{{},4}] - boxes[{{},2}]

  local x,y = im:size(3),im:size(2)
  local w = qtwidget.newwindow(x,y,"Detections")
  local qtimg = qt.QImage.fromTensor(im)
  w:image(0,0,x,y,qtimg)
  local fontsize = 15

  for i=1,num_boxes do
    local x,y = boxes[{i,1}],boxes[{i,2}]
    local width,height = widths[i], heights[i]
    
    -- add bbox
    w:setcolor("red")
    w:rectangle(x,y,width,height)
    w:moveto(x,y+fontsize)
    w:setfont(qt.QFont{serif=true,italic=true,size=fontsize,bold=true})

  end

  w:setcolor("red")
  w:setlinewidth(2)
  w:stroke()
  return w
end
