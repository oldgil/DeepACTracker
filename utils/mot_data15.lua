require 'image'		-- image load
require 'gnuplot'	-- plotting stuffgetTracksAndDetsTables
--require 'lfs' 		-- luaFileSystem, check for dir, etc...
local lfs = require 'lfs'

--------------------txt reader---------------------------------------
---------------------------------------------------------------------
function csvSplit(str,sep)
    sep = sep or ','
    fields = {}
    local matchfunc = string.gmatch(str,"([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields,str)

    end
    return fields
end

---------------------------------------------------------------------
function csvRead(path, sep, tonum)
    tonum = tonum or true
    sep = sep or ','
    local csvFile = {}
    local file = assert(io.open(path, "r"))
    for line in file:lines() do
        fields = csvSplit(line, sep)
        if tonum then -- convert numeric fields to numbers
            for i=1,#fields do
                fields[i] = tonumber(fields[i]) or fields[i]
            end
        end
        table.insert(csvFile, fields)
    end
    file:close()
    return csvFile
end

---------------------------------------------------------------------
---------------------------------------------------------------------
function csvWrite(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    
    if type(data)=='table' then
      for i=1,#data do
	  for j=1,#data[i] do
	      if j>1 then file:write(sep) end
	      file:write(data[i][j])
	  end
	  file:write('\n')
      end
    elseif torch.isTensor(data) then
      if data:nDimension() < 2 then -- zero solution case
-- 	file:write('\n')
      else
	for i=1,data:size(1) do
	    for j=1,data:size(2) do
		if j>1 then file:write(sep) end
		file:write(data[i][j])
	    end
	    file:write('\n')
	end      
      end
    else
      error('unknown data type in csvwrite')
    end  
    file:close()
end

---------------------------------------------------------------------
function tabLen(tab) 
  local count = 0
  for key in pairs(tab) do
    count = count + 1
  end
  return count
end


---------------------------------------------------------------------
function readTXT(datafile, mode)
  -- local datafile = '/media/sf_vmex/2DMOT2015/data/TUD-Campus/gt/gt.txt'

  local gtraw = csvRead(datafile)
  local data={}
  local confThr = -1e5
  --if opt.detConfThr ~= nil then confThr = opt.detConfThr end
  --if sopt~=nil and sopt.detConfThr ~= nil then confThr = sopt.detConfThr end
--   print(confThr)
    
  
  
  if not mode then
	-- figure out whether we are in GT (1) or in Det (2) mode
	mode = 1
	if gtraw[1][7] ~= -1 then mode = 2 end -- very simple, gt do not have scores
  end
  -- go through all lines
  for l = 1,tabLen(gtraw) do    
    fr=gtraw[l][1]
    id=gtraw[l][2]
    bx=gtraw[l][3]
    by=gtraw[l][4]
    bw=gtraw[l][5]
    bh=gtraw[l][6]
    --cl=gtraw[l][8]
    sc=gtraw[l][7]
    if data[fr] == nil then
      data[fr] = {}      
    end
    -- detections do not have IDs, simply increment
    if mode==2 then id = table.getn(data[fr]) + 1 end
    
    -- only use box for ground truth, and box + confidence for detections
    if mode == 1 and sc > 0.4 and cl == 1 then
      table.insert(data[fr],id,torch.Tensor({bx,by,bw,bh}):resize(1,4)) 
    elseif mode == 2 then      
      table.insert(data[fr],id,torch.Tensor({bx,by,bw,bh}):resize(1,4)) 
    end
  end
  
  -- shift (left,top) to (center_x, center_y)  
  for k,v in pairs(data) do
    for l,m in pairs(v) do
      local box = data[k][l]:squeeze()
      box[3] = box[1] + box[3]
      box[4] = box[2] + box[4]
      -- (x1,y1,x2,y2)
      --box[1]=box[1]+box[3]/2 -- this already changes data in place
      --box[2]=box[2]+box[4]/2
--       data[k][l] = box:reshape(1,box:nElement())
    end
  end    

  -- return table
  return data
end


--------------------------------------------------------------------------
--- Write a txt file format of MOTChallenge 2015 (frame, id, bbx,...)
-- @param data	The tensor containing bounding boxes, a FxNxD tensor.
-- @param datafile Path to data file.
-- @param thr 	A threshold for ignoring boxes.
-- @param mode 	(optional)  1 = result, 2 = detections
-- @see readTXT
function writeTXT(dataTable, datafile,im_scale)
  
--   
  
  -- Shift cx,cy back to left,top
--   print(data[1])
  local count = 0
  for i,boxInfo in pairs(dataTable) do
    count = count + 1
  end
  print(string.format('Total has %d boxes with im_scale %f',count,1/im_scale))
  local out = torch.Tensor(count, 7):fill(-1)	-- the tensor to be written

  for i,box in pairs(dataTable) do
    out[i][1] = box.fr
    out[i][2] = box.id
    box.box[{{1,4}}]:add(-1):mul(1/im_scale):add(1)
    for d = 1, 4 do
      out[i][d+2] = box.box[d]
    end
    out[i][5] = out[i][5] - out[i][3]
    out[i][6] = out[i][6] - out[i][4]

  end

  csvWrite(datafile, out)
  
end
---------------------------------txt reader ended------------------
-------------------------------------------------------------------



----------------image process -------------------------------------
-------------------------------------------------------------------
--crop image to patch
function im_crop(img,bb)
    local imSize = img:size()
    local x1 = math.min(math.max(bb[1],1),imSize[3]-1)
    local y1 = math.min(math.max(bb[2],1),imSize[2]-1)
    local x2 = math.max(math.min(bb[3],imSize[3]),1)
    local y2 = math.max(math.min(bb[4],imSize[2]),1)

    local dst = image.crop(img,x1,y1,x2,y2)
    return dst
end

------------------------------------------------------------------
--boxIoU caculate IoU of two bounding boxes
function boxIoU(box1,box2)
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

------------------------------------------------------------------
--getDimension


------------------------------------------------------------------
--find the closest box
function findClosestTrack(det, tracks)
    local CId = 0
    local CBox = torch.Tensor(1,5):zero()
    local CDist = 1e9
    for id,bbox in pairs(tracks) do
        local dist = math.pow(bbox[1][1] - det[1][1],2) + math.pow(bbox[1][2] - det[1][2],2)
        if dist < CDist then
            CId = id
            Cbox = bbox:clone()
            CDist = dist
        end

    end
    return CId,Cbox,CDist
end

------------------------------------------------------------------
--find the most closest IoU
function findClosestTrackIOU(det, tracks)
    local CId = 0
    local Cbox = torch.Tensor(1,5):zero()
    local CIoU = -1
    for id,bbox in pairs(tracks) do
        local iou = boxIoU(det[1],bbox[1])
        if iou > CIoU then
            CIoU = iou
            Cbox = bbox:clone()
            CId = id
        end
    end

    return CId, Cbox, CIoU
end

-------------------------------------------------------------------
----nms 
function nms(det,dets,threshold)
    threshold = threshold or 0.4
    local nmstable = {}
    for id,bbox in pairs(dets) do
        local iou = boxIoU(det[1],bbox[1])
        if iou > threshold and (bbox[1][6] < 0.3 or det[1][5] == bbox[1][5]) then
            table.insert(nmstable,#nmstable+1,id)
        end
    end
    for ind,id in pairs(nmstable) do
        table.remove(dets,id)
    end
    nmstable = {}
    --return dets
end

----------------------------------------------------------------
--sign()
function sign(a)
    if a > 0 then return 1 end
    if a == 0 then return 0 end
    if a < 0 then return -1 end
end

-------------------------------------------------------------------
-- generate motion smoothness
function motion_smooth(bbox,n)
    n = n or 10
    local smooth = {}
    local mu1 = 0
    local b1 = 1/15
    local mu2 = 1
    local b2 = 1/40
    math.randomseed(os.time())
    for i = 1,n do
        local c1 = math.random() - 0.5
        local c2 = math.random() - 0.5
        local s1 = math.random() - 0.5
        local s2 = math.random() - 0.5

        local dcx = mu1 - b1*sign(c1)*math.log(1-2*math.abs(c1))
        local dcy = mu1 - b1*sign(c2)*math.log(1-2*math.abs(c2))
        local dw = mu2 - b2*sign(s1)*math.log(1-2*math.abs(s1))
        local dh = mu2 - b2*sign(s2)*math.log(1-2*math.abs(s2))
        if dw < 0.8 then dw = 0.8 end
        if dw > 1.2 then dw = 1.2 end
        if dh < 0.8 then dh = 0.8 end
        if dh > 1.2 then dh = 1.2 end
        local cx = bbox[1]+bbox[3]*dcx
        local cy = bbox[2]+bbox[4]*dcy
        local w = bbox[3]*dw
        local h = bbox[4]*dh
        --cx = math.floor(math.min(math.max(cx,1),imSize[3]-1))
        --cy = math.floor(math.min(math.max(cy,1),imSize[2]-1))
        --w = math.floor(math.min(math.max(w,1),imSize[3]-1))
        --h = math.floor(math.min(math.max(h,1),imSize[2]-1))

        local bbox_s = torch.Tensor(1,4)

        bbox_s[1][1] = cx
        bbox_s[1][2] = cy
        bbox_s[1][3] = w
        bbox_s[1][4] = h
        
        table.insert(smooth,i,bbox_s)
    end
    return smooth
end

---------------------------------------------------------------------
---------------------------------------------------------------------
function randomBbox(n,imgSize,bboxs)
  local W = imgSize[3]
  local H = imgSize[2]
  local x1 = 0
  local y1 = 0
  local x2 = 0
  local y2 = 0
  local box = torch.Tensor(4):fill(0)
  local iou = 0

  local index = 0
  local flag = true
  local randBoxs = {}
  while flag do
    x1 = torch.random(1,W-1)
    x2 = torch.random(x1+1,W)
    y1 = torch.random(1,H-1)
    y2 = torch.random(y1+1,H)
    box[1] = x1
    box[2] = y1
    box[3] = x2
    box[4] = y2
    local isUseful = true
    local maxIoU = 0
    local maxBox = torch.Tensor()
    for id,bbox in pairs(bboxs) do
      iou = boxIoU(box,bbox)
      
      if iou > 0.3 then 
        isUseful = false
        break
      end
      if iou > maxIoU then maxIoU = iou end 
    end

    if isUseful then
      index = index + 1
      --print(string.format('Got %dth box the maxIoU is %f',index,maxIoU))
      --if maxIoU > 0 then
       --   print(string.format('box1 (%f,%f,%f,%f)',box[1],box[2],box[3],box[4]))
      --    print(string.format('box2 (%f,%f,%f,%f)',maxBox[1],maxBox[2],maxBox[3],maxBox[4]))
     -- end
      table.insert(randBoxs,#randBoxs+1,box:clone())
      if index == n then
        flag = false
      end
    end
  end
  return randBoxs
end



---------------------------------------------------------------------
---------------------------------------------------------------------
function loadImages(path)
    local imgs = {}
    local count = 0
    for file in lfs.dir(path) do
        if file ~= "." and file ~= ".." then
            local f = path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            if attr.mode ~= "directory" then count = count + 1 end
        end
    end
    local filename = ''
    for i = 1, count do
        filename = string.format("%06d",i)
        filename = path..filename..'.jpg'
        print(string.format('loading image %d/%d %s',i,count,filename))
        local im = image.load(filename)
        table.insert(imgs,i,im)
    end

    return imgs       
end

------------------------------------------------------------------------
function countImages(path)
   local count = 0
    for file in lfs.dir(path) do
        if file ~= "." and file ~= ".." then
            local f = path..'/'..file
            local attr = lfs.attributes (f)
            assert (type(attr) == "table")
            if attr.mode ~= "directory" then count = count + 1 end
        end
    end
    return count
end

-------------------------------------------------------------------------
-------------------------------------------------------------------------
function readImage(path,i)
   local filename = string.format("%06d",i)
   filename = path..filename..'.jpg'
   print(string.format('loading image %d %s',i,filename))
   local im = image.load(filename)
   --print(im:size())
   return im
end
