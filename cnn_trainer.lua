--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'tracking' 

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train CNN used in Tracking:')
cmd:text()
cmd:text('Options:')

cmd:option('-maxm', 120, 'number of tracked trajectories at the same time')
cmd:option('-feat_size', 256, 'number of cnn feature_size')
cmd:option('-model_path','./models/VGG16/TRCNN.lua','Path to the FRCNN model definition')
cmd:option('-model_weights','./data/trained_models/frcnn_alexnet_VOC2007_iter_40000.t7','Path to the FRCNN weights (used for testing)')
cmd:option('-scale', 600, 'Scale used for training and testing, currently only single scale is supported.')
cmd:option('-max_size', 1000, 'Max pixel size of the longest side of a scaled input image')
cmd:option('-pixel_means', {122.7717,115.9465,102.9801}, 'Pixel mean values (RGB order)')
--cmd:option('-pixel_means', {0,0,0}, 'Pixel mean values (BGR order)')
cmd:option('-weight_file_path','./pretrained_models/imgnet_VGG16.t7','')
cmd:option('-train',true,'')
cmd:option('-lr',0.01,'')
cmd:option('-lr_end',0.00001,'')
cmd:option('-wc',0,'')
cmd:option('-lr_endt',250000,'')
cmd:option('-steps',250000,'')
cmd:option('-discount',0.95,'')
cmd:option('-clip_delta',5,'')
--cmd:option('-optim_snapshot_iters', 10000, 'Iterations between snapshots (used for saving the network)')
cmd:option('-save_path','./data/trained_models','Path to be used for saving the trained models')

cmd:option('-data_cache','./data/cache/MOT16/')
cmd:option('-cache','./cache','Directory used for saving cache data')
cmd:option('-log_path','./cache','Path used for saving log data')
cmd:option('-dataset','MOT2016','Dataset to be used')
cmd:option('-dataPath','./data/MOT16/train/','Path to the dataset root folder')

cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', 1, 'gpu flag')
cmd:option('-t',15,'time steps for update network')
cmd:option('-T',32,'time steps for update target network')
cmd:text()

local opt = cmd:parse(arg)
cmd:addTime('VGGNet','%F %T')
cutorch.setDevice(opt.gpu)
torch.manualSeed(3)

local trainSequence = {'MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'}
--local trainSequence = {'MOT16-02'}

print(opt.model_path)
--opt.network = tracking.TrackNet(opt)

opt.cnn = tracking.VGGNet(opt)
--opt.cnn = torch.load('./shortcuts/cnn_params20_15.t7')
opt.cnn:training()
opt.dataloader = {}
opt.filename = ''
local Trainer = tracking.CNNLearner(opt)
local step = 0
local epochs = 50
local iteration = 0

logInfo2 = ''
cmd:log('./log/VGGNet_201701131818.log',{logInfo2})
--cmd:log('./log/CLS_201611102229.log',{logInfo2})

while iteration < epochs do
    for id,filename in pairs(trainSequence) do
        opt.filename = filename
        print(string.format('Begin training cnn with '..filename..' start loading data...at iteration %d',iteration))
        opt.dataloader = {}
        opt.dataloader = tracking.DataLoader(opt)     
        Trainer:setData(opt.dataloader)
        local term = Trainer:oneStepClassifier()
        while term == 0 do
	        term = Trainer:oneStepClassifier()
            step = step + 1
            if step%1000 == 0 then collectgarbage() end
        end
    end
    
    iteration = iteration + 1
    if iteration%5 == 0 then
        torch.save(string.format('./shortcuts/VGGNet_params50_%02d.t7',iteration),opt.cnn)
        --torch.save(string.format('./shortcuts/NCNet_with_params2_%02d.t7',iteration-1),opt.ncnet)
    end
    io.flush()
    collectgarbage()
end

opt.cnn:save(string.format('./shortcuts/VGGNet_%04d.t7',iteration))
