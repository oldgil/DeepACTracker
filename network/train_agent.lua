--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'tracking' 

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Tracking:')
cmd:text()
cmd:text('Options:')

cmd:option('-feat_size', 256, 'number of cnn feature_size')
cmd:option('-input_size', 256, 'number of lstm input_size')
cmd:option('-maxm', 120, 'number of tracked trajectories at the same time')
cmd:option('-rnn_size', 128, 'rnn hiden state dimensions')

cmd:option('-hist_len', 7, 'history length')
cmd:option('-eGreedy', 0, 'use eGreedy policy or not')

cmd:option('-_n_rois', 120, 'rois number')
cmd:option('-model_path','./models/AlexNet/TRCNN.lua','Path to the FRCNN model definition')
cmd:option('-model_weights','./data/trained_models/frcnn_alexnet_VOC2007_iter_40000.t7','Path to the FRCNN weights (used for testing)')
cmd:option('-scale', 600, 'Scale used for training and testing, currently only single scale is supported.')
cmd:option('-max_size', 1000, 'Max pixel size of the longest side of a scaled input image')
cmd:option('-pixel_means', {122.7717,115.9465,102.9801}, 'Pixel mean values (RGB order)')
--cmd:option('-pixel_means', {0,0,0}, 'Pixel mean values (BGR order)')
cmd:option('-weight_file_path','./pretrained_models/imgnet_alexnet.t7','')
cmd:option('-train',true,'')
    cmd:option('-lr',0.001,'')
    cmd:option('-lr_end',0.0001,'')
    cmd:option('-wc',0,'')
    cmd:option('-lr_endt',200000,'')
    cmd:option('-steps',200000,'')
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
cmd:option('-t',10,'time steps for update network')
cmd:option('-T',32,'time steps for update target network')
cmd:text()

local opt = cmd:parse(arg)
cmd:addTime('TrackNet','%F %T')
cutorch.setDevice(opt.gpu)
torch.manualSeed(3)

local trainSequence = {'MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'}
--local trainSequence = {'MOT16-02'}

print(opt.model_path)
--opt.network = tracking.TrackNet(opt)
opt.ncnet   = tracking.NCNet(opt)
opt.ncnet:training()
opt.network = torch.load('./shortcuts/Tracknet_with_params40_39.t7')
opt.network:training()
opt.dataloader = {}
opt.featurepool = {}
opt.filename = ''
local Trainer = tracking.RNNQLearner(opt)
local step = 0
local epochs = 40
local iteration = 0

logInfo = ''
logInfo2 = ''
cmd:log('./log/DQN_201611172137.log',{logInfo})
--cmd:log('./log/CLS_201611102229.log',{logInfo2})

while iteration < epochs do
    if iteration > 0 then opt.loadStored = 1 end
    for id,filename in pairs(trainSequence) do
        opt.filename = filename
        logInfo = string.format('Begin training '..filename..' start loading data...at iteration %d',iteration)
        --logInfo2 = string.format('Begin training '..filename..' start loading data...')
        --print(logInfo)

        opt.dataloader = {}
	    opt.featurepool = {}
        opt.dataloader = tracking.DataLoader(opt)
        opt.featurepool = tracking.Featurepool(opt)
        opt.featurepool:setStart()       
        Trainer:setData(opt.featurepool)
        local term, termS= Trainer:oneStep()
        --local term = Trainer:oneStepClassifier()
        --print(string.format('Total reward at 2nd frame is %f',opt.featurepool.totalReward))

        local fr = 2
        while term == 0 do
            term,termS = Trainer:oneStep()
            
	    --term = Trainer:oneStepClassifier()

            step = step + 1
            if step%1000 == 0 then collectgarbage() end
            fr = fr + 1
            --print(string.format('Total reward at %dth frame is %f',fr,opt.featurepool.totalReward))
        end
	--break
    end
    
    iteration = iteration + 1
    if iteration%10 == 0 then
        torch.save(string.format('./shortcuts/Tracknet_with_params40_%02d.t7',iteration+39),opt.network)
        --torch.save(string.format('./shortcuts/NCNet_with_params2_%02d.t7',iteration-1),opt.ncnet)
    end
    io.flush()
    collectgarbage()
end

opt.network:save(string.format('./shortcuts/Tracknet_%04d.t7',iteration),opt.network)
