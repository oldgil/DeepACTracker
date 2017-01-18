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

cmd:option('-feat_size', 4096, 'number of cnn feature_size')
cmd:option('-input_size', 4096, 'number of lstm input_size')
cmd:option('-maxm', 120, 'number of tracked trajectories at the same time')
cmd:option('-rnn_size', 512, 'rnn hiden state dimensions')

cmd:option('-hist_len',5, 'history length')
cmd:option('-eGreedy', 0, 'use eGreedy policy or not')

cmd:option('-_n_rois', 80, 'rois number')
cmd:option('-model_path','./models/AlexNet/TRCNN.lua','Path to the FRCNN model definition')
cmd:option('-model_weights','./data/trained_models/frcnn_alexnet_VOC2007_iter_40000.t7','Path to the FRCNN weights (used for testing)')
cmd:option('-scale', 600, 'Scale used for training and testing, currently only single scale is supported.')
cmd:option('-max_size', 1000, 'Max pixel size of the longest side of a scaled input image')
cmd:option('-pixel_means', {122.7717,115.9465,102.9801}, 'Pixel mean values (RGB order)')
--cmd:option('-pixel_means', {0,0,0}, 'Pixel mean values (BGR order)')
cmd:option('-weight_file_path','./pretrained_models/imgnet_alexnet.t7','')
cmd:option('-train',true,'')
cmd:option('-lr',0.001,'')
cmd:option('-lr_end',0.00001,'')
cmd:option('-wc',0,'')
cmd:option('-lr_endt',400000,'')
cmd:option('-steps',400000,'')
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
cmd:option('-T',21,'time steps for update target network')
cmd:text()

local opt = cmd:parse(arg)
cmd:addTime('TrackNet','%F %T')
cutorch.setDevice(opt.gpu)
torch.manualSeed(323)

local trainSequence = {'MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'}
--local trainSequence = {'MOT16-02'}

--print(opt.model_path)
--opt.network = tracking.TrackNet(opt)
opt.cnn   = torch.load('./shortcuts/AlexNet_0065.t7')
opt.cnn:evaluate()
opt.ddlnet = tracking.DDRLNet(opt)
--opt.ddlnet = torch.load('./shortcuts/DDRLNet_params_noise_80_20.t7') 
opt.dataloader = {}
opt.memorypool = {}
opt.filename = ''
local Trainer = tracking.DDRLLearner(opt)
local step = 0
local epochs = 60
local iteration = 0

logInfo = ''
cmd:log('./log/MOT16_512_5_noise_neg_clip_delta_20170118_0.log',{logInfo})
--cmd:log('./log/CLS_201611102229.log',{logInfo2})

while iteration < epochs do
    
    for i=1,#trainSequence do
        local n = torch.random(1,#trainSequence)
        local tmpFileName = trainSequence[n]
        trainSequence[n] = trainSequence[i]
        trainSequence[i] = tmpFileName
    end

    for id,filename in pairs(trainSequence) do
        opt.filename = filename
        logInfo = string.format('Begin training '..filename..' start loading data...at iteration %d',iteration)
        print(logInfo)

        opt.dataloader = nil
	    opt.memorypool = nil
        opt.dataloader = tracking.DataLoader(opt)
        opt.memorypool = tracking.MemoryPool(opt)
        opt.memorypool:setStart()       
        Trainer:setData(opt.memorypool)
        local term = Trainer:oneStep()       local fr = 2
        while term == 0 do
            term = Trainer:oneStep()
            step = step + 1
            if step%1000 == 0 then collectgarbage() end
            fr = fr + 1
            --print(string.format('Total reward at %dth frame is %f',fr,opt.featurepool.totalReward))
        end
    end
    
    iteration = iteration + 1
    if iteration%10 == 0 then
        torch.save(string.format('./shortcuts/DDRLNet_MOT16_512_5_params_noise_100_%02d.t7',iteration),opt.ddlnet)
    end
    io.flush()
    collectgarbage()
end

torch.save(string.format('./shortcuts/DDRLNet_MOT16_512_5_noise_100_%04d.t7',iteration),opt.ddlnet)
