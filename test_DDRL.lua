--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'tracking' 
require 'utils.mot_data'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test Agent in Tracking:')
cmd:text()
cmd:text('Options:')

cmd:option('-feat_size', 4096, 'number of cnn feature_size')
cmd:option('-input_size', 4096, 'number of lstm input_size')
cmd:option('-maxm', 120, 'number of tracked trajectories at the same time')
cmd:option('-rnn_size', 256, 'rnn hiden state dimensions')

cmd:option('-hist_len', 5, 'history length')
cmd:option('-eGreedy', 0, 'use eGreedy policy or not')

cmd:option('-_n_rois', 120, 'rois number')
cmd:option('-model_path','./models/AlexNet/TRCNN.lua','Path to the FRCNN model definition')
cmd:option('-model_weights','./data/trained_models/frcnn_alexnet_VOC2007_iter_40000.t7','Path to the FRCNN weights (used for testing)')
cmd:option('-scale', 600, 'Scale used for training and testing, currently only single scale is supported.')
cmd:option('-max_size', 1000, 'Max pixel size of the longest side of a scaled input image')
cmd:option('-pixel_means', {122.7717,115.9465,102.9801}, 'Pixel mean values (RGB order)')
--cmd:option('-pixel_means', {0,0,0}, 'Pixel mean values (BGR order)')

cmd:option('-train',false,'')
 --cmd:option('-optim_snapshot_iters', 10000, 'Iterations between snapshots (used for saving the network)')

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

--local testSequence = {'MOT16-02','MOT16-04','MOT16-05','MOT16-09','MOT16-10','MOT16-11','MOT16-13'}
--local testSequence = {'MOT16-01','MOT16-03','MOT16-06','MOT16-07','MOT16-08','MOT16-12','MOT16-14'}
--local testSequence = {'MOT16-14'}
--local testSequence = {'ADL-Rundle-6','PETS09-S2L1','ETH-Bahnhof','ETH-Pedcross2','KITTI-17','TUD-Campus','Venice-2'}
local testSequence = {'MOT16-02'}
opt.AlexNet   = torch.load('./shortcuts/CNN_0025.t7')
opt.AlexNet:evaluate()
--opt.cnn   = torch.load('./shortcuts/cnn_params40_50.t7')
opt.cnn   = torch.load('./shortcuts/AlexNet_0065.t7')
opt.cnn:evaluate()
opt.ddlnet = torch.load('./shortcuts/DDRLNet_MOT1602_128_5_noise_60_0060.t7')
opt.ddlnet:evaluate()
opt.dataloader = {}
opt.memorypool = {}
opt.filename = ''
local Trainer = tracking.DDRLLearner(opt)
local step = 0
logInfo = ''
cmd:log('./log/Test_201701171940.log',{logInfo})

    for id,filename in pairs(testSequence) do
        opt.filename = filename
        logInfo = string.format('Begin testing '..filename..' start loading data')
        --logInfo2 = string.format('Begin training '..filename..' start loading data...')
        print(logInfo)

         opt.dataloader = nil
	    opt.memorypool = nil
        opt.dataloader = tracking.DataLoader(opt)
         opt.memorypool = tracking.MemoryPool(opt)
        opt.memorypool:setStart()       
        Trainer:setData(opt.memorypool)
        local term = Trainer:TestOneStep()
        local fr = 2
        while term == 0 do
            term = Trainer:TestOneStep()
            step = step + 1
            if step%1000 == 0 then collectgarbage() end
            fr = fr + 1

            print(string.format(' %dth frame is evaluated!',fr))
        end
		local file = io.open('./results/01171940/')
		if file then file:close() end
		if file == nil then os.execute("mkdir ./results/01171940/") end
        opt.memorypool:filter()
        writeTXT(opt.memorypool.filtedRes,'./results/01171940/'..filename..'.txt',opt.dataloader.im_scale)

        io.flush()
        collectgarbage()
    end




