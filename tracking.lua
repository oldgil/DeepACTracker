require 'image'
require 'cudnn'
require 'inn'
require 'nn'
require 'torch'
require 'xlua'
--matio = require 'matio'
--tds = require 'tds'
--config = dofile 'config.lua'
--config = config.parse(arg)
--cutorch.setDevice(config.GPU_ID)

-- Setting the random seed
torch.manualSeed(123)

tracking = {}

-- General Utilities
--torch.include('tracking','utils/GeneralUtils.lua')
--torch.include('tracking','models/RNNs/LSTM.lua')
--torch.include('tracking','models/Linker/LNET.lua')



torch.include('tracking','ROI/ROIPooling.lua')
torch.include('tracking','network/TrackNet.lua')
torch.include('tracking','network/DDRLNet.lua')
torch.include('tracking','network/AlexNet.lua')
torch.include('tracking','network/VGGNet.lua')
torch.include('tracking','network/RNNQLearner.lua')
torch.include('tracking','network/DDRLLearner.lua')
torch.include('tracking','network/CNNLearner.lua')
torch.include('tracking','train/DataLoader.lua')
torch.include('tracking','train/Featurepool.lua')
torch.include('tracking','train/MemoryPool.lua')
