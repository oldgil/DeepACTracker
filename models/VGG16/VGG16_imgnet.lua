require 'cudnn'
 local function create_model(opt) 
  	local name = 'vgg16'
  	backend = backend or cudnn

	-- SHARED PART
  	local shared   = nn.Sequential()
	local conv1_1 = cudnn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1, 1)
	conv1_1.name = 'conv1_1'
	-- Freeze conv1_1
	conv1_1.accGradParameters = function() end 
  	shared:add(conv1_1)
  	shared:add(cudnn.ReLU(true))
  	-- Freeze conv1_2
  	local conv1_2 = cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1, 1)
  	conv1_2.name = 'conv1_2'
  	conv1_2.accGradParameters = function() end 
	shared:add(conv1_2)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	-- Freeze conv2_1
	local conv2_1 = cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1, 1)
	conv2_1.name = 'conv2_1'
	conv2_1.accGradParameters = function() end 
	shared:add(conv2_1)
	shared:add(cudnn.ReLU(true))
	-- Freeze conv2_2
	local conv2_2 = cudnn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1, 1)
	conv2_2.name = 'conv2_2'
	conv2_2.accGradParameters = function() end 
	shared:add(conv2_2)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	-- Freeze conv3_1
	local conv3_1 = cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1, 1)
	conv3_1.name = 'conv3_1'
	conv3_1.accGradParameters = function() end
	shared:add(conv3_1)
	shared:add(cudnn.ReLU(true))

	local conv3_2 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
	conv3_2.name = 'conv3_2'
	conv3_2.accGradParameters = function() end
	shared:add(conv3_2)
	shared:add(cudnn.ReLU(true))


	local conv3_3 = cudnn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1, 1)
	conv3_3.name = 'conv3_3'
	conv3_3.accGradParameters = function() end
	shared:add(conv3_3)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())


	local conv4_1 = cudnn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1, 1)
	conv4_1.name = 'conv4_1'
	conv4_1.accGradParameters = function() end
	shared:add(conv4_1)
	shared:add(cudnn.ReLU(true))


	local conv4_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	conv4_2.name = 'conv4_2'
	conv4_2.accGradParameters = function() end
	shared:add(conv4_2)
	shared:add(cudnn.ReLU(true))
	
	local conv4_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	conv4_3.name = 'conv4_3'
	conv4_3.accGradParameters = function() end
	shared:add(conv4_3)
	shared:add(cudnn.ReLU(true))
	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())


	local conv5_1 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	conv5_1.name = 'conv5_1'
	conv5_1.accGradParameters = function() end
	shared:add(conv5_1)
	shared:add(cudnn.ReLU(true))

	local conv5_2 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	conv5_2.name = 'conv5_2'

	conv5_2.accGradParameters = function() end
	shared:add(conv5_2)
	shared:add(cudnn.ReLU(true))

	local conv5_3 = cudnn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1, 1)
	conv5_3.name = 'conv5_3'
	conv5_3.accGradParameters = function() end
	shared:add(conv5_3)
	shared:add(cudnn.ReLU(true))

	shared:add(cudnn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	shared:add(nn.View(-1):setNumInputDims(3))
	local fc6 = nn.Linear(25088, 4096)
	fc6.name = 'fc6'
	shared:add(fc6)
	shared:add(cudnn.ReLU(true))
	shared:add(nn.Dropout(0.500000))
	local fc7 = nn.Linear(4096, 4096)
	fc7.name = 'fc7'
	shared:add(fc7)
	shared:add(cudnn.ReLU(true))
	shared:add(nn.Dropout(0.500000))
	shared:add(nn.Linear(4096, 1000))

	shared:cuda()
	return shared
end
return create_model























