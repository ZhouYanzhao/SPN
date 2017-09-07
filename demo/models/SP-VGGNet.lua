require 'cudnn'

local utils = paths.dofile '../utils.lua'

return function(opt)
    -- load pretrained model for fine-tuning
    model = torch.load('./models/convert/VGG-conv5-3Relu.t7')
    model:add(cudnn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1, 2))
    model:add(cudnn.ReLU(true))
    -- Soft Proposal module
    model:add(nn.SoftProposal())
    -- classifier
    model:add(nn.SpatialSumOverMap())
    model:add(nn.View(-1, 1024))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(1024, opt.numClasses))

    return model
end