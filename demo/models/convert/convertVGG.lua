require 'cudnn'
require 'loadcaffe'

model = loadcaffe.load('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', 'cudnn')

for l = 1, 10 do model:remove() end

torch.save('./VGG-conv5-3Relu.t7', model)
print('done!')
