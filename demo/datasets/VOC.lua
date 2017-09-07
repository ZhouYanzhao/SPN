local function normalizeGlobal(data, mean_, std_)
    local std = std_ or data:std()
    local mean = mean_ or data:mean()
    data:add(-mean)
    data:mul(1/std)
    return mean, std
end

local function initDataset(dataset_name, datadir)
    local datasetInfo = {
        classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
                   'cat','chair','cow','diningtable','dog','horse','motorbike',
                   'person','pottedplant','sheep','sofa','train','tvmonitor'},
        imgsetpath = paths.concat(datadir, dataset_name,'ImageSets','Main','%s.txt'),
        imgpath = paths.concat(datadir,dataset_name,'JPEGImages','%s.jpg'),
        classmap = torch.load('./datasets/classmap_'..dataset_name..'.t7') 
    }

    return datasetInfo
end

local function hflip(img)
    return torch.random(0,1) == 1 and img or image.hflip(img)
end

local function randomCrop(img, imsize) 
    local x = torch.random(1, img:size(2) - imsize - 1)
    local y = torch.random(1, img:size(3) - imsize - 1)
 
     local img = image.crop(img, x, y, x + imsize, y + imsize)
    return img
end

local function preprocess(img)
    img = image.scale(img, 256, 256, 'bilinear') 
    img = randomCrop(img, 224)
    img = hflip(img)
 
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)

    img:add(-1, mean_pixel)
 
    return img
end

return function (opt)
    torch.manualSeed(1234)
    local tnt = require 'torchnet'
    local provider = initDataset(opt.dataset..opt.year, opt.datadir)
 
    return function (mode)
        return tnt.ParallelDatasetIterator{
            nthread = 16,
            init = function()
                require 'torchnet'
                require 'nn'
                require 'image'
            end,
            closure = function()
                local list_dataset = tnt.ListDataset{
                    filename = string.format(provider.imgsetpath, mode),
                    load = function(imageName)
                        local img = image.load(string.format(provider.imgpath, imageName))
                        local input = preprocess(img)

                        return {
                            input = input:float(),
                            target = torch.LongTensor(provider.classmap[imageName]:long()),
                        }
                    end,
                }
 
                if mode == 'train' or mode == 'trainval' then
                    if opt.batchSize > 1 then
                        return list_dataset
                               :shuffle()
                               :batch(opt.batchSize, 'skip-last')
                    else
                        return list_dataset
                               :shuffle()
                    end
                else
                    if opt.batchSize > 1 then
                        return list_dataset
                               :batch(opt.batchSize, 'include-last')
                    else 
                        return list_dataset
                    end
                end
            end,
        }
    end
end