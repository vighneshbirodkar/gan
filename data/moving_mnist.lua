th = require 'torch'
require 'paths'
require 'image'
tnt = require 'torchnet'

local path_dataset = '/scratch/vnb222/data/moving_mnist/mnist.t7/'
local path_trainset = paths.concat(path_dataset, 'train_32x32.t7')
local path_testset = paths.concat(path_dataset, 'test_32x32.t7')
th.manualSeed(0)


function getImage(img)
   img = img:double()
   local out = image.scale(img, opt.imageSize, opt.imageSize)
   return {input=((out:float() - 0)/255)}
end


function random_idx(idx, size)
   return th.random(1, size)
end


function movingMNISTData(dataset)
   local data = nil
   if dataset == 'train' then
      data = th.load(path_trainset, 'ascii')

   else
      data = th.load(path_testset, 'ascii')
   end

   local listDataset = tnt.ListDataset(data.data:long(), getImage)
   local batchDataset = tnt.BatchDataset{dataset=listDataset, batchsize=opt.batchSize,
					 perm=random_idx}
   local iter = tnt.DatasetIterator{dataset=batchDataset}
   return iter, listDataset:size()
end

