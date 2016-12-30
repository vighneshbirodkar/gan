th = require 'torch'
require 'paths'
require 'image'
tnt = require 'torchnet'

local path_dataset = '/scratch/vnb222/data/moving_mnist/mnist.t7/'
local path_trainset = paths.concat(path_dataset, 'train_32x32.t7')
local path_testset = paths.concat(path_dataset, 'test_32x32.t7')
th.manualSeed(0)


function getImage(image)
   return {input=((image:float() - 128)/255)}
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
   local batchDataset = tnt.BatchDataset{dataset=listDataset, batchsize=128,
					 perm=random_idx}
   local iter = tnt.DatasetIterator{dataset=batchDataset}
   return iter
end

trainData = movingMNISTData('train')

--print(trainData:get(1))
for batch in trainData:run() do
   print(batch)
end
