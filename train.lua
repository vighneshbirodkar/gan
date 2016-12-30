require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'pl'
require 'paths'
require 'image'
require 'utils.plot'
require 'utils.base'

opt = lapp[[  
  --lrG              (default 0.0002)            learning rate  
  --lrD              (default 0.0002)            learning rate  
  --beta1            (default 0.5)               momentum term for adam
  -b,--batchSize     (default 100)               batch size  
  -g,--gpu           (default 0)                 gpu to use  
  --name             (default "default")           base directory to save logs  
  --optimizer        (default "adam")            "adam" | "sgd" | "adagrad"  
  --nEpochs          (default 100)               max training epochs  
  --seed             (default 1)                 random seed  
  --epochSize        (default 20000)             number of samples per epoch  
  --noiseDim         (default 100)               dimensionality of noise space
  --imageSize        (default 64)                size of image
  --dataset          (default moving_mnist)      dataset
  --movingDigits     (default 1)
  --dataPool         (default 200)
  --dataWarmup       (default 10)
  --nThreads         (default 1)                 number of dataloading threads
]]  

optSetup()
local nc = opt.geometry[1] 

require(('models/dcgan_%s'):format(opt.imageSize))
require(('data.%s'):format(opt.dataset))


local netG = makeG()
initModel(netG)
netG:cuda()
local params_G, grads_G = netG:getParameters()

local netD = makeD()
initModel(netD)
netD:cuda()
local params_D, grads_D = netD:getParameters()

local criterion = nn.BCECriterion()
criterion:cuda()


local inputImg = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local noise = torch.CudaTensor(opt.batchSize, opt.noiseDim, 1, 1)
local target = torch.CudaTensor(opt.batchSize)
local label_real = 1.0
local label_fake = 0
local noise = torch.CudaTensor(opt.batchSize, opt.noiseDim, 1, 1)
local currentBatch = nil

optimStateD = {learningRate = opt.lrD, beta1=opt.beta1, weightDecay=opt.weightDecay}
optimStateG = {learningRate = opt.lrG, beta1=opt.beta1, weightDecay=opt.weightDecay}


local fDx = function(x)
   grads_D:zero()

   -- Train Disc. on real images
   inputImg:copy(currentBatch.input)
   target:fill(label_real)
   local output_D = netD:forward(inputImg)
   local errD_real = criterion:forward(output_D, target)
   local gradReal = criterion:backward(output_D, target)
   netD:backward(inputImg, gradReal)

   --Train Disc. on generated images
   sampleNoise(noise)
   local genImage = netG:forward(noise)

   local output_D = netD:forward(genImage)
end

trainData = movingMNISTData('train')


for batch in trainData:run() do
   currentBatch = batch
   opt.optimizer(fDx, params_D, optimStateD)
end
