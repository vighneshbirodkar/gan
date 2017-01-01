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
  --printEvery       (default 10)                How often to print stats.
  --dispEvery        (default 50)                How often to display generated images.
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


local timer = tnt.TimeMeter({unit=true})
local gErrorMeter = tnt.MovingAverageValueMeter({windowsize=1})
local dErrorMeter = tnt.MovingAverageValueMeter({windowsize=1})


optimStateD = {learningRate = opt.lrD, beta1=opt.beta1, weightDecay=opt.weightDecay}
optimStateG = {learningRate = opt.lrG, beta1=opt.beta1, weightDecay=opt.weightDecay}


local fDx = function(_)
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
   target:fill(label_fake)

   local genImage = netG:forward(noise)
   
   local output_D = netD:forward(genImage)
   local errD_fake = criterion:forward(output_D, target)
   local gradFake = criterion:backward(output_D, target)
   netD:backward(genImage, gradFake)

   local errD = errD_real + errD_fake
   dErrorMeter:add(errD)

   return errD, grads_D
end


local fGx = function(_)
   grads_G:zero()

   target:fill(label_real)

   local output = netD.output
   local errG = criterion:forward(output, target)
   local grad_C = criterion:backward(output, target)
   local grad_D = netD:updateGradInput(inputImg, grad_C)

   netG:backward(noise, grad_D)

   gErrorMeter:add(errG)
   return errG, grads_G
end

local trainData, trainSize = movingMNISTData('train')
--local trainSize = trainData:size()

for epoch=1,opt.nEpochs do
   local seenSamples = 0
   timer:reset()
   gErrorMeter:reset()
   dErrorMeter:reset()

   local numBatches = 0
   for batch in trainData:run() do
      currentBatch = batch
      numBatches = numBatches + 1
      local samples = numBatches*opt.batchSize

      opt.optimizer(fDx, params_D, optimStateD)
      opt.optimizer(fGx, params_G, optimStateG)

      timer:incUnit()
      if (numBatches%opt.printEvery) == 0 then
	 print(('%d/%d samples %f secs/batch G-Err-Prob=%f D-Err-Prob=%f'):
	       format(samples, trainSize, timer:value(), gErrorMeter:value(),
		      dErrorMeter:value()))
      end
      if (numBatches%opt.dispEvery) == 0 then
	 sampleNoise(noise, opt.seed)
	 local name = ('Epoch_%03d-Step_%05d'):format(epoch, numBatches)
	 print(('Saving image %s'):format(name))
	 saveGen(netG, noise, name)
      end
   end
end
 
