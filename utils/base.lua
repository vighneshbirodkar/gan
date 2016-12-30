require 'torch'


function sampleNoise(z, seed)

   if seed == nil then
      seed = torch.uniform()
   end
   local gen = torch.Generator()
   torch.manualSeed(gen, seed)
   local z_cpu = nil
   if opt.noise == 'uniform' then
      z_cpu = torch.rand(gen, z:size())
   else
      z_cpu = torch.randn(gen, z:size())
   end
   z:copy(z_cpu)
   
end

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') or name:find('Linear') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

function initModel(model)
  for _, m in pairs(model:listModules()) do
    weights_init(m)
  end
end

function optSetup()
   
   opt.save = 'logs/' .. opt.name 
   os.execute('mkdir -p ' .. opt.save .. '/gen/')

   if opt.optimizer == "adam" then
      opt.optimizer = optim.adam
   elseif opt.optimizer == "sgd" then
      opt.optimizer = optim.sgd
   elseif opt.optimizer == "adagrad" then
      opt.optimizer = optim.adagrad
   else
      error('Unknown optimizer: ' .. opt.optimizer)
   end
   print(opt)

   torch.manualSeed(opt.seed)
   cutorch.manualSeed(opt.seed)
   math.randomseed(opt.seed)

   if opt.dataset:find('mnist') then 
      opt.geometry = {1, opt.imageSize, opt.imageSize}
      opt.output = 'sigmoid'
   else
      opt.geometry = {3, opt.imageSize, opt.imageSize}
      opt.output = 'tanh'
   end
end
