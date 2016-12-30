local SpatialConvolution = cudnn.SpatialConvolution
local SpatialMaxPooling =cudnn.SpatialMaxPooling
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization  or nn.SpatialBatchNormalization 
local SpatialFullConvolution = nn.SpatialFullConvolution
local ReLU = nn.ReLU
local LeakyReLU = nn.LeakyReLU
local nc = opt.geometry[1]


function makeG(ngf)
  local ngf = ngf or 64
  -- net
  local net = nn.Sequential()
  -- 1x1
  net:add(SpatialFullConvolution(opt.noiseDim, ngf*8, 4, 4))
  net:add(SpatialBatchNormalization(ngf*8)):add(nn.ReLU())
  -- 4x4
  net:add(SpatialFullConvolution(ngf*8, ngf*4, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ngf*4)):add(nn.ReLU())
  -- 8x8
  net:add(SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU())
  -- 16x16
  net:add(SpatialFullConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ngf)):add(nn.ReLU())
  -- 32x32
  net:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
  if opt.output == 'sigmoid' then
    net:add(nn.Sigmoid())
  elseif opt.output =='tanh' then
    net:add(nn.Tanh())
  else
    error('Unknown output: ' .. opt.output)
  end
  return net
end


function makeD(ndf)
  local ndf = ndf or 64
  local net = nn.Sequential()
  -- 64x64
  net:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ndf)):add(nn.ReLU(true))
  -- 32x32
  net:add(SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ndf*2)):add(nn.ReLU(true))
  -- 16x16
  net:add(SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ndf*4)):add(nn.ReLU(true))
  -- 8x8
  net:add(SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1))
  net:add(SpatialBatchNormalization(ndf*8)):add(nn.ReLU(true))
  -- 4x4
  net:add(SpatialConvolution(ndf*8, 1, 4, 4))
  net:add(nn.Sigmoid())
  return net
end
