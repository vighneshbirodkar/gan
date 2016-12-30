require 'image'
require 'os'

function saveGen(netG, noise, name)
   local N = opt.batchSize
   local gen = netG:forward(noise)
   local to_plot = {}
   for i=1,N do
      table.insert(to_plot, gen[i]:float())
   end
   local genImg = image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=math.sqrt(N)}
   image.save(('%s/gen/%s.png'):format(opt.save, name), genImg)
end

