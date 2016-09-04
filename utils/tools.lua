function tile2d (input, tw, th) 
   assert(input:nDimension() == 2, 'tile2d: only 2 dimension input can be used')
   local sz1 = input:size(1)
   local sz2 = input:size(2)
   local vspan = torch.reshape(input, sz1*sz2, 1)
   local htile = torch.reshape(torch.repeatTensor(vspan,1,tw), sz1, sz2*tw)
   local tiled = torch.reshape(torch.repeatTensor(htile,1,th), sz1*th, sz2*tw)
   return tiled
end

function tileBatchImg (input, nCol)
   assert(input:nDimension() >= 3, 'tileBatchImg: more than 3 dimensions is required')
   local nBatch = input:size(1)
   local nRow = torch.ceil(nBatch / nCol)
   local nDim = input:nDimension()
   local nChannel
   local r,c
   if nDim == 3 then
      nChannel = 1
      r = input:size(2)
      c = input:size(3)
   else
      nChannel = input:size(2)
      r = input:size(3)
      c = input:size(4)
   end
   local palette
   if nDim == 3 then
      palette  = torch.Tensor(nRow*(r+1)-1, nCol*(c+1)-1):fill(1)
   else
      palette  = torch.Tensor(nChannel, nRow*(r+1)-1, nCol*(c+1)-1):fill(1)
   end
   for i = 1, nBatch do
      local irow = torch.floor((i-1)/nCol)
      local icol = (i-1) % nCol
      local irstart = irow*(r+1) + 1
      local irend = irstart+r-1
      local icstart = icol*(c+1) + 1
      local icend = icstart+c-1
      if nDim == 3 then
         palette[{{irstart,irend},{icstart,icend}}]:copy(input[i])
      else
         palette[{{},{irstart,irend},{icstart,icend}}]:copy(input[i])
      end
   end
   return palette 
end
