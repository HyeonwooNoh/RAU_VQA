require 'image'
require 'hdf5'
local stringx = require 'pl.stringx'
local file_utils = require 'utils.file_utils'
local qa_utils = require 'utils.qa_utils'
local threads = require 'threads'
local cjson = require 'cjson'
local transform = require 'utils.transforms'

local vqa_prepro_loader = {}
vqa_prepro_loader.__index = vqa_prepro_loader

local dataclass = {}
dataclass.__index = dataclass

function dataclass.load_data(dataSubTypes, info, ques, imgPath_list,
                             batch_size, opt_prefetch)

   local data = {}
   setmetatable(data, dataclass)
   -- ques
   --    .question: tensor[nExample, SeqLen] (word index)
   --    .lengths_q: tensor[nExample] (length of question) 
   --    .img_list: (img_pos_train[qid] = imgidx (in json file))
   --    .question_id: tensor[nExample] (question_id_train[qid] = question_id(of mscoco))
   --    .answers: tensor[nExample] (answer index) (multiple choice answer)

   -- construct image list
   local img_list = {}
   for i=1, ques.img_list:size(1) do img_list[i] = imgPath_list[ques.img_list[i]] end

   -- set this data option
   data.data_subtype = table.concat(dataSubTypes)
   data.split = 'train'

   -- set initial values
   data.img_list = img_list
   data.questions = ques.question
   data.question_len = ques.lengths_q
   data.question_id = ques.question_id
   data.datatype = ques.datatype
   data.answers = ques.answers
   data.seq_len = info.seq_len
   data.ex_num_train = data.questions:size(1)

   -- init for mini-batch fetcher
   data.iter_index = 0
   data.batch_index = 0
   data.batch_size = batch_size
   data.batch_order = torch.range(1, data.ex_num_train) -- in order
   data.iter_per_epoch = torch.floor(data.ex_num_train / data.batch_size)
   data.opt_batch_order = 1
  
   -- mean_bgr
   data.mean_bgr = torch.Tensor({103.939, 116.779, 123.68}) 
   data.meanstd = {
      mean = { 0.485, 0.456, 0.406 },
      std = { 0.229, 0.224, 0.225 },
   }
   data.normalize = transform.Compose{
                       transform.ColorNormalize(data.meanstd),
                    }

   -- prefetching
   data.opt_prefetch = opt_prefetch
   if data.opt_prefetch then
      -- thread for prefetching
      data.pool = threads.Threads(1, function ()
                                        require 'image'
                                        transform = require 'utils.transforms'
                                        meanstd = {
                                           mean = { 0.485, 0.456, 0.406 },
                                           std = { 0.229, 0.224, 0.225 },
                                        }  
                                        normalize = transform.Compose{
                                           transform.ColorNormalize(meanstd),
                                        }
                                     end)
      data.prefetch_init = false
   end

   return data
end

function dataclass.load_testdata(dataSubTypes, info, ques, imgPath_list,
                             batch_size, opt_prefetch)

   local data = {}
   setmetatable(data, dataclass)

   -- ques
   --    .question: tensor[nExample, SeqLen] (word index)
   --    .lengths_q: tensor[nExample] (length of question) 
   --    .img_list: (img_pos_test[qid] = imgidx (in json file))
   --    .question_id: tensor[nExample] (question_id_train[qid] = question_id(of mscoco))
   --    .mc_ans: tensor[nExample, nMultChoice] (answer index) (multiple choices of answers)

   -- construct image list
   local img_list = {}
   for i=1,ques.img_list:size(1) do img_list[i] = imgPath_list[ques.img_list[i]] end

   -- set this data option
   data.data_subtype = dataSubTypes
   data.split = 'test'

   -- set initial values
   data.img_list = img_list
   data.questions = ques.question
   data.question_len = ques.lengths_q
   data.question_id = ques.question_id
   data.datatype = ques.datatype
   data.mc_ans = ques.mc_ans
   data.seq_len = info.seq_len
   data.ex_num_train = data.questions:size(1)

   -- init for mini-batch fetcher
   data.iter_index = 0
   data.batch_index = 0
   data.batch_size = batch_size
   data.batch_order = torch.range(1, data.ex_num_train) -- in order
   data.iter_per_epoch = torch.floor(data.ex_num_train / data.batch_size)
   data.opt_batch_order = 1

   -- mean_bgr
   data.mean_bgr = torch.Tensor({103.939, 116.779, 123.68}) 
   data.meanstd = {
      mean = { 0.485, 0.456, 0.406 },
      std = { 0.229, 0.224, 0.225 },
   }
   data.normalize = transform.Compose{
                       transform.ColorNormalize(data.meanstd),
                    }

   -- prefetching
   data.opt_prefetch = opt_prefetch
   if data.opt_prefetch then
      -- thread for prefetching
      data.pool = threads.Threads(1, function ()
                                        require 'image'
                                        transform = require 'utils.transforms'
                                        meanstd = {
                                           mean = { 0.485, 0.456, 0.406 },
                                           std = { 0.229, 0.224, 0.225 },
                                        }  
                                        normalize = transform.Compose{
                                           transform.ColorNormalize(meanstd),
                                        }
                                     end)
      data.prefetch_init = false
   end

   return data
end
function vqa_prepro_loader:qtable_to_tokens(qtable)
   local q = torch.zeros(self.seq_len,1) + 1
   local q_len = torch.zeros(1)
   for k, v in pairs(qtable) do
      if vqa_data.vocab_map[v] ~= nil then
         q[k][1] = vqa_data.vocab_map[v]
      else
         q[k][1] = vqa_data.vocab_map['UNK']
      end
      q_len[1] = k
   end
   return q, q_len
end
function vqa_prepro_loader:question_to_tokens(str)
   local q = torch.zeros(self.seq_len,1) + 1
   local q_len = torch.zeros(1)
   local str_tokens = stringx.split(str)
   for k, v in pairs(str_tokens) do
      if vqa_data.vocab_map[v] ~= nil then
         q[k][1] = vqa_data.vocab_map[v]
      else
         q[k][1] = vqa_data.vocab_map['UNK']
      end
      q_len[1] = k
   end
   return q, q_len
end
function vqa_prepro_loader:tokens_to_q_table(tokens)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens:size()[1] do
      table.insert(str, self.vocab_dict[tokens[i]])
   end
   return str
end
function vqa_prepro_loader:tokens_to_q_table_with_len(tokens, tokens_len)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens_len do
      table.insert(str, self.vocab_dict[tokens[i]])
   end
   return str
end
function vqa_prepro_loader:tokens_to_question_with_len(tokens, tokens_len)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens_len do
      table.insert(str, self.vocab_dict[tokens[i]])
      table.insert(str, ' ')
   end
   return table.concat(str)
end
function vqa_prepro_loader:tokens_to_question(tokens)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens:size()[1] do
      table.insert(str, self.vocab_dict[tokens[i]])
      table.insert(str, ' ')
   end
   return table.concat(str)
end
function vqa_prepro_loader:tokens_to_answer(token)
   assert(type(token) == 'number')
   return self.answer_dict[token]
end

function dataclass:next_batch_twofeats(cocofeatpath1, feat_dim1, feat_w1, feat_h1,
                                       cocofeatpath2, feat_dim2, feat_w2, feat_h2)
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a
   local batch_feat1
   local batch_feat2
   local loc_feat_list

   feat_w1 = feat_w1 or 1
   feat_h1 = feat_h1 or 1
   feat_w2 = feat_w2 or 1
   feat_h2 = feat_h2 or 1

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_twofeats' then
         if feat_w1 > 1 or feat_h1 > 1 then
            self.batch_feat1 = torch.zeros(self.batch_size, feat_dim1, feat_w1, feat_h1)
         else 
            self.batch_feat1 = torch.zeros(self.batch_size, feat_dim1)
         end
         if feat_w2 > 1 or feat_h2 > 1 then
            self.batch_feat2 = torch.zeros(self.batch_size, feat_dim2, feat_w2, feat_h2)
         else 
            self.batch_feat2 = torch.zeros(self.batch_size, feat_dim2)
         end
         self.feat_dim1 = feat_dim1
         self.feat_w1 = feat_w1
         self.feat_h1 = feat_h1
         self.feat_dim2 = feat_dim2
         self.feat_w2 = feat_w2
         self.feat_h2 = feat_h2
         loc_feat_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local cocoimg_name = paths.basename(self.img_list[ann_idx])
            local img_ext = paths.extname(cocoimg_name)
            -- ex) 'COCO_train2014_000000357413.t7'
            local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
            loc_feat_list[i] = cocofeat_name
         end
         for i = 1, self.batch_size do
            -- feature1
            local feat_path1 = paths.concat(cocofeatpath1, loc_feat_list[i]) 
            local feature1 = torch.load(feat_path1)
            if feat_w1 > 1 or feat_h1 > 1 then
               assert(feature1:size(1) == feat_dim1, 'feat_dim1 mismatch') 
               assert(feature1:size(2) == feat_w1, 'feat_w1 mismatch') 
               assert(feature1:size(3) == feat_h1, 'feat_h1 mismatch') 
            else
               assert(feature1:dim() == 1, 'only 1 dim feature could be loaded with this method')
               assert(feature1:nElement() == feat_dim1, 
                   string.format('dim mismatch: dim of saved feature is: %d', feature1:nElement()))
            end
            self.batch_feat1[i] = feature1
            -- feature2
            local feat_path2 = paths.concat(cocofeatpath2, loc_feat_list[i]) 
            local feature2 = torch.load(feat_path2)
            if feat_w2 > 1 or feat_h2 > 1 then
               assert(feature2:size(1) == feat_dim2, 'feat_dim2 mismatch') 
               assert(feature2:size(2) == feat_w2, 'feat_w2 mismatch') 
               assert(feature2:size(3) == feat_h2, 'feat_h2 mismatch') 
            else
               assert(feature2:dim() == 1, 'only 1 dim feature could be loaded with this method')
               assert(feature2:nElement() == feat_dim2, 
                   string.format('dim mismatch: dim of saved feature is: %d', feature2:nElement()))
            end
            self.batch_feat2[i] = feature2
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_twofeats'
      end

      assert(self.feat_dim1 == feat_dim1, 'feat_dim1 have to be save all the time')
      assert(self.feat_w1 == feat_w1, 'feat_w1 have to be save all the time')
      assert(self.feat_h1 == feat_h1, 'feat_h1 have to be save all the time')
      assert(self.feat_dim2 == feat_dim2, 'feat_dim2 have to be save all the time')
      assert(self.feat_w2 == feat_w2, 'feat_w2 have to be save all the time')
      assert(self.feat_h2 == feat_h2, 'feat_h2 have to be save all the time')
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      batch_feat1 = self.batch_feat1:clone()
      batch_feat2 = self.batch_feat2:clone()
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_feat_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local cocoimg_name = paths.basename(self.img_list[ann_idx])
         local img_ext = paths.extname(cocoimg_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
         loc_feat_list[i] = cocofeat_name
      end
 
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      self.pool:addjob(
         function ()
            local pre_feature1
            local pre_feature2
            if feat_w1 > 1 or feat_h1 > 1 then
               pre_feature1 = torch.zeros(loc_batch_size, feat_dim1, feat_w1, feat_h1)
            else 
               pre_feature1 = torch.zeros(loc_batch_size, feat_dim1)
            end
            if feat_w2 > 1 or feat_h2 > 1 then
               pre_feature2 = torch.zeros(loc_batch_size, feat_dim2, feat_w2, feat_h2)
            else 
               pre_feature2 = torch.zeros(loc_batch_size, feat_dim2)
            end
            for i = 1, loc_batch_size do
               local feat_path1 = paths.concat(cocofeatpath1, loc_feat_list[i])
               local feature1 = torch.load(feat_path1)
               if feat_w1 > 1 or feat_h1 > 1 then
                  assert(feature1:size(1) == feat_dim1, 'feat_dim1 mismatch') 
                  assert(feature1:size(2) == feat_w1, 'feat_w1 mismatch') 
                  assert(feature1:size(3) == feat_h1, 'feat_h1 mismatch') 
               else
                  assert(feature1:dim() == 1, 'only 1 dim feature could be loaded with this method')
                  assert(feature1:nElement() == feat_dim1, 
                      string.format('dim mismatch: dim of saved feature is: %d', feature1:nElement()))
               end
               pre_feature1[i] = feature1

               local feat_path2 = paths.concat(cocofeatpath2, loc_feat_list[i])
               local feature2 = torch.load(feat_path2)
               if feat_w2 > 1 or feat_h2 > 1 then
                  assert(feature2:size(1) == feat_dim2, 'feat_dim2 mismatch') 
                  assert(feature2:size(2) == feat_w2, 'feat_w2 mismatch') 
                  assert(feature2:size(3) == feat_h2, 'feat_h2 mismatch') 
               else
                  assert(feature2:dim() == 1, 'only 1 dim feature could be loaded with this method')
                  assert(feature2:nElement() == feat_dim2, 
                      string.format('dim mismatch: dim of saved feature is: %d', feature2:nElement()))
               end
               pre_feature2[i] = feature2
            end
            return pre_feature1, pre_feature2
         end,
         function (pre_feature1, pre_feature2)
            self.batch_feat1 = pre_feature1
            self.batch_feat2 = pre_feature2
         end
      )
   else
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      if feat_w1 > 1 or feat_h1 > 1 then
         batch_feat1 = torch.zeros(self.batch_size, feat_dim1, feat_w1, feat_h1)
      else 
         batch_feat1 = torch.zeros(self.batch_size, feat_dim1)
      end
      if feat_w2 > 1 or feat_h2 > 1 then
         batch_feat2 = torch.zeros(self.batch_size, feat_dim2, feat_w2, feat_h2)
      else 
         batch_feat2 = torch.zeros(self.batch_size, feat_dim2)
      end
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end

      loc_feat_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local cocoimg_name = paths.basename(self.img_list[ann_idx])
         local img_ext = paths.extname(cocoimg_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
         loc_feat_list[i] = cocofeat_name
      end
      for i = 1, self.batch_size do
         -- feature1
         local feat_path1 = paths.concat(cocofeatpath1, loc_feat_list[i]) 
         local feature1 = torch.load(feat_path1)
         if feat_w1 > 1 or feat_h1 > 1 then
            assert(feature1:size(1) == feat_dim1, 'feat_dim1 mismatch') 
            assert(feature1:size(2) == feat_w1, 'feat_w1 mismatch') 
            assert(feature1:size(3) == feat_h1, 'feat_h1 mismatch') 
         else
            assert(feature1:dim() == 1, 'only 1 dim feature could be loaded with this method')
            assert(feature1:nElement() == feat_dim1, 
                string.format('dim mismatch: dim of saved feature is: %d', feature1:nElement()))
         end
         batch_feat1[i] = feature1

         -- feature2
         local feat_path2 = paths.concat(cocofeatpath2, loc_feat_list[i]) 
         local feature2 = torch.load(feat_path2)
         if feat_w2 > 1 or feat_h2 > 1 then
            assert(feature2:size(1) == feat_dim2, 'feat_dim2 mismatch') 
            assert(feature2:size(2) == feat_w2, 'feat_w2 mismatch') 
            assert(feature2:size(3) == feat_h2, 'feat_h2 mismatch') 
         else
            assert(feature2:dim() == 1, 'only 1 dim feature could be loaded with this method')
            assert(feature2:nElement() == feat_dim2, 
                string.format('dim mismatch: dim of saved feature is: %d', feature2:nElement()))
         end
         batch_feat2[i] = feature2
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat1:clone(), batch_feat2:clone(),
          batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end
function dataclass:depreprocess_image(image)
   local height = image:size(2)
   local width = image:size(3)
   local mean_bgr = self.mean_bgr:repeatTensor(height,width,1)
         mean_bgr = mean_bgr:permute(3,2,1)
   image = image:add(mean_bgr):div(255):index(1,torch.LongTensor{3,2,1})
   return image
end
function dataclass:next_batch_feat_image_qid(question_id, 
                                             cocofeatpath, feat_dim, feat_w, feat_h,
                                             cocoimgpath, height, width,
                                             randcrop, reheight, rewidth)
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a
   local batch_feat
   local loc_feat_list
   local batch_img
   local loc_img_list

   feat_w = feat_w or 1
   feat_h = feat_h or 1
   randcrop = randcrop or false

   local mean_bgr = self.mean_bgr:repeatTensor(height,width,1)
         mean_bgr = mean_bgr:permute(3,2,1)

   local idx
   print(question_id)
   print(self.question_id[1])
   for i=1,self.question_id:size(1) do
      if self.question_id[i] == question_id then
         idx = i
         break
      end
   end 
   print(self.question_id:size())
   print(idx)
   print(self.question_id[idx])
   bInds = torch.Tensor({idx}):repeatTensor(self.batch_size):long()

   if feat_w > 1 or feat_h > 1 then
      batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
   else 
      batch_feat = torch.zeros(self.batch_size, feat_dim)
   end
   batch_img = torch.zeros(self.batch_size, 3, height, width)
   batch_q_len = self.question_len:index(1,bInds):clone()
   batch_qid = self.question_id:index(1,bInds):clone()
   batch_q = self.questions:index(1,bInds):clone()
   if self.split == 'train' then
      batch_a = self.answers:index(1,bInds):clone()
   elseif self.split == 'test' then
      batch_a = self.mc_ans:index(1,bInds):clone()
   end
 
   loc_img_list = {}
   loc_feat_list = {}
   for i = 1, self.batch_size do
      local ann_idx = bInds[i]
      -- ex) 'train2014/COCO_train2014_000000357413.jpg'
      local cocoimg_name = self.img_list[ann_idx]
      loc_img_list[i] = cocoimg_name

      cocoimg_name = paths.basename(cocoimg_name)
      local img_ext = paths.extname(cocoimg_name)
      -- ex) 'COCO_train2014_000000357413.t7'
      local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
      loc_feat_list[i] = cocofeat_name
   end
   for i = 1, self.batch_size do
      -- fetch images
      local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
      local img = image.load(img_path)
      if randcrop then
         img = image.scale(img, rewidth, reheight)
      else
         img = image.scale(img, width, height)
      end
      if img:size()[1] == 1 then
         img = img:repeatTensor(3,1,1)
      end
      if randcrop then
         local cx1 = torch.random() % (rewidth-width) + 1
         local cy1 = torch.random() % (reheight-height) + 1
         local cx2 = cx1 + width
         local cy2 = cy1 + height
         img = image.crop(img, cx1, cy1, cx2, cy2)
      end
      img = img:index(1, torch.LongTensor{3,2,1})
      img = img * 255 - mean_bgr
      img = img:contiguous()
      batch_img[i] = img
      -- fetch features
      local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
      local feature = torch.load(feat_path)
      if feat_w > 1 or feat_h > 1 then
         assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
         assert(feature:size(2) == feat_w, 'feat_w mismatch') 
         assert(feature:size(3) == feat_h, 'feat_h mismatch') 
      else
         assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
         assert(feature:nElement() == feat_dim, 
             string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
      end
      batch_feat[i] = feature
   end

   return batch_feat:clone(), batch_img:clone(), 
          batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end
function dataclass:next_batch_feat_image(cocofeatpath, feat_dim, feat_w, feat_h,
                                         cocoimgpath, height, width,
                                         randcrop, reheight, rewidth)
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a
   local batch_feat
   local loc_feat_list
   local batch_img
   local loc_img_list

   feat_w = feat_w or 1
   feat_h = feat_h or 1
   randcrop = randcrop or false

   local mean_bgr = self.mean_bgr:repeatTensor(height,width,1)
         mean_bgr = mean_bgr:permute(3,2,1)
   
   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or 
         self.prefetch_op ~= 'batch_feat_image' or self.randcrop ~= randcrop then

         if feat_w > 1 or feat_h > 1 then
            self.batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
         else 
            self.batch_feat = torch.zeros(self.batch_size, feat_dim)
         end
         self.feat_dim = feat_dim
         self.feat_w = feat_w
         self.feat_h = feat_h

         self.height = height
         self.width = width
         self.batch_img = torch.zeros(self.batch_size, 3, height, width)
         self.randcrop = randcrop    

         loc_feat_list = {}
         loc_img_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            -- ex) 'train2014/COCO_train2014_000000357413.jpg'
            local cocoimg_name = self.img_list[ann_idx]
            loc_img_list[i] = cocoimg_name

            cocoimg_name = paths.basename(cocoimg_name)
            local img_ext = paths.extname(cocoimg_name)
            -- ex) 'COCO_train2014_000000357413.t7'
            local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
            loc_feat_list[i] = cocofeat_name
         end
         for i = 1, self.batch_size do
            -- image
            local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
            local img = image.load(img_path)
            if randcrop then
               img = image.scale(img, rewidth, reheight)
            else
               img = image.scale(img, width, height)
            end
            if img:size()[1] == 1 then
               img = img:repeatTensor(3,1,1)
            end
            if randcrop then
               local cx1 = torch.random() % (rewidth-width) + 1
               local cy1 = torch.random() % (reheight-height) + 1
               local cx2 = cx1 + width
               local cy2 = cy1 + height
               img = image.crop(img, cx1, cy1, cx2, cy2)
            end
            img = img:index(1, torch.LongTensor{3,2,1})
            img = img * 255 - mean_bgr
            img = img:contiguous()
            self.batch_img[i] = img
            -- feature
            local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
            local feature = torch.load(feat_path)
            if feat_w > 1 or feat_h > 1 then
               assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
               assert(feature:size(2) == feat_w, 'feat_w mismatch') 
               assert(feature:size(3) == feat_h, 'feat_h mismatch') 
            else
               assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
               assert(feature:nElement() == feat_dim, 
                   string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
            end
            self.batch_feat[i] = feature
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_feat_image'
      end
      assert(self.feat_dim == feat_dim, 'feat_dim have to be save all the time')
      assert(self.feat_w == feat_w, 'feat_w have to be save all the time')
      assert(self.feat_h == feat_h, 'feat_h have to be save all the time')
      assert(self.height == height, 'height have to be same all the time')
      assert(self.width == width, 'width have to be same all the time')
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      batch_img = self.batch_img:clone()
      batch_feat = self.batch_feat:clone()
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
 
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_img_list = {}
      loc_feat_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         -- ex) 'train2014/COCO_train2014_000000357413.jpg'
         local cocoimg_name = self.img_list[ann_idx]
         loc_img_list[i] = cocoimg_name

         cocoimg_name = paths.basename(cocoimg_name)
         local img_ext = paths.extname(cocoimg_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
         loc_feat_list[i] = cocofeat_name
      end

      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      self.pool:addjob(
         function ()
            -- image preparation
            local pre_img = torch.zeros(loc_batch_size, 3, height, width)
            -- feature preparation
            local pre_feature
            if feat_w > 1 or feat_h > 1 then
               pre_feature = torch.zeros(loc_batch_size, feat_dim, feat_w, feat_h)
            else 
               pre_feature = torch.zeros(loc_batch_size, feat_dim)
            end
            for i = 1, loc_batch_size do
               -- image fetching
               local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
               local img = image.load(img_path)
               if randcrop then
                  img = image.scale(img, rewidth, reheight)
               else
                  img = image.scale(img, width, height)
               end
               if img:size()[1] == 1 then
                  img = img:repeatTensor(3,1,1)
               end
               if randcrop then
                  local cx1 = torch.random() % (rewidth-width) + 1
                  local cy1 = torch.random() % (reheight-height) + 1
                  local cx2 = cx1 + width
                  local cy2 = cy1 + height
                  img = image.crop(img, cx1, cy1, cx2, cy2)
               end
               img = img:index(1, torch.LongTensor{3,2,1})
               img = img * 255 - mean_bgr
               img = img:contiguous()
               pre_img[i] = img
               -- feature fetching
               local feat_path = paths.concat(cocofeatpath, loc_feat_list[i])
               local feature = torch.load(feat_path)
               if feat_w > 1 or feat_h > 1 then
                  assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
                  assert(feature:size(2) == feat_w, 'feat_w mismatch') 
                  assert(feature:size(3) == feat_h, 'feat_h mismatch') 
               else
                  assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
                  assert(feature:nElement() == feat_dim, 
                      string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
               end
               pre_feature[i] = feature
            end
            return pre_img, pre_feature
         end,
         function (pre_img, pre_feature)
            self.batch_img = pre_img
            self.batch_feat = pre_feature
         end
      )
   else
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      if feat_w > 1 or feat_h > 1 then
         batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
      else 
         batch_feat = torch.zeros(self.batch_size, feat_dim)
      end
      batch_img = torch.zeros(loc_batch_size, 3, height, width)
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
 
      loc_img_list = {}
      loc_feat_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         -- ex) 'train2014/COCO_train2014_000000357413.jpg'
         local cocoimg_name = self.img_list[ann_idx]
         loc_img_list[i] = cocoimg_name

         cocoimg_name = paths.basename(cocoimg_name)
         local img_ext = paths.extname(cocoimg_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local cocofeat_name = stringx.replace(cocoimg_name, img_ext, 't7')
         loc_feat_list[i] = cocofeat_name
      end
      for i = 1, self.batch_size do
         -- fetch images
         local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
         local img = image.load(img_path)
         if randcrop then
            img = image.scale(img, rewidth, reheight)
         else
            img = image.scale(img, width, height)
         end
         if img:size()[1] == 1 then
            img = img:repeatTensor(3,1,1)
         end
         if randcrop then
            local cx1 = torch.random() % (rewidth-width) + 1
            local cy1 = torch.random() % (reheight-height) + 1
            local cx2 = cx1 + width
            local cy2 = cy1 + height
            img = image.crop(img, cx1, cy1, cx2, cy2)
         end
         img = img:index(1, torch.LongTensor{3,2,1})
         img = img * 255 - mean_bgr
         img = img:contiguous()
         batch_img[i] = img
         -- fetch features
         local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
         local feature = torch.load(feat_path)
         if feat_w > 1 or feat_h > 1 then
            assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
            assert(feature:size(2) == feat_w, 'feat_w mismatch') 
            assert(feature:size(3) == feat_h, 'feat_h mismatch') 
         else
            assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
            assert(feature:nElement() == feat_dim, 
                string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
         end
         batch_feat[i] = feature
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat:clone(), batch_img:clone(), 
          batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end


function dataclass:next_batch_feat(tab_featpaths, feat_dim, feat_w, feat_h)
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a
   local batch_feat
   local loc_feat_list
   local loc_datatype_list

   feat_w = feat_w or 1
   feat_h = feat_h or 1
   if type(tab_featpaths) ~= 'table' then
      tab_featpaths = {tab_featpaths}
   end

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_feat' then
         if feat_w > 1 or feat_h > 1 then
            self.batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
         else 
            self.batch_feat = torch.zeros(self.batch_size, feat_dim)
         end
         self.feat_dim = feat_dim
         self.feat_w = feat_w
         self.feat_h = feat_h
         loc_feat_list = {}
	 loc_datatype_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local img_name = paths.basename(self.img_list[ann_idx])
            local img_ext = paths.extname(img_name)
            -- ex) 'COCO_train2014_000000357413.t7'
            local feat_name = stringx.replace(img_name, img_ext, 't7')
            loc_feat_list[i] = feat_name
	    loc_datatype_list[i] = self.datatype[ann_idx]
         end
         for i = 1, self.batch_size do
            -- feature
            local feat_path = paths.concat(tab_featpaths[loc_datatype_list[i]], loc_feat_list[i]) 
            local feature = torch.load(feat_path)
            if feat_w > 1 or feat_h > 1 then
               assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
               assert(feature:size(2) == feat_w, 'feat_w mismatch') 
               assert(feature:size(3) == feat_h, 'feat_h mismatch') 
            else
               assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
               assert(feature:nElement() == feat_dim, 
                   string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
            end
            self.batch_feat[i] = feature
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_feat'
      end

      assert(self.feat_dim == feat_dim, 'feat_dim have to be save all the time')
      assert(self.feat_w == feat_w, 'feat_w have to be save all the time')
      assert(self.feat_h == feat_h, 'feat_h have to be save all the time')
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      batch_feat = self.batch_feat:clone()
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_feat_list = {}
      loc_datatype_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local img_name = paths.basename(self.img_list[ann_idx])
         local img_ext = paths.extname(img_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local feat_name = stringx.replace(img_name, img_ext, 't7')
         loc_feat_list[i] = feat_name
	 loc_datatype_list[i] = self.datatype[ann_idx]
      end
 
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      self.pool:addjob(
         function ()
            local pre_feature
            if feat_w > 1 or feat_h > 1 then
               pre_feature = torch.zeros(loc_batch_size, feat_dim, feat_w, feat_h)
            else 
               pre_feature = torch.zeros(loc_batch_size, feat_dim)
            end
            for i = 1, loc_batch_size do
               local feat_path = paths.concat(tab_featpaths[loc_datatype_list[i]], loc_feat_list[i])
               local feature = torch.load(feat_path)
               if feat_w > 1 or feat_h > 1 then
                  assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
                  assert(feature:size(2) == feat_w, 'feat_w mismatch') 
                  assert(feature:size(3) == feat_h, 'feat_h mismatch') 
               else
                  assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
                  assert(feature:nElement() == feat_dim, 
                      string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
               end
               pre_feature[i] = feature
            end
            return pre_feature
         end,
         function (pre_feature)
            self.batch_feat = pre_feature
         end
      )
   else
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      if feat_w > 1 or feat_h > 1 then
         batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
      else 
         batch_feat = torch.zeros(self.batch_size, feat_dim)
      end
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end

      loc_feat_list = {}
      loc_datatype_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local img_name = paths.basename(self.img_list[ann_idx])
         local img_ext = paths.extname(img_name)
         -- ex) 'COCO_train2014_000000357413.t7'
         local feat_name = stringx.replace(img_name, img_ext, 't7')
         loc_feat_list[i] = feat_name
	 loc_datatype_list[i] = self.datatype[ann_idx]
      end
      for i = 1, self.batch_size do
         -- feature
         local feat_path = paths.concat(tab_featpaths[loc_datatype_list[i]], loc_feat_list[i]) 
         local feature = torch.load(feat_path)
         if feat_w > 1 or feat_h > 1 then
            assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
            assert(feature:size(2) == feat_w, 'feat_w mismatch') 
            assert(feature:size(3) == feat_h, 'feat_h mismatch') 
         else
            assert(feature:dim() == 1, 'only 1 dim feature could be loaded with this method')
            assert(feature:nElement() == feat_dim, 
                string.format('dim mismatch: dim of saved feature is: %d', feature:nElement()))
         end
         batch_feat[i] = feature
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end

function dataclass:next_batch_image(cocoimgpath, height, width,
                                    randcrop, reheight, rewidth,
                                    opt_prepro)
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a
   local batch_img
   local loc_img_list

   if randcrop == nil then
      randcrop = false
   end
   opt_prepro = opt_prepro or 'vgg'

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or 
         self.prefetch_op ~= 'batch_image' or
         self.randcrop ~= randcrop then

         self.height = height
         self.width = width
         self.batch_img = torch.zeros(self.batch_size, 3, height, width)
         self.randcrop = randcrop    

         loc_img_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            -- ex) 'train2014/COCO_train2014_000000357413.jpg'
            local cocoimg_name = self.img_list[ann_idx]
            loc_img_list[i] = cocoimg_name
         end
         for i = 1, self.batch_size do
            -- image
            local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
            local img = image.load(img_path, 3, 'float')
            if randcrop then
               img = image.scale(img, rewidth, reheight)
            else
               img = image.scale(img, width, height)
            end
            if randcrop then
               local cx1 = torch.random() % (rewidth-width) + 1
               local cy1 = torch.random() % (reheight-height) + 1
               local cx2 = cx1 + width
               local cy2 = cy1 + height
               img = image.crop(img, cx1, cy1, cx2, cy2)
            end
            if opt_prepro == 'vgg' then
               img = img:index(1, torch.LongTensor{3,2,1})
               img = img * 255 - mean_bgr
               img = img:contiguous()
            else
               img = self.normalize(img)
            end
            self.batch_img[i] = img
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_image'
      end

      assert(self.height == height, 'height have to be same all the time')
      assert(self.width == width, 'width have to be same all the time')
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      batch_img = self.batch_img:clone()
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
 
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_img_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         -- ex) 'train2014/COCO_train2014_000000357413.jpg'
         local cocoimg_name = self.img_list[ann_idx]
         loc_img_list[i] = cocoimg_name
      end

      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      self.pool:addjob(
         function ()
            local pre_img = torch.zeros(loc_batch_size, 3, height, width)
            for i = 1, loc_batch_size do
               local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
               local img = image.load(img_path, 3, 'float')
               if randcrop then
                  img = image.scale(img, rewidth, reheight)
               else
                  img = image.scale(img, width, height)
               end
               if randcrop then
                  local cx1 = torch.random() % (rewidth-width) + 1
                  local cy1 = torch.random() % (reheight-height) + 1
                  local cx2 = cx1 + width
                  local cy2 = cy1 + height
                  img = image.crop(img, cx1, cy1, cx2, cy2)
               end
               if opt_prepro == 'vgg' then
                  img = img:index(1, torch.LongTensor{3,2,1})
                  img = img * 255 - mean_bgr
                  img = img:contiguous()
               else
                  img = normalize(img)
               end
               pre_img[i] = img
            end
            return pre_img
         end,
         function (pre_img)
            self.batch_img = pre_img
         end
      )
   else
      local sIdx = self.batch_index+1
      local eIdx = self.batch_index+self.batch_size
      local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
      batch_img = torch.zeros(loc_batch_size, 3, height, width)
      batch_q_len = self.question_len:index(1,bInds):clone()
      batch_qid = self.question_id:index(1,bInds):clone()
      batch_q = self.questions:index(1,bInds):clone()
      if self.split == 'train' then
         batch_a = self.answers:index(1,bInds):clone()
      elseif self.split == 'test' then
         batch_a = self.mc_ans:index(1,bInds):clone()
      end
 
      loc_img_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         -- ex) 'train2014/COCO_train2014_000000357413.jpg'
         local cocoimg_name = self.img_list[ann_idx]
         loc_img_list[i] = cocoimg_name
      end
      for i = 1, self.batch_size do
         local img_path = paths.concat(cocoimgpath, loc_img_list[i]) 
         local img = image.load(img_path, 3, 'float')
         if randcrop then
            img = image.scale(img, rewidth, reheight)
         else
            img = image.scale(img, width, height)
         end
         if randcrop then
            local cx1 = torch.random() % (rewidth-width) + 1
            local cy1 = torch.random() % (reheight-height) + 1
            local cx2 = cx1 + width
            local cy2 = cy1 + height
            img = image.crop(img, cx1, cy1, cx2, cy2)
         end
         if opt_prepro == 'vgg' then
            img = img:index(1, torch.LongTensor{3,2,1})
            img = img * 255 - mean_bgr
            img = img:contiguous()
         else
            img = self.normalize(img)
         end
         batch_img[i] = img
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_img:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end

function dataclass:next_batch()
   local batch_q
   local batch_q_len
   local batch_qid
   local batch_a

   local sIdx = self.batch_index+1
   local eIdx = self.batch_index+self.batch_size
   local bInds = self.batch_order[{{sIdx,eIdx}}]:long()
   batch_q_len = self.question_len:index(1,bInds):clone()
   batch_qid = self.question_id:index(1,bInds):clone()
   batch_q = self.questions:index(1,bInds):clone()
   if self.split == 'train' then
      batch_a = self.answers:index(1,bInds):clone()
   elseif self.split == 'test' then
      batch_a = self.mc_ans:index(1,bInds):clone()
   end
 
   self.batch_index = self.batch_index + self.batch_size
   if (self.batch_index + self.batch_size) > self.ex_num_train then
      self:reorder()
   end

   return batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qid:clone()
end
function dataclass:set_batch_order_option(opt_batch_order)
   if opt_batch_order == 1 then
      print(string.format('[%s] set batch order option 1 : shuffle', self.data_subtype))
   elseif opt_batch_order == 2 then
      print(string.format('[%s] set batch order option 2 : inorder', self.data_subtype))
   elseif opt_batch_order == 3 then
      print(string.format('[%s] set batch order option 3 : sort', self.data_subtype))
   elseif opt_batch_order == 4 then
      print(string.format('[%s] set batch order option 4 : randsort', self.data_subtype))
   else
      assert(true, 'set_opt_batch_order error: this batch order option is not yet defined')
   end
   self.opt_batch_order = opt_batch_order
end
function dataclass:reorder()
   if self.opt_batch_order == 1 then
      self:shuffle()
   elseif self.opt_batch_order == 2 then
      self:inorder()
   elseif self.opt_batch_order == 3 then
      self:sort()
   elseif self.opt_batch_order == 4 then
      self:randsort()
   else
      assert(true, 'reorder error: this batch order option is not yet defined')
   end
end
function dataclass:inorder()
   -- in order
   self.batch_index = 0
   self.batch_order = torch.range(1, self.ex_num_train)
   self.prefetch_init = false
end
function dataclass:shuffle()
   -- random order
   self.batch_index = 0
   self.batch_order = torch.randperm(self.ex_num_train)
   self.prefetch_init = false
end
function dataclass:randsort()
   -- sort according to sequence lenth, but if sequence lengths are equal, shuffle
   self.prefetch_init = false
   self.batch_index = 0
   local sorted, loc_batch_order = self.question_len:sort()
   self.batch_order = loc_batch_order:clone()
   local i = 1
   while i < sorted:nElement()-1 do
      local i_start = i
      local i_end
      for j = i_start, sorted:nElement() do
         if sorted[j] > sorted[i_start] then
            i_end = j
            break
         end
         if j == sorted:nElement() then
            i_end = j+1
         end
      end
      local rand_order = torch.randperm(i_end - i_start)
      for k = 1, i_end-i_start do
         self.batch_order[i_start+k-1] = loc_batch_order[rand_order[k]+i_start-1]
      end
      i = i_end
   end
end
function dataclass:sort()
   self.prefetch_init = false
   self.batch_index = 0
   sorted, self.batch_order = self.question_len:sort()
end
function dataclass:reset_batch_pointer()
   self.batch_index = 0
end


function vqa_prepro_loader.load_data(vqa_dir, batch_size, opt_prefetch, opt_split, test_batch_size, valid_ratio)
   local vqa_data = {}
   setmetatable(vqa_data, vqa_prepro_loader)

   valid_ratio = valid_ratio or 0

   local ques_h5 = paths.concat(vqa_dir, 'data_prepro.h5')
   local info_json = paths.concat(vqa_dir, 'data_prepro.json')

   ---------------------------------------
   -- load dataset
   ---------------------------------------
   -- info (json)
   --    format:
   --       json_file.ix_to_ans: {'1': 'yes', ...}
   --       json_file.ix_to_word: {'1': 'raining', ...} 
   --       json_file.unique_img_train: {1: 'train2014/COCO_train2014_000000357413.jpg'}
   --       json_file.unique_img_test: {1: 'test2015/COCO_test2015_000000006350.jpg'} or
   --                                  {1: 'val2014/COCO_val2014_000000533942.jpg'}
   local file = io.open(info_json, 'r')
   local text = file:read()
   local json_file = cjson.decode(text)
   file:close()
   -- question (hdf5)
   --    format:
   --       [train]
   --       '/ques_train': tensor[nExample, SeqLen] (word index)
   --       '/ques_length_train': tensor[nExample] (length of question) 
   --       '/img_pos_train': tensor[nExample]: (img_pos_train[qid] = imgidx (in json file))
   --       '/question_id_train: tensor[nExample] (question_id_train[qid] = question_id(of mscoco))
   --       '/answers: tensor[nExample] (answer index) (multiple choice answer)
   --       [test]
   --       '/ques_test': tensor[nExample, SeqLen] (word index)
   --       '/ques_length_test': tensor[nExample] (length of question) 
   --       '/img_pos_test': tensor[nExample]: (img_pos_test[qid] = imgidx (in json file))
   --       '/question_id_test: tensor[nExample] (question_id_test[qid] = question_id(of mscoco))
   --       '/MC_ans_test: tensor[nExample, nMultChoice] (answer index) (multiple choice answers)
   local h5_file = hdf5.open(ques_h5, 'r')
   local h5_file_data = h5_file:all()
   local train_ques = {}
   train_ques.question = h5_file_data['ques_train']
   train_ques.question:add(1) -- make zero padding to one
   train_ques.lengths_q = h5_file_data['ques_length_train']
   train_ques.img_list = h5_file_data['img_pos_train']
   train_ques.question_id = h5_file_data['question_id_train']
   train_ques.answers = h5_file_data['answers']
   if h5_file_data['datatype_train'] ~= nil then
      train_ques.datatype = h5_file_data['datatype_train']
   else
      train_ques.datatype = train_ques.answers:clone():fill(1)
   end
   local val_ques = {}
   if valid_ratio > 0 then
      assert(valid_ratio <= 1, 'validation ratio should be smaller than 1')
      local ori_trainsz = train_ques.answers:size(1)
      local ori_randorder = torch.randperm(ori_trainsz)
      local val_sz = math.floor(ori_trainsz * valid_ratio)
      local train_sz = ori_trainsz - val_sz
      local val_idx = ori_randorder:narrow(1,1,val_sz):long()
      local train_idx = ori_randorder:narrow(1,val_sz+1,train_sz):long()
      -- val ques copy
      val_ques.question = train_ques.question:index(1,val_idx)
      val_ques.lengths_q = train_ques.lengths_q:index(1,val_idx)
      val_ques.img_list = train_ques.img_list:index(1,val_idx)
      val_ques.question_id = train_ques.question_id:index(1,val_idx)
      val_ques.answers = train_ques.answers:index(1,val_idx) 
      val_ques.datatype = train_ques.datatype:index(1,val_idx)
      -- train ques copy
      train_ques.question = train_ques.question:index(1,train_idx)
      train_ques.lengths_q = train_ques.lengths_q:index(1,train_idx)
      train_ques.img_list = train_ques.img_list:index(1,train_idx)
      train_ques.question_id = train_ques.question_id:index(1,train_idx)
      train_ques.answers = train_ques.answers:index(1,train_idx) 
      train_ques.datatype = train_ques.datatype:index(1,train_idx)
      -- garbage collection
      collectgarbage()
   end
   local test_ques = {}
   test_ques.question = h5_file_data['ques_test']
   test_ques.question:add(1) -- make zero padding to one
   test_ques.lengths_q = h5_file_data['ques_length_test']
   test_ques.img_list = h5_file_data['img_pos_test']
   test_ques.question_id = h5_file_data['question_id_test']
   test_ques.mc_ans = h5_file_data['MC_ans_test']
   test_ques.datatype = test_ques.question_id:clone():fill(1)
   h5_file:close() 
   -- usage
   --   get filename of [i]th quenstion image:
   --      json_file.unique_img_train[train_ques.img_list[i]]
   --   get answer word of [i]th question answer:
   --      json_file.ix_to_ans[tostring(train_ques.answers[i])]
   --   ...

   -- vocab size count
   local qcount=1 -- including padding symbol 'ZEROPAD'
   local acount=0
   for i, w in pairs(json_file.ix_to_word) do qcount = qcount + 1 end
   for i, w in pairs(json_file.ix_to_ans) do acount = acount + 1 end
   -- vocab dict construction
   local vocab_dict = {[1]='ZEROPAD'}
   local answer_dict = {}
   for i, w in pairs(json_file.ix_to_word) do vocab_dict[tonumber(i)+1] = w end
   for i, w in pairs(json_file.ix_to_ans) do answer_dict[tonumber(i)] = w end
   -- vocab map construction
   local vocab_map = {['ZEROPAD']=1}
   local answer_map = {}
   for i, w in pairs(json_file.ix_to_word) do vocab_map[w] = tonumber(i)+1 end
   for i, w in pairs(json_file.ix_to_ans) do answer_map[w] = tonumber(i) end

   -- important options
   opt_prefetch = opt_prefetch or false
   opt_split = opt_split or 'trainval'
   test_batch_size = test_batch_size or batch_size

   -- vocabulary
   vqa_data.img_train = json_file.unique_img_train
   vqa_data.img_test = json_file.unique_img_test
   vqa_data.vocab_map = vocab_map
   vqa_data.vocab_dict = vocab_dict
   vqa_data.vocab_size = qcount
   vqa_data.answer_map = answer_map
   vqa_data.answer_dict = answer_dict
   vqa_data.answer_size = acount
   vqa_data.max_sentence_len = train_ques.question:size(2)
   vqa_data.seq_len = train_ques.question:size(2)

   if opt_split == 'train2014' then
      vqa_data.train_data = dataclass.load_data({'train2014train'}, 
                                             vqa_data, train_ques, json_file.unique_img_train,
                                             batch_size, opt_prefetch)
      if valid_ratio > 0 then
         vqa_data.val_data = dataclass.load_data({'train2014val'}, 
                                                vqa_data, val_ques, json_file.unique_img_train,
                                                batch_size, opt_prefetch)
      end
      vqa_data.test_data = dataclass.load_testdata('train2014', 
                                             vqa_data, test_ques, json_file.unique_img_test,
                                             test_batch_size, opt_prefetch)
   elseif opt_split == 'val2014' then
      vqa_data.train_data = dataclass.load_data({'train2014train'}, 
                                             vqa_data, train_ques, json_file.unique_img_train,
                                             batch_size, opt_prefetch)
      if valid_ratio > 0 then
         vqa_data.val_data = dataclass.load_data({'train2014val'}, 
                                                vqa_data, val_ques, json_file.unique_img_train,
                                                batch_size, opt_prefetch)
      end
      vqa_data.test_data = dataclass.load_testdata('val2014', 
                                             vqa_data, test_ques, json_file.unique_img_test,
                                             test_batch_size, opt_prefetch)
   elseif opt_split == 'test2015' then
      vqa_data.train_data = dataclass.load_data({'train2014train', 'val2014train'}, 
                                             vqa_data, train_ques, json_file.unique_img_train,
                                             batch_size, opt_prefetch)
      if valid_ratio > 0 then
         vqa_data.val_data = dataclass.load_data({'train2014val', 'val2014val'}, 
                                                vqa_data, val_ques, json_file.unique_img_train,
                                                batch_size, opt_prefetch)
      end
      vqa_data.test_data = dataclass.load_testdata('test2015',
                                             vqa_data, test_ques, json_file.unique_img_test,
                                             test_batch_size, opt_prefetch)
   elseif opt_split == 'test-dev2015' then
      vqa_data.train_data = dataclass.load_data({'train2014train', 'val2014train'}, 
                                             vqa_data, train_ques, json_file.unique_img_train,
                                             batch_size, opt_prefetch)
      if valid_ratio > 0 then
         vqa_data.val_data = dataclass.load_data({'train2014val', 'val2014val'}, 
                                                vqa_data, val_ques, json_file.unique_img_train,
                                                batch_size, opt_prefetch)
      end
      vqa_data.test_data = dataclass.load_testdata('test-dev2015',
                                             vqa_data, test_ques, json_file.unique_img_test,
                                             test_batch_size, opt_prefetch)
   else
      assert(false, 'undefined split option')
   end

   return vqa_data
end

return vqa_prepro_loader
