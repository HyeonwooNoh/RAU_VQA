require 'hdf5'
require 'image'
require 'torch'
local stringx = require 'pl.stringx'
local file_utils = require 'utils.file_utils'
local qa_utils = require 'utils.qa_utils'
local threads = require 'threads'
local cjson = require 'cjson'
local transform = require 'utils.transforms'

local function LoadJson(json_path)
	local file = io.open(json_path, 'r')
	local text = file:read()
	local json_data = cjson.decode(text)
	file:close()
	return json_data
end

local function LoadHdf5(hdf5_path)
	local file = hdf5.open(hdf5_path, 'r')
	local hdf5_data = file:all()
	return hdf5_data
end

-- DataFetcher Class
do
	local DataFetcher = torch.class('DataFetcher')

	function DataFetcher:__init(config)
		self:CheckConfig(config)

		unique_image_list = config['unique_image_list']
		questions = config['questions']
		split = config['split']
		sequence_length = config['sequence_length']
		use_prefetch = config['use_prefetch']
		batch_size = config['batch_size']

		-- Construct image list
		self.image_list = {}
		for i=1, questions['image_index']:size(1) do
			self.image_list[i] = unique_image_list[questions['image_index'][i]]
		end

		-- Set split of this data fetcher
		self.split = split

		-- Set initial values
		self.question = questions['question']
		self.question_length = questions['question_length']
		self.question_id = questions['question_id']
		self.is_best = questions['is_best']
		self.best_shortest = questions['best_shortest']
		self.has_correct_answer = questions['has_correct_answer']
		self.num_steps = self.is_best:size(2)
		self.sequence_length = sequence_length
		self.num_examples = self.question:size(1)

		-- Initialization for mini-batch fetching
		self.iter_index = 0
		self.batch_index = 0
		self.batch_size = batch_size
		self.batch_order = torch.range(1, self.num_examples) -- in order
		self.iter_per_epoch = torch.floor(self.num_examples / self.batch_size)
		self.batch_order_option = 1

		-- Data normalization
		self.mean_bgr = torch.Tensor({103.939, 116.779, 123.68}) 
		self.meanstd = {
			['mean'] = { 0.485, 0.456, 0.406 },
			['std'] = { 0.229, 0.224, 0.225 },
		}
		self.normalize = transform.Compose{
         transform.ColorNormalize(self.meanstd),
		}
		-- Prefetching
		self.use_prefetch = use_prefetch
		if self.use_prefetch then
			-- thread for prefetching
			self.pool = threads.Threads(1, function ()
				require 'image'
				transform = require 'utils.transforms'
				meanstd = {
					['mean'] = { 0.485, 0.456, 0.406 },
               ['std'] = { 0.229, 0.224, 0.225 },
            }  
            normalize = transform.Compose{
					transform.ColorNormalize(meanstd),
				}
         end)
			self.prefetch_init = false
		end
	end

	function DataFetcher:CheckConfig(config)
		if config['use_prefetch'] == nil then
			error("Missing values in config: use_prefetch")
		elseif config['batch_size'] == nil then
			error("Missing values in config: batch_size")
		elseif config['unique_image_list'] == nil then
			error("Missing values in config: unique_image_list")
		elseif config['questions'] == nil then
			error("Missing values in config: questions")
		elseif config['split'] == nil then
			error("Missing values in config: split")
		elseif config['sequence_length'] == nil then
			error("Missing values in config: sequence_length")
		end
	end

	function DataFetcher:NextBatchFeature(feat_dir, feat_dim, feat_w, feat_h)
		local batch_q
		local batch_q_len
		local batch_qid
		local batch_is_best
		local batch_best_shortest
		local batch_has_correct_answer
		local batch_feat
		local local_feat_list

		feat_w = feat_w or 1
		feat_h = feat_h or 1

		if self.use_prefetch then
			self.pool:synchronize()
			if self.prefetch_init == false or
				self.prefetch_option ~= 'batch_feat' then
				if feat_w > 1 or feat_h > 1 then
					self.batch_feat = torch.zeros(self.batch_size, feat_dim, feat_w, feat_h)
				else 
					self.batch_feat = torch.zeros(self.batch_size, feat_dim)
				end
				self.feat_dim = feat_dim
				self.feat_w = feat_w
				self.feat_h = feat_h
				local_feat_list = {}
				for i = 1, self.batch_size do
					local ann_idx = self.batch_order[i + self.batch_index]
					local image_name = paths.basename(self.image_list[ann_idx])
					local image_ext = paths.extname(image_name)
					-- ex) 'COCO_train2014_000000357413.t7'
					local feat_name = stringx.replace(image_name, image_ext, 't7')
					local_feat_list[i] = feat_name
				end
				for i = 1, self.batch_size do
					-- feature
					local feat_path = paths.concat(feat_dir, local_feat_list[i]) 
					local feature = torch.load(feat_path)
					if feat_w > 1 or feat_h > 1 then
						assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
						assert(feature:size(2) == feat_w, 'feat_w mismatch') 
						assert(feature:size(3) == feat_h, 'feat_h mismatch') 
					else
						assert(feature:dim() == 1,
							'only 1 dim feature could be loaded with this method')
						assert(feature:nElement() == feat_dim, 
							string.format('dim mismatch: dim of saved feature is: %d',
							feature:nElement()))
					end
					self.batch_feat[i] = feature
				end
				self.prefetch_init = true
				self.prefetch_option = 'batch_feat'
			end

			assert(self.feat_dim == feat_dim,
                'feat_dim have to be save all the time')
			assert(self.feat_w == feat_w, 'feat_w have to be save all the time')
			assert(self.feat_h == feat_h, 'feat_h have to be save all the time')
			local sIdx = self.batch_index + 1
			local eIdx = self.batch_index + self.batch_size
			local bInds = self.batch_order[{{sIdx, eIdx}}]:long()
			batch_feat = self.batch_feat:clone()
			batch_q = self.question:index(1, bInds):clone()
			batch_q_len = self.question_length:index(1, bInds):clone()
			batch_qid = self.question_id:index(1, bInds):clone()
			batch_is_best = self.is_best:index(1, bInds):clone()
			batch_best_shortest = self.best_shortest:index(1, bInds):clone()
			batch_has_correct_answer = self.has_correct_answer:index(1, bInds):clone()
  
			-- update batch counter
			self.batch_index = self.batch_index + self.batch_size
			if (self.batch_index + self.batch_size) > self.num_examples then
				self:Reorder()
			end
  
			-- light weight fetching
			local_feat_list = {}
			for i = 1, self.batch_size do
				local ann_idx = self.batch_order[i + self.batch_index]
				local image_name = paths.basename(self.image_list[ann_idx])
				local image_ext = paths.extname(image_name)
				-- ex) 'COCO_train2014_000000357413.t7'
				local feat_name = stringx.replace(image_name, image_ext, 't7')
				local_feat_list[i] = feat_name
			end
 
			-- heavy weight fetching (use thread)
			local local_batch_size = self.batch_size
			self.pool:addjob(
				function ()
					local pre_feature
					if feat_w > 1 or feat_h > 1 then
						pre_feature = torch.zeros(local_batch_size, feat_dim,
                                            feat_w, feat_h)
					else 
						pre_feature = torch.zeros(loc_batch_size, feat_dim)
					end
					for i = 1, local_batch_size do
						local feat_path = paths.concat(feat_dir, local_feat_list[i])
						local feature = torch.load(feat_path)
						if feat_w > 1 or feat_h > 1 then
							assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
							assert(feature:size(2) == feat_w, 'feat_w mismatch') 
							assert(feature:size(3) == feat_h, 'feat_h mismatch') 
						else
							assert(feature:dim() == 1,
								'only 1 dim feature could be loaded with this method')
							assert(feature:nElement() == feat_dim, 
								string.format('dim mismatch: dim of saved feature is: %d',
								feature:nElement()))
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
			batch_q = self.question:index(1,bInds):clone()
			batch_is_best = self.is_best:index(1, bInds):clone()
			batch_best_shortest = self.best_shortest:index(1, bInds):clone()
			batch_has_correct_answer = self.has_correct_answer:index(1, bInds):clone()

			local_feat_list = {}
			for i = 1, self.batch_size do
				local ann_idx = self.batch_order[i + self.batch_index]
				local image_name = paths.basename(self.image_list[ann_idx])
				local image_ext = paths.extname(image_name)
				-- ex) 'COCO_train2014_000000357413.t7'
				local feat_name = stringx.replace(image_name, image_ext, 't7')
				local_feat_list[i] = feat_name
			end
			for i = 1, self.batch_size do
				-- feature
				local feat_path = paths.concat(feat_dir, local_feat_list[i]) 
				local feature = torch.load(feat_path)
				if feat_w > 1 or feat_h > 1 then
					assert(feature:size(1) == feat_dim, 'feat_dim mismatch') 
					assert(feature:size(2) == feat_w, 'feat_w mismatch') 
					assert(feature:size(3) == feat_h, 'feat_h mismatch') 
				else
					assert(feature:dim() == 1,
						'only 1 dim feature could be loaded with this method')
					assert(feature:nElement() == feat_dim, 
						string.format('dim mismatch: dim of saved feature is: %d',
						feature:nElement()))
				end
				batch_feat[i] = feature
			end

			self.batch_index = self.batch_index + self.batch_size
			if (self.batch_index + self.batch_size) > self.num_examples then
				self:Reorder()
			end
		end
		return batch_feat:clone(), batch_q:transpose(1,2):clone(),
			batch_q_len:clone(), batch_is_best:clone(),
			batch_best_shortest:clone(), batch_has_correct_answer:clone(),
			batch_qid:clone()
	end

	function DataFetcher:SetBatchOrderOption(batch_order_option)
		if batch_order_option == 1 then
			print('Set batch order option 1 : shuffle')
		elseif batch_order_option == 2 then
			print('Set batch order option 2 : inorder')
		elseif batch_order_option == 3 then
			print('Set batch order option 3 : sort')
		elseif batch_order_option == 4 then
			print('Set batch order option 4 : randsort')
		else
			error('Unknown batch_order_option')
		end
		self.batch_order_option = batch_order_option
	end

	function DataFetcher:Reorder()
		if self.batch_order_option == 1 then
			self:Shuffle()
		elseif self.batch_order_option == 2 then
			self:Inorder()
		elseif self.batch_order_option == 3 then
			self:Sort()
		elseif self.batch_order_option == 4 then
			self:RandSort()
		else
			error('Unknown batch_order_option')
		end
	end

	function DataFetcher:Inorder()
		self.batch_index = 0
		self.batch_order = torch.range(1, self.num_examples)
		self.prefetch_init = false
	end

	function DataFetcher:Shuffle()
		self.batch_index = 0
		self.batch_order = torch.randperm(self.num_examples)
		self.prefetch_init = false
	end

	function DataFetcher:RandSort()
		-- Sort according to sequence lenth, but if sequence lengths are equal,
		-- shuffle.
		self.prefetch_init = false
		self.batch_index = 0
		local sorted, local_batch_order = self.question_length:sort()
		self.batch_order = local_batch_order:clone()
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
				self.batch_order[i_start+k-1] =
					local_batch_order[rand_order[k]+i_start-1]
			end
			i = i_end
		end
	end

	function DataFetcher:Sort()
		self.prefetch_init = false
		self.batch_index = 0
		sorted, self.batch_order = self.question_len:sort()
	end

	function DataFetcher:ResetBatchPointer()
		self.batch_index = 0
	end
end

-- OracleLoader
do 
	local OracleLoader = torch.class('OracleLoader')

	function OracleLoader:__init(config)
		-- config:
		-- The input config is table containing following variables:
		--		* data_dir: data directory containing data_prepro.h5 and
		--	     data_prepro.json
		--    * train_batch_size: batch_size of the loaded training data
		--    * test_batch_size: batch_size of the loaded testing data
      --    * use_prefetch: whether to use prefetch or not
		self:CheckConfig(config)

		-- data_prepro.json	
		--		format:
		--       json_file.ix_to_word: {'1': 'raining', ...} 
		--       json_file.unique_img_train:
		--				{1: 'val2014/COCO_val2014_000000533942.jpg'}
		--       json_file.unique_img_test:
		--				{1: 'val2014/COCO_val2014_000000533942.jpg'}
		local json_path = paths.concat(config['data_dir'], 'data_prepro.json')	
		local json_data = LoadJson(json_path)	

		-- data_prepro.h5 
		--		format:
		--       [train]
		--       '/ques_train': tensor[nExample, SeqLen] (word index)
		--       '/ques_length_train': tensor[nExample] (length of question) 
		--       '/img_pos_train': tensor[nExample] 
		--				(img_pos_train[qid] = imgidx (in json file))
		--       '/question_id_train': tensor[nExample]
		--				(question_id_train[qid] = question_id(of mscoco))
		--			'/is_best_train': tensor[nExample, num_steps]
		--			'/best_shortest_train': tensor[nExample]
		--			'/has_correct_answer_train': tensor[nExample]
		--           
		--       [test]
		--       '/ques_test': tensor[nExample, SeqLen] (word index)
		--       '/ques_length_test': tensor[nExample] (length of question) 
		--       '/img_pos_test': tensor[nExample] 
		--				(img_pos_test[qid] = imgidx (in json file))
		--       '/question_id_test': tensor[nExample]
		--				(question_id_test[qid] = question_id(of mscoco))
		--			'/is_best_test': tensor[nExample, num_steps]
		--			'/best_shortest_test': tensor[nExample]
		--			'/has_correct_answer_test': tensor[nExample]
		local hdf5_path = paths.concat(config['data_dir'], 'data_prepro.h5')
		local hdf5_data = LoadHdf5(hdf5_path)

		-- Loading question data
		local train_questions = self:GetQuestionSplit(hdf5_data, 'train')
		local test_questions = self:GetQuestionSplit(hdf5_data, 'test')

		-- vocab size count
		local vocab_size=1 -- including padding symbol 'ZEROPAD'
		for i, w in pairs(json_data['ix_to_word']) do vocab_size = vocab_size + 1 end
		-- vocab dict construction
		local vocab_dict = {[1]='ZEROPAD'}
		for i, w in pairs(json_data['ix_to_word']) do vocab_dict[tonumber(i)+1] = w end
		-- vocab map construction
		local vocab_map = {['ZEROPAD']=1}
		for i, w in pairs(json_data['ix_to_word']) do vocab_map[w] = tonumber(i)+1 end

		-- Set vocabulary
		self.train_images = json_data['unique_img_train']
		self.test_images = json_data['unique_img_test']
		self.vocab_map = vocab_map
		self.vocab_dict = vocab_dict
		self.vocab_size = vocab_size
		self.max_sentence_length = train_questions['question']:size(2)
		self.sequence_length = train_questions['question']:size(2)

		-- Data fetcher
		self.train_data = DataFetcher({
			['questions']=train_questions,
			['unique_image_list']=self.train_images,
			['split']='train',
			['sequence_length']=self.sequence_length,
			['use_prefetch']=config['use_prefetch'],
			['batch_size']=config['train_batch_size'],
		})
		self.test_data = DataFetcher({
			['questions']=test_questions,
			['unique_image_list']=self.test_images,
			['split']='test',
			['sequence_length']=self.sequence_length,
			['use_prefetch']=config['use_prefetch'],
			['batch_size']=config['test_batch_size'],
		})
	end

	function OracleLoader:CheckConfig(config)
		if config['data_dir'] == nil then
			error("Missing values in config: data_dir")
		elseif config['train_batch_size'] == nil then
			error("Missing values in config: train_batch_size")
		elseif config['test_batch_size'] == nil then
			error("Missing values in config: test_batch_size")
		elseif config['use_prefetch'] == nil then
			error("Missing values in config: use_prefetch")
		end
	end

	function OracleLoader:GetQuestionSplit(hdf5_data, split)
		if split ~= 'train' and split ~= 'test' then
			error("Unknown split name")
		end
		split_data = {}
		split_data['question'] = hdf5_data['ques_'..split]
		split_data['question']:add(1)
		split_data['question_length'] = hdf5_data['ques_length_'..split]
		split_data['image_index'] = hdf5_data['img_pos_'..split]
		split_data['question_id'] = hdf5_data['question_id_'..split]
		split_data['is_best'] = hdf5_data['is_best_'..split]
		split_data['best_shortest'] = hdf5_data['best_shortest_'..split]
		split_data['has_correct_answer'] = hdf5_data['has_correct_answer_'..split]
		return split_data
	end
end
