require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'utils.oracle_loader'
require 'utils.optim_updates'
require 'utils.tools'
require 'gnuplot'
require 'image'

local DeepLSTM = require 'model.DeepLSTM'
local model_utils = require 'utils.model_utils'
local cjson = require 'cjson'

-- read input parameter
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train meta controller')
cmd:text()

cmd:text('GENERAL')
cmd:option('-init_from', '',
           'initialize network parameters from checkpoint at this path')
cmd:option('-continue_learning', 'false', '[true|false]')
cmd:option('-loss_explod_threshold', 3, 'loss explosion threshold')

cmd:text('DATA')
cmd:option('-data_dir', './experiments/Ours_MS/save_result_vqa_448_val2014'..
           '/oracle_prediction/OpenEnded_08_steps_40_epochs',
           'oracle data directory')
cmd:option('-feat_dir', './data/vqa_VGG16Conv_pool5_448/feat_448x448',
           'vqa feature directory')
cmd:option('-cnnout_w', 14, 'cnnout_w')
cmd:option('-cnnout_h', 14, 'cnnout_h')
cmd:option('-cnnout_dim', 512, 'cnnout_dim')

cmd:text('OPTIMIZATION')
cmd:option('-optim', 'adam', 'option: adam')
cmd:option('-batch_order_option', 1,
           '[1]:shuffle, [2]:inorder, [3]:sort, [4]:randsort')
cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-learning_rate',3e-3,'learning rate')
cmd:option('-lr_decay', 0.9,
           'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-mult_learning_rate', 3e-4, 'learning rate')
cmd:option('-mult_lr_decay', 0.9,
           'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-lr_decay_interval', 1, 'learning rate decay interval in epoch')
cmd:option('-batch_size', 100, 'number of examples for each batch')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value')
cmd:option('-free_interval',10, 'interval for collect garbage')

cmd:text('VISUALIZATION')
cmd:option('-test_interval', 1, 'interval of testing in epoch')
cmd:option('-graph_interval', 10, 'interval of drawing graph')
cmd:option('-print_iter', 1, 'interval of printing training loss')
cmd:option('-denseloss_saveinterval', 50,
           'interval for saving dense training loss in iteration')
cmd:option('-visatt', 'true', 'whether visualize attention or not')

cmd:text('DISPLAY')
cmd:option('-display', 'true', 'display result while training')
cmd:option('-display_id', 10, 'display window id')
cmd:option('-display_host', 'localhost', 'display hostname 0.0.0.0')
cmd:option('-display_port', 8000, 'display port number')
cmd:option('-display_interval', 10, 'display interval')

cmd:text('CONTROL')
cmd:option('-prediction_type', 'best_shortest',
           'Type of oracle prediction [is_best|best_shortest]')

cmd:text('SAVE')
cmd:option('-save_dir', './experiments/Ours_MS/save_result_vqa_448_val2014'..
           '/oracle_prediction/OpenEnded_08_steps_40_epochs',
           'subdirectory to save results [log, snapshot]')
cmd:option('-seed', 123, 'torch manual random number generator seed')

cmd:text('GPU')
cmd:option('-gpuid', 0, 'which GPU to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', '[cunn|cudnn]')

-- parse input params
opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

-- load vqa dataset

--30378 = 2 * 3 * 61 * 83
opt.test_batch_size = 83

-- constant options
opt.save_dir = paths.concat(opt.save_dir, 'meta_controller')
opt.log_dir = 'training_log'
opt.snapshot_dir = 'snapshot'
opt.results_dir = 'results'
opt.graph_dir = 'graphs'
opt.figure_dir = 'figures'

-- create directory and log file
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.log_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.snapshot_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.graph_dir))
cmd:log(string.format('%s/%s/log_cmdline', opt.save_dir, opt.log_dir), opt)
print('create directory and log file')

-- initialize GPU
if opt.gpuid >= 0 then
   local ok, cunn = pcall(require, 'cunn')
   local ok2, cutorch = pcall(require, 'cutorch')
   if not ok then print('package cunn not found!') end
   if not ok2 then print('package cutorch not found!') end
   if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        math.randomseed(opt.seed)
        torch.manualSeed(opt.seed)
        cutorch.manualSeed(opt.seed)
        LookupTable = nn.LookupTable
    else
        print('If cutorch and cunn are installed, '..
              'your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, '..
              'rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

if opt.backend == 'cudnn' then
   require 'cudnn'
   math.randomseed(opt.seed)
   torch.manualSeed(opt.seed)
   cutorch.manualSeed(opt.seed)
   LookupTable = nn.LookupTable
   cudnn.fastest = true
end

local SpatialBatchNormalization
local BatchNormalization
local ReLU
local Tanh
local Sigmoid
local SpatialConvolution
local SpatialAveragePooling
local SpatialMaxPooling
local SoftMax
local LogSoftMax
if opt.backend == 'cudnn' then
   SpatialBatchNormalization = cudnn.SpatialBatchNormalization
   BatchNormalization = cudnn.BatchNormalization
   ReLU = cudnn.ReLU
   Tanh = cudnn.Tanh
   Sigmoid = cudnn.Sigmoid
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialAveragePooling = cudnn.SpatialAveragePooling
   SpatialMaxPooling = cudnn.SpatialMaxPooling
   SoftMax = cudnn.SoftMax
   LogSoftMax = cudnn.LogSoftMax
else
   SpatialBatchNormalization = nn.SpatialBatchNormalization
   BatchNormalization = nn.BatchNormalization
   ReLU = nn.ReLU
   Tanh = nn.Tanh
   Sigmoid = nn.Sigmoid
   SpatialConvolution = nn.SpatialConvolution
   SpatialAveragePooling = nn.SpatialAveragePooling
   SpatialMaxPooling = nn.SpatialMaxPooling
   SoftMax = nn.SoftMax
   LogSoftMax = nn.LogSoftMax
end

local oracle_data = OracleLoader({
	['data_dir']=opt.data_dir,
	['train_batch_size']=opt.batch_size,
	['test_batch_size']=opt.test_batch_size,
	['use_prefetch']=true})

-- Network Definition
print('creating a neural network with random initialization')
local protos = {}
-- Word Embedding
local embed_dim = 200
protos.word_embed = nn.Sequential()
                                    :add(LookupTable(oracle_data.vocab_size, 200))
                                    :add(nn.Dropout(0.5))
                                    :add(Tanh())

-- LSTM
local rnn_size = 512
local nrnn_layer = 2
local rnn_dropout = 0.5
local rnnout_dim = 2*rnn_size*nrnn_layer
protos.rnn = DeepLSTM.create(embed_dim, rnn_size, nrnn_layer, rnn_dropout)

-- MULTIMODAL (Average Pooling and Aggregation)
local cnnout_dim = tonumber(opt.cnnout_dim)
local cnnout_w = tonumber(opt.cnnout_w)
local cnnout_h = tonumber(opt.cnnout_h)
local cnnout_spat = cnnout_w * cnnout_h
local multfeat_dim = 512
local attfeat_dim = 256
local netout_dim = oracle_data.train_data.num_steps

-- [q_embed] in: {rnnout_dim, att_rnn_s_dim} out: multfeat_dim
local q_embed = nn.Sequential()
	q_embed:add(nn.Dropout(0.5))
	q_embed:add(nn.Linear(rnnout_dim, multfeat_dim))
	q_embed:add(Tanh())
-- [i_embed] in: cnnout_dimxcnnout_hxcnnout_w, out: multfeat_dimxcnnout_spat
local i_embed = nn.Sequential()
   i_embed:add(nn.Dropout(0.5))
   i_embed:add(SpatialConvolution(cnnout_dim, multfeat_dim,3,3,1,1,1,1))
   i_embed:add(Tanh())
	i_embed:add(SpatialAveragePooling(cnnout_w, cnnout_h))
	i_embed:add(nn.Reshape(multfeat_dim))
-- [classifier] in:{multfeat_dim, multfeat_dim}, out: multfeat_dim
local in_qfeat = nn.Identity()()
local in_ifeat = nn.Identity()()
	local join_input = nn.CAddTable()({in_qfeat, in_ifeat})
   local resfeat = nn.Linear(multfeat_dim, multfeat_dim)(ReLU()(
		nn.Linear(multfeat_dim, multfeat_dim)(join_input)))
   local merge_feat = nn.Dropout(0.5)(nn.CAddTable()({join_input, resfeat}))
   local out_logit = nn.Linear(multfeat_dim, netout_dim)(merge_feat)
local classifier = nn.gModule({in_qfeat, in_ifeat}, {out_logit})
-- [multimodal]
local in_q = nn.Identity()()
local in_i = nn.Identity()()
   local qfeat = q_embed(in_q)
   local ifeat = i_embed(in_i)
	local logit = classifier({qfeat, ifeat})
protos.multimodal = nn.gModule({in_q, in_i}, {logit})

-- Criterion
if opt.prediction_type == 'is_best' then
	protos.criterion = nn.BCECriterion()
elseif opt.prediction_type == 'best_shortest' then
	protos.criterion = nn.CrossEntropyCriterion()
else
	error('Unknown prediction type')
end

-- Transfer to gpu
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

local embed_param, embed_grad = protos.word_embed:getParameters()
local rnn_param, rnn_grad = protos.rnn:getParameters()
local mult_param, mult_grad = protos.multimodal:getParameters()

-- clone many times for lstm sequential reading
print('clone many times for lstm sequential reading')
local mult_protos = {}
      mult_protos.rnns = {}
      mult_protos.word_embeds = {}

for t = 1, oracle_data.sequence_length do
   mult_protos.rnns[t] = protos.rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
   mult_protos.word_embeds[t] = protos.word_embed:clone('weight', 'bias', 'gradWeight', 'gradBias')
end

if string.len(opt.init_from) == 0 then
   -- if weight is not initialized then initialize weights
   print('initialing weights..')
   embed_param:uniform(-0.08, 0.08)
   rnn_param:uniform(-0.08, 0.08)
   mult_param:uniform(-0.08, 0.08)
end

-- LSTM Init State [TRAIN]
local init_state = torch.zeros(opt.batch_size, rnnout_dim)
local rnn_out = torch.zeros(opt.batch_size, rnnout_dim)
local drnn_out = torch.zeros(opt.batch_size, rnnout_dim)
local gradattprob = torch.zeros(opt.batch_size, cnnout_spat)
if opt.gpuid >= 0 then
   init_state = init_state:cuda()
   rnn_out = rnn_out:cuda()
   drnn_out = drnn_out:cuda()
   gradattprob = gradattprob:cuda()
end
-- LSTM Init State [TEST]
local test_init_state = torch.zeros(opt.test_batch_size, rnnout_dim)
local test_rnn_out = torch.Tensor(opt.test_batch_size, rnnout_dim)
if opt.gpuid >= 0 then
   test_init_state = test_init_state:cuda()
   test_rnn_out = test_rnn_out:cuda()
end
train_acc = 0
train_num_data = 0

iter_print = ''
--------------------------------------------------------
-- feval: Function for Optimization
--------------------------------------------------------
function feval(step_t)
   embed_grad:zero()
   rnn_grad:zero()
   mult_grad:zero()

   ------------------- get minibatch -------------------
   local feats, x, x_len, is_best, best_shortest, has_correct_answer, qids =
		oracle_data.train_data:NextBatchFeature(opt.feat_dir, cnnout_dim,
                                              cnnout_w, cnnout_h)
   if opt.gpuid >= 0 then
      feats = feats:cuda()
      x = x:cuda()
		is_best = is_best:cuda()
		best_shortest = best_shortest:cuda()
		has_correct_answer = has_correct_answer:cuda()
   end
   -- make sure we are in correct omde (this is cheap, sets flag)
   -------------------- forward pass -------------------
   -- RNN FORWARD
   local max_len = x_len:max()
   local min_len = x_len:min()
   local we_vecs = {}
   local rnn_state = {[0] = init_state}
   rnn_out:zero()
   for t = 1, max_len do
      mult_protos.rnns[t]:training()
      mult_protos.word_embeds[t]:training()
      local we = mult_protos.word_embeds[t]:forward(x[{t,{}}])
      local lst = mult_protos.rnns[t]:forward({we, rnn_state[t-1]})
      we_vecs[t] = we
      rnn_state[t] = lst
      if t >= min_len then
         for k = 1, opt.batch_size do
            if x_len[k] == t then
               rnn_out[k] = lst[k]
            end
         end
      end
   end
   -- MULTIMODAL FORWARD
	protos.multimodal:training()
	local logit = protos.multimodal:forward({rnn_out, feats})

	-- Compute accuracy
	if opt.prediction_type == 'is_best' then
		error('Not implemented yet')
	elseif opt.prediction_type == 'best_shortest' then
		local max_score, ans = torch.max(logit, 2)
			ans = torch.squeeze(ans):cuda()
		local is_correct = torch.eq(ans, best_shortest):cuda()
		train_acc = train_acc + is_correct:sum()
		train_num_data = train_num_data + best_shortest:nElement()
	end

   -- Loss forward, backward
   local loss
	local dlogit
	if opt.prediction_type == 'is_best' then
		loss = protos.criterion:forward(logit, is_best)
		dlogit = protos.criterion:backward(logit, is_best)
	elseif opt.prediction_type == 'best_shortest' then
		loss = protos.criterion:forward(logit, best_shortest)	
		dlogit = protos.criterion:backward(logit, best_shortest)
	end

   -------------------- backward pass -------------------
   -- MULTIMODAL BACKWARD
	local dmultimodal = protos.multimodal:backward({rnn_out, feats}, dlogit)

   -- rnn backward
   local drnn_state = {[max_len+1] = init_state:clone()}
                    -- true also zeros the clones
   for t = max_len, 1, -1 do
      drnn_out:copy(drnn_state[t+1])
      if t >= min_len then
         for k = 1, opt.batch_size do
            if x_len[k] == t then
               drnn_out[k] = dmultimodal[1][k]
            end
         end
      end
      local dlst = mult_protos.rnns[t]:backward({we_vecs[t], rnn_state[t-1]}, drnn_out)
      local dwe = mult_protos.word_embeds[t]:backward(x[{t, {}}], dlst[1])

      drnn_state[t] = dlst[2] -- dlst[1] is gradient on x, which we don't need
   end

   ------------------ gradient clipping ---------------
   local grad_norm
   grad_norm = embed_grad:norm()
   if grad_norm > opt.grad_clip then
      embed_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(
			' - embed grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(
			' - embed grad is not clipped norm: %f\n', grad_norm)
   end
   grad_norm = rnn_grad:norm()
   if grad_norm > opt.grad_clip then
      rnn_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(
			' - rnn grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(
			' - rnn grad is not clipped norm: %f\n', grad_norm)
   end
   grad_norm = mult_grad:norm()
   if grad_norm > opt.grad_clip then
      mult_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(
			' - multimodal grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(
			' - multimodal grad is not clipped norm: %f\n', grad_norm)
   end
   return loss
end

function predict_result (feats, x, x_len)
   -- transfer to gpu
   if opt.gpuid >= 0 then
      feats = feats:cuda()
      x = x:cuda()
   end
   -- make sure we are in correct mode (this is cheap, sets flag)
   -------------------- forward pass -------------------
   -- RNN FORWARD
   local max_len = x_len:max()
   local min_len = x_len:min()
   local we_vecs = {}
   local rnn_state = {[0] = test_init_state}
   test_rnn_out:zero()
   for t = 1, max_len do
      mult_protos.rnns[t]:evaluate()
      mult_protos.word_embeds[t]:evaluate()
      local we = mult_protos.word_embeds[t]:forward(x[{t,{}}])
      local lst = mult_protos.rnns[t]:forward({we, rnn_state[t-1]})
      we_vecs[t] = we
      rnn_state[t] = lst
      if t >= min_len then
         for k = 1, opt.test_batch_size do
            if x_len[k] == t then
               test_rnn_out[k] = lst[k]
            end
         end
      end
   end
   -- MULTIMODAL FORWARDl
	protos.multimodal:evaluate()
	local logit = protos.multimodal:forward({test_rnn_out, feats})

   return logit
end

-- create log file for optimization
testLogger = optim.Logger(paths.concat(string.format('%s/%s',opt.save_dir,opt.log_dir), 'test.log'))

-- reorder train / val data
oracle_data.train_data:SetBatchOrderOption(opt.batch_order_option)
oracle_data.train_data:Reorder()

------------------------------------------------------------------------
------------------------- Main Training Loop ---------------------------
------------------------------------------------------------------------
-- stats for drawing figures
local epoch_history = {}
local trainacc_history = {}
local testacc_history = {}
local dense_epoch_history = {}
local dense_trainloss = {}
local trainloss_history = {}
local lr_history = {}
local mult_lr_history = {}
local accum_trainloss = 0
local avgloss = nil
-- optim state
local multlearningrate = opt.mult_learning_rate
local learningrate = opt.learning_rate
local embed_optimstate = {}
local rnn_optimstate = {}
local mult_optimstate = {}
local alpha
local beta
local epsilon

-- initialize display for visualization
if opt.display == 'true' then
   disp = require 'display'
   disp.url = string.format('http://%s:%d/events', opt.display_host, opt.display_port)
end

iter_epoch = oracle_data.train_data.iter_per_epoch
total_iter = opt.max_epochs * iter_epoch 
for it = 1, total_iter do
   local epoch = it / iter_epoch
   iter_print = iter_print .. '-----------------------------------------------------------------------\n'
   local timer = torch.Timer()
   local loss = feval(it)
   if opt.optim == 'adam' then
      adam(embed_param, embed_grad, learningrate, alpha, beta, epsilon, embed_optimstate)
      adam(rnn_param, rnn_grad, learningrate, alpha, beta, epsilon, rnn_optimstate)
      adam(mult_param, mult_grad, multlearningrate, alpha, beta, epsilon, mult_optimstate)
   else
      error('bad option opt.optim')
   end
   local time = timer:time().real
   
   local train_loss = loss -- the loss is inside a list, pop it
	avgloss = avgloss or train_loss
	avgloss = avgloss * 0.9 + train_loss
   if it % opt.denseloss_saveinterval == 1 then
      table.insert(dense_trainloss, avgloss)
      table.insert(dense_epoch_history, epoch)
   end
   if opt.display == 'true' then
      local base_display_id = opt.display_id + 100
      if it % opt.denseloss_saveinterval == 1 then
         local line_dense_epoch = torch.Tensor(dense_epoch_history)
         local tab_label = {'epoch'}
         local line_dense_trainloss = torch.Tensor(dense_trainloss)
         line_dense_epoch = line_dense_epoch:cat(line_dense_trainloss, 2)
         table.insert(tab_label, 'train')
         disp.plot(line_dense_epoch,
                   {title='training loss',
                    labels=tab_label,
                    ylabel='loss', win=base_display_id})
         base_display_id = base_display_id + 1
      end
   end
   accum_trainloss = accum_trainloss + train_loss
   
   if it % opt.print_iter == 0 then
      iter_print = iter_print .. string.format(
			'%d (epoch %.3f), lr=%f, multlr=%f, time=%.2fs\n',
         it, epoch, learningrate, multlearningrate, time)
      local loss_print = string.format('[loss]:%6.8f\n', train_loss)
      iter_print = iter_print .. (loss_print .. '\n')

      iter_print = iter_print .. string.format('[norm] EP:%6.4f, RP:%6.4f, MP:%6.4f\n',
                           embed_param:norm(), rnn_param:norm(), mult_param:norm())
      iter_print = iter_print .. string.format('[grad] EG:%6.4f, RG:%6.4f, MG:%6.4f\n',
                           embed_grad:norm(), rnn_grad:norm(), mult_grad:norm())
   end
   if (it % iter_epoch == 0 and epoch % opt.test_interval == 0) or it == total_iter then
      -- inorder
      oracle_data.test_data:Inorder()
      local test_iter = oracle_data.test_data.iter_per_epoch
		local test_acc = 0
		local test_num_data = 0
		local oracle_predict_results_table = {}
      print(string.format('start test'))
      for k = 1, test_iter do
         print(string.format('test -- [%d/%d]', k, test_iter))
         local feats, x, x_len, is_best, best_shortest, has_correct_answer,
				qids = oracle_data.test_data:NextBatchFeature(opt.feat_dir,
				cnnout_dim, cnnout_w, cnnout_h)
			if opt.gpuid >= 0 then
				is_best = is_best:cuda()
				best_shortest = best_shortest:cuda()
			end
			local logit = predict_result(feats, x, x_len)
			local max_score, ans
			if opt.prediction_type == 'is_best' then
				error('Not implemented yet')
			elseif opt.prediction_type == 'best_shortest' then
				-- Compute accuracy
			   max_score, ans = torch.max(logit, 2)
				ans = torch.squeeze(ans):cuda()
				local is_correct = torch.eq(ans, best_shortest):cuda()
				test_acc = test_acc + is_correct:sum()
				test_num_data = test_num_data + best_shortest:nElement()
			end
			-- Save results
			for bidx=1, qids:size(1) do
				local oracle_predict_result = {}
				oracle_predict_result['question_id'] = qids[bidx]
				if opt.prediction_type == 'is_best' then
					error('Not implemented yet')
				elseif opt.prediction_type == 'best_shortest' then
					oracle_predict_result['best_shortest'] = ans[bidx]
				end
				table.insert(oracle_predict_results_table, oracle_predict_result)
			end
         if k % opt.free_interval == 0 then collectgarbage() end
      end
      if train_num_data ~= 0 then 
         train_acc = train_acc / train_num_data
      end
		if test_num_data ~= 0 then
			test_acc = test_acc / test_num_data
		end
      if iter_epoch ~= 0 then accum_trainloss = accum_trainloss / iter_epoch end
      -- draw figures      
      table.insert(epoch_history, epoch)
      table.insert(trainacc_history, train_acc)
      table.insert(trainloss_history, accum_trainloss)
      table.insert(lr_history, learningrate)
      table.insert(mult_lr_history, multlearningrate)
		table.insert(testacc_history, test_acc)

      local base_display_id = opt.display_id + 200

      -- epoch history (used globally)
      local line_epoch = torch.Tensor(epoch_history)

      -- accuracy curve
      local line_trainacc = torch.Tensor(trainacc_history)
      local line_testacc = torch.Tensor(testacc_history)
      if epoch % opt.graph_interval == 0 or it == total_iter then
         local fname_accplot = paths.concat(string.format('%s/%s', opt.save_dir,
				opt.graph_dir), 'accuracy_curve.png')
         gnuplot.pngfigure(fname_accplot)
         gnuplot.plot({'train', line_epoch, line_trainacc},
                      {'test', line_epoch, line_testacc})
         gnuplot.xlabel('epoch')
         gnuplot.ylabel('accuracy')
         gnuplot.movelegend('right','bottom')
         gnuplot.title(string.format('train / test accuracy'))
         gnuplot.plotflush()
      end
      if opt.display == 'true' then
         disp.plot(line_epoch:cat(line_trainacc, 2):cat(line_testacc, 2),
                   {title='train / test accuracy',
                    labels={'epoch', 'train', 'test'},
                    ylabel='accuracy', win=base_display_id})
         base_display_id = base_display_id + 1
      end
      -- loss curve
      local line_trainloss = torch.Tensor(trainloss_history)
      if epoch % opt.graph_interval == 0 or it == total_iter then
         local fname_lossplot = paths.concat(string.format('%s/%s',
				opt.save_dir, opt.graph_dir), 'loss_curve_hop%02d.png')
         gnuplot.pngfigure(fname_lossplot)
         gnuplot.plot({'train', line_epoch, line_trainloss})
         gnuplot.xlabel('epoch')
         gnuplot.ylabel('loss')
         gnuplot.movelegend('right','top')
         gnuplot.title('training loss')
         gnuplot.plotflush()
      end
      -- learning rate curve
      local line_lr_curve = torch.Tensor(lr_history)
      if epoch % opt.graph_interval == 0 or it == total_iter then
         local fname_lr = paths.concat(string.format('%s/%s', opt.save_dir,
				opt.graph_dir), 'learning_rate.png')
         gnuplot.pngfigure(fname_lr)
         gnuplot.plot({'lr', line_epoch, line_lr_curve})
         gnuplot.xlabel('iter')
         gnuplot.ylabel('learning rate')
         gnuplot.movelegend('right','top')
         gnuplot.title('learning rate')
         gnuplot.plotflush()
      end

      local mult_line_lr_curve = torch.Tensor(mult_lr_history)
      if epoch % opt.graph_interval == 0 or it == total_iter then
         -- mult learning rate curve
         local fname_multlr = paths.concat(string.format('%s/%s', opt.save_dir,
				opt.graph_dir), 'mult_learning_rate.png')
         gnuplot.pngfigure(fname_multlr)
         gnuplot.plot({'lr', line_epoch, mult_line_lr_curve})
         gnuplot.xlabel('iter')
         gnuplot.ylabel('learning rate')
         gnuplot.movelegend('right','top')
         gnuplot.title('multimodal learning rate')
         gnuplot.plotflush()
      end
      local tab_testlog = {}
      tab_testlog['epoch'] = epoch
      tab_testlog['trainacc'] = train_acc * 100
      tab_testlog['testacc'] = test_acc * 100
      testLogger:add(tab_testlog)
      print(string.format('iter: %d, epoch: %f', it, epoch))
      print_acc = ''
      print_acc = print_acc .. string.format('trainacc: %f, ', train_acc * 100)
      print_acc = print_acc .. string.format('testacc: %f, ', test_acc * 100)
      print_acc = print_acc .. string.format('trainloss: %f, ', accum_trainloss)
      print_acc = print_acc .. '\n'
      print(print_acc)

		-- SAVE RESULTS
		local fn_result = string.format('oracle_selection_%.2f_epoch.json', epoch)	
		local save_result_path = paths.concat(opt.save_dir, opt.results_dir, fn_result)
		local result_json = cjson.encode(oracle_predict_results_table)
		local wf = io.open(save_result_path, 'w')
		wf:write(result_json)
		wf:close()	
      -- SAVE TRAINED PARAMETERS
      local savefile = string.format('%s/%s/snapshot_iter%06d_epoch%.2f.t7',
                                     opt.save_dir, opt.snapshot_dir, it, epoch)
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.it = it
      checkpoint.opt = opt
      checkpoint.epoch = epoch
      checkpoint.params = {[1]=embed_param, [2]=rnn_param, [3]=mult_param}
      torch.save(savefile, checkpoint)

      train_acc = 0
      train_num_data = 0
      accum_trainloss = 0
   end
   print (iter_print)
   iter_print = ''

   -- exponential learning rate decay
   if it % iter_epoch == 0 and opt.lr_decay < 1 then
      if epoch % opt.lr_decay_interval == 0 then
         local lr_decay = opt.lr_decay
         learningrate = learningrate * lr_decay -- decay it
         local mult_lr_decay = opt.mult_lr_decay
         multlearningrate = multlearningrate * mult_lr_decay
         print('decayed learning rate by a factor ' .. lr_decay .. ' to ' .. learningrate)
      end
   end

   if it % opt.free_interval == 0 then collectgarbage() end
end











