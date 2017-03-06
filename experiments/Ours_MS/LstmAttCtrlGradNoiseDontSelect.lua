require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'utils.optim_updates'
require 'utils.tools'
require 'gnuplot'
require 'image'

local DeepLSTM = require 'model.DeepLSTM'
local ATTLSTM = require 'model.ATTLSTM'
local vqa_prepro_loader = require 'utils.vqa_prepro_loader'
local model_utils = require 'utils.model_utils'
local cjson = require 'cjson'

-- read input parameter
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Multi Hop Attention (content + memory based) with LSTM (resdual) for VQA')
cmd:text('[trained with noisy gradient, prediction step selection is added, Prediction Loss is BCE]')
cmd:text('[Dont care the case when no hop have correct answer]')
cmd:text()
cmd:text('ALGORITHM NAME')
cmd:option('-alg_name', 'LstmAttCtrlGradNoiseDontSelect224Pool5', 'algorithm name for vqa eval result file')

cmd:text('GENERAL')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-continue_learning', 'false', '[true|false]')
cmd:option('-loss_explod_threshold', 3, 'loss explosion threshold')

cmd:text('DATA')
cmd:option('-split', 'val2014', 'option: [val2014|test2015|test-dev2015]')
cmd:option('-vqa_dir', './data/VQA_prepro/data_train_val','vqa data directory')
cmd:option('-feat_dir', './data/vqa_VGG16Conv_pool5/feat_224x224', 'vqa feature directory')
cmd:option('-gt_anno', './data/VQA_prepro/comprehend/comprehend_val2014_annotations.json', '')
cmd:option('-cnnout_w', 7, 'cnnout_w')
cmd:option('-cnnout_h', 7, 'cnnout_h')
cmd:option('-cnnout_dim', 512, 'cnnout_dim')

cmd:text('OPTIMIZATION')
cmd:option('-optim', 'adam', 'option: adam')
cmd:option('-batch_order_option', 1, '[1]:shuffle, [2]:inorder, [3]:sort, [4]:randsort')
cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-learning_rate',3e-3,'learning rate')
cmd:option('-lr_decay',0.9,'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-mult_learning_rate',3e-4,'learning rate')
cmd:option('-mult_lr_decay',0.9,'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-lr_decay_interval', 1, 'learning rate decay interval in epoch')
cmd:option('-batch_size', 100, 'number of examples for each batch')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value')
cmd:option('-free_interval',10, 'interval for collect garbage')

cmd:text('HYPER PARAMETER')
cmd:option('-nhop', 8, 'number of hop to use')
cmd:option('-noisy_eta', 0.01,'eta for gradient noise injection')
cmd:option('-noisy_gamma', 0.55,'gamma for gradient noise injection')

cmd:text('VISUALIZATION')
cmd:option('-test_interval', 1, 'interval of testing in epoch')
cmd:option('-graph_interval', 10, 'interval of drawing graph')
cmd:option('-print_iter', 1, 'interval of printing training loss')
cmd:option('-denseloss_saveinterval', 50, 'interval for saving dense training loss in iteration')
cmd:option('-visatt', 'true', 'whether visualize attention or not')

cmd:text('DISPLAY')
cmd:option('-display', 'true', 'display result while training')
cmd:option('-display_id', 10, 'display window id')
cmd:option('-display_host', 'localhost', 'display hostname 0.0.0.0')
cmd:option('-display_port', 8000, 'display port number')
cmd:option('-display_interval', 10, 'display interval')

cmd:text('SAVE')
cmd:option('-save_dir', 'save_result_vqa', 'subdirectory to save results [log, snapshot]')
cmd:option('-seed', 123, 'torch manual random number generator seed')

cmd:text('GPU')
cmd:option('-gpuid', 0, 'which GPU to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', '[cunn|cudnn]')

-- parse input params
opt = cmd:parse(arg or {})
torch.manualSeed(opt.seed)

-- load vqa dataset
print('load vqa dataset')
if opt.split == 'val2014' then
   opt.test_batch_size = 83
elseif opt.split == 'test2015' then
   -- 244302
   opt.test_batch_size = 57 -- or 114
elseif opt.split == 'test-dev2015' then
   -- 60864
   opt.test_batch_size = 32
else
   assert(false, 'unknown split option')
end

local nHop = opt.nhop

print('')
print('')
print('')
print(string.format('nHop: %d', nHop))
print('')
print('')
print('')

-- constant options
opt.save_dir = string.format('%s_%s', opt.save_dir,opt.split) 
opt.log_dir = 'training_log'
opt.snapshot_dir = 'snapshot'
opt.results_dir = 'results'
opt.graph_dir = 'graphs'
opt.figure_dir = 'figures'

-- create directory and log file
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.log_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.snapshot_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
for h=1,nHop+2 do
   os.execute(string.format('mkdir -p %s/%s/hop_%02d', opt.save_dir, opt.results_dir, h))
end
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.graph_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.figure_dir))
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
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
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
   SpatialMaxPooling = nn.SpatialMaxPooling
   SoftMax = nn.SoftMax
   LogSoftMax = nn.LogSoftMax
end

local opt_prefetch = true
local vqa_data = vqa_prepro_loader.load_data(opt.vqa_dir, opt.batch_size, opt_prefetch, 
                                             opt.split, opt.test_batch_size)

local val_gt_anno
if opt.split == 'val2014' then
   local cjson = require 'cjson'
   val_gt_anno = cjson.decode(io.open(opt.gt_anno,'r'):read())
end

-- Network Definition
print('creating a neural network with random initialization')
local protos = {}
-- Word Embedding
local embed_dim = 200
protos.word_embed = nn.Sequential()
                                    :add(LookupTable(vqa_data.vocab_size, 200))
                                    :add(nn.Dropout(0.5))
                                    :add(Tanh())

-- LSTM
local rnn_size = 512
local nrnn_layer = 2
local rnn_dropout = 0.5
local rnnout_dim = 2*rnn_size*nrnn_layer
protos.rnn = DeepLSTM.create(embed_dim, rnn_size, nrnn_layer, rnn_dropout)

-- MULTIMODAL (ATTENTION)
local cnnout_dim = tonumber(opt.cnnout_dim)
local cnnout_w = tonumber(opt.cnnout_w)
local cnnout_h = tonumber(opt.cnnout_h)
local cnnout_spat = cnnout_w * cnnout_h
local multfeat_dim = 512
local attfeat_dim = 256
local netout_dim = vqa_data.answer_size --(1000)

-- [attlstm] in: {2*multfeat_dim, att_rnn_s_dim} {att_rnn_size, att_rnn_s_dim}
local att_rnn_size = 512
local att_rnn_nlayer = 1
local att_rnn_dropout = 0.0
local att_rnn_s_dim = att_rnn_size*att_rnn_nlayer
local attlstm = ATTLSTM.create(multfeat_dim, att_rnn_size, att_rnn_nlayer, att_rnn_dropout)
-- [q_embed] in: {rnnout_dim, att_rnn_s_dim} out: multfeat_dim
local in_q = nn.Identity()()
local in_prev_h = nn.Identity()()
      local q_proj = nn.Linear(rnnout_dim, multfeat_dim)(nn.Dropout(0.5)(in_q))
      local h_proj = nn.Linear(att_rnn_s_dim, multfeat_dim)(in_prev_h)
      local out_q_embed = Tanh()(nn.CAddTable()({q_proj,h_proj}))
local q_embed = nn.gModule({in_q, in_prev_h}, {out_q_embed})
-- [i_embed] in: cnnout_dimxcnnout_hxcnnout_w, out: multfeat_dimxcnnout_spat
local i_embed = nn.Sequential()
      i_embed:add(nn.Dropout(0.5))
      i_embed:add(SpatialConvolution(cnnout_dim, multfeat_dim,1,1,1,1,0,0))
      i_embed:add(Tanh())
      i_embed:add(nn.Reshape(multfeat_dim, cnnout_spat))
-- [attbycontent] in: {multfeat_dim, cnnout_dimxcnnout_spat}, out: cnnout_spat
local in_qfeat = nn.Identity()()
local in_ifeat = nn.Identity()()
      local qfeatatt = nn.Replicate(cnnout_spat, 3)(nn.Linear(multfeat_dim, attfeat_dim)(in_qfeat))
      local ifeatproj = SpatialConvolution(multfeat_dim, attfeat_dim,1,1,1,1,0,0)(
                                           nn.Reshape(multfeat_dim,cnnout_spat, 1)(in_ifeat))
      local ifeatatt = nn.Reshape(attfeat_dim, cnnout_spat)(ifeatproj)
      local addfeat = nn.Reshape(attfeat_dim,cnnout_spat,1)(Tanh()(nn.CAddTable()({ifeatatt, qfeatatt})))
      local attscore = nn.Reshape(cnnout_spat)(SpatialConvolution(attfeat_dim,1,1,1,1,1,0,0)(addfeat))
local attbycontent = nn.gModule({in_qfeat, in_ifeat}, {attscore})
-- [attselect] in: {multfeat_dimxcnnout_spat, cnnout_spat}, out: multfeat_dim
local attselect = nn.Sequential()
      local in_attselect = nn.ParallelTable()
            local in_ifeat = nn.Identity()
            local in_attprob = nn.Sequential()
                  in_attprob:add(nn.Replicate(multfeat_dim, 2))
            in_attselect:add(in_ifeat)
            in_attselect:add(in_attprob)
      attselect:add(in_attselect)
      attselect:add(nn.CMulTable())
      attselect:add(nn.Sum(3))
-- [classifier] in:{multfeat_dim, multfeat_dim}, out: multfeat_dim
local in_qfeat = nn.Identity()()
local in_attfeat = nn.Identity()()
local in_attprob = nn.Identity()()
local in_prev_c = nn.Identity()()
local in_prev_h = nn.Identity()()
      local q_n_att_feat = nn.CAddTable()({in_qfeat, in_attfeat})
      local feat_attprob = nn.Linear(cnnout_spat, multfeat_dim)(in_attprob)
      local join_input = nn.CAddTable()({q_n_att_feat, feat_attprob})
      local tab_lstmout = attlstm({join_input, in_prev_c, in_prev_h})
      local out_next_c = nn.SelectTable(1)(tab_lstmout)
      local out_next_h = nn.SelectTable(2)(tab_lstmout)
      local lstmfeat = nn.Dropout(att_rnn_dropout)(out_next_h)
      local merge_feat = nn.Dropout(0.5)(
                              nn.CAddTable()({join_input, 
                                              nn.Linear(att_rnn_s_dim, multfeat_dim)(lstmfeat)}))
      local out_score = nn.Linear(multfeat_dim, netout_dim)(merge_feat)
      local out_do_pred = nn.Sum(2)(Sigmoid()(nn.Linear(multfeat_dim, 1)(merge_feat)))
local classifier = nn.gModule({in_qfeat, in_attfeat, in_attprob, in_prev_c, in_prev_h}, 
                              {out_score, out_do_pred, out_next_c, out_next_h})
-- [attbymemory] in: {cnnout_spat, att_rnn_s_dim}, out: cnnout_spat
local in_attscore = nn.Identity()()
local in_prev_h = nn.Identity()()
      local attscore_bymem = nn.Linear(att_rnn_s_dim, cnnout_spat)(in_prev_h)
      local new_attscore = nn.CAddTable()({in_attscore, attscore_bymem})
      local out_attprob = SoftMax()(new_attscore)
local attbymemory = nn.gModule({in_attscore, in_prev_h}, {out_attprob})
-- [multimodal]
local in_q = nn.Identity()()
local in_i = nn.Identity()()
local in_prev_c = nn.Identity()()
local in_prev_h = nn.Identity()()
      local qfeat = q_embed({in_q, in_prev_h})
      local ifeat = i_embed(in_i)
      local attscore = attbycontent({qfeat, ifeat})
      local attprob = attbymemory({attscore, in_prev_h})
      local attfeat = attselect({ifeat, attprob})
      local tab_clsout = classifier({qfeat, attfeat, attprob, in_prev_c, in_prev_h})
      local multout = nn.SelectTable(1)(tab_clsout)
      local do_pred = nn.SelectTable(2)(tab_clsout)
      local next_c = nn.SelectTable(3)(tab_clsout)
      local next_h = nn.SelectTable(4)(tab_clsout)
protos.multimodal = nn.gModule({in_q, in_i, in_prev_c, in_prev_h}, 
                               {multout, do_pred, attprob, next_c, next_h})

-- Criterion
protos.criterion = nn.CrossEntropyCriterion()
protos.bincriterion = nn.BCECriterion()

protos.branch = nn.ConcatTable()
for h = 1, nHop do
   protos.branch:add(nn.Identity())
end
-- transfer to gpu
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

local embed_param, embed_grad = protos.word_embed:getParameters()
local rnn_param, rnn_grad = protos.rnn:getParameters()
local mult_param, mult_grad = protos.multimodal:getParameters()

local embed_noise = embed_grad:clone()
local rnn_noise =rnn_grad:clone()
local mult_noise = mult_grad:clone()

-- clone many times for lstm sequential reading
print('clone many times for lstm sequential reading')
local mult_protos = {}
      mult_protos.rnns = {}
      mult_protos.word_embeds = {}
      mult_protos.multimodals = {}
      mult_protos.criterions = {}
      mult_protos.bincriterions = {}

for t = 1, vqa_data.seq_len do
   mult_protos.rnns[t] = protos.rnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
   mult_protos.word_embeds[t] = protos.word_embed:clone('weight', 'bias', 'gradWeight', 'gradBias')
end
for h = 1, nHop do
   mult_protos.multimodals[h] = protos.multimodal:clone('weight', 'bias', 'gradWeight', 'gradBias')
   mult_protos.criterions[h] = protos.criterion:clone() 
   mult_protos.bincriterions[h] = protos.bincriterion:clone()
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
local att_c = torch.zeros(opt.batch_size, att_rnn_s_dim)
local grad_att_c = torch.zeros(opt.batch_size, att_rnn_s_dim)
local att_h = torch.zeros(opt.batch_size, att_rnn_s_dim)
local grad_att_h = torch.zeros(opt.batch_size, att_rnn_s_dim)
local uni_pred = torch.zeros(opt.batch_size, netout_dim)
local did_pred = torch.zeros(opt.batch_size)
local did_correct = torch.zeros(opt.batch_size)
local do_pred_gt = torch.zeros(opt.batch_size)
local select_pred = torch.zeros(opt.batch_size, netout_dim)
if opt.gpuid >= 0 then
   init_state = init_state:cuda()
   rnn_out = rnn_out:cuda()
   drnn_out = drnn_out:cuda()
   gradattprob = gradattprob:cuda()
   att_c = att_c:cuda()
   grad_att_c = grad_att_c:cuda()
   att_h = att_h:cuda()
   grad_att_h = grad_att_h:cuda()
   uni_pred = uni_pred:cuda()
   did_pred = did_pred:cuda()
   did_correct = did_correct:cuda()
   do_pred_gt = do_pred_gt:cuda()
   select_pred = select_pred:cuda()
end
-- LSTM Init State [TEST]
local test_init_state = torch.zeros(opt.test_batch_size, rnnout_dim)
local test_rnn_out = torch.Tensor(opt.test_batch_size, rnnout_dim)
local test_att_c = torch.zeros(opt.test_batch_size, att_rnn_s_dim)
local test_att_h = torch.zeros(opt.test_batch_size, att_rnn_s_dim)
local test_uni_pred = torch.zeros(opt.test_batch_size, netout_dim)
local test_uni_att = torch.zeros(opt.test_batch_size, cnnout_spat)
local test_did_pred = torch.zeros(opt.test_batch_size)
local test_did_correct = torch.zeros(opt.test_batch_size)
local test_select_pred = torch.zeros(opt.test_batch_size, netout_dim)
local test_select_att = torch.zeros(opt.test_batch_size, cnnout_spat)
if opt.gpuid >= 0 then
   test_init_state = test_init_state:cuda()
   test_rnn_out = test_rnn_out:cuda()
   test_att_c = test_att_c:cuda()
   test_att_h = test_att_h:cuda()
   test_uni_pred = test_uni_pred:cuda()
   test_uni_att = test_uni_att:cuda()
   test_did_pred = test_did_pred:cuda()
   test_did_correct = test_did_correct:cuda()
   test_select_pred = test_select_pred:cuda()
   test_select_att = test_select_att:cuda()
end
-- mask for computing MultipleChoice Answers
test_mc_mask = torch.zeros(opt.test_batch_size, vqa_data.answer_size)
if opt.gpuid >= 0 then
   test_mc_mask = test_mc_mask:cuda()
end
tab_train_acc = {} 
tab_train_do_pred_acc = {}
tab_train_num_data = {}
tab_train_do_pred_num_data = {}
for h=1,nHop+2 do
   tab_train_acc[h] = 0
   tab_train_do_pred_acc[h] = 0
   tab_train_num_data[h] = 0
   tab_train_do_pred_num_data[h] = 0
end
iter_print = ''
--------------------------------------------------------
-- feval: Function for Optimization
--------------------------------------------------------
function feval(step_t)
   embed_grad:zero()
   rnn_grad:zero()
   mult_grad:zero()

   ------------------- get minibatch -------------------
   local feats, x, x_len, y, qids = vqa_data.train_data:next_batch_feat(opt.feat_dir, cnnout_dim, cnnout_w, cnnout_h)
   if opt.gpuid >= 0 then
      feats = feats:float():cuda()
      x = x:float():cuda()
      y = y:float():cuda()
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
   -- branch feat
   protos.branch:training()
   local brcfeat = protos.branch:forward({rnn_out, feats})
   -- MULTIMODAL FORWARD
   local tab_mult_c = {[0] = att_c}
   local tab_mult_h = {[0] = att_h}
   local tab_pred = {} 
   local tab_loss = {}
   local tab_loss_do_pred = {}
   local tab_do_pred = {}
   local tab_do_pred_gt = {}
   uni_pred:zero()
   select_pred:zero()
   did_pred:zero()
   did_correct:zero()
   for h=1,nHop do
      mult_protos.multimodals[h]:training()
      local pred = mult_protos.multimodals[h]:forward({brcfeat[h][1], brcfeat[h][2], 
                                                       tab_mult_c[h-1], tab_mult_h[h-1]})
      uni_pred:add(pred[1])
      tab_pred[h] = pred[1]
      tab_do_pred[h] = pred[2]
      tab_mult_c[h] = pred[4]
      tab_mult_h[h] = pred[5]
      -- compute accuracy
      local max_score, ans = torch.max(pred[1], 2)
            ans = torch.squeeze(ans):cuda()
      local is_correct = torch.eq(ans, y):cuda()
      tab_train_acc[h] = tab_train_acc[h] + is_correct:sum()
      tab_train_num_data[h] = tab_train_num_data[h] + y:nElement()
      
      -- find do pred ground truth and compute loss and accuracies
      if opt.gpuid < 0 then is_correct = is_correct:float() end
      -- did correct is do_pred_gt
      do_pred_gt:copy(is_correct)
      tab_do_pred_gt[h] = do_pred_gt:clone()
 
      -- compute do pred accuracy
      local do_pred = torch.gt(pred[2], 0.5):cuda()
      if opt.gpuid < 0 then do_pred = do_pred:float() end

      -- accumulate select_pred
      local pred_cur_hop = torch.add(do_pred, -did_pred):clamp(0,1)
            pred_cur_hop = pred_cur_hop:reshape(opt.batch_size,1):repeatTensor(1,netout_dim)
      select_pred:add(torch.cmul(pred[1], pred_cur_hop))

      iter_print = iter_print .. string.format('[%d hop] do_pred min: %f, max: %f, mean: %f, pred_cur_mean: %f\n', 
                               h, do_pred:min(), do_pred:max(), do_pred:mean(), pred_cur_hop:mean())

      -- update did correct
      did_correct:add(is_correct):clamp(0,1)
      -- update did_pred
      did_pred:add(do_pred):clamp(0,1)

      -- loss forward
      local loss = mult_protos.criterions[h]:forward(pred[1], y)
      tab_loss[h] = loss
   end
   -- uni pred (addition)
   uni_pred:div(nHop)
   -- compute accuracy
   local max_score, ans = torch.max(uni_pred, 2)
         ans = torch.squeeze(ans):cuda()
   tab_train_acc[nHop+1] = tab_train_acc[nHop+1] + torch.eq(ans, y):cuda():sum()
   tab_train_num_data[nHop+1] = tab_train_num_data[nHop+1] + y:nElement()
   -- loss forward
   local loss = protos.criterion:forward(uni_pred, y)
   tab_loss[nHop+1] = loss

   -- select pred (selective merge)
   -- compute accuracy
   local max_score, ans = torch.max(select_pred, 2)
         ans = torch.squeeze(ans):cuda()
   tab_train_acc[nHop+2] = tab_train_acc[nHop+2] + torch.eq(ans, y):cuda():sum()
   tab_train_num_data[nHop+2] = tab_train_num_data[nHop+2] + y:nElement()
   -- loss forward
   local loss = protos.criterion:forward(select_pred, y)
   tab_loss[nHop+2] = loss

   -- compute do_pred accuracy and loss
   iter_print = iter_print .. string.format('[At least one answer ratio] min: %f, max: %f, mean: %f\n',did_correct:min(), did_correct:max(), did_correct:mean()) 
   for h=1,nHop do
      iter_print = iter_print .. string.format('[%d hop] do_pred_gt min: %f, max: %f, mean: %f\n', 
                               h, tab_do_pred_gt[h]:min(), tab_do_pred_gt[h]:max(), tab_do_pred_gt[h]:mean())
 
      -- compute do pred accuracy
      local do_pred = torch.gt(tab_do_pred[h], 0.5):cuda()
      if opt.gpuid < 0 then do_pred = do_pred:float() end
      -- if did_correct: 0, don care this batch as no hop have correct answer
      tab_train_do_pred_acc[h] = tab_train_do_pred_acc[h] + torch.cmul(torch.eq(do_pred, tab_do_pred_gt[h]):cuda(), did_correct):sum()
      tab_train_do_pred_num_data[h] = tab_train_do_pred_num_data[h] + did_correct:sum()     

      local loss_do_pred = mult_protos.bincriterions[h]:forward(tab_do_pred[h], tab_do_pred_gt[h])
      tab_loss_do_pred[h] = loss_do_pred
   end

   -------------------- backward pass -------------------
   -- MULTIMODAL BACKWARD
   local d_mult_c = {[nHop+1] = grad_att_c}
   local d_mult_h = {[nHop+1] = grad_att_h}
   local d_brcfeat = {}
   for h=nHop,1,-1 do
      local d_do_pred = mult_protos.bincriterions[h]:backward(tab_do_pred[h], tab_do_pred_gt[h])
            d_do_pred:mul(0)
            --don care this batch as no hop have correct answer
      local dpred = mult_protos.criterions[h]:backward(tab_pred[h], y)
      
      local dmult = mult_protos.multimodals[h]:backward({brcfeat[h][1], brcfeat[h][2],
                                                         tab_mult_c[h-1], tab_mult_h[h-1]},
                                                        {dpred, d_do_pred, gradattprob, 
                                                         d_mult_c[h+1], d_mult_h[h+1]}) 
      d_mult_h[h] = dmult[4]
      d_mult_c[h] = dmult[3]
      d_brcfeat[h] = {dmult[1], dmult[2]}
   end
   local d_feats = protos.branch:backward({rnn_out, feats}, d_brcfeat)
   -- rnn backward
   local drnn_state = {[max_len+1] = init_state:clone()}
                    -- true also zeros the clones
   for t = max_len, 1, -1 do
      drnn_out:copy(drnn_state[t+1])
      if t >= min_len then
         for k = 1, opt.batch_size do
            if x_len[k] == t then
               drnn_out[k] = d_feats[1][k]
            end
         end
      end
      local dlst = mult_protos.rnns[t]:backward({we_vecs[t], rnn_state[t-1]}, drnn_out)
      local dwe = mult_protos.word_embeds[t]:backward(x[{t, {}}], dlst[1])

      drnn_state[t] = dlst[2] -- dlst[1] is gradient on x, which we don't need
   end
   -------------- gradient noise injection ------------
   local noisy_var = opt.noisy_eta / ((step_t+1)*opt.noisy_gamma)
   local noisy_std = math.sqrt(noisy_var)
         embed_noise:normal(0, noisy_std)
         rnn_noise:normal(0, noisy_std)
         mult_noise:normal(0, noisy_std)
   embed_grad:add(embed_noise)
   rnn_grad:add(rnn_noise)
   mult_grad:add(mult_noise)

   ------------------ gradient clipping ---------------
   local grad_norm
   grad_norm = embed_grad:norm()
   if grad_norm > opt.grad_clip then
      embed_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(' - embed grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(' - embed grad is not clipped norm: %f\n', grad_norm)
   end
   grad_norm = rnn_grad:norm()
   if grad_norm > opt.grad_clip then
      rnn_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(' - rnn grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(' - rnn grad is not clipped norm: %f\n', grad_norm)
   end
   grad_norm = mult_grad:norm()
   if grad_norm > opt.grad_clip then
      mult_grad:mul(opt.grad_clip / grad_norm)
      iter_print = iter_print .. string.format(' - multimodal grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
   else
      iter_print = iter_print .. string.format(' - multimodal grad is not clipped norm: %f\n', grad_norm)
   end
   return tab_loss, tab_loss_do_pred
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
   -- branch feat
   protos.branch:evaluate()
   local brcfeat = protos.branch:forward({test_rnn_out, feats})
   -- MULTIMODAL FORWARD
   local tab_mult_c = {[0] = test_att_c}
   local tab_mult_h = {[0] = test_att_h}
   local tab_pred = {}
   local tab_do_pred = {}
   local tab_att = {}
   test_uni_pred:zero()
   test_uni_att:zero()
   test_select_pred:zero()
   test_did_pred:zero()
   for h=1,nHop do
      mult_protos.multimodals[h]:evaluate()
      local pred = mult_protos.multimodals[h]:forward({brcfeat[h][1], brcfeat[h][2], 
                                                       tab_mult_c[h-1], tab_mult_h[h-1]})
      -- uni pred accumulate
      test_uni_pred:add(pred[1])
      test_uni_att:add(pred[3])
      -- select pred accumulate
      local do_pred = torch.gt(pred[2], 0.5):cuda()
      if opt.gpuid < 0 then do_pred = do_pred:float() end
      if h == nHop then do_pred:fill(1) end -- for testing, always predict in the final step
      local pred_cur_hop = torch.add(do_pred, -test_did_pred):clamp(0,1):reshape(opt.test_batch_size,1)
      test_select_pred:add(torch.cmul(pred[1], pred_cur_hop:repeatTensor(1,netout_dim)))
      test_select_att:add(torch.cmul(pred[3], pred_cur_hop:repeatTensor(1,cnnout_spat)))

      tab_pred[h] = pred[1]
      tab_do_pred[h] = do_pred
      tab_att[h] = pred[3]
      tab_mult_c[h] = pred[4]
      tab_mult_h[h] = pred[5]

      -- update did_pred
      test_did_pred:add(do_pred):clamp(0,1)
   end
   tab_pred[nHop+1] = test_uni_pred:div(nHop)
   tab_att[nHop+1] = test_uni_att:div(nHop) 
   tab_pred[nHop+2] = test_select_pred
   tab_att[nHop+2] = test_select_att 

   return tab_pred, tab_att
end

-- create log file for optimization
testLogger = optim.Logger(paths.concat(string.format('%s/%s',opt.save_dir,opt.log_dir), 'test.log'))

-- reorder train / val data
vqa_data.train_data:set_batch_order_option(opt.batch_order_option)
vqa_data.train_data:reorder()

------------------------------------------------------------------------
------------------------- Main Training Loop ---------------------------
------------------------------------------------------------------------
-- stats for drawing figures
local epoch_history = {}
local tab_trainacc_history = {}
local tab_trainacc_do_pred_history = {}
local dense_epoch_history = {}
local dense_trainloss = {}
local dense_trainloss_do_pred = {}
local tab_trainloss_history = {}
local tab_trainloss_do_pred_history = {}
local tab_testOEacc_history = {}
local tab_testMCacc_history = {}
local lr_history = {}
local mult_lr_history = {}
local tab_accum_trainloss = {}
local tab_accum_trainloss_do_pred = {}
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
local tab_avgloss = {}
local tab_avgloss_do_pred = {}
for h=1,nHop+2 do
   dense_trainloss[h] = {}
   dense_trainloss_do_pred[h] = {}
   tab_accum_trainloss[h] = 0
   tab_accum_trainloss_do_pred[h] = 0
   tab_trainacc_history[h] = {}
   tab_trainacc_do_pred_history[h] = {}
   tab_trainloss_history[h] = {}
   tab_trainloss_do_pred_history[h] = {}
   tab_testOEacc_history[h] = {}
   tab_testMCacc_history[h] = {}
end

iter_epoch = vqa_data.train_data.iter_per_epoch
total_iter = opt.max_epochs * iter_epoch 
for it = 1, total_iter do
   local epoch = it / iter_epoch
   iter_print = iter_print .. '-----------------------------------------------------------------------\n'
   local timer = torch.Timer()
   local tab_loss, tab_loss_do_pred = feval(it)
   if opt.optim == 'adam' then
      adam(embed_param, embed_grad, learningrate, alpha, beta, epsilon, embed_optimstate)
      adam(rnn_param, rnn_grad, learningrate, alpha, beta, epsilon, rnn_optimstate)
      adam(mult_param, mult_grad, multlearningrate, alpha, beta, epsilon, mult_optimstate)
   else
      error('bad option opt.optim')
   end
   local time = timer:time().real
   
   local tab_train_loss = tab_loss -- the loss is inside a list, pop it
   local tab_train_loss_do_pred = tab_loss_do_pred -- the loss is inside a list, pop it
   for h=1,nHop+2 do
      tab_avgloss[h] = tab_avgloss[h] or tab_train_loss[h]
      tab_avgloss[h] = tab_avgloss[h] * 0.9 + tab_train_loss[h] * 0.1
   end
   for h=1,nHop do
      tab_avgloss_do_pred[h] = tab_avgloss_do_pred[h] or tab_train_loss_do_pred[h]
      tab_avgloss_do_pred[h] = tab_avgloss_do_pred[h] * 0.9 + tab_train_loss_do_pred[h] * 0.1
   end
   if it % opt.denseloss_saveinterval == 1 then
      for h=1,nHop+2 do table.insert(dense_trainloss[h], tab_avgloss[h]) end
      for h=1,nHop do table.insert(dense_trainloss_do_pred[h], tab_avgloss_do_pred[h]) end
      table.insert(dense_epoch_history, epoch)
   end
   if opt.display == 'true' then
      local base_display_id = opt.display_id + 100
      if it % opt.denseloss_saveinterval == 1 then
         local line_dense_epoch = torch.Tensor(dense_epoch_history)
         local tab_label = {'epoch'}
         for h=1,nHop+2 do
            local line_dense_trainloss = torch.Tensor(dense_trainloss[h])
            line_dense_epoch = line_dense_epoch:cat(line_dense_trainloss, 2)
            table.insert(tab_label, string.format('train_%d', h))
         end
         disp.plot(line_dense_epoch,
                   {title='training loss',
                    labels=tab_label,
                    ylabel='loss', win=base_display_id})
         base_display_id = base_display_id + 1

         local line_dense_epoch = torch.Tensor(dense_epoch_history)
         local tab_label = {'epoch'}
         for h=1,nHop do
            local line_dense_trainloss_do_pred = torch.Tensor(dense_trainloss_do_pred[h])
            line_dense_epoch = line_dense_epoch:cat(line_dense_trainloss_do_pred, 2)
            table.insert(tab_label, string.format('train_%d', h))
         end
         disp.plot(line_dense_epoch,
                   {title='training loss (do pred)',
                    labels=tab_label,
                    ylabel='loss', win=base_display_id})
         base_display_id = base_display_id + 1
      end
   end
   for h=1,nHop+2 do tab_accum_trainloss[h] = tab_accum_trainloss[h] + tab_train_loss[h] end
   for h=1,nHop do 
      tab_accum_trainloss_do_pred[h] = tab_accum_trainloss_do_pred[h] + tab_train_loss_do_pred[h] 
   end
   
   if it % opt.print_iter == 0 then
      iter_print = iter_print .. string.format('%d (epoch %.3f), lr=%f, multlr=%f, time=%.2fs\n',
                           it, epoch, learningrate, multlearningrate, time)
      local loss_print = '[loss]:\n'
      for h=1,nHop+2 do
         loss_print = loss_print .. string.format('%02d=%6.8f, ', h, tab_train_loss[h])
         if h % 3 == 0 then loss_print = loss_print .. '\n' end
      end
      iter_print = iter_print .. (loss_print .. '\n')

      local loss_print_do_pred = '[loss (do pred)]: \n'
      for h=1,nHop do
         loss_print_do_pred = loss_print_do_pred .. 
                              string.format('%02d=%6.8f, ', h, tab_train_loss_do_pred[h])
         if h % 3 == 0 then loss_print_do_pred = loss_print_do_pred .. '\n' end
      end
      iter_print = iter_print .. (loss_print_do_pred .. '\n')

      iter_print = iter_print .. string.format('[norm] EP:%6.4f, RP:%6.4f, MP:%6.4f\n',
                           embed_param:norm(), rnn_param:norm(), mult_param:norm())
      iter_print = iter_print .. string.format('[grad] EG:%6.4f, RG:%6.4f, MG:%6.4f\n',
                           embed_grad:norm(), rnn_grad:norm(), mult_grad:norm())
   end
   if (it % iter_epoch == 0 and epoch % opt.test_interval == 0) or it == total_iter then
      -- inorder
      vqa_data.test_data:inorder()
      local test_iter = vqa_data.test_data.iter_per_epoch
      local tab_results_OE = {} -- OpenEnded
      local tab_results_MC = {} -- MultipleChoices
      local tab_testOE_acc = {}
      local tab_testMC_acc = {}      
      local tab_test_num_data = {}
      for h=1,nHop+2 do
         tab_results_OE[h] = {}
         tab_results_MC[h] = {}
         tab_testOE_acc[h] = 0
         tab_testMC_acc[h] = 0
         tab_test_num_data[h] = 0 
      end
      if opt.visatt == 'true' then
         for h=1,nHop+2 do
            os.execute(string.format('mkdir -p %s/%s/epoch_%03d/hop_%03d', 
                                      opt.save_dir, opt.figure_dir, epoch, h))
         end
      end
      print(string.format('start test'))
      for k = 1, test_iter do
         print(string.format('test -- [%d/%d]', k, test_iter))
         local feats, x, x_len, ans_mc, qids = vqa_data.test_data:next_batch_feat(opt.feat_dir, cnnout_dim, 
                                                                                  cnnout_w, cnnout_h)
         local tab_oe_pred, tab_att_pred = predict_result(feats, x, x_len)
        
         for h=1,nHop+2 do
            local oe_pred = tab_oe_pred[h]
            local att_pred = tab_att_pred[h]
            -- compute MultipleChoice Answer
            local mc_pred = oe_pred:clone()
            test_mc_mask:zero()
            for nb=1,ans_mc:size(1) do
               for nmc=1,ans_mc:size(2) do
                  if ans_mc[{nb,nmc}] ~= 0 then
                     test_mc_mask[{nb, ans_mc[{nb,nmc}]}] = 1
                  end
               end
            end
            mc_pred:cmul(test_mc_mask)
            local mcmax, mc_ans = torch.max(mc_pred, 2)
            mc_ans = torch.reshape(mc_ans, mc_ans:nElement())
 
            -- compute OpenEnded Answers
            local oemax, oe_ans = torch.max(oe_pred, 2)
            oe_ans = torch.reshape(oe_ans, oe_ans:nElement())
            if opt.visatt == 'true' then
               att_pred = att_pred:reshape(opt.test_batch_size, cnnout_w, cnnout_h)
            end
            -- Save Answers
            for bidx=1,qids:size(1) do
               local mc_result = {}
               local oe_result = {}
               mc_result.answer = vqa_data.answer_dict[mc_ans[bidx]]
               oe_result.answer = vqa_data.answer_dict[oe_ans[bidx]]
               mc_result.question_id = qids[bidx]
               oe_result.question_id = qids[bidx]
               table.insert(tab_results_MC[h], mc_result)
               table.insert(tab_results_OE[h], oe_result)

               if opt.split == 'val2014' then
                  -- fast validation
                  local gtans = val_gt_anno[('%d'):format(oe_result.question_id)]
                  assert(gtans ~= nil, 'fast validataion: this question id is not exist')
                  if gtans == oe_result.answer then
                     tab_testOE_acc[h] = tab_testOE_acc[h] + 1
                  end
                  if gtans == mc_result.answer then
                     tab_testMC_acc[h] = tab_testMC_acc[h] + 1
                  end
                  tab_test_num_data[h] = tab_test_num_data[h] + 1
               end
               if opt.visatt == 'true' then
                  local sattname = string.format('%d.png', qids[bidx])
                  local sattpath = paths.concat(string.format('%s/%s/epoch_%03d/hop_%03d',
                                   opt.save_dir,opt.figure_dir,epoch,h),sattname)
                  local att_result = att_pred[bidx]
                  image.save(sattpath, att_result)
               end
            end
         end 
         if k % opt.free_interval == 0 then collectgarbage() end
      end
      for h=1,nHop+2 do 
         if tab_train_num_data[h] ~= 0 then 
            tab_train_acc[h] = tab_train_acc[h] / tab_train_num_data[h] 
         end
         if iter_epoch ~= 0 then tab_accum_trainloss[h] = tab_accum_trainloss[h] / iter_epoch end
      end
      for h=1,nHop do
         if tab_train_do_pred_num_data[h] ~= 0 then 
            tab_train_do_pred_acc[h] = tab_train_do_pred_acc[h] / tab_train_do_pred_num_data[h] 
         end
         if iter_epoch ~= 0 then 
            tab_accum_trainloss_do_pred[h] = tab_accum_trainloss_do_pred[h] / iter_epoch 
         end
      end
      -- draw figures      
      table.insert(epoch_history, epoch)
      for h=1,nHop+2 do
         table.insert(tab_trainacc_history[h], tab_train_acc[h])
         table.insert(tab_trainloss_history[h], tab_accum_trainloss[h])
      end
      for h=1,nHop do
         table.insert(tab_trainacc_do_pred_history[h], tab_train_do_pred_acc[h])
         table.insert(tab_trainloss_do_pred_history[h], tab_accum_trainloss_do_pred[h])
      end
      table.insert(lr_history, learningrate)
      table.insert(mult_lr_history, multlearningrate)
      if opt.split == 'val2014' then
         for h=1,nHop+2 do
            if tab_test_num_data[h] ~= 0 then 
               tab_testOE_acc[h] = tab_testOE_acc[h] / tab_test_num_data[h] 
            end
            if tab_test_num_data[h] ~= 0 then 
               tab_testMC_acc[h] = tab_testMC_acc[h] / tab_test_num_data[h] 
            end
            table.insert(tab_testOEacc_history[h], tab_testOE_acc[h])
            table.insert(tab_testMCacc_history[h], tab_testMC_acc[h])
         end
      end

      local base_display_id = opt.display_id + 200

      -- epoch history (used globally)
      local line_epoch = torch.Tensor(epoch_history)

      -- accuracy curve
      if opt.split == 'val2014' then
         for h=1,nHop+2 do
            local line_trainacc = torch.Tensor(tab_trainacc_history[h])
            local line_testOEacc = torch.Tensor(tab_testOEacc_history[h])
            local line_testMCacc = torch.Tensor(tab_testMCacc_history[h])

            if epoch % opt.graph_interval == 0 or it == total_iter then
               local fname_accplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                                  string.format('accuracy_curve_hop%02d.png', h))
               gnuplot.pngfigure(fname_accplot)
               gnuplot.plot({'train', line_epoch, line_trainacc},
                            {'test OE', line_epoch, line_testOEacc},
                            {'test MC', line_epoch, line_testMCacc})
               gnuplot.xlabel('epoch')
               gnuplot.ylabel('accuracy')
               gnuplot.movelegend('right','bottom')
               gnuplot.title(string.format('train / test accuracy [hop %02d]', h))
               gnuplot.plotflush()
            end
            if opt.display == 'true' then
               disp.plot(line_epoch:cat(line_trainacc,2):cat(line_testOEacc,2),
                         {title=string.format('train / test accuracy [hop %02d]', h),
                          labels={'epoch', 'train', 'testOE'},
                          ylabel='accuracy', win=base_display_id})
               base_display_id = base_display_id + 1
            end
         end
      else
         for h=1,nHop+2 do
            local line_trainacc = torch.Tensor(tab_trainacc_history[h])
            if epoch % opt.graph_interval == 0 or it == total_iter then
               local fname_accplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                                  string.format('accuracy_curve_hop%02d.png', h))
               gnuplot.pngfigure(fname_accplot)
               gnuplot.plot({'train', line_epoch, line_trainacc})
               gnuplot.xlabel('epoch')
               gnuplot.ylabel('accuracy')
               gnuplot.movelegend('right','bottom')
               gnuplot.title(string.format('training accuracy [hop %02d]', h))
               gnuplot.plotflush()
            end
            if opt.display == 'true' then
               disp.plot(line_epoch:cat(line_trainacc,2),
                         {title=string.format('train accuracy [hop %02d]', h),
                          labels={'epoch', 'train'},
                          ylabel='accuracy', win=base_display_id})
               base_display_id = base_display_id + 1
            end
         end
      end
      -- do pred accuracy (training)
      for h=1,nHop do
         local line_trainacc_do_pred = torch.Tensor(tab_trainacc_do_pred_history[h])
         if epoch % opt.graph_interval == 0 or it == total_iter then
            local fname_accplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                               string.format('do_pred_accuracy_curve_hop%02d.png', h))
            gnuplot.pngfigure(fname_accplot)
            gnuplot.plot({'train', line_epoch, line_trainacc_do_pred})
            gnuplot.xlabel('epoch')
            gnuplot.ylabel('accuracy')
            gnuplot.movelegend('right','bottom')
            gnuplot.title(string.format('do_pred training accuracy [hop %02d]', h))
            gnuplot.plotflush()
         end
         if opt.display == 'true' then
            disp.plot(line_epoch:cat(line_trainacc_do_pred,2),
                      {title=string.format('do_pred train accuracy [hop %02d]', h),
                       labels={'epoch', 'train'},
                       ylabel='accuracy', win=base_display_id})
            base_display_id = base_display_id + 1
         end
      end
      for h=1,nHop+2 do
         -- loss curve
         local line_trainloss = torch.Tensor(tab_trainloss_history[h])
         if epoch % opt.graph_interval == 0 or it == total_iter then
            local fname_lossplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                                string.format('loss_curve_hop%02d.png', h))
            gnuplot.pngfigure(fname_lossplot)
            gnuplot.plot({'train', line_epoch, line_trainloss})
            gnuplot.xlabel('epoch')
            gnuplot.ylabel('loss')
            gnuplot.movelegend('right','top')
            gnuplot.title(string.format('training loss [hop%02d]', h))
            gnuplot.plotflush()
         end
      end
      for h=1,nHop do
         -- loss curve
         local line_trainloss = torch.Tensor(tab_trainloss_do_pred_history[h])
         if epoch % opt.graph_interval == 0 or it == total_iter then
            local fname_lossplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                                string.format('do_pred_loss_curve_hop%02d.png', h))
            gnuplot.pngfigure(fname_lossplot)
            gnuplot.plot({'train', line_epoch, line_trainloss})
            gnuplot.xlabel('epoch')
            gnuplot.ylabel('loss')
            gnuplot.movelegend('right','top')
            gnuplot.title(string.format('training do_pred loss [hop%02d]', h))
            gnuplot.plotflush()
         end
      end
      -- learning rate curve
      local line_lr_curve = torch.Tensor(lr_history)
      if epoch % opt.graph_interval == 0 or it == total_iter then
         local fname_lr = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                       'learning_rate.png')
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
         local fname_multlr = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 
                                           'mult_learning_rate.png')
         gnuplot.pngfigure(fname_multlr)
         gnuplot.plot({'lr', line_epoch, mult_line_lr_curve})
         gnuplot.xlabel('iter')
         gnuplot.ylabel('learning rate')
         gnuplot.movelegend('right','top')
         gnuplot.title('multimodal learning rate')
         gnuplot.plotflush()
      end
      if opt.split == 'val2014' then
         local tab_testlog = {}
               tab_testlog['epoch'] = epoch
         for h=1,nHop+2 do
            tab_testlog[string.format('trainacc_%02d', h)] = tab_train_acc[h] * 100
            tab_testlog[string.format('testOEacc_%02d', h)] = tab_testOE_acc[h] * 100
            --tab_testlog[string.format('testMCacc_%02d', h)] = tab_testMC_acc[h] * 100
         end
         testLogger:add(tab_testlog)
         print(string.format('iter: %d, epoch: %f',
                              it, epoch))
         print_acc = ''
         for h=1,nHop+2 do
            print_acc = print_acc .. string.format('trainacc[%02d]: %f, ', h, tab_train_acc[h]*100)
            print_acc = print_acc .. string.format('testOE[%02d]: %f, ', h, tab_testOE_acc[h]*100)
            print_acc = print_acc .. string.format('testMC[%02d]: %f, ', h, tab_testMC_acc[h]*100)
            print_acc = print_acc .. string.format('trainloss[%02d]: %f, ', h, tab_accum_trainloss[h])
            print_acc = print_acc .. '\n'
         end
         print(print_acc)

         print_acc = ''
         for h=1,nHop+2 do
            print_acc = print_acc .. 
               string.format('train_do_pred_acc[%02d]: %f, ', h, tab_train_do_pred_acc[h]*100)
            print_acc = print_acc .. 
               string.format('train_do_pred_loss[%02d]: %f, ', h, tab_accum_trainloss_do_pred[h])
            print_acc = print_acc .. '\n'
         end
         print(print_acc)   
      else
         local tab_testlog = {}
               tab_testlog['epoch'] = epoch
         for h=1,nHop+2 do
            tab_testlog[string.format('trainacc_%02d', h)] = tab_train_acc[h] * 100
         end
         testLogger:add(tab_testlog)
         print(string.format('iter: %d, epoch: %f',
                              it, epoch))
         print_acc = ''
         for h=1,nHop+2 do
            print_acc = print_acc .. string.format('trainacc[%02d]: %f, ', h, tab_train_acc[h]*100)
            print_acc = print_acc .. string.format('trainloss[%02d]: %f, ', h, tab_accum_trainloss[h])
            print_acc = print_acc .. '\n'
         end
         print(print_acc)
         print_acc = ''
         for h=1,nHop+2 do
            print_acc = print_acc .. 
               string.format('train_do_pred_acc[%02d]: %f, ', h, tab_train_do_pred_acc[h]*100)
            print_acc = print_acc .. 
               string.format('train_do_pred_loss[%02d]: %f, ', h, tab_accum_trainloss_do_pred[h])
            print_acc = print_acc .. '\n'
         end
         print(print_acc)   
      end
      for h=1,nHop+2 do
         local dir_result = string.format('%s/%s/hop_%02d',
                                           opt.save_dir, opt.results_dir, h)
         -- SAVE OPEN-ENDED RESULTS
         local fn_oe_result = string.format('vqa_%s_mscoco_%s_%s%02dhop-%.2f_results.json',
                                            'OpenEnded', opt.split, opt.alg_name, h, epoch)
         local save_oe_result = paths.concat(dir_result, fn_oe_result)
         local oe_json = cjson.encode(tab_results_OE[h])
         local wf = io.open(save_oe_result, 'w')
         wf:write(oe_json)
         wf:close()

         -- SAVE MULTIPLE-CHOICE RESULTS
         local fn_mc_result = string.format('vqa_%s_mscoco_%s_%s%02dhop-%.2f_results.json',
                                            'MultipleChoice', opt.split, opt.alg_name, h, epoch)
         local save_mc_result = paths.concat(dir_result, fn_mc_result)
         local mc_json = cjson.encode(tab_results_MC[h])
         local wf = io.open(save_mc_result, 'w')
         wf:write(mc_json)
         wf:close()    
      end
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

      for h=1,nHop+2 do
         tab_train_acc[h] = 0
         tab_train_num_data[h] = 0
         tab_train_do_pred_acc[h] = 0
         tab_train_do_pred_num_data[h] = 0
         tab_accum_trainloss[h] = 0
         tab_accum_trainloss_do_pred[h] = 0
      end
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











