
local LSTM = {}

local function lstm(input_size, rnn_size, x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(input_size, 4*rnn_size)(x)
  local h2h = nn.Linear(rnn_size, 4*rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})

  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)

  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

function LSTM.create(input_size, rnn_size, num_layers, dropout)
  dropout = dropout or 0
  init_weight = init_weight or 0
  local in_x             = nn.Identity()()
  local in_prev_c        = nn.Identity()()
  local in_prev_h        = nn.Identity()()

  -- input processing
  local i                = {[0] = in_x}
  local i_sz             = {[0] = input_size}
  local tab_prev_c       = {} 
  local tab_prev_h       = {}
  for L = 1, num_layers do
    table.insert(tab_prev_c, nn.Narrow(2, (L-1)*rnn_size+1, rnn_size)(in_prev_c))
    table.insert(tab_prev_h, nn.Narrow(2, (L-1)*rnn_size+1, rnn_size)(in_prev_h))
  end
  -- configure LSTM
  local tab_next_c           = {}
  local tab_next_h           = {}
  for layer_idx = 1, num_layers do
    local prev_c         = tab_prev_c[layer_idx]
    local prev_h         = tab_prev_h[layer_idx]
    local curr_x        = nn.Dropout(dropout)(i[layer_idx - 1])
    local curr_x_sz     = i_sz[layer_idx-1]
    local next_c, next_h = lstm(curr_x_sz, rnn_size, curr_x, prev_c, prev_h)
    table.insert(tab_next_c, next_c)
    table.insert(tab_next_h, next_h)
    i[layer_idx] = next_h
    i_sz[layer_idx] = rnn_size
  end
  -- set output
  local next_c
  local next_h
  if num_layers > 1 then
     next_c    = nn.JoinTable(1,1)(tab_next_c)
     next_h    = nn.JoinTable(1,1)(tab_next_h)
  else
     next_c    = nn.Identity()(tab_next_c)
     next_h    = nn.Identity()(tab_next_h)
  end
  local module = nn.gModule({in_x, in_prev_c, in_prev_h},
                            {next_c, next_h})

  return module
end

return LSTM
