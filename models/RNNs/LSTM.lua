-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}
--local LSTM = torch.class('tracking.LSTM')
-- Creates one timestep of one LSTM
function LSTM.lstm(opt)
    local x1 = nn.Identity()()
    local x2 = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(opt.input_size, opt.rnn_size)(x1)
        local p2h            = nn.Linear(opt.input_size,opt.rnn_size,false)(x2)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h, p2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    local g = nn.gModule({x1, x2, prev_c, prev_h}, {next_c, next_h})

    g:getParameters():uniform(-0.08, 0.08)

    return g
end

return LSTM
