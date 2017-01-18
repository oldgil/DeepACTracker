-- 20161030

 local LNET = {}
--local LNET = torch.class('tracking.LNET')
-- 
function LNET.lnet(opt)

    local x = nn.Identity()()   -- a table t * maxm x (maxm+1)
    local y = nn.Identity()()   -- mask (m x 1)

    local x1 = nn.JoinTable(1)(x) -- a tensor t x m x (m+1)
    local sumt = nn.Sum(1)(x1)   -- m x (m+1)
    --local sum = torch.Tensor()
    --sum = sumt:clone()
    --local q, action = sum:max(2)  -- action m x 1
    local maxq = nn.Max(2)(sumt) -- m
    maxq = nn.View(1,-1)(maxq)
    local qmax = nn.MM()({maxq,y})  -- 1x1

    local g = nn.gModule({x, y}, {qmax, sumt})

    --g:getParameters():uniform(-0.08, 0.08)

    return g
end

return LNET
