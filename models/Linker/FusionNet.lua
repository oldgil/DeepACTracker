-- 20161030

 local FusionNet = {}
--local LNET = torch.class('tracking.LNET')
-- 
function FusionNet.fusionnet(opt)
    -- output maxm x (maxm + 1) * t * 2 * n 
    local x = nn.Identity()()   -- a table maxm x n
    local y = nn.Identity()()   -- (maxm )* n
    local f = nn.Identity()()   -- 1 x n

    local yt = nn.JoinTable(1)({y,f})
    local x1 = nn.Replicate(opt.maxm+1,2)(x) --maxm x (maxm + 1) * t * n 
    local y1 = nn.Replicate(opt.maxm,1)(yt) --maxm  x (maxm + 1)
    
    local x2 = nn.Reshape(opt.maxm*(opt.maxm+1),opt.feat_size)(x1)
    local y2 = nn.Reshape(opt.maxm*(opt.maxm+1),opt.feat_size)(y1)

    local g = nn.gModule({x, y,f}, {x2, y2})

    --g:getParameters():uniform(-0.08, 0.08)

    return g
end

return FusionNet
