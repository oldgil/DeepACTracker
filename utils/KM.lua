function dfs(N,parms,love,i)
    parms.vis_traj[i] = 1
    for j=1,N do
        if parms.vis_cand[j] == 0 then 
            local gap = parms.ex_traj[i] + parms.ex_cand[j] - love[i][j]
            if  gap == 0 then
                parms.vis_cand[j] = 1
                if parms.match[j] == -1 or dfs(N,parms,love,parms.match[j]) then
                    parms.match[j] = i 
                    return true
                end
            else
                parms.slack[j] = math.min(parms.slack[j],gap)
            end
        end
    end

    return false
end

function KM(love)
    -- parms NxN tensor
    local N = love:size(1)
    local parms = {}
    local INF = 10000000
    --parms.love = love
    parms.match = torch.Tensor(N):fill(-1)
    parms.ex_cand = torch.Tensor(N):fill(0)
    parms.ex_traj = love:max(2):squeeze()
    if type(parms.ex_traj) == 'number' then
        parms.match[1] = 1
        
        return love[1][1],parms.match
    end
    parms.vis_traj = torch.Tensor(N):fill(0)
    parms.vis_cand = torch.Tensor(N):fill(0)
    parms.slack    = torch.Tensor(N):fill(0)
    local steps = 1
    for i=1,N do
        parms.slack:fill(INF)
        while 1 do
            
            steps = steps + 1
            parms.vis_traj:fill(0)
            parms.vis_cand:fill(0)

            local bool = dfs(N,parms,love,i)
            if bool then break end

            local d = INF

            for j=1,N do
                if parms.vis_cand[j] == 0 and d > parms.slack[j] then 
                    d = parms.slack[j] 
                end    
            end
            for j=1,N do
                if parms.vis_traj[j] == 1 then
                    parms.ex_traj[j] = parms.ex_traj[j] - d
                end
                if parms.vis_cand[j] == 1 then
                    parms.ex_cand[j] = parms.ex_cand[j] + d
                else
                    parms.slack[j] = parms.slack[j] - d
                end
            end
        end
    
    end

    local res = 0
    for i = 1, N do
        res = res + love[parms.match[i]][i]
    end
    return res,parms.match
end

function testKM()
    local testlove = torch.Tensor(3,3):zero()
    testlove[1][1] = 3
    testlove[1][3] = 4
    testlove[2][1] = 2
    testlove[2][2] = 1
    testlove[2][3] = 3
    testlove[3][3] = 5
    local res,match = KM(testlove)

    print(string.format('match result is %f',res))
    for i=1,match:size(1) do
        print(string.format('match for %d is %d',i,match[i]))
    end

    testlove = torch.Tensor(5,5):zero()
    testlove[1][1] = 3;testlove[1][2] = 4;testlove[1][3] = 6;testlove[1][4] = 4;testlove[1][5] = 9;
    testlove[2][1] = 6;testlove[2][2] = 4;testlove[2][3] = 5;testlove[2][4] = 3;testlove[2][5] = 8;
    testlove[3][1] = 7;testlove[3][2] = 5;testlove[3][3] = 3;testlove[3][4] = 4;testlove[3][5] = 2;
    testlove[4][1] = 6;testlove[4][2] = 3;testlove[4][3] = 2;testlove[4][4] = 2;testlove[4][5] = 5;
    testlove[5][1] = 8;testlove[5][2] = 4;testlove[5][3] = 5;testlove[5][4] = 4;testlove[5][5] = 7;
    res,match = KM(testlove)

    print(string.format('match result is %f',res))
    for i=1,match:size(1) do
        print(string.format('match for %d is %d',i,match[i]))
    end

    testlove = torch.Tensor(5,5):zero()
    testlove[1][1] = 7;testlove[1][2] = 6;testlove[1][3] = 4;testlove[1][4] = 6;testlove[1][5] = 1;
    testlove[2][1] = 4;testlove[2][2] = 6;testlove[2][3] = 5;testlove[2][4] = 7;testlove[2][5] = 2;
    testlove[3][1] = 3;testlove[3][2] = 5;testlove[3][3] = 7;testlove[3][4] = 6;testlove[3][5] = 8;
    testlove[4][1] = 4;testlove[4][2] = 7;testlove[4][3] = 8;testlove[4][4] = 8;testlove[4][5] = 5;
    testlove[5][1] = 2;testlove[5][2] = 6;testlove[5][3] = 5;testlove[5][4] = 6;testlove[5][5] = 3;
    res,match = KM(testlove)

    print(string.format('match result is %f',res))
    for i=1,match:size(1) do
        print(string.format('match for %d is %d',i,match[i]))
    end
end