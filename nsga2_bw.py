import os, random, numpy as np
def feasible_modes():
    m=[]
    if os.getenv('REDIS_URL',''): m.append('hot')
    if os.getenv('OSS_URI_PREFIX',''): m.append('cold')
    m.append('http'); return m
def estimate_objectives(inst, mode, req, deadline_ms):
    bw={'hot': float(os.getenv('HOT_BW_MBPS','800')), 'cold': float(os.getenv('COLD_BW_MBPS','200')), 'http': float(os.getenv('HTTP_BW_MBPS','100'))}[mode]
    rtt={'hot': float(os.getenv('HOT_RTT_MS','2')), 'cold': float(os.getenv('COLD_RTT_MS','15')), 'http': float(os.getenv('HTTP_RTT_MS','8'))}[mode]
    q=float(inst.get('dyn',{}).get('avg_q_ms',0.0)); grad_bytes=float(req.get('grad_bytes',1.0)); mbps=max(1e-3,bw)
    tx_ms=(grad_bytes*8.0)/(mbps*1e6)*1000.0; comm_ms=rtt+tx_ms+q; bw_time=comm_ms
    price=float(inst.get('meta',{}).get('price_cents_s', req.get('price_cents_s',0.0)))
    cost=price*(bw_time/1000.0)
    if mode=='cold': cost+=float(os.getenv('OSS_PUT_CENTS','0.002'))
    stall=max(0.0, bw_time - float(deadline_ms or 0.0))
    return np.array([bw_time, comm_ms, cost, stall], dtype=np.float32)
def dominates(a,b): return np.all(a<=b) and np.any(a<b)
def fast_nondominated(objs):
    S=[[] for _ in objs]; n=[0]*len(objs); rank=[0]*len(objs); fronts=[[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if p==q: continue
            if dominates(objs[p],objs[q]): S[p].append(q)
            elif dominates(objs[q],objs[p]): n[p]+=1
        if n[p]==0: rank[p]=0; fronts[0].append(p)
    i=0
    while fronts[i]:
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q: Q.append(q)
        i+=1; fronts.append(Q)
    fronts.pop(); return fronts, rank
def crowding_distance(front, objs):
    m=len(objs[0]); dist={i:0.0 for i in front}
    if len(front)<=2:
        for i in front: dist[i]=float('inf'); return dist
    for j in range(m):
        values=[(i, objs[i][j]) for i in front]; values.sort(key=lambda x:x[1])
        dist[values[0][0]]=dist[values[-1][0]]=float('inf')
        minv=values[0][1]; maxv=values[-1][1]; rng=max(1e-9, maxv-minv)
        for k in range(1,len(values)-1):
            dist[values[k][0]] += (values[k+1][1]-values[k-1][1])/rng
    return dist
def nsga2_select(inst_list, req, deadline_ms, pop_size=24, generations=8, seed=None):
    rnd=random.Random(seed or 1234); modes=feasible_modes()
    if not inst_list or not modes: return None
    def rand_ind(): return [rnd.randrange(0,len(inst_list)), rnd.randrange(0,len(modes))]
    def eval_ind(ind): return estimate_objectives(inst_list[ind[0]], modes[ind[1]], req, deadline_ms)
    pop=[rand_ind() for _ in range(pop_size)]
    for _ in range(generations):
        objs=[eval_ind(ind) for ind in pop]
        fronts,rank=fast_nondominated(objs); new=[]; f=0
        while len(new)<pop_size and f<len(fronts):
            front=fronts[f]; cd=crowding_distance(front, objs)
            fs=sorted(front, key=lambda i:(rank[i], -cd[i]))
            for idx in fs:
                if len(new)<pop_size: new.append(pop[idx])
            f+=1
        off=[]
        while len(off)<pop_size:
            a,b=rnd.randrange(0,len(new)), rnd.randrange(0,len(new)); pa=new[a][:]; pb=new[b][:]
            if rnd.random()<0.7: pa[1],pb[1]=pb[1],pa[1]
            if rnd.random()<0.2: pa[0]=rnd.randrange(0,len(inst_list))
            if rnd.random()<0.2: pa[1]=rnd.randrange(0,len(modes))
            if rnd.random()<0.2: pb[0]=rnd.randrange(0,len(inst_list))
            if rnd.random()<0.2: pb[1]=rnd.randrange(0,len(modes))
            off.extend([pa,pb])
        pop=off[:pop_size]
    objs=[eval_ind(ind) for ind in pop]; fronts,rank=fast_nondominated(objs); best=fronts[0] if fronts else list(range(len(pop)))
    w=np.array([1.0,0.5,0.2,3.0],dtype=np.float32)
    i_best=min([(i,float((objs[i]*w).sum())) for i in best], key=lambda x:x[1])[0]
    inst_idx, mode_idx = pop[i_best]
    return inst_list[inst_idx], modes[mode_idx]
