import argparse, csv, json, math, random, statistics, os, multiprocessing as mp
from collections import Counter
from itertools import product
import numpy as np

STATES=("wake","n2","rem","n3"); LOOPS=("thalamo_cortical","fronto_parietal","dmn","ct_cingulate")
FEATURES=("peak_alpha_hz","L_eff_m","v_eff_m_per_s","tau_eff_s","pl_local_aw","pl_global_aw","cfc","w_prop","s_struct")
SI={s:i for i,s in enumerate(STATES)}; LI={l:i for i,l in enumerate(LOOPS)}
STATE_INERTIA=np.array([.18,.08,.10,.28],float)
TRANSITION_PENALTY={("wake","n2"):.02,("n2","wake"):.015,("n2","n3"):.01,("n3","n2"):.03,("wake","rem"):.015,("rem","wake"):.02}; DEFAULT_TRANSITION_PENALTY=.015
TRANS=np.zeros((4,4),float)
for a in STATES:
    for b in STATES:
        if a!=b: TRANS[SI[a],SI[b]]=TRANSITION_PENALTY.get((a,b),DEFAULT_TRANSITION_PENALTY)
SHUFFLE=np.array([SI["n3"],SI["wake"],SI["n2"],SI["rem"]],int) # destination index for source wake,n2,rem,n3

# PREREGISTERED before results inspection.
GRID={"memory_scale":[0.,.5,1.,2.,4.,8.],"transition_scale":[0.,.5,1.,2.,4.],"rollout_steps":[4,8,16],"emission_width":[1.,2.,4.],"temperature":[.10,.25,.50],"loop_scale":[0.,1.,4.]}
COARSE_ALPHAS=[1.5,5.,8.,11.]; COARSE_CHIS=[.28,.44,.60,.76]
FULL_ALPHAS=[.75+.5*i for i in range(27)]; FULL_CHIS=[.22+.04*i for i in range(15)]
SEEDS=[11,23,42,77,101]
CONDITIONS=("full","memory_off","previous_state_shuffled","transition_penalties_zero","loop_state_coupling_off","recurrence_fully_off","legacy_target_positive_control")
PASS={"min_state_occupancy":.05,"max_state_occupancy":.70,"min_final_entropy_bits":1.20,"min_history_mi_bits":.10,"min_mi_gain_vs_recurrence_off":.08,"min_mi_gain_vs_shuffled":.05,"min_mean_pairwise_js_bits":.03,"max_same_state_fraction":.90,"min_away_transition_fraction":.10,"min_seed_pass_fraction":.80,"min_perturb_pass_fraction":.70,"min_robust_component_size":3}

def clamp(x,a,b): return max(a,min(b,x))
def jitter(r,x,f): return x*r.uniform(1-f,1+f)
def synthetic_rows(seed=42,n=120):
    r=random.Random(seed); out=[]
    for i in range(n):
        for s in STATES:
            d={"state":s}
            if s=="wake": d.update(candidate_loop=r.choice(["thalamo_cortical","fronto_parietal","ct_cingulate"]),L_eff_m=jitter(r,.030,.10),v_eff_m_per_s=jitter(r,8,.10),tau_eff_s=jitter(r,.008,.10),pl_local_aw=clamp(jitter(r,.74,.12),.05,.99),pl_global_aw=clamp(jitter(r,.83,.12),.05,.99),cfc=clamp(jitter(r,.68,.12),.05,.99),w_prop=clamp(jitter(r,.84,.10),.05,.99),s_struct=clamp(jitter(r,.86,.10),.05,.99),peak_alpha_hz=clamp(jitter(r,9.8,.12),8,12))
            elif s=="n2": d.update(candidate_loop=r.choice(["dmn","fronto_parietal","thalamo_cortical"]),L_eff_m=jitter(r,.040,.12),v_eff_m_per_s=jitter(r,6.1,.12),tau_eff_s=jitter(r,.012,.12),pl_local_aw=clamp(jitter(r,.58,.15),.05,.95),pl_global_aw=clamp(jitter(r,.70,.15),.05,.95),cfc=clamp(jitter(r,.52,.15),.05,.90),w_prop=clamp(jitter(r,.66,.14),.05,.95),s_struct=clamp(jitter(r,.75,.12),.05,.95),peak_alpha_hz=clamp(jitter(r,7.2,.18),5.5,9))
            elif s=="rem": d.update(candidate_loop=r.choice(["dmn","fronto_parietal","thalamo_cortical"]),L_eff_m=jitter(r,.034,.10),v_eff_m_per_s=jitter(r,6.9,.10),tau_eff_s=jitter(r,.010,.12),pl_local_aw=clamp(jitter(r,.64,.14),.05,.95),pl_global_aw=clamp(jitter(r,.76,.12),.05,.95),cfc=clamp(jitter(r,.60,.15),.05,.95),w_prop=clamp(jitter(r,.74,.12),.05,.95),s_struct=clamp(jitter(r,.80,.10),.05,.95),peak_alpha_hz=clamp(jitter(r,8.9,.15),6,10))
            else: d.update(candidate_loop=r.choice(["dmn","ct_cingulate"]),L_eff_m=jitter(r,.050,.18),v_eff_m_per_s=jitter(r,4,.22),tau_eff_s=jitter(r,.020,.22),pl_local_aw=clamp(jitter(r,.42,.18),.05,.95),pl_global_aw=clamp(jitter(r,.30,.20),.05,.85),cfc=clamp(jitter(r,.25,.25),.01,.80),w_prop=clamp(jitter(r,.40,.18),.05,.85),s_struct=clamp(jitter(r,.55,.15),.05,.95),peak_alpha_hz=clamp(jitter(r,2,.45),.5,4))
            out.append(d)
    return out

def build_model(rows):
    x=np.array([[float(r[k]) for k in FEATURES] for r in rows]); labels=np.array([SI[r["state"]] for r in rows])
    refs=np.stack([x[labels==i].mean(0) for i in range(4)]); glob=x.mean(0); sd=x.std(0); sd[sd==0]=1
    counts=np.ones((4,4),float)
    for r in rows: counts[SI[r["state"]],LI[r["candidate_loop"]]]+=1
    priors=counts/counts.sum(1,keepdims=True)
    best=np.empty((4,4),int)
    for s in range(4):
        for prev in range(4):
            scores=np.log(np.maximum(priors[s],1e-12)); scores[prev]+=0.05; best[s,prev]=int(np.argmax(scores))
    default_loop=np.argmax(priors,axis=1)
    return {"refs":refs,"global":glob,"sd":sd,"priors":priors,"best_loop":best,"default_loop":default_loop}

def make_target_vector(base,alpha,chi):
    o=np.array(base,float,copy=True); o[0]=alpha; o[1]=chi*o[2]*o[3]; return o

def make_target(base,alpha,chi):
    # Public helper used by tests; returns a deterministic tuple.
    return tuple(make_target_vector(base,alpha,chi))
def condition_flags(c):
    return {"full":(1,1,1,0,0),"memory_off":(0,1,1,0,0),"previous_state_shuffled":(1,1,1,1,0),"transition_penalties_zero":(1,0,1,0,0),"loop_state_coupling_off":(1,1,0,0,0),"recurrence_fully_off":(0,0,0,0,0),"legacy_target_positive_control":(1,1,1,0,1)}[c]
def rollout_batch(model,cfg,condition,alphas,chis,record=False):
    mem,trs,loop,shuf,legacy=condition_flags(condition); refs=model["refs"]; glob=model["global"]; sd=model["sd"]; pri=model["priors"]
    combos=[(s,a,c) for s in range(4) for a in alphas for c in chis]; n=len(combos)
    starts=np.array([z[0] for z in combos],int); startx=refs[starts].copy(); targets=np.empty_like(startx)
    for i,(s,a,c) in enumerate(combos): targets[i]=make_target_vector(refs[s] if legacy else glob,a,c)
    probs=np.eye(4)[starts]; prev=starts.copy(); prev_loop=model["default_loop"][starts].copy(); tc=np.zeros((4,4),int)
    for k in range(cfg["rollout_steps"]):
        t=(k+1)/cfg["rollout_steps"]; probe=startx*(1-t)+targets*t
        d2=np.sum(((probe[:,None,:]-refs[None,:,:])/sd[None,None,:])**2,axis=2)
        scores=np.exp(-.5*d2/(cfg["emission_width"]**2))
        hs=prev; hp=probs; hl=prev_loop
        if shuf:
            hs=SHUFFLE[prev]; hp=np.zeros_like(probs)
            for src,dst in enumerate(SHUFFLE): hp[:,dst]=probs[:,src]
        if mem: scores += cfg["memory_scale"]*hp*STATE_INERTIA[None,:]
        if trs: scores -= cfg["transition_scale"]*TRANS[hs]
        if loop:
            compat=pri[:,hl].T-.25 # n x state
            scores += cfg["loop_scale"]*.10*compat
        temp=max(cfg["temperature"],1e-9); z=(scores-scores.max(1,keepdims=True))/temp; e=np.exp(z); probs=e/e.sum(1,keepdims=True)
        nxt=np.argmax(probs,axis=1)
        if record: np.add.at(tc,(prev,nxt),1)
        prev_loop=model["best_loop"][nxt,prev_loop]; prev=nxt
    return starts,prev,tc

def entropy_counts(c,n): return -sum((x/n)*math.log2(x/n) for x in c if x)
def mi_from_arrays(a,b):
    n=len(a); ca=np.bincount(a,minlength=4); cb=np.bincount(b,minlength=4); joint=np.zeros((4,4),int);np.add.at(joint,(a,b),1);mi=0.
    for i in range(4):
        for j in range(4):
            if joint[i,j]: p=joint[i,j]/n;mi+=p*math.log2(p/((ca[i]/n)*(cb[j]/n)))
    return mi
def js(p,q):
    m=(p+q)/2;ans=0.
    for a,b in ((p,m),(q,m)):
        nz=a>0;ans+=.5*np.sum(a[nz]*np.log2(a[nz]/b[nz]))
    return float(ans)
def evaluate(model,cfg,condition,alphas=COARSE_ALPHAS,chis=COARSE_CHIS,details=False):
    start,final,tc=rollout_batch(model,cfg,condition,alphas,chis,details);n=len(final);fc=np.bincount(final,minlength=4);same=float(np.mean(start==final))
    dist=[]
    for s in range(4): dist.append(np.bincount(final[start==s],minlength=4)/np.sum(start==s))
    jss=[js(dist[i],dist[j]) for i in range(4) for j in range(i+1,4)]
    out={"history_mi_bits":mi_from_arrays(start,final),"final_entropy_bits":entropy_counts(fc,n),"same_state_fraction":same,"away_transition_fraction":1-same,"mean_pairwise_js_bits":statistics.mean(jss),"max_final_occupancy":float(fc.max()/n),"min_final_occupancy":float(fc.min()/n)}
    for i,s in enumerate(STATES): out[f"{s}_final_fraction"]=float(fc[i]/n)
    if details:
        out["transition_matrix"]={STATES[i]:{STATES[j]:int(tc[i,j]) for j in range(4)} for i in range(4)}
        out["endpoint_distributions"]={STATES[i]:{STATES[j]:float(dist[i][j]) for j in range(4)} for i in range(4)}
        for i,s in enumerate(STATES): out[f"self_transition_{s}"]=float(tc[i,i]/tc[i].sum()) if tc[i].sum() else 0.
    return out
def basic_pass(f,o,s):
    ck={"occupancy_min":f["min_final_occupancy"]>=PASS["min_state_occupancy"],"occupancy_max":f["max_final_occupancy"]<=PASS["max_state_occupancy"],"entropy":f["final_entropy_bits"]>=PASS["min_final_entropy_bits"],"history_mi":f["history_mi_bits"]>=PASS["min_history_mi_bits"],"gain_vs_off":f["history_mi_bits"]-o["history_mi_bits"]>=PASS["min_mi_gain_vs_recurrence_off"],"gain_vs_shuffle":f["history_mi_bits"]-s["history_mi_bits"]>=PASS["min_mi_gain_vs_shuffled"],"pairwise_js":f["mean_pairwise_js_bits"]>=PASS["min_mean_pairwise_js_bits"],"not_frozen":f["same_state_fraction"]<=PASS["max_same_state_fraction"],"transitions_exist":f["away_transition_fraction"]>=PASS["min_away_transition_fraction"]}
    return all(ck.values()),ck
def config_id(c): return "m{memory_scale:g}_t{transition_scale:g}_s{rollout_steps}_w{emission_width:g}_T{temperature:g}_l{loop_scale:g}".format(**c)
def iter_grid():
    ks=list(GRID)
    for v in product(*(GRID[k] for k in ks)): yield dict(zip(ks,v))
def flatten(prefix,m,row):
    for k,v in m.items():
        if isinstance(v,(int,float)): row[(prefix+"_" if prefix else "")+k]=round(v,9)
def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def neighbors(c):
    o=[]
    for k in ("memory_scale","transition_scale","emission_width","temperature","loop_scale"):
        if c[k]==0:continue
        for f in (-.10,-.05,.05,.10):n=dict(c);n[k]=c[k]*(1+f);o.append((f"{k}_{f:+.0%}",n))
    vals=GRID["rollout_steps"];i=vals.index(c["rollout_steps"])
    if i:o.append(("rollout_steps_lower",{**c,"rollout_steps":vals[i-1]}))
    if i<len(vals)-1:o.append(("rollout_steps_upper",{**c,"rollout_steps":vals[i+1]}))
    return o
def adjacent(a,b):
    d=0
    for k,v in GRID.items():
        ia,ib=v.index(a[k]),v.index(b[k])
        if ia!=ib:
            if abs(ia-ib)!=1:return False
            d+=1
    return d==1
def components(configs):
    ids={config_id(c):c for c in configs};u=set(ids);out=[]
    while u:
        root=u.pop();q=[root];comp=[root]
        while q:
            x=q.pop();hits=[y for y in list(u) if adjacent(ids[x],ids[y])]
            for y in hits:u.remove(y);q.append(y);comp.append(y)
        out.append(comp)
    return out


def perturb_validate_worker(cfg):
    model=build_model(synthetic_rows(42)); rows=[]
    for label,ncfg in neighbors(cfg):
        f=evaluate(model,ncfg,"full"); o=evaluate(model,ncfg,"recurrence_fully_off"); sh=evaluate(model,ncfg,"previous_state_shuffled")
        ok,_=basic_pass(f,o,sh)
        rows.append({"config_id":config_id(cfg),"perturbation":label,"pass":int(ok),"history_mi_bits":f["history_mi_bits"],"off_mi_bits":o["history_mi_bits"],"shuffled_mi_bits":sh["history_mi_bits"]})
    frac=sum(r["pass"] for r in rows)/len(rows) if rows else 0.0
    return config_id(cfg),frac,rows

def seed_validate_worker(cfg):
    rows=[]; passes=0
    for seed in SEEDS:
        model=build_model(synthetic_rows(seed))
        full=evaluate(model,cfg,"full",FULL_ALPHAS,FULL_CHIS)
        off=evaluate(model,cfg,"recurrence_fully_off",FULL_ALPHAS,FULL_CHIS)
        shuf=evaluate(model,cfg,"previous_state_shuffled",FULL_ALPHAS,FULL_CHIS)
        ok,_=basic_pass(full,off,shuf); passes+=int(ok)
        rows.append({"config_id":config_id(cfg),"seed":seed,"pass":int(ok),"history_mi_bits":full["history_mi_bits"],"off_mi_bits":off["history_mi_bits"],"shuffled_mi_bits":shuf["history_mi_bits"],"same_state_fraction":full["same_state_fraction"],"min_occupancy":full["min_final_occupancy"],"max_occupancy":full["max_final_occupancy"],"entropy_bits":full["final_entropy_bits"],"pairwise_js_bits":full["mean_pairwise_js_bits"]})
    return config_id(cfg),passes/len(SEEDS),rows

def nearest_to_baseline(configs):
    baseline={"memory_scale":1.0,"transition_scale":1.0,"rollout_steps":8,"emission_width":2.0,"temperature":0.25,"loop_scale":1.0}
    def d(c): return sum((GRID[k].index(c[k])-GRID[k].index(baseline[k]))**2 for k in GRID)
    return min(configs,key=lambda c:(d(c),c["memory_scale"],c["loop_scale"]))
