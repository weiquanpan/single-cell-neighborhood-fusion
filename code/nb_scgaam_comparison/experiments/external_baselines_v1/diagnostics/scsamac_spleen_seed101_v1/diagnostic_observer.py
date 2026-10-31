"""Read-only observations of tensors/RNG; hooks return None, never alter gradients."""
from pathlib import Path
import json, time, random, math, hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
ACTIVE=False
STEP=0
FIRST=None
CURRENT=None
PRE=None
LAST_BEFORE=None
LAST_AFTER=None
LAST_RECORD=None
FORWARD={}
GRAD_VALUES={}
COUNTS={}
FIRST_STEP=None
TRIGGER_SAVED=False
START=time.perf_counter()
LOG=None
HISTORICAL={}
EPOCH_MATCH=[]

def number(v):
    v=float(v)
    return v if math.isfinite(v) else str(v)

def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def stats(t):
    a=t.detach().cpu().numpy()
    finite=np.isfinite(a); f=a[finite]
    return {'shape':list(a.shape),'dtype':str(a.dtype),'nan':int(np.isnan(a).sum()),'posinf':int(np.isposinf(a).sum()),'neginf':int(np.isneginf(a).sum()),'finite':bool(finite.all()),'min':float(f.min()) if f.size else None,'max':float(f.max()) if f.size else None,'max_abs':float(np.abs(f).max()) if f.size else None,'zero':int((a==0).sum())}

def clone(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:clone(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clone(v) for v in obj]
    if isinstance(obj,tuple):return tuple(clone(v) for v in obj)
    return obj

def rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'torch_cuda':torch.cuda.get_rng_state_all()}

def snapshot(model,optimizer,batch=None):
    return {'model':clone(model.state_dict()),'optimizer':clone(optimizer.state_dict()) if optimizer else None,'rng':rng(),'training':model.training,'step':STEP,'batch':batch,'scope':'Exact batch replay state; DataLoader iterator is not serialized'}

def event(stage,name,detail):
    global FIRST,FIRST_STEP
    item={'stage':stage,'name':name,'stats':detail,'step':STEP,'epoch':CURRENT['epoch'],'batch':CURRENT['batch']}
    CURRENT['nonfinite_events'].append(item)
    if FIRST is None:
        FIRST=item;FIRST_STEP=STEP
        save_json(ROOT/'first_nonfinite.json',FIRST)
        print('FIRST_NONFINITE '+json.dumps(FIRST),flush=True)

def detailed():return CURRENT is not None and (FIRST_STEP is None or STEP<=FIRST_STEP+2)

def observe(name,tensor,gradient=True):
    if not ACTIVE or not detailed():return
    COUNTS[name]=COUNTS.get(name,0)+1
    key=name if COUNTS[name]==1 else name+'#'+str(COUNTS[name])
    detail=stats(tensor);CURRENT['forward'][key]=detail;FORWARD[key]=tensor.detach()
    if not detail['finite']:event('forward',key,detail)
    if gradient and tensor.requires_grad:
        def hook(g):
            ds=stats(g)
            CURRENT['backward_order'].append({'tensor':key,**ds})
            if g.numel()<=4096 or not ds['finite'] or FIRST_STEP==STEP:
                GRAD_VALUES[key]=g.detach().cpu().clone()
            if not ds['finite']:event('backward',key,ds)
            return None
        tensor.register_hook(hook)

def module_hook(name):
    def hook(module,args,output):
        if isinstance(output,torch.Tensor):observe('module.'+name,output)
    return hook

def begin_batch(epoch,batch_idx,indices,x,raw,sf,model,optimizer):
    global ACTIVE,STEP,CURRENT,PRE,FORWARD,GRAD_VALUES,COUNTS
    ACTIVE=True;STEP+=1;FORWARD={};GRAD_VALUES={};COUNTS={}
    CURRENT={'step':STEP,'epoch':epoch+1,'batch':batch_idx+1,'n':len(indices),'cell_positions':indices.tolist(),'forward':{},'backward_order':[],'nonfinite_events':[]}
    PRE=None
    if FIRST_STEP is None or STEP<=FIRST_STEP+2:
        batch={'indices':indices.clone(),'x':x.clone(),'raw':raw.clone(),'size_factor':sf.clone()}
        PRE=snapshot(model,optimizer,batch)

def losses(nb,contrast,total):
    CURRENT['losses']={'nb':number(nb.detach()),'contrast':number(contrast.detach()),'weighted_contrast':number(contrast.detach()*.01),'total':number(total.detach())}
    observe('total_loss',total)

def before_step(model,optimizer):
    if detailed():
        CURRENT['parameter_gradients']={}
        for name,p in model.named_parameters():
            if p.grad is not None:
                detail=stats(p.grad);CURRENT['parameter_gradients'][name]=detail
                if not detail['finite']:event('parameter_gradient',name,detail)
    if FIRST_STEP==STEP:
        (ROOT/'checkpoints').mkdir(exist_ok=True)
        torch.save(PRE,ROOT/'checkpoints/first_nonfinite_before.pt')
        torch.save({'gradients':{n:clone(p.grad) for n,p in model.named_parameters()},'forward':clone(FORWARD),'intermediate_gradients':clone(GRAD_VALUES),'record':CURRENT},ROOT/'checkpoints/first_nonfinite_tensors.pt')
        if LAST_BEFORE is not None:
            torch.save(LAST_BEFORE,ROOT/'checkpoints/last_finite_before.pt')
            torch.save(LAST_AFTER,ROOT/'checkpoints/last_finite_after.pt')
            save_json(ROOT/'last_finite_batch.json',LAST_RECORD)

def after_step(model,optimizer):
    global LAST_BEFORE,LAST_AFTER,LAST_RECORD,PRE,TRIGGER_SAVED,ACTIVE
    post=None
    if detailed():
        post=snapshot(model,optimizer)
        CURRENT['model_after']={n:stats(v) for n,v in model.state_dict().items()}
        for name,detail in CURRENT['model_after'].items():
            if not detail['finite']:event('model_after_step',name,detail)
        if FIRST_STEP is None:
            LAST_BEFORE=PRE;LAST_AFTER=post;LAST_RECORD=CURRENT
        elif STEP==FIRST_STEP:
            if not (ROOT/'checkpoints/first_nonfinite_before.pt').exists():
                (ROOT/'checkpoints').mkdir(exist_ok=True)
                torch.save(PRE,ROOT/'checkpoints/first_nonfinite_before.pt')
            torch.save(post,ROOT/'checkpoints/first_nonfinite_after.pt')
            save_json(ROOT/'first_nonfinite_batch.json',CURRENT)
            TRIGGER_SAVED=True
        elif STEP<=FIRST_STEP+2:
            torch.save(PRE,ROOT/f'checkpoints/after_trigger_{STEP-FIRST_STEP}_before.pt')
            torch.save(post,ROOT/f'checkpoints/after_trigger_{STEP-FIRST_STEP}_after.pt')
            save_json(ROOT/f'after_trigger_{STEP-FIRST_STEP}_batch.json',CURRENT)
    # Keep batch losses and exact sampled indices for all 300 epochs.
    LOG.write(json.dumps(CURRENT,allow_nan=False)+'\n');LOG.flush()
    PRE=None;ACTIVE=False
    if FIRST_STEP is not None:
        LAST_BEFORE=None;LAST_AFTER=None;LAST_RECORD=None

def epoch_end(epoch,total):
    observed='%.8f'%total;expected=HISTORICAL.get(epoch+1)
    match=observed==expected
    EPOCH_MATCH.append({'epoch':epoch+1,'observed':observed,'historical':expected,'exact_printed_match':match})
    save_json(ROOT/'state.json',{'status':'RUNNING','epoch':epoch+1,'steps':STEP,'elapsed_seconds':time.perf_counter()-START,'first_nonfinite':FIRST,'matched_epochs':sum(e['exact_printed_match'] for e in EPOCH_MATCH),'observed_epochs':len(EPOCH_MATCH)})

def initialize(historical):
    global LOG,HISTORICAL
    LOG=(ROOT/'batches.jsonl').open('x',encoding='utf8')
    HISTORICAL=historical

def finish():
    if LOG:LOG.close()
    save_json(ROOT/'epoch_trace_comparison.json',EPOCH_MATCH)
