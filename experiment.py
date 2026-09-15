from __future__ import annotations
import json, math, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parent
AI_MEM = ROOT / 'ai-memory'
sys.path.insert(0, str(AI_MEM))
from dimensional_memory_final import DimensionalMemoryFinal

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(8, torch.get_num_threads()))

# ---------- Vocabulary ----------
SPECIAL = ['<pad>', '<bos>', '<eos>', '<ans>', '.', ';', '?']
WORDS = ['fatos', 'pergunta', 'qual', 'valor', 'de', 'relacao', 'projeto', 'usa', 'linguagem', 'resposta', 'entidade']
SUBJECTS = [f'E{i}' for i in range(64)]
RELATIONS = [f'R{i}' for i in range(8)]
VALUES = [f'V{i}' for i in range(96)]
PROJECTS = [f'P{i}' for i in range(64)]
LANGS = ['Python', 'Rust', 'Go', 'Java', 'CSharp', 'Kotlin', 'Swift', 'Ruby']
DBS = ['Postgres', 'SQLite', 'MySQL', 'Redis']
VOCAB_LIST = SPECIAL + WORDS + SUBJECTS + RELATIONS + VALUES + PROJECTS + LANGS + DBS
# preserve order and uniqueness
VOCAB_LIST = list(dict.fromkeys(VOCAB_LIST))
STOI = {t:i for i,t in enumerate(VOCAB_LIST)}
ITOS = {i:t for t,i in STOI.items()}
PAD, BOS, EOS, ANS = [STOI[x] for x in ['<pad>','<bos>','<eos>','<ans>']]

def ids(tokens): return [STOI[t] for t in tokens]

# ---------- AI-Memory external project memory ----------
PROJECT_META = {}
records = []
for i,p in enumerate(PROJECTS):
    lang = LANGS[(i * 5 + 3) % len(LANGS)]
    db = DBS[(i * 3 + 1) % len(DBS)]
    PROJECT_META[p] = (lang, db)
    records.append({'id': p, 'text': f'projeto {p} usa linguagem {lang} banco {db}', 'recurrence': 2 + (i % 3)})

MEM_DIR = ROOT / 'runtime'
MEM_DIR.mkdir(exist_ok=True)
PROJECT_MEMORY = DimensionalMemoryFinal(MEM_DIR / 'project_memory.bin').fit(records)

# Cache actual repository retrieval output as token IDs. This is the second memory.
PROJECT_CONTEXT_IDS: Dict[str, List[int]] = {}
PROJECT_RETRIEVAL_OK = 0
for p in PROJECTS:
    got = PROJECT_MEMORY.search(f'qual linguagem usa projeto {p}', k=1)
    if got and got[0]['id'] == p:
        PROJECT_RETRIEVAL_OK += 1
    text = got[0]['text'] if got else ''
    toks = text.split()
    PROJECT_CONTEXT_IDS[p] = [STOI[t] for t in toks if t in STOI]

# ---------- Synthetic samples ----------
@dataclass
class Sample:
    kind: str
    facts: List[Tuple[str,str,str,int]]  # subject, relation, value, global_fact_id
    query_tokens: List[str]
    target_tokens: List[str]
    q_subject: str
    q_relation: str
    project_context: List[int]

GLOBAL_FACT_COUNTER = 1

def make_rel_sample(nfacts:int, rng:random.Random) -> Sample:
    global GLOBAL_FACT_COUNTER
    # Avoid duplicate (subject, relation), so the symbolic fact identity is unambiguous.
    pairs = rng.sample([(s,r) for s in SUBJECTS for r in RELATIONS], nfacts)
    facts=[]
    for s,r in pairs:
        v=rng.choice(VALUES)
        facts.append((s,r,v,GLOBAL_FACT_COUNTER)); GLOBAL_FACT_COUNTER += 1
    target_idx=rng.randrange(nfacts)
    s,r,v,_=facts[target_idx]
    rng.shuffle(facts)
    q=['pergunta','qual','valor','de',s,'relacao',r,'?']
    target=['resposta',v,'.','<eos>']
    return Sample('rel',facts,q,target,s,r,[])

def make_project_sample(nfacts:int, rng:random.Random, project_pool:List[str]) -> Sample:
    # internal relational memory remains present as distractor; the answer comes from AI-Memory.
    global GLOBAL_FACT_COUNTER
    pairs = rng.sample([(s,r) for s in SUBJECTS for r in RELATIONS], nfacts)
    facts=[]
    for s,r in pairs:
        v=rng.choice(VALUES)
        facts.append((s,r,v,GLOBAL_FACT_COUNTER)); GLOBAL_FACT_COUNTER += 1
    p=rng.choice(project_pool); lang,_=PROJECT_META[p]
    q=['pergunta','qual','linguagem','usa','projeto',p,'?']
    target=['resposta',lang,'.','<eos>']
    return Sample('project',facts,q,target,SUBJECTS[0],RELATIONS[0],PROJECT_CONTEXT_IDS[p])

def make_dataset(n:int, fact_range:Tuple[int,int], seed:int, project_pool:List[str], project_ratio=.25):
    rng=random.Random(seed); out=[]
    lo,hi=fact_range
    for _ in range(n):
        nf=rng.randint(lo,hi)
        if rng.random() < project_ratio:
            out.append(make_project_sample(nf,rng,project_pool))
        else:
            out.append(make_rel_sample(nf,rng))
    return out

TRAIN_PROJECTS = PROJECTS[:48]
TEST_PROJECTS = PROJECTS[48:]
train = make_dataset(1400,(6,20),11,TRAIN_PROJECTS,.25)
valid = make_dataset(260,(6,20),12,TRAIN_PROJECTS,.25)
test_sets = {
    'in_dist_12': make_dataset(320,(12,12),21,TEST_PROJECTS,.25),
    'scale_32': make_dataset(320,(32,32),22,TEST_PROJECTS,.25),
    'scale_64': make_dataset(320,(64,64),23,TEST_PROJECTS,.25),
    'scale_128': make_dataset(320,(128,128),24,TEST_PROJECTS,.25),
}

# ---------- Batching ----------
def tensorize_batch(samples:List[Sample], baseline:bool=False):
    B=len(samples); max_f=max(len(s.facts) for s in samples)
    fact_sub=torch.full((B,max_f), PAD, dtype=torch.long)
    fact_rel=torch.full((B,max_f), PAD, dtype=torch.long)
    fact_val=torch.full((B,max_f), PAD, dtype=torch.long)
    fact_mask=torch.zeros((B,max_f), dtype=torch.bool)
    fact_ids=torch.zeros((B,max_f), dtype=torch.long)
    qsub=[]; qrel=[]; kind=[]
    ext_lists=[]
    prompts=[]; targets=[]
    for b,s in enumerate(samples):
        for j,(su,re,va,fid) in enumerate(s.facts):
            fact_sub[b,j]=STOI[su]; fact_rel[b,j]=STOI[re]; fact_val[b,j]=STOI[va]
            fact_mask[b,j]=1; fact_ids[b,j]=fid
        qsub.append(STOI[s.q_subject]); qrel.append(STOI[s.q_relation]); kind.append(0 if s.kind=='rel' else 1)
        ext_lists.append(s.project_context)
        if baseline:
            seq=['<bos>','fatos']
            for su,re,va,_ in s.facts:
                seq += [su,re,va,';']
            if s.project_context:
                seq += [ITOS[i] for i in s.project_context] + [';']
            seq += s.query_tokens
        else:
            seq=['<bos>'] + s.query_tokens
        prompts.append(ids(seq)); targets.append(ids(s.target_tokens))
    max_p=max(map(len,prompts)); max_e=max(1,max(map(len,ext_lists)))
    pids=torch.full((B,max_p),PAD,dtype=torch.long); plen=torch.zeros(B,dtype=torch.long)
    eids=torch.full((B,max_e),PAD,dtype=torch.long); emask=torch.zeros((B,max_e),dtype=torch.bool)
    for b,seq in enumerate(prompts): pids[b,:len(seq)]=torch.tensor(seq); plen[b]=len(seq)
    for b,seq in enumerate(ext_lists):
        if seq: eids[b,:len(seq)]=torch.tensor(seq); emask[b,:len(seq)]=1
    tgt=torch.tensor(targets,dtype=torch.long)
    return dict(prompt=pids, plen=plen, target=tgt,
                fact_sub=fact_sub,fact_rel=fact_rel,fact_val=fact_val,fact_mask=fact_mask,fact_ids=fact_ids,
                qsub=torch.tensor(qsub),qrel=torch.tensor(qrel),kind=torch.tensor(kind),
                ext=eids,ext_mask=emask)

def batches(data, bs, shuffle, baseline=False):
    idx=list(range(len(data)))
    if shuffle: random.shuffle(idx)
    for i in range(0,len(idx),bs):
        ss=[data[j] for j in idx[i:i+bs]]
        yield tensorize_batch(ss,baseline=baseline)

# ---------- Baseline 2-layer GRU ----------
class BaselineGRU(nn.Module):
    def __init__(self, vocab, emb=48, hid=72):
        super().__init__()
        self.emb=nn.Embedding(vocab,emb,padding_idx=PAD)
        self.gru=nn.GRU(emb,hid,2,batch_first=True)
        self.out=nn.Linear(hid,vocab)
    def encode(self,prompt,plen):
        x=self.emb(prompt)
        packed=pack_padded_sequence(x,plen.cpu(),batch_first=True,enforce_sorted=False)
        _,h=self.gru(packed)
        return h
    def forward(self,b):
        h=self.encode(b['prompt'],b['plen'])
        dec_in=torch.cat([torch.full((len(b['target']),1),ANS,dtype=torch.long), b['target'][:,:-1]],1)
        y,h=self.gru(self.emb(dec_in),h)
        return self.out(y)
    @torch.no_grad()
    def generate(self,b,steps=4):
        h=self.encode(b['prompt'],b['plen']); tok=torch.full((len(b['target']),1),ANS,dtype=torch.long); outs=[]
        for _ in range(steps):
            y,h=self.gru(self.emb(tok[:,-1:]),h); nxt=self.out(y[:,-1]).argmax(-1); outs.append(nxt); tok=torch.cat([tok,nxt[:,None]],1)
        return torch.stack(outs,1)

# ---------- Distributed relational memory RNN ----------
class DistributedMemoryRNN(nn.Module):
    """Two recurrent layers with ID-aligned distributed K/V fact memory.

    Each layer owns a complete, differently projected view of every fact.
    Keys are Transformer-like vectors used for comparison; values preserve the
    full fact representation. The same row corresponds to the same global
    fact_id in both matrices. K/V are built once per sequence and reused.
    """
    def __init__(self,vocab,emb=48,hid=72,mem=64):
        super().__init__(); self.hid=hid; self.mem=mem; self.vocab=vocab
        self.emb=nn.Embedding(vocab,emb,padding_idx=PAD)
        pair_dim=emb*2; fact_dim=emb*3
        # Shared key transforms between query and fact on each memory layer.
        self.key1=nn.Linear(pair_dim,mem,bias=False)
        self.key2=nn.Sequential(nn.Linear(pair_dim,mem),nn.Tanh(),nn.Linear(mem,mem,bias=False))
        # Complete fact values on both layers, but with different projections.
        self.v1=nn.Linear(fact_dim,mem,bias=False)
        self.v2=nn.Sequential(nn.Linear(fact_dim,mem),nn.Tanh(),nn.Linear(mem,mem,bias=False))
        self.hidden_q1=nn.Linear(hid,mem,bias=False); self.hidden_q2=nn.Linear(hid+mem,mem,bias=False)
        self.r1=nn.Linear(mem,hid,bias=False); self.r2=nn.Linear(mem,hid,bias=False)
        self.ext_encoder=nn.GRU(emb,emb//2,1,batch_first=True,bidirectional=True)
        self.ext=nn.Linear(emb,hid,bias=False)
        self.cell1=nn.GRUCell(emb+hid,hid); self.cell2=nn.GRUCell(hid+hid,hid)
        self.out=nn.Linear(hid,vocab)
        self.ext_q=nn.Linear(hid,emb,bias=False)
        # generation vs copy gate. Source masks prevent impossible memory source.
        self.mix=nn.Linear(hid,3)

    def build_memory(self,b):
        fs=self.emb(b['fact_sub']); fr=self.emb(b['fact_rel']); fv=self.emb(b['fact_val'])
        pair=torch.cat([fs,fr],-1); full=torch.cat([fs,fr,fv],-1)
        K1=F.normalize(self.key1(pair),dim=-1); V1=self.v1(full)
        K2=F.normalize(self.key2(pair),dim=-1); V2=self.v2(full)
        qpair=torch.cat([self.emb(b['qsub']),self.emb(b['qrel'])],-1)
        qk1=F.normalize(self.key1(qpair),dim=-1); qk2=F.normalize(self.key2(qpair),dim=-1)
        ex_raw=self.emb(b['ext']); mask=b['ext_mask'].unsqueeze(-1)
        ext_len=b['ext_mask'].sum(1).clamp_min(1)
        packed=pack_padded_sequence(ex_raw,ext_len.cpu(),batch_first=True,enforce_sorted=False)
        packed_out,_=self.ext_encoder(packed)
        ex,_=torch.nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True,total_length=b['ext'].shape[1])
        denom=mask.sum(1).clamp_min(1); extmean=(ex*mask).sum(1)/denom
        extv=self.ext(extmean)*(b['kind']==1).float().unsqueeze(-1)
        internal_gate=(b['kind']==0).float().unsqueeze(-1)
        return K1,V1,K2,V2,qk1,qk2,extv,internal_gate,ex

    def step(self,tok,h1,h2,cache,b,return_attn=False):
        K1,V1,K2,V2,qk1,qk2,extv,internal_gate,ex=cache
        x=self.emb(tok)
        h1=self.cell1(torch.cat([x,extv],-1),h1)
        q1=F.normalize(qk1 + 0.15*self.hidden_q1(h1),dim=-1)
        s1=8.0*torch.einsum('bd,bfd->bf',q1,K1)
        s1=s1.masked_fill(~b['fact_mask'],-1e9); a1=F.softmax(s1,-1)
        read1=torch.einsum('bf,bfd->bd',a1,V1)*internal_gate
        h1m=h1+self.r1(read1)
        h2=self.cell2(torch.cat([h1m,extv],-1),h2)
        q2=F.normalize(qk2 + 0.15*self.hidden_q2(torch.cat([h2,read1],-1)),dim=-1)
        s2=8.0*torch.einsum('bd,bfd->bf',q2,K2)
        s2=s2.masked_fill(~b['fact_mask'],-1e9); a2=F.softmax(s2,-1)
        # Same row == same global fact_id. Agreement between layer matrices.
        joint=a1*a2; joint=joint/(joint.sum(-1,keepdim=True)+1e-8)
        read2=torch.einsum('bf,bfd->bd',joint,V2)*internal_gate
        h2m=h2+self.r2(read2)

        # Generator distribution.
        pgen=F.softmax(self.out(h2m),-1)
        # Copy exact relational value from the selected fact; no recompression required.
        pfact=torch.zeros(h2m.size(0),self.vocab)
        pfact.scatter_add_(1,b['fact_val'],joint)
        pfact=pfact*(b['kind']==0).float().unsqueeze(-1)
        # Copy from external AI-Memory retrieved text.
        eq=F.normalize(self.ext_q(h2m),dim=-1)
        ek=F.normalize(ex,dim=-1)
        es=6.0*torch.einsum('bd,bed->be',eq,ek).masked_fill(~b['ext_mask'],-1e9)
        # avoid NaN for rows with no external context
        has_ext=b['ext_mask'].any(-1)
        ea=torch.zeros_like(es)
        if has_ext.any(): ea[has_ext]=F.softmax(es[has_ext],-1)
        pext=torch.zeros(h2m.size(0),self.vocab)
        pext.scatter_add_(1,b['ext'],ea)
        pext=pext*(b['kind']==1).float().unsqueeze(-1)

        mix_logits=self.mix(h2m)
        # relational task cannot use external; project task cannot use fact-copy.
        mix_logits[:,1]=mix_logits[:,1].masked_fill(b['kind']==1,-1e9)
        mix_logits[:,2]=mix_logits[:,2].masked_fill(b['kind']==0,-1e9)
        mix=F.softmax(mix_logits,-1)
        probs=mix[:,0:1]*pgen + mix[:,1:2]*pfact + mix[:,2:3]*pext
        logp=(probs+1e-9).log()
        if return_attn:return h1,h2m,logp,joint,ea
        return h1,h2m,logp

    def encode(self,b,cache):
        B=len(b['target']); h1=torch.zeros(B,self.hid); h2=torch.zeros(B,self.hid)
        for t in range(b['prompt'].shape[1]):
            active=(t<b['plen']).float().unsqueeze(-1)
            nh1,nh2,_=self.step(b['prompt'][:,t],h1,h2,cache,b)
            h1=nh1*active+h1*(1-active); h2=nh2*active+h2*(1-active)
        return h1,h2

    def forward(self,b):
        cache=self.build_memory(b); h1,h2=self.encode(b,cache)
        prev=torch.full((len(b['target']),),ANS,dtype=torch.long); logs=[]
        for t in range(b['target'].shape[1]):
            h1,h2,logp=self.step(prev,h1,h2,cache,b); logs.append(logp); prev=b['target'][:,t]
        return torch.stack(logs,1)

    @torch.no_grad()
    def generate(self,b,steps=4,with_attention=False):
        cache=self.build_memory(b); h1,h2=self.encode(b,cache); prev=torch.full((len(b['target']),),ANS,dtype=torch.long)
        outs=[]; attn_steps=[]; ext_steps=[]
        for _ in range(steps):
            h1,h2,logp,attn,extattn=self.step(prev,h1,h2,cache,b,return_attn=True)
            prev=logp.argmax(-1); outs.append(prev); attn_steps.append(attn); ext_steps.append(extattn)
        return torch.stack(outs,1), torch.stack(attn_steps,1)

# ---------- Train/eval ----------
def parameter_count(m): return sum(p.numel() for p in m.parameters())

def train_model(model, baseline, epochs=4, lr=2e-3):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=None; best_acc=-1
    hist=[]
    for ep in range(1,epochs+1):
        model.train(); total=0; loss_sum=0
        for b in batches(train,64,True,baseline):
            opt.zero_grad(set_to_none=True); logits=model(b)
            loss=(F.cross_entropy(logits.reshape(-1,logits.size(-1)),b['target'].reshape(-1)) if baseline else F.nll_loss(logits.reshape(-1,logits.size(-1)),b['target'].reshape(-1)))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=len(b['target']); loss_sum+=loss.item()*len(b['target'])
        metrics=evaluate(model,valid,baseline)
        acc=metrics['value_acc']
        hist.append({'epoch':ep,'loss':loss_sum/total,'valid_exact':metrics['exact'],'valid_value_acc':acc})
        print(f"{'baseline' if baseline else 'memory'} epoch {ep}: loss={loss_sum/total:.4f} val_value={acc:.3f} val_exact={metrics['exact']:.3f}",flush=True)
        if acc>best_acc:
            best_acc=acc; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    return hist

@torch.no_grad()
def evaluate(model,data,baseline):
    model.eval(); n=0; exact=0; value_ok=0; rel_n=rel_ok=0; proj_n=proj_ok=0
    for b in batches(data,96,False,baseline):
        if baseline: pred=model.generate(b,4)
        else: pred,_=model.generate(b,4)
        eq=(pred==b['target']); ex=eq.all(-1); val=eq[:,1]  # answer value/language token
        exact+=ex.sum().item(); value_ok+=val.sum().item(); n+=len(ex)
        relmask=b['kind']==0; pmask=b['kind']==1
        rel_n+=relmask.sum().item(); proj_n+=pmask.sum().item()
        rel_ok+=(val&relmask).sum().item(); proj_ok+=(val&pmask).sum().item()
    return {'exact':exact/n,'value_acc':value_ok/n,'rel_value_acc':rel_ok/max(1,rel_n),'project_value_acc':proj_ok/max(1,proj_n),'n':n}

@torch.no_grad()
def attention_diagnostics(model,data,n=120):
    model.eval(); top1=0; total=0; entropy=[]; samples=[]
    for b in batches(data[:n],48,False,False):
        pred,attn=model.generate(b,4,True)
        for i in range(len(b['target'])):
            if b['kind'][i].item()!=0: continue
            # relevant fact is the row whose subject+relation matches structured query.
            mask=(b['fact_sub'][i]==b['qsub'][i]) & (b['fact_rel'][i]==b['qrel'][i]) & b['fact_mask'][i]
            true_idx=int(mask.nonzero()[0])
            value_attn=attn[i,1]
            got=int(value_attn.argmax())
            top1 += (got==true_idx); total+=1
            a=value_attn[b['fact_mask'][i]]; entropy.append(float(-(a*(a+1e-9).log()).sum()))
            if len(samples)<3:
                samples.append({'true_fact_id':int(b['fact_ids'][i,true_idx]),'attended_fact_id':int(b['fact_ids'][i,got]),'attention':float(value_attn[got])})
    return {'fact_top1':top1/max(1,total),'mean_attention_entropy':sum(entropy)/max(1,len(entropy)),'examples':samples}


def decode(seq): return ' '.join(ITOS[int(x)] for x in seq)

def generation_examples(base,mem,data,count=5):
    ss=data[:count]; bb=tensorize_batch(ss,True); bm=tensorize_batch(ss,False)
    pb=base.generate(bb,4); pm,att=mem.generate(bm,4,True); out=[]
    for i,s in enumerate(ss):
        out.append({'kind':s.kind,'query':' '.join(s.query_tokens),'expected':' '.join(s.target_tokens),
                    'baseline':decode(pb[i]),'memory_model':decode(pm[i]),
                    'facts':len(s.facts)})
    return out

if __name__=='__main__':
    print(f'AI-Memory retrieval: {PROJECT_RETRIEVAL_OK}/{len(PROJECTS)} top-1 correct')
    base=BaselineGRU(len(VOCAB_LIST),hid=128); mem=DistributedMemoryRNN(len(VOCAB_LIST))
    print('params baseline',parameter_count(base),'memory',parameter_count(mem))
    t=time.time(); hb=train_model(base,True,epochs=5); base_time=time.time()-t
    t=time.time(); hm=train_model(mem,False,epochs=3); mem_time=time.time()-t
    results={'seed':SEED,'ai_memory_retrieval_top1':PROJECT_RETRIEVAL_OK/len(PROJECTS),
             'params':{'baseline':parameter_count(base),'memory_model':parameter_count(mem)},
             'train_seconds':{'baseline':base_time,'memory_model':mem_time},
             'history':{'baseline':hb,'memory_model':hm},'tests':{}}
    for name,ds in test_sets.items():
        rb=evaluate(base,ds,True); rm=evaluate(mem,ds,False); diag=attention_diagnostics(mem,ds)
        results['tests'][name]={'baseline':rb,'memory_model':rm,'memory_attention':diag}
        print(name,'baseline',rb,'memory',rm,'attn',diag['fact_top1'])
    results['examples']=generation_examples(base,mem,test_sets['scale_64'],6)
    (ROOT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
    torch.save({'state_dict':base.state_dict(),'vocab':VOCAB_LIST},ROOT/'baseline_gru.pt')
    torch.save({'state_dict':mem.state_dict(),'vocab':VOCAB_LIST},ROOT/'distributed_memory_rnn.pt')
    PROJECT_MEMORY.close()
    print(json.dumps(results['examples'],indent=2,ensure_ascii=False))
