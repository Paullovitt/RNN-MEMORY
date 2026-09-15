import bisect, heapq, math, mmap, re, unicodedata
from array import array
from pathlib import Path

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

def norm(s):
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))

def tokens(text):
    return [norm(x) for x in TOKEN_RE.findall(text or "") if x]

class UltraCompactDimensionalMemoryV3:
    """Memória dimensional com vocabulário textual fora da RAM."""
    def __init__(self, docstore_path, terms_path=None):
        self.docstore_path=Path(docstore_path)
        self.terms_path=Path(terms_path) if terms_path else self.docstore_path.with_suffix('.terms.bin')
        self.vocab={}; self._build_terms=[]
        self.lookup_ids=array('I')
        self.term_off=array('Q'); self.term_len=array('I')
        self.post_flat=array('I'); self.post_off=array('Q'); self.post_len=array('I')
        self.freq=array('Q')
        self.edge_out_key=array('Q'); self.edge_out_count=array('Q')
        self.edge_in_key=array('Q'); self.edge_in_count=array('Q')
        self.offsets=array('Q'); self.lengths=array('I'); self.recurrence=bytearray()
        self._fh=None; self._tfh=None; self._tmmap=None; self._n=0; self._dimensions=0

    @staticmethod
    def _pack(a,b): return (a<<32)|b

    def _term_id(self,t,posting_lists):
        tid=self.vocab.get(t)
        if tid is not None:return tid
        tid=len(self._build_terms); self.vocab[t]=tid; self._build_terms.append(t)
        posting_lists.append(array('I')); self.freq.append(0)
        return tid

    def fit(self,records):
        posting_lists=[]; edges={}
        self.docstore_path.parent.mkdir(parents=True,exist_ok=True)
        with self.docstore_path.open('wb') as f:
            for doc,r in enumerate(records):
                rid=str(r.get('id',doc+1)); text=str(r['text'])
                rec=max(1,min(255,int(r.get('recurrence',1))))
                raw=(rid+'\0'+text).encode('utf-8')
                self.offsets.append(f.tell()); self.lengths.append(len(raw)); f.write(raw)
                self.recurrence.append(rec)
                ts=tokens(text); tids=[self._term_id(t,posting_lists) for t in ts]
                for tid in set(tids):posting_lists[tid].append(doc)
                for tid in tids:self.freq[tid]+=rec
                for a,b in zip(tids,tids[1:]):
                    key=self._pack(a,b); edges[key]=edges.get(key,0)+rec
                self._n=doc+1

        for p in posting_lists:
            self.post_off.append(len(self.post_flat)); self.post_len.append(len(p)); self.post_flat.extend(p)
        del posting_lists
        out_items=sorted(edges.items())
        self.edge_out_key=array('Q',(k for k,_ in out_items)); self.edge_out_count=array('Q',(c for _,c in out_items))
        in_items=sorted((((k & 0xffffffff)<<32)|(k>>32),c) for k,c in edges.items())
        self.edge_in_key=array('Q',(k for k,_ in in_items)); self.edge_in_count=array('Q',(c for _,c in in_items))
        del edges,out_items,in_items
        order=sorted(range(len(self._build_terms)), key=self._build_terms.__getitem__)
        self.lookup_ids=array('I',order); del order
        self._dimensions=len(self._build_terms)
        with self.terms_path.open('wb') as tf:
            for term in self._build_terms:
                raw=term.encode('utf-8'); self.term_off.append(tf.tell()); self.term_len.append(len(raw)); tf.write(raw)
        self.vocab=None; self._build_terms=None
        self._fh=self.docstore_path.open('rb')
        self._tfh=self.terms_path.open('rb')
        self._tmmap=mmap.mmap(self._tfh.fileno(),0,access=mmap.ACCESS_READ)
        return self

    @property
    def dimensions(self):return self._dimensions
    @property
    def edge_count(self):return len(self.edge_out_key)
    def _term_bytes(self,tid):
        s=self.term_off[tid]; return self._tmmap[s:s+self.term_len[tid]]
    def _term_str(self,tid):return self._term_bytes(tid).decode('utf-8','replace')
    def _lookup(self,term):
        q=term.encode('utf-8'); ids=self.lookup_ids; lo=0; hi=len(ids)
        while lo<hi:
            mid=(lo+hi)//2; tid=ids[mid]; v=self._term_bytes(tid)
            if v<q:lo=mid+1
            else:hi=mid
        if lo<len(ids):
            tid=ids[lo]
            if self._term_bytes(tid)==q:return tid
        return None
    def close(self):
        if self._tmmap:self._tmmap.close(); self._tmmap=None
        if self._tfh:self._tfh.close(); self._tfh=None
        if self._fh:self._fh.close(); self._fh=None
    def _record(self,doc):
        if self._fh is None:self._fh=self.docstore_path.open('rb')
        self._fh.seek(self.offsets[doc]); raw=self._fh.read(self.lengths[doc]).decode('utf-8','replace')
        return raw.split('\0',1)
    def _plen(self,tid):return self.post_len[tid]
    def _posting_contains(self,tid,doc):
        start=self.post_off[tid]; lo=start; hi=start+self.post_len[tid]; a=self.post_flat
        while lo<hi:
            mid=(lo+hi)//2; v=a[mid]
            if v<doc:lo=mid+1
            else:hi=mid
        return lo<start+self.post_len[tid] and a[lo]==doc
    def search(self,query,k=4,candidate_floor=512):
        qids=[]; seen=set()
        for t in tokens(query):
            tid=self._lookup(t)
            if tid is not None and tid not in seen:qids.append(tid); seen.add(tid)
        if not qids:return []
        qids.sort(key=self._plen); rare=qids[0]; rlen=self._plen(rare); rs=self.post_off[rare]
        if rlen<=max(64,k*16): cand=list(self.post_flat[rs:rs+rlen])
        else:
            cs=set()
            for tid in qids:
                s=self.post_off[tid]; e=s+self.post_len[tid]
                for pos in range(s,e):cs.add(self.post_flat[pos])
                if len(cs)>=candidate_floor:break
            cand=list(cs)
        n=max(1,self._n); scored=[]
        for doc in cand:
            score=0.0; matched=0
            for tid in qids:
                if self._posting_contains(tid,doc):
                    idf=math.log((1+n)/(1+self._plen(tid)))+1.0; score+=idf*idf; matched+=1
            if matched:scored.append((score+0.02*math.log1p(self.recurrence[doc]),doc))
        scored.sort(reverse=True); out=[]
        for score,doc in scored[:k]:
            rid,text=self._record(doc)
            out.append({'id':rid,'text':text,'recurrence':self.recurrence[doc],'score':round(score,4)})
        return out
    def _edge_neighbors(self,keys,counts,wid,top):
        lo=bisect.bisect_left(keys,wid<<32); hi=bisect.bisect_left(keys,(wid+1)<<32)
        if hi<=lo:return []
        return heapq.nlargest(top, ((self._term_str(keys[i]&0xffffffff),int(counts[i])) for i in range(lo,hi)), key=lambda x:x[1])
    def word_info(self,word,top=15):
        w=norm(word); wid=self._lookup(w)
        if wid is None:return {'word':w,'dimension':None,'frequency':0,'before':[],'after':[]}
        return {'word':w,'dimension':wid,'frequency':int(self.freq[wid]),
                'before':self._edge_neighbors(self.edge_in_key,self.edge_in_count,wid,top),
                'after':self._edge_neighbors(self.edge_out_key,self.edge_out_count,wid,top)}
