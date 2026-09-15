import json,time,torch
import experiment as e

torch.set_num_threads(min(8,torch.get_num_threads()))
base=e.BaselineGRU(len(e.VOCAB_LIST),hid=128)
mem=e.DistributedMemoryRNN(len(e.VOCAB_LIST))
base.load_state_dict(torch.load(e.ROOT/'baseline_gru.pt',map_location='cpu')['state_dict'])
mem.load_state_dict(torch.load(e.ROOT/'distributed_memory_rnn.pt',map_location='cpu')['state_dict'])
base.eval(); mem.eval()

results={}
for nf,n in [(128,160),(256,160),(512,120)]:
    ds=e.make_dataset(n,(nf,nf),1000+nf,e.TEST_PROJECTS,.25)
    t=time.perf_counter(); rb=e.evaluate(base,ds,True); tb=time.perf_counter()-t
    t=time.perf_counter(); rm=e.evaluate(mem,ds,False); tm=time.perf_counter()-t
    diag=e.attention_diagnostics(mem,ds,n=min(100,n))
    results[str(nf)]={
        'baseline':rb,'memory_model':rm,'fact_attention_top1':diag['fact_top1'],
        'seconds_total':{'baseline':tb,'memory_model':tm},
        'ms_per_sample':{'baseline':tb/n*1000,'memory_model':tm/n*1000}
    }
    print(nf,results[str(nf)],flush=True)

(e.ROOT/'stress_results.json').write_text(json.dumps(results,indent=2))
e.PROJECT_MEMORY.close()
