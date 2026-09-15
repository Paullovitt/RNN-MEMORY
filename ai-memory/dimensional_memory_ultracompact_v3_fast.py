import bisect, heapq
from dimensional_memory_ultracompact_v3 import UltraCompactDimensionalMemoryV3

class UltraCompactDimensionalMemoryV3Fast(UltraCompactDimensionalMemoryV3):
    def _edge_neighbors(self,keys,counts,wid,top):
        lo=bisect.bisect_left(keys,wid<<32); hi=bisect.bisect_left(keys,(wid+1)<<32)
        if hi<=lo:return []
        best=heapq.nlargest(top, ((int(counts[i]), int(keys[i]&0xffffffff)) for i in range(lo,hi)))
        return [(self._term_str(tid),count) for count,tid in best]
