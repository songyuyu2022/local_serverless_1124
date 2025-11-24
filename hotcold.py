import os, math, time
from collections import defaultdict
class HotColdManager:
  def __init__(self,alpha=0.9):
    self.alpha=float(os.getenv('HOT_EMA_ALPHA',alpha)); self.ema=defaultdict(float); self.count=defaultdict(int)
    self.last=time.time(); self.win=int(os.getenv('HOT_WINDOW_S','60')); self.pct=float(os.getenv('HOT_PCT','0.2')); self.minf=int(os.getenv('HOT_MIN_FREQ','128')); self.cold=int(os.getenv('COLD_ACC_STEPS','10'))
  def observe_batch(self,ids):
    for e in ids: self.count[e]+=1
  def _roll(self):
    if time.time()-self.last>=self.win:
      for e,c in list(self.count.items()): self.ema[e]=self.alpha*self.ema[e]+(1-self.alpha)*c; self.count[e]=0
      self.last=time.time()
  def classify(self):
    self._roll()
    if not self.ema: return set(), set()
    items=sorted(self.ema.items(), key=lambda x:x[1], reverse=True)
    k=max(1,int(math.ceil(len(items)*self.pct))); hot=set([e for e,_ in items[:k]])
    for e,c in self.count.items():
      if c>=self.minf: hot.add(e)
    allids=set(list(self.ema.keys())+list(self.count.keys())); cold=allids-hot; return hot,cold
  def cold_delay_steps(self): return self.cold
