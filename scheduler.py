import numpy as np, os
class LGBMScheduler:
  def __init__(self):
    self.alpha=float(os.getenv('SCORE_ALPHA','1.0'))
    self.beta=float(os.getenv('SCORE_BETA','0.2'))
    self.gamma=float(os.getenv('SCORE_GAMMA','0.5'))
    self.delta=float(os.getenv('SCORE_DELTA','0.3'))
  def score_instances(self, req, insts, cap):
    rows=[]; ids=[]
    for i in insts:
      rtt=float(i['meta'].get('rtt_ms',3.0))
      price=float(i['meta'].get('price_cents_s',0.01))
      q=float(i.get('dyn',{}).get('avg_q_ms',0.0))
      rows.append([rtt,price,q, float(req.get('tokens',1)), float(req.get('emb_dim',256))])
      ids.append(i['id'])
    X=np.array(rows,dtype=np.float32)
    rtt=X[:,0]; price=X[:,1]; q=X[:,2]
    load = np.array([1.0/max(1,cap.get(i,1)) for i in ids], dtype=np.float32)
    score=self.alpha*(rtt+q)+self.beta*price+self.gamma*load+self.delta*rtt
    order=np.argsort(score)
    return [(ids[i], float(score[i]), float(rtt[i]+q[i]), float(price[i])) for i in order]
