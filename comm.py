import os, json, base64, numpy as np, torch
def _maybe_fp16(t): return t.half() if os.getenv('GRAD_FP16','1')=='1' else t
def _maybe_topk(t):
  r=float(os.getenv('GRAD_TOPK','0.0'))
  if r<=0 or t.numel()==0: return t, None
  k=int(max(1,t.numel()*r)); import torch as T
  vals,idx=T.topk(t.view(-1).abs(),k); mask=T.zeros_like(t.view(-1),dtype=T.bool); mask[idx]=True; sparse=t.view(-1)[mask]; return sparse, idx
def _pack_tensor(t):
  t=_maybe_fp16(t.detach().cpu()); sparse,idx=_maybe_topk(t)
  if idx is None: return {'dtype':str(t.dtype),'shape':list(t.shape),'data':base64.b64encode(t.numpy().tobytes()).decode()}
  else: return {'dtype':str(t.dtype),'shape':list(t.shape),'idx':idx.tolist(),'data':base64.b64encode(sparse.numpy().tobytes()).decode()}
def _unpack_tensor(obj):
  dt=np.dtype(obj['dtype'].replace('torch.',''))
  if 'idx' in obj:
    dense=np.zeros(int(np.prod(obj['shape'])),dtype=dt); buf=np.frombuffer(base64.b64decode(obj['data']),dtype=dt)
    import numpy as np
    dense[np.array(obj['idx'],dtype=np.int64)] = buf
    import torch as T
    return T.from_numpy(dense.reshape(obj['shape']))
  else:
    import torch as T
    buf=np.frombuffer(base64.b64decode(obj['data']),dtype=dt); return T.from_numpy(buf.reshape(obj['shape']))
class Comm:
  def __init__(self):
    self.redis_url=os.getenv('REDIS_URL',''); self._redis=None
    if self.redis_url:
      try:
        import redis; self._redis=redis.Redis.from_url(self.redis_url, decode_responses=False)
      except Exception: self._redis=None
    self.oss_prefix=os.getenv('OSS_URI_PREFIX','')
    self._oss_bucket=None
    if self.oss_prefix:
      import oss2
      auth=None
      if os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID') and os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'):
        auth=oss2.Auth(os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID'), os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET'))
      elif os.getenv('OSS_AK') and os.getenv('OSS_SK'):
        auth=oss2.Auth(os.getenv('OSS_AK'), os.getenv('OSS_SK'))
      else:
        auth=oss2.AnonymousAuth()
      self._oss_bucket=oss2.Bucket(auth, os.getenv('OSS_ENDPOINT'), os.getenv('OSS_BUCKET'))
  def send_hot(self,key,obj):
    if self._redis is None: return False
    self._redis.set(key, json.dumps(obj).encode()); return True
  def pull_hot(self,key):
    if self._redis is None: return None
    b=self._redis.get(key); 
    if not b: return None
    return json.loads(b.decode())
  def send_cold(self,key,obj):
    if not self._oss_bucket: return False
    data=json.dumps(obj).encode('utf-8')
    path=f"{self.oss_prefix.strip('/')}/{key}.json"
    self._oss_bucket.put_object(path, data); return True
  def pull_cold(self,key):
    if not self._oss_bucket: return None
    path=f"{self.oss_prefix.strip('/')}/{key}.json"
    try:
      r=self._oss_bucket.get_object(path); return json.loads(r.read().decode('utf-8'))
    except Exception: return None
