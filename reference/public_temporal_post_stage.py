"""Pure second-frame public block70 A/B/C live tail for zero motion."""
from pathlib import Path
import struct,numpy as np
ROOT=Path(__file__).resolve().parent
def _table(d,i):
 b=(d/f'APPROX_PHASE{i}_INPUT.bin').read_bytes();u=struct.unpack('<'+'I'*(len(b)//4),b);n=u[0];o=struct.unpack('<'+'I'*n,(d/f'APPROX_PHASE{i}_OUTPUT.bin').read_bytes());return {tuple(u[1+3*j:4+3*j]):o[j] for j in range(n)}
class PublicTemporalPostSM120:
 def __init__(self,live_module,lut_dir=ROOT/'native_approx'):
  self.live=live_module;self.lut={};d=Path(lut_dir)
  for i in (1,3,4,5,12,13):self.lut.update(_table(d,i))
 def forward(self,learned,color,history_half,blend_half_bits=0x39eb):
  live=self.live;lut=self.lut
  if not isinstance(learned,np.ndarray) or learned.dtype!=np.float16 or learned.shape!=(512,512,4) or not np.all(np.isfinite(learned)):raise ValueError('learned')
  if not isinstance(color,np.ndarray) or color.dtype!=np.float32 or color.shape!=(512,512,4):raise ValueError('color')
  if not isinstance(history_half,np.ndarray) or history_half.dtype!=np.uint16 or history_half.shape!=(512,512,4):raise ValueError('history')
  class AP(live.f.SyntheticRNApprox):
   def get(self,op,a,b=0):
    k=(op,live.f.ftz(a),live.f.ftz(b))
    if k not in lut:raise live.f.F32DomainError('SM120 post approximation domain miss')
    return lut[k]
   def div_approx(self,a,b):return self.get(0,a,b)
   def rcp_approx(self,a):return self.get(5,a)
   def ex2_approx(self,a):return self.get(6,a)
  cb=np.ascontiguousarray(color).view(np.uint32);hf=history_half.view(np.float16).astype(np.float32).view(np.uint32)
  def center(bits):
   v=struct.unpack('<f',struct.pack('<I',bits))[0]*512-.5;r=round(v)
   if v!=r:raise ValueError('zero-motion linear coordinate is not exact center')
   return min(511,max(0,int(r)))
  class Fetch:
   def fetch(self,slot,mode,u,v):
    x,y=center(u),center(v)
    if slot=='A':return tuple(map(int,cb[y,x]))
    if slot=='B_history':return tuple(map(int,hf[y,x]))
    if slot=='C_motion':return (0,0,0,0)
    raise ValueError('slot')
  cfg=live.FrameConfig(source_width=512,source_height=512,origin_x=-4,origin_y=-4,rgb_multiplier_bits=0x3d000000,decode_enabled_u32=1,blend_half_bits=blend_half_bits,a_base_x_bits=0,a_base_y_bits=0,a_scale_x_bits=0x44000000,a_scale_y_bits=0x44000000,a_post_x_bits=0x3b000000,a_post_y_bits=0x3b000000,p112_low_u32=1,p112_high_bits=0,p120_low_bits=0,p120_high_bits=0x44000000,p128_low_bits=0x44000000,p128_high_bits=0x3b000000,p136_low_bits=0x3b000000,p136_high_bits=0,p144_low_bits=0,p144_high_bits=0x44000000,p152_low_bits=0x44000000,p152_high_bits=0x3b000000,p160_low_bits=0x3b000000,p160_high_bits=0x3b000000,p168_bits=0x3b000000)
  return live.forward_half(learned,config=cfg,textures=live.LiveTextures(a=Fetch(),history_b=Fetch(),motion_c=Fetch()),approx=AP())
