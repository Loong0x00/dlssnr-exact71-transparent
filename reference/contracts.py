"""Inspectable topology, geometry and original synchronization/state contracts."""
from dataclasses import dataclass
from types import MappingProxyType
EDGES=MappingProxyType({0:(1,70),1:(2,),2:(3,),3:(4,),4:(5,66),5:(6,),6:(7,),7:(8,),8:(9,62),9:(10,),10:(11,),11:(12,),12:(13,),13:(14,),14:(15,56),15:(16,),16:(17,),17:(18,),18:(19,),19:(20,),20:(21,),21:(22,),22:(23,48),23:(24,),24:(25,),25:(26,),26:(27,),27:(28,),28:(29,),29:(30,),30:(31,39),31:(32,),32:(33,),33:(34,),34:(35,),35:(36,),36:(37,),37:(38,),38:(39,),39:(40,),40:(41,),41:(42,),42:(43,),43:(44,),44:(45,),45:(46,),46:(47,),47:(48,),48:(49,),49:(50,),50:(51,),51:(52,),52:(53,),53:(54,),54:(55,),55:(56,),56:(57,),57:(58,),58:(59,),59:(60,),60:(61,),61:(62,),62:(63,),63:(64,),64:(65,),65:(66,),66:(67,),67:(68,),68:(69,),69:(70,),70:()})
GEOMETRY=MappingProxyType({'pre0/skip0':(512,512,32),'block1':(256,256,32),'block4.high':(256,256,32),'block8.high':(128,128,64),'block14.high':(64,64,128),'block22.high':(32,32,256),'block22.low/split':(16,16,512),'block30.head/ViT':(8,8,1024),'ViT.tokens':64,'block39/40':(16,16,512),'block48':(32,32,256),'block56':(64,64,128),'block62':(128,128,64),'block66':(256,256,32),'block70':(512,512,4)})
SYNCHRONIZATION=MappingProxyType({'swin_1h_2h_wait':'signed-negative relaxed poll; producer release-store zero; no acquire/fence is invented','swin_local':'original CTA bar.sync and mbarrier arrival/try_wait/bulk complete_tx ordering retained in source-equivalence evidence; functional CPU forward schedules completed windows sequentially','split16':'4-warp CTA barriers and two mbarrier phases; original padding predicates and complete writer partitions','vit_expand_qkv_contract_projection':'fresh caller-owned counter epoch -1; z0 direct producer, z1..z3 relaxed poll counter >= z-1, original reductions, release-store z; temporary buffers start unproduced','vit_attention':'128-thread mbarrier arrival/try_wait and source K32 score/PV ordering','native_visibility':'GPU acquire/fence, ABA/lifetime and cross-submission visibility remain external and are not silently strengthened'})
@dataclass
class ViTCounterEpoch:
 words:list[int]
 @classmethod
 def fresh(cls,count):
  if type(count)is not int or count<=0:raise ValueError('positive counter count')
  return cls([-1]*count)
 def poll(self,index,z):
  if not(0<=index<len(self.words) and z in (0,1,2,3)):raise ValueError('counter/index')
  need=z-1
  if self.words[index]<need:raise RuntimeError(f'relaxed poll blocked: need {need}, got {self.words[index]}')
  return self.words[index]
 def release(self,index,z):
  if not(0<=index<len(self.words) and z in (0,1,2,3)):raise ValueError('counter/index')
  if self.words[index]!=z-1:raise RuntimeError('counter phase/ABA mismatch')
  self.words[index]=z
 def complete(self,index):
  for z in range(4):self.poll(index,z);self.release(index,z)
  return self.words[index]
def execution_contract():return MappingProxyType({'edges':EDGES,'geometry':GEOMETRY,'synchronization':SYNCHRONIZATION,'temporal':'full-u64 counter; reset suppresses T1/T2 pair for that frame; history resource/copy and synchronization are caller-owned explicit state','texture':'strict raw-F32 callback at exact source coordinates; A/C mode2, B mode0; no software filter fallback'})
