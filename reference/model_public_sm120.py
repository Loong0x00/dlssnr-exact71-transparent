"""Transparent public-path DLSS NR71 SM120 first- and second-frame model.

Every forward uses only Python/NumPy numeric and layout primitives. It does not
load or call a DLL, CUBIN, PTX, VM, CUDA API, subprocess, network, or GPU.
"""
from pathlib import Path
import hashlib,numpy as np
from public_pre_stage import PublicPre0SM120
from public_encoder_stage import PaddedPublicEncoderStage
from public_split_stage import PaddedPublicSplitStage
from public_vit_stage import PublicViT96SM120
from public_decoder_stage import PublicDecoderSM120
from public_temporal_stage import PublicTemporalPreSM120,rgba32f_to_history_rgba16f_rtz
from public_temporal_post_stage import PublicTemporalPostSM120
from weights import LogicalWeights649
import contracts as execution_contracts
ROOT=Path(__file__).resolve().parent
class ExactNR71PublicSM120:
 """Pinned public feature-18 route at visible 512x512 and internal 512x576."""
 numeric_target='nvngx_dlssnr.dll e16bcf... original embedded SM120 CUBIN'
 supported_state='first/reset and immediate second frame; 512x512 RGBA32F Color; second-frame zero RG32F motion; no control mask'
 def __init__(self,checkpoint=ROOT/'weights/dlssnr-310.8-weights.safetensors'):
  checkpoint=Path(checkpoint);self.pre=PublicPre0SM120(checkpoint);self.temporal_pre=PublicTemporalPreSM120(checkpoint);self.encoder=PaddedPublicEncoderStage(ROOT,checkpoint);self.split=PaddedPublicSplitStage(ROOT);self.vit=PublicViT96SM120(checkpoint);self.decoder=PublicDecoderSM120(checkpoint);self.temporal_post=PublicTemporalPostSM120(self.decoder.live)
 def forward(self,color_rgba_f32,*,tone=.25,structure=.75,style=0,skin=-1.,use_auto_mask=False,keep_boundaries=False):
  if not isinstance(color_rgba_f32,np.ndarray) or color_rgba_f32.dtype!=np.float32 or color_rgba_f32.shape!=(512,512,4):raise ValueError('float32[512,512,4] required')
  if not np.all(np.isfinite(color_rgba_f32)):raise ValueError('finite color required')
  p=self.pre.forward(color_rgba_f32,tone=tone,structure=structure,style=style,skin=skin,use_auto_mask=use_auto_mask);e=self.encoder.forward(p['primary_A_planar'],p['skip_C_packed']);s=self.split.forward(e['block22_low']);v=self.vit.forward(s['head_8x12']);d=self.decoder.forward(v['output_2d_physical'],s['block30_layer3'],e['block22_high'],e['skip14'],e['skip8'],e['skip4'],p['skip_C_packed'],color_rgba_f32);out={'output_RGBA_f32_bits':d['output_RGBA_f32_bits'],'learned_RGB_logit_half':d['learned_RGB_logit_half'],'numeric_target':self.numeric_target,'state_scope':self.supported_state,'geometry':{'visible':(512,512),'internal_high':(512,576),'bottleneck':(16,20),'vit':(8,12,96)}}
  if keep_boundaries:out['boundaries']={'block0':p,'encoder':e,'split':s,'vit':v}
  return out
 def forward_two_frames(self,color0_rgba_f32,color1_rgba_f32=None,*,keep_boundaries=False):
  """Run reset frame then counter=1 temporal frame with zero motion.

  The public history transition is an explicit formatted-surface F32->F16 RTZ
  conversion. The second-frame history sampler lands on exact texel centers
  for this zero-motion contract; any non-center coordinate fails closed.
  """
  if color1_rgba_f32 is None:color1_rgba_f32=color0_rgba_f32
  first=self.forward(color0_rgba_f32,keep_boundaries=keep_boundaries)
  history=rgba32f_to_history_rgba16f_rtz(first['output_RGBA_f32_bits'])
  p=self.temporal_pre.forward(color1_rgba_f32,history);e=self.encoder.forward(p['primary_A_planar'],p['skip_C_packed']);s=self.split.forward(e['block22_low']);v=self.vit.forward(s['head_8x12']);d=self.decoder.forward(v['output_2d_physical'],s['block30_layer3'],e['block22_high'],e['skip14'],e['skip8'],e['skip4'],p['skip_C_packed'],color1_rgba_f32);temporal=self.temporal_post.forward(d['learned_RGB_logit_half'],color1_rgba_f32,history);second={'output_RGBA_f32_bits':temporal.view(np.uint32),'learned_RGB_logit_half':d['learned_RGB_logit_half'],'counter_after':p['counter_after'],'history_contract':'RGBA32F -> formatted RGBA16F RTZ -> normalized linear exact-center sampling','numeric_target':self.numeric_target,'state_scope':self.supported_state}
  if keep_boundaries:second['boundaries']={'block0':p,'encoder':e,'split':s,'vit':v}
  return {'frame0':first,'frame1':second,'history_RGBA16F_bits':history}
 @staticmethod
 def logical_weights(path=ROOT/'weights/dlssnr-logical-v18.safetensors'):return LogicalWeights649(path)
 @staticmethod
 def execution_contract():return execution_contracts.execution_contract()
 @staticmethod
 def public_evidence_digest():return '571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e'
