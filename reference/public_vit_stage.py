#!/usr/bin/env python3
from pathlib import Path
from types import MappingProxyType
import importlib.util,sys,struct,json,hashlib,numpy as np
H=Path(__file__).resolve().parent;V=H;CP=H/'weights/dlssnr-310.8-weights.safetensors';M=96
def load(tag,path):s=importlib.util.spec_from_file_location(tag,path);m=importlib.util.module_from_spec(s);sys.modules[tag]=m;s.loader.exec_module(m);return m
sys.path.insert(0,str(V/'nr-vit-n64-tensor-main-20260912'));import predict as core
qkv=load('_package_public_qkv_sm120',H/'qkv_sm120_cubin.py');inrep=load('padded_public_inrepack',V/'nr-vit-repack-reference-20260912/repack_2d_to_1d_fp8_reference.py');outrep=load('padded_public_outrepack',V/'nr-vit-output-repack-main-20260912/repack.py');num=core.expand;axes=core.contract
mi=np.arange(M)[:,None];k1024=np.arange(1024)[None,:];n1024=np.arange(1024)[None,:]
def amap(rows,k,width):return axes.a_offset(rows,k,width)
def cmap(rows,n,width):return axes.c_offset(rows,n,width)
# Expand source maps at N96. Producer tail stores logical channels through the source pi permutation.
ex=core.expand;ex.M=M;ex.INPUT_BYTES=M*1024;ex.OUTPUT_BYTES=M*4096;ex.INPUT_MAP=amap(mi,k1024,1024);a4096=amap(mi,np.arange(4096)[None,:],4096);perm4096=np.array([axes.pi(k) for k in range(4096)]);ex.OUTPUT_MAP=np.empty_like(a4096);ex.OUTPUT_MAP[:,perm4096]=a4096
# Contract/projection maps and generic forwards.
A_CON=amap(mi,np.arange(4096)[None,:],4096);C1024=cmap(mi,n1024,1024);B_CON=axes.b_offset(np.arange(4096)[:,None],n1024,1024);A1024=amap(mi,k1024,1024);B1024=axes.b_offset(np.arange(1024)[:,None],n1024,1024)
def linear_residual(p0,p8,slab,*,K,A_MAP,B_MAP,C_MAP,coef_off,label):
 if type(p0)is not bytes or len(p0)!=M*K or type(p8)is not bytes or len(p8)!=M*1024:raise ValueError(label+' N96 spans')
 a=num.decode_e4m3(p0,A_MAP);b=num.decode_e4m3(slab,B_MAP);skip=num.decode_e4m3(p8,C_MAP);coef=np.frombuffer(slab,dtype='<f2',offset=coef_off).copy()
 if not np.all(np.isfinite(coef)):raise num.UnsupportedNonFinite(label+' coefficient')
 carry=num._rn_half(skip*coef.astype(np.float64),label+' residual');partials=[];combined=None
 chunk=K//4
 for z in range(4):
  c=carry.copy() if z==0 else np.zeros((M,1024),np.float16)
  for k0 in range(z*chunk,(z+1)*chunk,32):c=num._rn_half(a[:,k0:k0+32]@b[k0:k0+32]+c.astype(np.float64),label+' K32')
  partials.append(c);combined=c.copy() if z==0 else num._rn_half(combined.astype(np.float64)+c.astype(np.float64),label+' zadd')
 code=num.fp8_rn_satfinite(combined);phys=np.empty(M*1024,np.uint8);phys[C_MAP]=code;return {'output_physical':phys,'prequantization':combined,'partials':partials}
def contract(p0,p8,slab):return linear_residual(p0,p8,slab,K=4096,A_MAP=A_CON,B_MAP=B_CON,C_MAP=C1024,coef_off=4194304,label='contract')
def projection(p0,p8,slab):return linear_residual(p0,p8,slab,K=1024,A_MAP=A1024,B_MAP=B1024,C_MAP=C1024,coef_off=1048576,label='projection')
# QKV N96 SM120 maps.
qkv.M=M;qkv.m=mi;qkv.A_MAP=qkv.axes.a_offset(mi,k1024,1024);qkv.B_MAP=128+qkv.axes.b_offset(np.arange(1024)[:,None],np.arange(3072)[None,:],3072);mm=np.arange(M)[:,None,None];hh=np.arange(32)[None,:,None];dd=np.arange(32)[None,None,:];qkv.QK_MAP=qkv.axes.c_offset(mm,32*hh+dd,1024);qkv.K_MAP=(16384*(mm//16)+512*hh+16*(4*(mm%8)+(dd%8)//2)+8*((mm%16)//8)+4*(dd//16)+2*((dd%16)//8)+dd%2);qkv.V_MAP=(32768*(mm//32)+1024*hh+512*(dd//16)+16*(4*(dd%8)+(mm%8)//2)+8*((dd%16)//8)+4*((mm%32)//16)+2*((mm%16)//8)+mm%2)
att=core.attention;OUTPUT_MAP=C1024
def denom64(p):return att.denominator64(p)
def denom32(p):
 if p.shape[-1]!=32:raise ValueError('32-key tail')
 g=p.reshape(*p.shape[:-1],4,4,2);a=att.add(g[...,0,:,:],g[...,1,:,:],'tail pair0/8');b=att.add(g[...,2,:,:],g[...,3,:,:],'tail pair16/24');c=att.add(a,b,'tail pairgroups');d=att.add(c[...,0,:],c[...,1,:],'tail lanes01');d=att.add(d,c[...,2,:],'tail lane2');d=att.add(d,c[...,3,:],'tail lane3');return att.add(d[...,0],d[...,1],'tail half pair')
def attention(Qb,Kb,Vb):
 for b in (Qb,Kb,Vb):
  if type(b)is not bytes or len(b)!=M*1024:raise ValueError('N96 QKV span')
 q=num.decode_e4m3(Qb,qkv.QK_MAP).transpose(1,0,2);k=num.decode_e4m3(Kb,qkv.K_MAP).transpose(1,0,2);v=num.decode_e4m3(Vb,qkv.V_MAP).transpose(1,0,2);score=att.rn(q@k.transpose(0,2,1),'QK dot32');prob=att.score_exp(score);codes=num.fp8_rn_satfinite(prob);pv=num.E4M3_LUT[codes];numer=np.zeros((32,M,32),np.float16)
 # Source executes two complete K64 denominator trees. Its 32 invalid tail scores are zero,
 # therefore score_exp(0), then it subtracts one separately F32-multiplied/half-rounded
 # missing-count correction after the running half addition (PTX 2783..2850).
 base=att.score_exp(np.zeros((1,),np.float16))[0];tail=np.empty((32,M,64),np.float16);tail[...,:32]=prob[...,64:96];tail[...,32:]=base;part0=denom64(prob[...,:64]);part1=denom64(tail);den=att.add(part0,part1,'running denominator K64');correction=np.float16(np.float32(base)*np.float32(32));den=att.rn(den.astype(np.float64)-float(correction),'invalid-key correction')
 for k0 in (0,32,64):numer=att.rn(pv[...,k0:k0+32]@v[:,k0:k0+32]+numer.astype(np.float64),'PV K32')
 floor=np.maximum(den,att.EPS).astype(np.float16);bits=floor.view(np.uint16)
 if np.any((bits<0x0410)|(bits>=0x7c00)):raise num.UnsupportedNonFinite('N96 reciprocal domain')
 inv=att.rn(att.RCP_F32[bits].astype(np.float64),'RCP');res=att.rn(numer.astype(np.float64)*inv[...,None].astype(np.float64),'normalize');logical=num.fp8_rn_satfinite(res).transpose(1,0,2).reshape(M,1024);phys=np.empty(M*1024,np.uint8);phys[OUTPUT_MAP]=logical;return {'output_physical':phys,'denominator':den,'parts':[part0,part1]}
# Package stage: constructor I/O only; forward is pure array computation.
SIZES=(4194320,4196352,3145856,2,1050624)
class PublicViT96SM120:
 def __init__(self,checkpoint=CP):
  checkpoint=Path(checkpoint)
  if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e':raise ValueError('checkpoint identity')
  self.blocks={}
  with checkpoint.open('rb') as f:
   hn=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(hn))
   for b in range(31,39):
    self.blocks[b]=[]
    for j,size in enumerate(SIZES):
     lo,hi=head[f'block{b}.layer{j}.layer']['data_offsets'];f.seek(8+hn+lo);z=f.read(hi-lo)
     if len(z)!=size:raise ValueError('ViT slab size')
     self.blocks[b].append(z)
 def forward(self,head_2d):
  if type(head_2d)is not bytes or len(head_2d)!=98304:raise ValueError('8x12x1024 head bytes')
  cur=bytes(inrep.run_repack(head_2d,8,12));stages={}
  for b in range(31,39):
   before=cur;e=ex.forward(before,self.blocks[b][0]);F=e['output_physical'].tobytes();c=contract(F,before,self.blocks[b][1]);P=c['output_physical'].tobytes();q=qkv.forward(P,self.blocks[b][2]);Q=q['Q_physical'].tobytes();K=q['K_physical'].tobytes();VV=q['V_physical'].tobytes();a=attention(Q,K,VV);A=a['output_physical'].tobytes();p=projection(A,P,self.blocks[b][4]);cur=p['output_physical'].tobytes();stages[b]={'expand':F,'contract':P,'Q':Q,'K':K,'V':VV,'attention':A,'projection':cur}
  return {'output_1d_physical':cur,'output_2d_physical':outrep.forward(cur,8,12),'stages':stages,'logical_N':96}
