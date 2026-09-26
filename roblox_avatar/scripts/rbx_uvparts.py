import bpy, sys, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=sys.argv[1], anim_offset=0)
m=[o for o in bpy.data.objects if o.type=='MESH'][0]; uv=m.data.uv_layers.active.data; names=[g.name for g in m.vertex_groups]
mw=np.array(m.matrix_world); P=(np.array([v.co[:] for v in m.data.vertices])@mw[:3,:3].T+mw[:3,3])*100
dom=[names[max(v.groups,key=lambda g:g.weight).group] if v.groups else '?' for v in m.data.vertices]
alias={'LeftUpLeg':'LeftUpperLeg','LeftLeg':'LowerLeg_L'}
def part(n):
    n=n.replace('mixamorig:','')
    for s in ('Left','Right'):
        if n in (s+'Leg', s+'LowerLeg'): return s+'LowerLeg'
        if n in (s+'Foot',s+'ToeBase'): return s+'Foot'
    return None
stats={}
for f in m.data.polygons:
    ps=[part(dom[v]) for v in f.vertices]
    pn=max(set(ps),key=ps.count)
    if pn is None: continue
    U=np.array([uv[i].uv[:] for i in f.loop_indices])
    a=sum(np.cross(U[k]-U[0],U[k+1]-U[0]) for k in range(1,len(U)-1))
    st=stats.setdefault(pn,dict(n=0,flip=0,outside=0,zero=0,umin=[9,9],umax=[-9,-9],area=0))
    st['n']+=1; st['flip']+=a<0; st['zero']+=abs(a)<1e-9; st['outside']+=bool(((U<0)|(U>1)).any()); st['area']+=abs(a)/2
    st['umin']=np.minimum(st['umin'],U.min(0)); st['umax']=np.maximum(st['umax'],U.max(0))
for k,v in sorted(stats.items()):
    print(f"{k:14} faces {v['n']:4d} | UV flipped {v['flip']:4d} | zero-area {v['zero']} | outside 0-1 {v['outside']} | UV bbox {np.round(v['umin'],3)}..{np.round(v['umax'],3)} | UV area {v['area']:.4f}")
