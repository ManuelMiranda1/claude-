import bpy, sys, numpy as np
def load(p, f):
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=p)
    arm=bpy.data.objects['Armature']; m=[o for o in bpy.data.objects if o.type=='MESH'][0]
    names=[g.name for g in m.vertex_groups]
    W=[tuple(sorted((names[g.group],round(g.weight,5)) for g in v.groups)) for v in m.data.vertices]
    dom=[max(v.groups,key=lambda g:g.weight).group for v in m.data.vertices]
    mw=np.array(m.matrix_world); tw=lambda a: (a@mw[:3,:3].T+mw[:3,3])*f
    base=tw(np.array([v.co[:] for v in m.data.vertices]))
    keys={k.name:tw(np.array([p.co[:] for p in k.data])) for k in m.data.shape_keys.key_blocks}
    anim=[]
    sc=bpy.context.scene
    for fr in (sc.frame_start, (sc.frame_start+sc.frame_end)//2, sc.frame_end):
        sc.frame_set(fr); dg=bpy.context.evaluated_depsgraph_get(); e=m.evaluated_get(dg).to_mesh()
        d=np.array([(m.matrix_world@v.co)[:] for v in e.vertices])*f; m.evaluated_get(dg).to_mesh_clear()
        anim.append((fr, np.ptp(d,0).round(2), np.isfinite(d).all()))
    return names,W,dom,base,keys,anim
S=4.4
n0,W0,d0,b0,k0,a0=load(sys.argv[1],1); n1,W1,d1,b1,k1,a1=load(sys.argv[2],100)
print("vertex count", len(b0), len(b1), "| weights identical:", W0==W1)
dom=np.array([n0[i].replace('mixamorig:','') for i in d0])
for grp in (['Head','Neck','HeadTop_End'], ['LeftHand','LeftHandThumb3','LeftHandPinky1'], ['LeftFoot','LeftToeBase'], ['RightFoot','RightToeBase'], ['Spine','Spine1','Spine2','Hips']):
    idx=np.isin(dom,grp); A=b0[idx]*S; B=b1[idx]
    off=(B-A).mean(0); err=np.abs(B-A-off).max()
    print(f"{grp[0]:10} rigid-shape error after scale x{S}: {err:.4f} studs (offset {off.round(3)})")
for k in k0:
    if k=='Basis': continue
    d0_=(k0[k]-k0['Basis'])*S; d1_=k1[k]-k1['Basis']
    print(f"shape key '{k}': moved verts {int((np.abs(d0_).sum(1)>1e-6).sum())} -> {int((np.abs(d1_).sum(1)>1e-6).sum())}, delta diff max {np.abs(d1_-d0_).max():.4f}")
print("anim original(x4.4 bbox):", [(f,(s*S).round(2)) for f,s,_ in a0])
print("anim new     (bbox)     :", [(f,s,ok) for f,s,ok in a1])
