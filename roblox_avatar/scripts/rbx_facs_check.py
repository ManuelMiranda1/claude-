import bpy, sys, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=sys.argv[1], anim_offset=0)
rig=[o for o in bpy.data.objects if o.type=='ARMATURE'][0]; m=[o for o in bpy.data.objects if o.type=='MESH'][0]
def tree(b,d=0):
    out=[("  "*d)+b.name]
    for c in b.children: out+=tree(c,d+1)
    return out
roots=[b for b in rig.data.bones if not b.parent]
print("\n".join(tree(roots[0])[:24])); print("... total bones", len(rig.data.bones))
props={k:m[k] for k in m.keys() if k.startswith('Frame') or k=='RootFaceJoint'}
print("custom props:", len(props), props.get('RootFaceJoint'), [props[f'Frame{i}'] for i in range(len(props)-1)])
sc=bpy.context.scene; print("frame range", sc.frame_start, sc.frame_end, "| action", rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None)
mw=np.array(m.matrix_world)
def ev(f):
    sc.frame_set(f); dg=bpy.context.evaluated_depsgraph_get(); e=m.evaluated_get(dg).to_mesh()
    P=(np.array([v.co[:] for v in e.vertices])@mw[:3,:3].T+mw[:3,3])*100; m.evaluated_get(dg).to_mesh_clear(); return P
base=ev(0); rest=(np.array([v.co[:] for v in m.data.vertices])@mw[:3,:3].T+mw[:3,3])*100
print("frame0 vs rest max diff:", np.abs(base-rest).max().round(6))
np.save('facs_frames.npy', np.array([ev(f) for f in range(0, len(props)-1)]))
for f in range(1,len(props)-1):
    d=np.linalg.norm(ev(f)-base,axis=1); mv=d>0.005
    print(f"{props[f'Frame{f}']:24} moved verts {mv.sum():4d} max {d.max():.3f} | moved z-range {base[mv,2].min() if mv.any() else 0:.2f}-{base[mv,2].max() if mv.any() else 0:.2f} | body(z<3) moved: {(mv&(base[:,2]<3.0)).sum()}")
