import bpy, sys, numpy as np
def load(p):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=p)
    m=[o for o in bpy.data.objects if o.type=='MESH'][0]
    w=np.array([[g.weight for g in v.groups] and sum(g.weight*(g.group+1) for g in v.groups) for v in m.data.vertices])
    out={}
    for f in (0,20):
        bpy.context.scene.frame_set(f)
        dg=bpy.context.evaluated_depsgraph_get(); e=m.evaluated_get(dg).to_mesh()
        out[f]=np.array([(m.matrix_world@v.co)[:] for v in e.vertices]); m.evaluated_get(dg).to_mesh_clear()
    base=np.array([v.co[:] for v in m.data.vertices])
    return out,w,base
a,wa,ba=load(sys.argv[1]); b,wb,bb=load(sys.argv[2])
print("weights identical:", np.allclose(wa,wb,atol=1e-5))
print("base geometry = 1.6x:", np.abs(bb-1.6*ba).max())
for f in a: print("frame",f,"deformed max err vs 1.6x:", np.abs(b[f]-1.6*a[f]).max())
