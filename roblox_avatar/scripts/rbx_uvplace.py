import bpy, sys, numpy as np
from mathutils import Vector, geometry
from scipy.spatial import cKDTree
def load(p):
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=p)
    o=[o for o in bpy.data.objects if o.type=='MESH'][0]; m=o.data; uv=m.uv_layers.active.data
    mw=np.array(o.matrix_world); P=(np.array([v.co[:] for v in m.vertices])@mw[:3,:3].T+mw[:3,3])*100
    F=[]
    for f in m.polygons:
        vs=list(f.vertices); c=P[vs].mean(0)
        if c[0]>0 and c[2]<1.2:
            F.append((P[vs], np.array([uv[i].uv[:] for i in f.loop_indices])))
    return F
ref=load(sys.argv[1])
tris=[]; 
for Q,U in ref:
    for k in range(1,len(Q)-1): tris.append((Q[[0,k,k+1]],U[[0,k,k+1]]))
cent=np.array([t[0].mean(0) for t in tris]); tree=cKDTree(cent)
for p in sys.argv[2:]:
    errs=[]
    for Q,U in load(p):
        c=Q.mean(0); best=None
        for i in tree.query(c,k=8)[1]:
            a,b,cc=(Vector(x) for x in tris[i][0]); q=geometry.closest_point_on_tri(Vector(c),a,b,cc); d=(q-Vector(c)).length
            if best is None or d<best[0]: best=(d,i,q)
        d,i,q=best; a,b,cc=(Vector(x) for x in tris[i][0])
        u=geometry.barycentric_transform(q,a,b,cc,*(Vector((w[0],w[1],0)) for w in tris[i][1]))
        errs.append(np.linalg.norm(U.mean(0)-[u.x,u.y]))
    e=np.array(errs); print(f"{p}: face UV-centre error vs original: median {np.median(e):.4f} p95 {np.percentile(e,95):.4f} max {e.max():.4f} | faces off by >0.01: {(e>0.01).sum()} / {len(e)}")
