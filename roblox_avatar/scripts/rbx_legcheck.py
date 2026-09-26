import bpy, bmesh, sys, numpy as np
from scipy.spatial import cKDTree
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=sys.argv[1])
m=[o for o in bpy.data.objects if o.type=='MESH'][0]
mw=np.array(m.matrix_world); B=(np.array([v.co[:] for v in m.data.vertices])@mw[:3,:3].T+mw[:3,3])*100
bm=bmesh.new(); bm.from_mesh(m.data); bm.verts.ensure_lookup_table(); uvl=bm.loops.layers.uv.active
Lidx=[i for i in range(len(B)) if B[i,0]>0.005 and B[i,2]<1.2]; Ridx=[i for i in range(len(B)) if B[i,0]<-0.005 and B[i,2]<1.2]
d,j=cKDTree(B).query(B[Lidx]*[-1,1,1])
val_ok=all(len(bm.verts[i].link_edges)==len(bm.verts[k].link_edges) for i,k in zip(Lidx,j))
print(f"leg verts L/R {len(Lidx)}/{len(Ridx)} | bijective {len(set(j.tolist()))==len(Lidx)==len(Ridx)} | valences equal {val_ok} | max mirror dist {d.max():.5f}")
def uvstats(side):
    r=[]; big=0
    for f in bm.faces:
        c=np.mean([B[v.index] for v in f.verts],0)
        if side*c[0]>0 and c[2]<1.2:
            uv=np.array([lp[uvl].uv[:] for lp in f.loops]); p=B[[v.index for v in f.verts]]
            a3=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0])); a2=abs(np.cross(uv[1]-uv[0],uv[2]-uv[0]))
            r.append(a2/(a3+1e-12)); big+= np.ptp(uv,0).max()>0.1
    r=np.array(r); return len(r), np.median(r), np.percentile(r,99), r.max(), big
for s,n in ((1,'LEFT'),(-1,'RIGHT')):
    c,med,p99,mx,big=uvstats(s); print(f"{n}: faces {c} UV/3D area ratio median {med:.4f} p99 {p99:.4f} max {mx:.4f} | faces with UV span>0.1: {big}")
