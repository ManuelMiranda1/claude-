import bpy, sys, numpy as np
from scipy.spatial import cKDTree
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=sys.argv[1], anim_offset=0)
m=[o for o in bpy.data.objects if o.type=='MESH'][0]; names=[g.name for g in m.vertex_groups]
mw=np.array(m.matrix_world); P=(np.array([v.co[:] for v in m.data.vertices])@mw[:3,:3].T+mw[:3,3])*100
W=[{names[g.group].replace('mixamorig:',''):g.weight for g in v.groups} for v in m.data.vertices]
sw=lambda n: n.replace('Left','#').replace('Right','Left').replace('#','Right')
L=np.where((P[:,0]>0.005)&(P[:,2]<1.5))[0]; d,j=cKDTree(P).query(P[L]*[-1,1,1])
bad=[]
for i,k,dd in zip(L,j,d):
    a=W[i]; b={sw(n):w for n,w in W[k].items()}
    diff=max(abs(a.get(n,0)-b.get(n,0)) for n in set(a)|set(b))
    if diff>0.02: bad.append((diff,i,k))
bad.sort(reverse=True)
print(f"leg verts {len(L)} | pairs with weight mismatch >0.02: {len(bad)} | worst {bad[0][0] if bad else 0:.3f}")
for dd,i,k in bad[:6]: print(f"  L{i} z={P[i,2]:.2f} {dict((n,round(w,2)) for n,w in W[i].items())}  <->  R{k} {dict((n,round(w,2)) for n,w in W[k].items())}")
