import bpy, sys, numpy as np
for p in sys.argv[1:]:
    bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=p)
    o=[o for o in bpy.data.objects if o.type=='MESH'][0]; m=o.data; uv=m.uv_layers.active.data
    mw=np.array(o.matrix_world); P=(np.array([v.co[:] for v in m.vertices])@mw[:3,:3].T+mw[:3,3])*100
    out={}
    for side,n in ((1,'L'),(-1,'R')):
        r=[]
        for f in m.polygons:
            vs=list(f.vertices); c=P[vs].mean(0)
            if not (side*c[0]>0 and c[2]<1.2): continue
            U=np.array([uv[i].uv[:] for i in f.loop_indices]); Q=P[vs]
            a3=sum(np.linalg.norm(np.cross(Q[k]-Q[0],Q[k+1]-Q[0])) for k in range(1,len(vs)-1))
            a2=sum(abs(np.cross(U[k]-U[0],U[k+1]-U[0])) for k in range(1,len(vs)-1))
            r.append(a2/a3)
        r=np.array(r); med=np.median(r)
        out[n]=(len(r), int((r<0.25*med).sum()), int((r>4*med).sum()))
    print(p, "faces / collapsed(<25% med) / stretched(>4x med):", out)
