import bpy, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=sys.argv[1], anim_offset=0)
m=[o for o in bpy.data.objects if o.type=='MESH'][0]
names={m[f'Frame{i}']:i for i in range(0,21)}
Fr=np.load('facs_frames.npy'); base=Fr[0]; D=lambda n: Fr[names[n]]-base
polys=[list(p.vertices) for p in m.data.polygons]
tests={
 'Neutral':{},
 'Parpadeo izq (LeftEyeClosed+EyesLookDown)':{'LeftEyeClosed':1,'EyesLookDown':1},
 'Parpadeo der (RightEyeClosed+EyesLookDown)':{'RightEyeClosed':1,'EyesLookDown':1},
 'Boca abierta (JawDrop)':{'JawDrop':1},
 'Feliz (Pucker+LipCornerPuller L/R)':{'Pucker':1,'LeftLipCornerPuller':1,'RightLipCornerPuller':1},
 'Triste (combinación oficial)':{'ChinRaiser':1,'ChinRaiserUpperLip':1,'LeftCheekRaiser':.85,'RightCheekRaiser':.85,'LeftInnerBrowRaiser':1,'RightInnerBrowRaiser':1,'LeftLipCornerDown':1,'RightLipCornerDown':1,'LeftLowerLipDepressor':1,'RightLowerLipDepressor':1},
}
def normals(P):
    out=[]
    for p in polys:
        q=P[p]; out.append(np.cross(q[1]-q[0],q[2]-q[0]))
    return np.array(out)
n0=normals(base)
import bmesh
bm=bmesh.new(); bm.from_mesh(m.data); bm.verts.ensure_lookup_table()
seen=set(); eyeset=set()
for v in bm.verts:
    if v.index in seen: continue
    st=[v]; c=[]; seen.add(v.index)
    while st:
        x=st.pop(); c.append(x.index)
        for ed in x.link_edges:
            o=ed.other_vert(x)
            if o.index not in seen: seen.add(o.index); st.append(o)
    if len(c)==72: eyeset|=set(c)
# landmark proxies: mouth lip ring and eye domes (the regions Roblox projects landmarks onto)
kb=m.data.shape_keys.key_blocks
mw=np.array(m.matrix_world)
Bk=(np.array([p.co[:] for p in kb['Basis'].data])@mw[:3,:3].T+mw[:3,3])*100
Ok=(np.array([p.co[:] for p in kb['BOCA AAAA'].data])@mw[:3,:3].T+mw[:3,3])*100
lip=np.where(np.linalg.norm(Ok-Bk,axis=1)>1e-5)[0]
cL=lip[np.argmax(base[lip,0])]; cR=lip[np.argmin(base[lip,0])]
up=lip[(Ok-Bk)[lip,2]>0]; lo=lip[(Ok-Bk)[lip,2]<0]
fig,axs=plt.subplots(2,3,figsize=(16,10.5),dpi=95); axs=axs.ravel()
for ax,(t,combo) in zip(axs,tests.items()):
    P=base+sum((w*D(k) for k,w in combo.items()), np.zeros_like(base))
    nn=normals(P); flips=int(((nn*n0).sum(1)<0).sum())
    top_e=lambda s: None
    msg=""
    if 'Parpadeo' in t:
        side = 1 if 'izq' in t else -1
        e=np.where((side*base[:,0]>0.1)&(np.hypot(np.abs(base[:,0])-0.308, base[:,2]-3.94)<0.235)&(base[:,1]<-0.395)&(base[:,1]>-0.63))[0]
        e=[i for i in e if i in eyeset]
        cx=side*0.308; sk=[i for i in range(len(base)) if i not in eyeset and abs(base[i,0]-cx)<0.08 and base[i,1]<-0.35 and 0.2<abs(base[i,2]-3.94)<0.36]
        upk=[i for i in sk if base[i,2]>3.94]; lok=[i for i in sk if base[i,2]<3.94]
        skin_gap0=base[upk,2].mean()-base[lok,2].mean(); skin_gap1=P[upk,2].mean()-P[lok,2].mean()
        msg=f"globo {np.ptp(base[e,2]):.2f}->{np.ptp(P[e,2]):.2f} | piel arriba-abajo {skin_gap0:.2f}->{skin_gap1:.2f}"
    if 'Boca abierta' in t:
        cu=up[np.abs(base[up,0])<0.03]; cl=lo[np.abs(base[lo,0])<0.03]
        msg=f"apertura en el centro {base[cu,2].min()-base[cl,2].max():+.3f} -> {P[cu,2].min()-P[cl,2].max():+.3f}"
    if 'Feliz' in t or 'Triste' in t: msg=f"comisuras dz L {P[cL,2]-base[cL,2]:+.3f} R {P[cR,2]-base[cR,2]:+.3f}"
    print(f"{t:44} caras invertidas: {flips} | {msg}")
    vis=[p for p in polys if P[p][:,1].mean()<-0.2 and abs(P[p][:,0].mean())<0.95 and 3.1<P[p][:,2].mean()<4.75]
    vis.sort(key=lambda p:-P[p][:,1].mean())
    L=np.array([0.3,-0.8,0.5]); L/=np.linalg.norm(L)
    cols=[]
    for p in vis:
        q=P[p]; nv=np.cross(q[1]-q[0],q[2]-q[0]); nv/=np.linalg.norm(nv)+1e-12; sh=0.35+0.65*max(0,nv@L)
        cols.append((0.95*sh,0.62*sh,0.32*sh))
    ax.add_collection(PolyCollection([P[p][:,[0,2]] for p in vis],facecolors=cols,edgecolors='none'))
    ax.set_xlim(-0.95,0.95); ax.set_ylim(3.1,4.75); ax.set_aspect('equal'); ax.set_title(t+"\n"+msg,fontsize=10); ax.axis('off')
plt.tight_layout(); plt.savefig(sys.argv[2])
