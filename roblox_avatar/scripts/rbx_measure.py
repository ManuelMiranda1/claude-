import bpy, sys, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=sys.argv[1]); f=float(sys.argv[2])
arm=bpy.data.objects['Armature']; me=[o for o in bpy.data.objects if o.type=='MESH'][0]
arm.animation_data.action=None
for pb in arm.pose.bones: pb.matrix_basis.identity()
bpy.context.view_layer.update()
co=np.array([(me.matrix_world@v.co)[:] for v in me.data.vertices])*f
H=lambda n: np.array((arm.matrix_world@arm.data.bones['mixamorig:'+n].head_local)[:])*f
print(f"TOTAL  width={np.ptp(co[:,0]):.2f} height={np.ptp(co[:,2]):.2f} depth={np.ptp(co[:,1]):.2f}  zmin={co[:,2].min():.3f}")
hipz=H('LeftUpLeg')[2]; neckz=H('Neck')[2]
for s,sg in (('Left',1),('Right',-1)):
    leg=co[(co[:,2]<hipz)&(sg*co[:,0]>0)]
    x0=abs(H(s+'Shoulder')[0]); a=co[sg*co[:,0]>x0]
    print(f"{s}Leg  Y={np.ptp(leg[:,2]):.2f} X={np.ptp(leg[:,0]):.2f} Z={np.ptp(leg[:,1]):.2f}   {s}Arm length(Y)={np.ptp(a[:,0]):.2f} thick={np.ptp(a[:,2]):.2f}")
hd=co[co[:,2]>neckz]; print(f"Head   X={np.ptp(hd[:,0]):.2f} Y={np.ptp(hd[:,2]):.2f} Z={np.ptp(hd[:,1]):.2f}")
t=co[(co[:,2]>=hipz)&(co[:,2]<=neckz)&(np.abs(co[:,0])<abs(H('LeftShoulder')[0]))]
print(f"Torso  X={np.ptp(t[:,0]):.2f} Y={np.ptp(t[:,2]):.2f} Z={np.ptp(t[:,1]):.2f}")
