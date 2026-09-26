import bpy, sys
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=sys.argv[1])
sc=bpy.context.scene
print("unit scale", sc.unit_settings.scale_length)
for o in bpy.data.objects:
    line=f"{o.name:30} {o.type:9} parent={o.parent.name if o.parent else None} ptype={o.parent_type} pbone={o.parent_bone} loc={tuple(round(v,3) for v in o.location)} rot={tuple(round(v,3) for v in o.rotation_euler)} scl={tuple(round(v,3) for v in o.scale)}"
    if o.type=='MESH':
        ws=[o.matrix_world@Vector(c) for c in o.bound_box]
        mn=[min(w[i] for w in ws) for i in range(3)]; mx=[max(w[i] for w in ws) for i in range(3)]
        line+=f" size={[round(mx[i]-mn[i],3) for i in range(3)]} min={[round(v,3) for v in mn]} v={len(o.data.vertices)} vg={[g.name for g in o.vertex_groups][:6]} mods={[m.type for m in o.modifiers]}"
    print(line)
    if o.type=='ARMATURE':
        for b in o.data.bones:
            print("   bone",b.name,"parent",b.parent.name if b.parent else None,"head",tuple(round(v,3) for v in (o.matrix_world@b.head_local)),"tail",tuple(round(v,3) for v in (o.matrix_world@b.tail_local)))
for a in bpy.data.actions: print("action",a.name)
