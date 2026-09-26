import bpy, sys
from mathutils import Matrix
SRC, DST, S = sys.argv[1], sys.argv[2], float(sys.argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=SRC)
arm = bpy.data.objects['Armature']
mesh = [o for o in bpy.data.objects if o.type == 'MESH']
# go to rest pose frame reference; scale uniformly about world origin (feet at z~0)
arm.scale = (S, S, S)
bpy.context.view_layer.update()
bpy.ops.object.select_all(action='DESELECT')
for o in [arm] + mesh:
    o.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
# bone translation keys are in (now scaled) bone space units -> scale them too
for act in bpy.data.actions:
    for fc in getattr(act, 'fcurves', []) or []:
        pass
def all_fcurves(act):
    try:
        return list(act.fcurves)
    except AttributeError:
        out = []
        for layer in act.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    out += list(bag.fcurves)
        return out
ad = arm.animation_data
acts = {ad.action} if ad and ad.action else set()
for act in acts:
    for fc in all_fcurves(act):
        if fc.data_path.startswith('pose.bones') and fc.data_path.endswith('.location'):
            for k in fc.keyframe_points:
                k.co.y *= S; k.handle_left.y *= S; k.handle_right.y *= S
print("scales:", [(o.name, tuple(o.scale)) for o in [arm] + mesh])
bpy.ops.object.select_all(action='DESELECT')
for o in [arm] + mesh: o.select_set(True)
bpy.ops.export_scene.fbx(filepath=DST, use_selection=True, object_types={'ARMATURE', 'MESH'},
    apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE', add_leaf_bones=False,
    use_armature_deform_only=False, bake_anim=True, path_mode='COPY', embed_textures=True)
