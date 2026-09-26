"""Lengthen legs/arms via bone stretch (weights untouched), rebase rest pose,
scale to stud units and export an FBX Roblox imports at 1 unit = 1 stud."""
import bpy, sys, numpy as np
from mathutils import Matrix
SRC, DST = sys.argv[1], sys.argv[2]
K_LEG, K_ARM, STUDS = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=SRC)
arm = bpy.data.objects['Armature']
me = [o for o in bpy.data.objects if o.type == 'MESH'][0]
mod = [m for m in me.modifiers if m.type == 'ARMATURE'][0]
assert not mod.use_deform_preserve_volume
# source rig error: some thigh vertices were weighted to the OTHER leg's bones (left leg: 51 verts,
# up to 0.37). Each leg must only follow its own bones: drop cross-leg weights and renormalise.
_names = {g.index: g.name.replace('mixamorig:', '') for g in me.vertex_groups}
_leg = lambda n, s: n.startswith(s) and any(k in n for k in ('UpLeg', 'Leg', 'Foot', 'Toe'))
_fixed = 0
for v in me.data.vertices:
    if abs(v.co.x) < 0.01: continue  # crotch seam on the centre line is shared by both legs
    side = 'Left' if v.co.x > 0 else 'Right'; other = 'Right' if side == 'Left' else 'Left'
    bad = [g for g in v.groups if _leg(_names[g.group], other)]
    if not bad: continue
    keep = [(g.group, g.weight) for g in v.groups if g not in bad]
    tot = sum(w for _, w in keep)
    for g in bad: me.vertex_groups[g.group].remove([v.index])
    for gi, w in keep: me.vertex_groups[gi].add([v.index], w / tot, 'REPLACE')
    _fixed += 1
print("cross-leg weights removed on", _fixed, "verts")
action = arm.animation_data.action
slot = getattr(arm.animation_data, 'action_slot', None)
arm.animation_data.action = None
for pb in arm.pose.bones:
    pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0)
    pb.rotation_euler = (0, 0, 0); pb.scale = (1, 1, 1)
P = 'mixamorig:'
stretch = {}
for s in ('Left', 'Right'):
    stretch.update({P+s+'UpLeg': K_LEG, P+s+'Leg': K_LEG, P+s+'Arm': K_ARM, P+s+'ForeArm': K_ARM})
no_inherit = [P+s+n for s in ('Left', 'Right') for n in ('Leg', 'Foot', 'ForeArm', 'Hand')]
saved = {b.name: b.inherit_scale for b in arm.data.bones}
for n in no_inherit: arm.data.bones[n].inherit_scale = 'NONE'
for n, k in stretch.items(): arm.pose.bones[n].scale = (1, k, 1)
bpy.context.view_layer.update()
# linear blend skinning, same as Blender's armature modifier (normalized weights)
names = [g.name for g in me.vertex_groups]
defm = {}
for pb in arm.pose.bones:
    defm[pb.name] = np.array(pb.matrix @ pb.bone.matrix_local.inverted())
nv = len(me.data.vertices)
W = []
for v in me.data.vertices:
    W.append([(names[g.group], g.weight) for g in v.groups if names[g.group] in defm and g.weight > 0])
def skin(co):
    out = co.copy()
    for i, ws in enumerate(W):
        tot = sum(w for _, w in ws)
        if tot <= 0: continue
        M = sum(w * defm[n] for n, w in ws) / tot
        out[i] = M[:3, :3] @ co[i] + M[:3, 3]
    return out
# sanity: our skinning matches Blender's evaluation on the basis
dg = bpy.context.evaluated_depsgraph_get(); ev = me.evaluated_get(dg).to_mesh()
blender_def = np.array([v.co[:] for v in ev.vertices]); me.evaluated_get(dg).to_mesh_clear()
keys = me.data.shape_keys.key_blocks if me.data.shape_keys else []
for kb in keys: kb.value = 0.0
base = np.array([v.co[:] for v in me.data.vertices])
print("LBS vs blender maxdiff:", np.abs(skin(base) - blender_def).max())
for kb in keys:
    co = np.array([p.co[:] for p in kb.data]); new = skin(co)
    for i, p in enumerate(kb.data): p.co = new[i]
newbase = skin(base)
for i, v in enumerate(me.data.vertices): v.co = newbase[i]
me.data.update()
# pose -> rest
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode='OBJECT')
for n, v in saved.items(): arm.data.bones[n].inherit_scale = v
for pb in arm.pose.bones: pb.scale = (1, 1, 1)
# feet back to ground, then scale to studs (data-level, object transforms stay identity)
zmin = min(v.co.z for v in me.data.vertices)
T = Matrix.Scale(STUDS, 4) @ Matrix.Translation((0, 0, -zmin))
me.data.transform(T)
for kb in keys:
    for p in kb.data: p.co = T @ p.co
arm.data.transform(T)
me.data.update(); bpy.context.view_layer.update()
arm.animation_data.action = action
if slot is not None:
    try: arm.animation_data.action_slot = slot
    except Exception: pass
def fcurves(act):
    try: return list(act.fcurves)
    except AttributeError:
        return [fc for l in act.layers for s in l.strips for b in s.channelbags for fc in b.fcurves]
for fc in fcurves(action):
    if fc.data_path.endswith('.location'):
        for k in fc.keyframe_points:
            k.co.y *= STUDS; k.handle_left.y *= STUDS; k.handle_right.y *= STUDS
    if fc.data_path.endswith('.scale'):
        vals = {round(k.co.y, 4) for k in fc.keyframe_points}
        if vals != {1.0}: print("WARN scale keys", fc.data_path, vals)
print("scales", tuple(arm.scale), tuple(me.scale), "loc", tuple(arm.location))
# Avatar Setup mouthparts (teeth + tongue) and watertight fixes
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from rbx_mouthparts import add_mouthparts, enlarge_mouth
add_mouthparts(me.data, me.vertex_groups['mixamorig:Head'].index, [k.name for k in keys], 'BOCA AAAA')
import os, json
if os.environ.get('MOUTH', 'off') != 'off':  # v4-v6 enlarged the mouth; v7 keeps the original design
    enlarge_mouth(me.data, 'BOCA AAAA', **json.loads(os.environ['MOUTH']))
# left leg topology differed from the right one (source mesh artifact): mirror it
from rbx_kneepatch import mirror_left_leg
mirror_left_leg(me.data, {g.name: g.index for g in me.vertex_groups})
# source custom normals are exactly smooth vertex normals: drop them so new/edited faces shade the same
for name in ('custom_normal', 'sharp_edge', 'sharp_face'):
    if name in me.data.attributes: me.data.attributes.remove(me.data.attributes[name])
for p in me.data.polygons: p.use_smooth = True
me.data.update()
# R15 rig + hand-authored FACS face (Avatar Setup keeps it instead of generating one)
from rbx_r15face import build
arm, poses = build(arm, me)
# Roblox convention: Blender unit scale 0.01 -> FBX values in studs
bpy.context.scene.unit_settings.scale_length = 0.01
bpy.ops.object.select_all(action='DESELECT'); arm.select_set(True); me.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.export_scene.fbx(filepath=DST, use_selection=True, object_types={'ARMATURE', 'MESH'},
    apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE', add_leaf_bones=False,
    use_custom_props=True, bake_anim=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=False, bake_anim_simplify_factor=0.0, path_mode='COPY', embed_textures=True)
