"""Convert the Mixamo rig to a Roblox standard R15 rig and add a hand-authored FACS face
(RootFaceJoint 'DynamicHead' + face joints + 17 required poses and 3 eye-look poses),
so Avatar Setup keeps our facial animation instead of generating one."""
import bpy, bmesh, numpy as np
from mathutils import Vector, Quaternion, Matrix

P = 'mixamorig:'

# ---------------------------------------------------------------- R15 body
MERGE = {
    'LowerTorso': ['Hips', 'Spine'],
    'UpperTorso': ['Spine1', 'Spine2'],
    'Head': ['Neck', 'Head', 'HeadTop_End', 'HeadTop_End_end'],
}
for s in ('Left', 'Right'):
    MERGE[s + 'UpperArm'] = [s + 'Shoulder', s + 'Arm']
    MERGE[s + 'LowerArm'] = [s + 'ForeArm']
    MERGE[s + 'Hand'] = [s + 'Hand']  # + every finger bone, added below
    MERGE[s + 'UpperLeg'] = [s + 'UpLeg']
    MERGE[s + 'LowerLeg'] = [s + 'Leg']
    MERGE[s + 'Foot'] = [s + 'Foot', s + 'ToeBase', s + 'Toe_End', s + 'Toe_End_end']


def _islands(me):
    bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
    seen = set(); out = []
    for v in bm.verts:
        if v.index in seen: continue
        st = [v]; comp = []; seen.add(v.index)
        while st:
            x = st.pop(); comp.append(x.index)
            for e in x.link_edges:
                o = e.other_vert(x)
                if o.index not in seen: seen.add(o.index); st.append(o)
        out.append(comp)
    bm.free()
    return out


def build(arm, meobj, open_key='BOCA AAAA'):
    me = meobj.data
    heads = {b.name[len(P):]: b.head_local.copy() for b in arm.data.bones}
    tails = {b.name[len(P):]: b.tail_local.copy() for b in arm.data.bones}
    for n in heads:
        for s in ('Left', 'Right'):
            if n.startswith(s + 'Hand') and n != s + 'Hand': MERGE[s + 'Hand'].append(n)

    # --- merge vertex groups into R15 groups
    old = {g.index: g.name[len(P):] for g in meobj.vertex_groups}
    to_new = {o: new for new, olds in MERGE.items() for o in olds}
    W = np.zeros((len(me.vertices), 0)).tolist()
    acc = [dict() for _ in me.vertices]
    for v in me.vertices:
        for g in v.groups:
            nn = to_new.get(old[g.group])
            assert nn, f"unmapped bone {old[g.group]}"
            acc[v.index][nn] = acc[v.index].get(nn, 0) + g.weight
    for g in list(meobj.vertex_groups): meobj.vertex_groups.remove(g)
    groups = {n: meobj.vertex_groups.new(name=n) for n in MERGE}

    # --- new armature with R15 hierarchy
    arm.name = 'mixamo_old'; arm.data.name = 'mixamo_old'
    ad = bpy.data.armatures.new('Armature'); rig = bpy.data.objects.new('Armature', ad)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    eb = ad.edit_bones
    def bone(name, h, t, parent=None):
        b = eb.new(name); b.head = Vector(h); b.tail = Vector(t); b.roll = 0
        if parent: b.parent = eb[parent]; b.use_connect = False
        return b
    hip = heads['Hips'].copy(); hip.x = 0
    up = Vector((0, 0, 0.25))
    bone('Root', (0, 0, 0), up)
    bone('HumanoidRootNode', hip, hip + up, 'Root')
    bone('LowerTorso', hip, heads['Spine1'] * Vector((0, 1, 1)), 'HumanoidRootNode')
    bone('UpperTorso', heads['Spine1'] * Vector((0, 1, 1)), heads['Neck'] * Vector((0, 1, 1)), 'LowerTorso')
    bone('Head', heads['Neck'] * Vector((0, 1, 1)), heads['HeadTop_End'] * Vector((0, 1, 1)), 'UpperTorso')
    for s in ('Left', 'Right'):
        bone(s + 'UpperArm', heads[s + 'Shoulder'], heads[s + 'ForeArm'], 'UpperTorso')
        bone(s + 'LowerArm', heads[s + 'ForeArm'], heads[s + 'Hand'], s + 'UpperArm')
        bone(s + 'Hand', heads[s + 'Hand'], tails[s + 'HandMiddle4'], s + 'LowerArm')
        bone(s + 'UpperLeg', heads[s + 'UpLeg'], heads[s + 'Leg'], 'LowerTorso')
        bone(s + 'LowerLeg', heads[s + 'Leg'], heads[s + 'Foot'], s + 'UpperLeg')
        bone(s + 'Foot', heads[s + 'Foot'], tails[s + 'Toe_End'], s + 'LowerLeg')

    # --- facial features (studs, armature space == mesh space)
    co = np.array([v.co[:] for v in me.vertices])
    kb = me.shape_keys.key_blocks
    dopen = np.array([p.co[:] for p in kb[open_key].data]) - np.array([p.co[:] for p in kb['Basis'].data])
    lip = np.where(np.linalg.norm(dopen, axis=1) > 1e-6)[0]
    up_lip = lip[dopen[lip, 2] > 0]; lo_lip = lip[dopen[lip, 2] < 0]
    isl = _islands(me)
    body = max(isl, key=len)
    small = [c for c in isl if c is not body]
    eyes = sorted([c for c in small if len(c) > 50], key=lambda c: co[c, 0].mean())  # [Right(-x), Left(+x)]
    parts = sorted([c for c in small if len(c) <= 50], key=lambda c: -co[c, 2].mean())  # upper teeth first
    upper_teeth, lower_parts = parts[0], [i for c in parts[1:] for i in c]
    mouth_c = co[lip].mean(0)
    cornerL = co[lip[np.argmax(co[lip, 0])]]; cornerR = co[lip[np.argmin(co[lip, 0])]]
    mouth_w = cornerL[0] - cornerR[0]
    eye = {}
    for side, c in (('Right', eyes[0]), ('Left', eyes[1])):
        E = co[c]; R = np.ptp(E[:, 0]) / 2
        ctr = np.array([E[:, 0].mean(), E[:, 1].max(), (E[:, 2].max() + E[:, 2].min()) / 2])
        eye[side] = dict(idx=np.array(c), R=R, ctr=ctr)
    front_y = co[body][:, 1].min()
    face_z0 = mouth_c[2] - 0.45
    print(f"face: mouth centre {mouth_c.round(3)} width {mouth_w:.3f}; eye R {eye['Left']['R']:.3f} at {eye['Left']['ctr'].round(3)}")

    # face joints
    headpos = Vector((0, heads['Head'].y, heads['Head'].z + 0.2))
    fup = Vector((0, 0, 0.08))
    bone('DynamicHead', headpos, headpos + Vector((0, 0, 0.25)), 'Head')
    F = {}
    def fbone(name, p):
        p = Vector(p); bone(name, p, p + fup, 'DynamicHead'); F[name] = np.array(p[:])
    for s in ('Left', 'Right'):
        e = eye[s]; c = e['ctr']
        fbone(s + 'Eye', c)
        fbone(s + 'UpperLid', c + [0, -0.01, e['R'] * 0.5])
        fbone(s + 'LowerLid', c + [0, -0.01, -e['R'] * 0.5])
        sg = 1 if s == 'Left' else -1
        fbone(s + 'LipCorner', (cornerL if s == 'Left' else cornerR))
        fbone(s + 'LowerLip', mouth_c + [sg * mouth_w * 0.25, 0, -0.03])
        fbone(s + 'Cheek', [sg * (abs(c[0]) + 0.02), c[1] - 0.1, (c[2] + mouth_c[2]) / 2 - 0.05])
        fbone(s + 'InnerBrow', [sg * max(abs(c[0]) - e['R'] * 0.6, 0.06), c[1] - 0.1, c[2] + e['R'] * 1.25])
    fbone('Jaw', [0, mouth_c[1] + 0.35, mouth_c[2] + 0.03])
    fbone('UpperLip', mouth_c + [0, 0, 0.04])
    fbone('Chin', [0, mouth_c[1] + 0.02, mouth_c[2] - 0.2])
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- face weights: partition of the Head weight by smooth influences
    def fall(d, r):
        t = np.clip(1 - d / r, 0, 1); return t * t * (3 - 2 * t)
    n = len(co)
    infl = {}
    front = np.clip((-(co[:, 1] - (front_y + 0.45))) / 0.15, 0, 1)  # 1 on the face, 0 towards the back
    for s in ('Left', 'Right'):
        e = eye[s]; c = e['ctr']; R = e['R']; idx = e['idx']
        t = np.clip((co[idx, 2] - c[2]) / R, -1, 1)
        u = np.zeros(n); l = np.zeros(n); look = np.zeros(n)
        u[idx] = np.clip(t, 0, 1); l[idx] = np.clip(-t, 0, 1); look[idx] = 1 - np.abs(t)
        # skin ring around the eye follows the lids a little
        ring = np.hypot(co[:, 0] - c[0], co[:, 2] - c[2])
        skin = np.ones(n, bool); skin[idx] = False
        k = fall(np.abs(ring - R * 1.1), R * 0.8) * 0.18 * front * skin
        u += np.where(co[:, 2] > c[2], k, 0); l += np.where(co[:, 2] <= c[2], k, 0)
        infl[s + 'UpperLid'] = u; infl[s + 'LowerLid'] = l; infl[s + 'Eye'] = look
    d = lambda p, sx=1.0, sz=1.0: np.sqrt(((co[:, 0] - p[0]) / sx) ** 2 + ((co[:, 2] - p[2]) / sz) ** 2)
    notEye = np.ones(n); notEye[eye['Left']['idx']] = 0; notEye[eye['Right']['idx']] = 0
    face = front * notEye
    half = mouth_w / 2
    jaw = np.zeros(n); jaw[lo_lip] = 0.8 * np.clip(1 - (np.abs(co[lo_lip, 0] - mouth_c[0]) / half) ** 2, 0, 1); jaw[lower_parts] = 1.0
    below = co[:, 2] < mouth_c[2] - 0.01
    jaw += fall(d(mouth_c + [0, 0, -0.08], 1.0, 0.8), 0.32) * below * face * 0.8
    infl['Jaw'] = jaw
    for s, cr in (('Left', cornerL), ('Right', cornerR)):
        sg = 1 if s == 'Left' else -1
        infl[s + 'LipCorner'] = fall(d(cr), max(0.24, mouth_w * 1.1)) * face
        side = np.clip(0.5 + sg * (co[:, 0] - mouth_c[0]) / max(mouth_w, 1e-3), 0, 1)
        ll = np.zeros(n); ll[lo_lip] = 0.4 * side[lo_lip]
        ll += fall(d(mouth_c + [sg * mouth_w * 0.3, 0, -0.06], 1.0, 0.8), 0.16) * below * face * side * 0.7
        infl[s + 'LowerLip'] = ll
        infl[s + 'Cheek'] = fall(d(F[s + 'Cheek']), 0.28) * face
        infl[s + 'InnerBrow'] = fall(d(F[s + 'InnerBrow']), 0.25) * face
    ul = np.zeros(n); ul[up_lip] = 1.0; ul[upper_teeth] = 0.5
    ul += fall(d(mouth_c + [0, 0, 0.05], 1.0, 0.7), 0.18) * (co[:, 2] > mouth_c[2]) * face * 0.6
    infl['UpperLip'] = ul
    infl['Chin'] = fall(d(F['Chin']), 0.22) * face * below
    order = ['LeftEye', 'LeftUpperLid', 'LeftLowerLid', 'RightEye', 'RightUpperLid', 'RightLowerLid',
             'Jaw', 'LeftLowerLip', 'RightLowerLip', 'UpperLip', 'LeftLipCorner', 'RightLipCorner',
             'Chin', 'LeftCheek', 'RightCheek', 'LeftInnerBrow', 'RightInnerBrow']
    assert set(order) == set(infl)
    names = order
    M = np.zeros((len(names), n)); rem = np.ones(n)
    for k, nm in enumerate(names):  # each joint takes its share of what is still free
        r = np.clip(infl[nm], 0, 1); r[r < 1e-4] = 0
        M[k] = r * rem; rem = rem - M[k]
    for nm in names: groups[nm] = meobj.vertex_groups.new(name=nm)
    head_w = np.array([acc[i].get('Head', 0.0) for i in range(n)])
    for i in range(n):
        hw = head_w[i]
        facew = 0.0
        if hw > 0:
            for k, nm in enumerate(names):
                w = M[k, i] * hw
                if w > 1e-4: acc[i][nm] = w; facew += w
            acc[i]['Head'] = hw - facew
        # keep at most 4 influences, renormalise
        top = sorted(((w, g) for g, w in acc[i].items() if w > 1e-4), reverse=True)[:4]
        s_ = sum(w for w, _ in top)
        for w, g in top: groups[g].add([i], w / s_, 'REPLACE')

    # --- rebind mesh to the new rig, drop the old one and all old animation
    for m in meobj.modifiers:
        if m.type == 'ARMATURE': m.object = rig
    meobj.parent = rig
    for ob in (arm, meobj):
        if ob.animation_data: ob.animation_data_clear()
    if me.shape_keys and me.shape_keys.animation_data: me.shape_keys.animation_data_clear()
    for k in kb: k.value = 0.0
    old_data = arm.data; bpy.data.objects.remove(arm); bpy.data.armatures.remove(old_data)
    for a in list(bpy.data.actions): bpy.data.actions.remove(a)

    # --- FACS poses (world-space deltas in studs / rotations in degrees)
    Rl = eye['Left']['R']; mw = mouth_w
    OUT = lambda s: 1 if s == 'Left' else -1
    def pose_eyeclose(s):
        R = eye[s]['R']
        return {s + 'UpperLid': dict(t=(0, 0, -R)), s + 'LowerLid': dict(t=(0, 0, R * 0.9))}
    poses = [
        ('EyesLookDown', {'LeftEye': dict(r=('X', 18)), 'RightEye': dict(r=('X', 18))}),
        ('EyesLookUp', {'LeftEye': dict(r=('X', -18)), 'RightEye': dict(r=('X', -18))}),
        ('EyesLookLeft', {'LeftEye': dict(r=('Z', 20)), 'RightEye': dict(r=('Z', 20))}),
        ('EyesLookRight', {'LeftEye': dict(r=('Z', -20)), 'RightEye': dict(r=('Z', -20))}),
        ('LeftEyeClosed', pose_eyeclose('Left')),
        ('RightEyeClosed', pose_eyeclose('Right')),
        ('JawDrop', {'Jaw': dict(r=('X', 25))}),
        ('Pucker', {'LeftLipCorner': dict(t=(-mw * 0.18, -0.03, 0)), 'RightLipCorner': dict(t=(mw * 0.18, -0.03, 0)),
                    'UpperLip': dict(t=(0, -0.03, 0)), 'LeftLowerLip': dict(t=(0, -0.03, 0)), 'RightLowerLip': dict(t=(0, -0.03, 0))}),
    ]
    for s in ('Left', 'Right'):
        o = OUT(s)
        poses += [
            (s + 'LipCornerPuller', {s + 'LipCorner': dict(t=(o * mw * 0.22, 0.02, mw * 0.35)), s + 'Cheek': dict(t=(0, -0.01, 0.03))}),
            (s + 'LipCornerDown', {s + 'LipCorner': dict(t=(o * mw * 0.05, 0, -mw * 0.3))}),
            (s + 'LowerLipDepressor', {s + 'LowerLip': dict(t=(0, -0.01, -mw * 0.3))}),
            (s + 'CheekRaiser', {s + 'Cheek': dict(t=(0, -0.02, 0.07)), s + 'LowerLid': dict(t=(0, 0, Rl * 0.3))}),
            (s + 'InnerBrowRaiser', {s + 'InnerBrow': dict(t=(0, 0, 0.08))}),
        ]
    poses += [
        ('ChinRaiser', {'Chin': dict(t=(0, -0.02, 0.06)), 'LeftLowerLip': dict(t=(0, -0.01, 0.03)), 'RightLowerLip': dict(t=(0, -0.01, 0.03))}),
        ('ChinRaiserUpperLip', {'UpperLip': dict(t=(0, -0.02, 0.05))}),
    ]
    required = ['LeftEyeClosed', 'RightEyeClosed', 'EyesLookDown', 'JawDrop', 'Pucker', 'LeftLipCornerPuller',
                'RightLipCornerPuller', 'ChinRaiser', 'ChinRaiserUpperLip', 'LeftCheekRaiser', 'RightCheekRaiser',
                'LeftInnerBrowRaiser', 'RightInnerBrowRaiser', 'LeftLipCornerDown', 'RightLipCornerDown',
                'LeftLowerLipDepressor', 'RightLowerLipDepressor']
    assert set(required) <= {p for p, _ in poses}

    face_bones = ['DynamicHead'] + names
    rig.animation_data_create()
    act = bpy.data.actions.new('FACS'); rig.animation_data.action = act
    for pb in rig.pose.bones: pb.rotation_mode = 'QUATERNION'
    def key_all(frame):
        for bn in face_bones:
            pb = rig.pose.bones[bn]
            pb.keyframe_insert('location', frame=frame); pb.keyframe_insert('rotation_quaternion', frame=frame)
    def reset():
        for pb in rig.pose.bones: pb.location = (0, 0, 0); pb.rotation_quaternion = (1, 0, 0, 0)
    reset(); key_all(0)
    meobj['RootFaceJoint'] = 'DynamicHead'; meobj['Frame0'] = 'Neutral'
    for f, (pname, spec) in enumerate(poses, start=1):
        reset()
        for bn, tr in spec.items():
            pb = rig.pose.bones[bn]; B = pb.bone.matrix_local.to_3x3()
            if 't' in tr: pb.location = B.inverted() @ Vector(tr['t'])
            if 'r' in tr:
                ax, deg = tr['r']
                qw = Quaternion(Vector({'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[ax]), np.radians(deg))
                pb.rotation_quaternion = (B.to_quaternion().inverted() @ qw @ B.to_quaternion())
        key_all(f)
        meobj[f'Frame{f}'] = pname
    for fc in (act.fcurves if hasattr(act, 'fcurves') else [c for l in act.layers for st in l.strips for b in st.channelbags for c in b.fcurves]):
        for k in fc.keyframe_points: k.interpolation = 'CONSTANT'
    sc = bpy.context.scene; sc.frame_start = 0; sc.frame_end = len(poses); sc.frame_set(0)
    print(f"R15 rig: {len(rig.data.bones)} bones ({len(face_bones)} face); FACS poses: {len(poses)} + Neutral")
    return rig, poses
