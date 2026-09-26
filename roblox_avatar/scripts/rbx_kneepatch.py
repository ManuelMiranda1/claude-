"""Replace the asymmetric patch on the left inner knee with a mirror copy of the right knee,
so Avatar Setup processes both lower legs identically."""
import bmesh, numpy as np
from mathutils import Vector
from scipy.spatial import cKDTree


def _boundary(faces):
    fs = set(faces); out = set()
    for f in faces:
        for e in f.edges:
            if sum(1 for g in e.link_faces if g in fs) == 1:
                out.update(v.index for v in e.verts)
    return out


def mirror_knee_patch(me, vgroups, center_right, radius=0.13, tol=0.02):
    bm = bmesh.new(); bm.from_mesh(me)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify(); uvl = bm.loops.layers.uv.active
    shapes = list(bm.verts.layers.shape.values())
    co = np.array([v.co[:] for v in bm.verts]); tree = cKDTree(co)
    cR = np.array(center_right); cL = cR * [-1, 1, 1]

    # right patch and its left counterparts
    fR = [f for f in bm.faces if np.linalg.norm(np.array(f.calc_center_median()[:]) - cR) < radius]
    vR = {v.index for f in fR for v in f.verts}
    bR = _boundary(fR)
    corr = {}
    for i in vR:
        d, j = tree.query(co[i] * [-1, 1, 1])
        if d < tol and co[j, 0] > 0:
            corr[i] = int(j)
    missing = [i for i in bR if i not in corr]
    assert not missing, f"patch boundary without counterpart: {missing}"
    bL = {corr[i] for i in bR}

    # left patch: faces made only of mapped verts plus left verts that have no mirror partner
    dl, _ = cKDTree(co * [-1, 1, 1]).query(co)
    extra = {i for i in range(len(co)) if co[i, 0] > 0 and dl[i] >= tol
             and np.linalg.norm(co[i] - cL) < radius + 0.1}
    SL = {corr[i] for i in vR if i in corr} | extra
    fL = [f for f in bm.faces if all(v.index in SL for v in f.verts)]
    assert _boundary(fL) == bL, "left/right patch boundaries differ"
    nL = len({v.index for f in fL for v in f.verts})
    print(f"knee patch: right {len(fR)} faces/{len(vR)} verts, left {len(fL)} faces/{nL} verts")

    # snapshot right-side data before editing
    right_faces = [[v.index for v in f.verts] for f in fR]
    right_w = {i: dict(bm.verts[i][deform]) for i in vR}
    left_uv = {}
    for f in fL:
        for lp in f.loops: left_uv.setdefault(lp.vert.index, lp[uvl].uv.copy())
    swap = {g: vgroups[n.replace('Right', 'Left')] for n, g in vgroups.items()
            if 'Right' in n and n.replace('Right', 'Left') in vgroups}
    keep = {i: bm.verts[i] for i in bL}

    # delete the left patch (faces, then its now-loose interior verts)
    interior = [v for f in fL for v in f.verts if v.index not in bL]
    bmesh.ops.delete(bm, geom=list(fL), context='FACES_ONLY')
    bmesh.ops.delete(bm, geom=list({v for v in interior if v.is_valid and not v.link_faces}), context='VERTS')

    new = {}
    def left_vert(i):
        if i in bR: return keep[corr[i]]
        if i not in new:
            p = Vector((-co[i, 0], co[i, 1], co[i, 2]))
            v = bm.verts.new(p)
            for g, w in right_w[i].items(): v[deform][swap.get(g, g)] = w
            for lay in shapes: v[lay] = p.copy()
            new[i] = v
        return new[i]

    made = []
    for vs in right_faces:
        f = bm.faces.new([left_vert(i) for i in vs][::-1]); made.append(f)
        for lp in f.loops:
            # UV from the left side's own texture: nearest original left patch vertex
            k = min(left_uv, key=lambda k: (Vector(co[k]) - lp.vert.co).length)
            lp[uvl].uv = left_uv[k]
    bmesh.ops.recalc_face_normals(bm, faces=made)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bm.to_mesh(me); bm.free(); me.update()
    print(f"knee patch rebuilt: {len(made)} faces, {len(new)} new verts, non-manifold edges now {nm}")


def mirror_left_leg(me, vgroups, z_top=1.2, tol=0.02):
    """Rebuild every left-leg face below z_top as a mirror copy of the right leg (same topology),
    keeping the left side's own UVs (per face, seam-aware) and swapping Right*->Left* weights."""
    bm = bmesh.new(); bm.from_mesh(me)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify(); uvl = bm.loops.layers.uv.active
    shapes = list(bm.verts.layers.shape.values())
    co = np.array([v.co[:] for v in bm.verts]); tree = cKDTree(co)
    cen = lambda f: co[[v.index for v in f.verts]].mean(0)
    fR = [f for f in bm.faces if cen(f)[0] < 0 and cen(f)[2] < z_top]
    fL = [f for f in bm.faces if cen(f)[0] > 0 and cen(f)[2] < z_top]
    bR, bL = _boundary(fR), _boundary(fL)
    corr = {}
    for i in bR:
        d, j = tree.query(co[i] * [-1, 1, 1])
        assert d < tol and j in bL, f"boundary vertex {i} has no mirrored partner"
        corr[i] = int(j)
    assert set(corr.values()) == bL and len(bR) == len(bL), "leg boundary loops differ"
    # old left loops, for seam-aware UV lookup: (vertex co, face centre, uv)
    old = [(np.array(lp.vert.co[:]), cen(f), lp[uvl].uv.copy()) for f in fL for lp in f.loops]
    old_v = np.array([o[0] for o in old]); old_c = np.array([o[1] for o in old])
    right_faces = [[v.index for v in f.verts] for f in fR]
    right_w = {v.index: dict(v[deform]) for f in fR for v in f.verts}
    swap = {g: vgroups[n.replace('Right', 'Left')] for n, g in vgroups.items()
            if 'Right' in n and n.replace('Right', 'Left') in vgroups}
    keep = {i: bm.verts[i] for i in bL}
    interior = {v for f in fL for v in f.verts if v.index not in bL}
    nfL, nvL = len(fL), len(interior)
    bmesh.ops.delete(bm, geom=fL, context='FACES_ONLY')
    bmesh.ops.delete(bm, geom=[v for v in interior if v.is_valid and not v.link_faces], context='VERTS')
    new = {}
    def left_vert(i):
        if i in corr: return keep[corr[i]]
        if i not in new:
            p = Vector((-co[i, 0], co[i, 1], co[i, 2]))
            v = bm.verts.new(p)
            for g, w in right_w[i].items(): v[deform][swap.get(g, g)] = w
            for lay in shapes: v[lay] = p.copy()
            new[i] = v
        return new[i]
    made = []
    for vs in right_faces:
        f = bm.faces.new([left_vert(i) for i in vs][::-1]); made.append(f)
        c = np.array(f.calc_center_median()[:])
        for lp in f.loops:
            p = np.array(lp.vert.co[:])
            # nearest old loop by vertex position, tie-broken by face centre (keeps UV islands apart)
            k = np.argmin(np.linalg.norm(old_v - p, axis=1) * 4 + np.linalg.norm(old_c - c, axis=1))
            lp[uvl].uv = old[k][2]
    bmesh.ops.recalc_face_normals(bm, faces=made)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bm.to_mesh(me); bm.free(); me.update()
    print(f"left leg mirrored: replaced {nfL} faces/{nvL} verts with {len(made)} faces/{len(new)} verts; non-manifold edges {nm}")
