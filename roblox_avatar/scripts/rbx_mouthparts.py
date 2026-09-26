"""Add Avatar Setup mouthparts (upper teeth, lower teeth, tongue) as separate closed
islands inside the existing mouth pocket, and close tiny holes. Called from rbx_fix.py."""
import bmesh, numpy as np
from mathutils import Vector

def _box_strip(bm, xs, zlo, zhi, y0, y1):
    """Closed curved slab: for each x sample, a quad section (y0/y1 x zlo/zhi)."""
    secs = []
    for x, a, b in zip(xs, zlo, zhi):
        secs.append([bm.verts.new((x, y0, a)), bm.verts.new((x, y1, a)),
                     bm.verts.new((x, y1, b)), bm.verts.new((x, y0, b))])
    faces = [bm.faces.new(secs[0][::-1]), bm.faces.new(secs[-1])]
    for s, t in zip(secs, secs[1:]):
        for i in range(4):
            j = (i + 1) % 4
            faces.append(bm.faces.new((s[i], s[j], t[j], t[i])))
    return [v for s in secs for v in s], faces

def add_mouthparts(me, head_group, key_names, open_key):
    co = np.array([v.co[:] for v in me.vertices])
    kb = me.shape_keys.key_blocks
    d = np.array([p.co[:] for p in kb[open_key].data]) - np.array([p.co[:] for p in kb['Basis'].data])
    lip = np.where(np.linalg.norm(d, axis=1) > 1e-6)[0]
    back = lip[co[lip, 1] > np.median(co[lip, 1])]          # inner lip row
    up = back[d[back, 2] > 0]; lo = back[d[back, 2] < 0]    # rows that open up / down
    ux, uz = co[up, 0], co[up, 2]; lx, lz = co[lo, 0], co[lo, 2]
    u = lambda x: np.interp(x, ux[np.argsort(ux)], uz[np.argsort(ux)])
    l = lambda x: np.interp(x, lx[np.argsort(lx)], lz[np.argsort(lx)])
    dz_up, dz_lo = d[up, 2].mean(), d[lo, 2].mean()
    ylip = co[back, 1].mean()
    # pocket floor: verts behind the lips inside the mouth outline
    xs_all = co[:, 0]; inside = (np.abs(xs_all) < np.abs(co[back, 0]).max()) & \
        (co[:, 2] > lz.min() - 0.01) & (co[:, 2] < uz.max() + 0.01) & (co[:, 1] > ylip + 0.05) & (co[:, 1] < ylip + 0.2)
    ycap = co[inside, 1].min()
    depth = ycap - ylip
    print(f"mouth: lip y={ylip:.3f} pocket cap y={ycap:.3f} depth={depth:.3f}")
    bm = bmesh.new(); bm.from_mesh(me)
    # close tiny holes (<= 6 boundary edges) so the body is watertight
    bm.edges.ensure_lookup_table()
    bnd = [e for e in bm.edges if e.is_boundary]
    loops = []; seen = set()
    for e in bnd:
        if e in seen: continue
        comp = set(); st = [e]
        while st:
            x = st.pop()
            if x in seen: continue
            seen.add(x); comp.add(x)
            for v in x.verts: st += [y for y in v.link_edges if y.is_boundary and y not in seen]
        loops.append(comp)
    for comp in loops:
        if len(comp) <= 6:
            bmesh.ops.holes_fill(bm, edges=list(comp), sides=len(comp))
            print("filled hole with", len(comp), "edges")
    n0 = len(bm.verts)
    xs = np.linspace(-0.055, 0.055, 7)
    y_a, y_b = ylip + 0.25 * depth, ylip + 0.45 * depth
    parts = {
        'upper_teeth': (xs, u(xs) - 0.020, u(xs) - 0.007, y_a, y_b, dz_up),
        'lower_teeth': (xs, l(xs) + 0.006, l(xs) + 0.018, y_a, y_b, dz_lo),
        'tongue': (np.linspace(-0.045, 0.045, 7), l(np.linspace(-0.045, 0.045, 7)) + 0.004,
                   l(np.linspace(-0.045, 0.045, 7)) + 0.016, ylip + 0.5 * depth, ylip + 0.85 * depth, dz_lo),
    }
    deform = bm.verts.layers.deform.verify()
    uvl = bm.loops.layers.uv.active
    # UV for new parts: take the pocket cap's UV (mouth interior colour)
    cap_idx = int(np.where(inside)[0][np.argmax(co[inside, 1])])
    bm.verts.ensure_lookup_table()
    cap_uv = bm.verts[cap_idx].link_loops[0][uvl].uv.copy()
    made = {}
    for name, (x, zl, zh, y0, y1, dz) in parts.items():
        vs, fs = _box_strip(bm, x, zl, zh, y0, y1)
        for v in vs: v[deform][head_group] = 1.0
        for f in fs:
            for lp in f.loops: lp[uvl].uv = cap_uv
        made[name] = (vs, dz)
    for kname in key_names:
        lay = bm.verts.layers.shape[kname]
        for name, (vs, dz) in made.items():
            for v in vs:
                v[lay] = v.co.copy()  # parts stay inside the pocket in every key
    bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.verts[0].index == -1 or f.verts[0].index >= n0])
    bm.to_mesh(me); bm.free(); me.update()
    print("mouthparts added:", {k: len(v[0]) for k, v in made.items()})
