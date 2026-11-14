"""
03_valves_chordae.py
Complete valvular apparatus, built as thin solid (two-sided, closed) leaflets fitted to the real
orifices of the stage-02 mesh:
  * mitral (anterior + scalloped posterior leaflet) and tricuspid (anterior / posterior / septal)
    valves: continuous skirt hinged on the fibrous annulus, coaptation zone, saddle-shaped annulus;
  * aortic and pulmonary valves: three semilunar cusps each (swallow's-nest pockets), lunulae and
    nodules of Arantius / Morgagni;
  * chordae tendineae: Bezier trees (1st -> 2nd -> 3rd order) from the papillary-muscle tips to the
    free margin and rough zone of the leaflets, plus direct septal chordae for the tricuspid.
Rest pose = every valve closed.  Shape keys: `av_valves_open`, `semilunar_valves_open`.
Leaflets are collision-projected against the myocardial SDF in both poses (no interpenetration).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heartlib import *      # noqa
from scipy import ndimage as ndi
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
g = Grid()
SOL = load('solid02', mmap=True)
log('stage 03')


def solid(P):
    P = np.asarray(P, float)
    return g.sample(SOL, P.reshape(-1, 3)).reshape(P.shape[:-1])


def ray_wall(org, dirs, rmax=2.4, step=0.02, r0=0.25):
    """Distance from each origin along its direction to the first tissue crossing."""
    ts = np.arange(r0, rmax, step)
    P = org[:, None, :] + ts[None, :, None] * dirs[:, None, :]
    s = solid(P)
    inside = s < 0
    first = np.argmax(inside, axis=1)
    none = ~inside.any(axis=1)
    i1 = np.clip(first, 1, len(ts) - 1)
    ar = np.arange(len(org))
    s0, s1 = s[ar, i1 - 1], s[ar, i1]
    d = ts[i1 - 1] + np.clip(s0 / (s0 - s1 + 1e-9), 0, 1) * step
    d[none] = np.nan
    return d


def bezier(B0, B1, B2, B3, vs):
    v = vs[None, :, None]
    return ((1 - v) ** 3 * B0[:, None, :] + 3 * (1 - v) ** 2 * v * B1[:, None, :]
            + 3 * (1 - v) * v ** 2 * B2[:, None, :] + v ** 3 * B3[:, None, :])


def arclen(P):
    return np.linalg.norm(np.diff(P, axis=1), axis=2).sum(axis=1)


def grid_normals(P, closed_u):
    du = (np.roll(P, -1, 0) - np.roll(P, 1, 0)) if closed_u else np.gradient(P, axis=0)
    dv = np.gradient(P, axis=1)
    n = np.cross(du, dv)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    if closed_u:
        n = ndi.gaussian_filter1d(n, 1.0, axis=0, mode='wrap')
    else:
        n = ndi.gaussian_filter1d(n, 1.0, axis=0, mode='nearest')
    n = ndi.gaussian_filter1d(n, 1.0, axis=1, mode='nearest')
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    return n


def shell(states, thick, closed_u):
    """Solidify a (nu, nv) grid surface given in several poses.  Returns [verts per pose], quads, uv."""
    nu, nv, _ = states[0].shape
    outs = []
    for P in states:
        n = grid_normals(P, closed_u)
        top = P + n * (thick[..., None] * 0.5)
        bot = P - n * (thick[..., None] * 0.5)
        outs.append(np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3)]).astype(np.float32))
    N = nu * nv

    def idx(i, j, layer):
        return layer * N + (i % nu) * nv + j

    F = []
    iu = range(nu if closed_u else nu - 1)
    for i in iu:
        for j in range(nv - 1):
            F.append((idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0)))
            F.append((idx(i, j, 1), idx(i, j + 1, 1), idx(i + 1, j + 1, 1), idx(i + 1, j, 1)))
        F.append((idx(i, 0, 0), idx(i, 0, 1), idx(i + 1, 0, 1), idx(i + 1, 0, 0)))
        F.append((idx(i, nv - 1, 0), idx(i + 1, nv - 1, 0), idx(i + 1, nv - 1, 1), idx(i, nv - 1, 1)))
    if not closed_u:
        for j in range(nv - 1):
            F.append((idx(0, j, 0), idx(0, j + 1, 0), idx(0, j + 1, 1), idx(0, j, 1)))
            F.append((idx(nu - 1, j, 0), idx(nu - 1, j, 1), idx(nu - 1, j + 1, 1), idx(nu - 1, j + 1, 0)))
    uu, vv = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing='ij')
    uv = np.stack([uu.ravel(), vv.ravel()], 1)
    uv = np.concatenate([uv, uv])
    return outs, np.array(F, np.int32), uv.astype(np.float32)


def unpenetrate(P, inward, skip_rows=3, margin=0.045, closed_u=True, iters=40, weight=None):
    """Push grid points out of the tissue along `inward` (nu, nv, 3) until they clear the wall."""
    P = P.copy()
    disp = np.zeros(P.shape[:2])
    if weight is None:
        weight = np.ones(P.shape[:2])
        weight[:, :skip_rows] = 0.0
    for _ in range(iters):
        s = solid(P + inward * disp[..., None])
        pen = (s < margin) * (weight > 0.5)
        if not pen.any():
            break
        disp += 0.03 * pen
    if disp.max() > 0:
        m = 'wrap' if closed_u else 'nearest'
        sm = ndi.gaussian_filter1d(ndi.gaussian_filter1d(disp, 2.0, axis=0, mode=m), 1.5, axis=1, mode='nearest')
        disp = np.maximum(disp, sm) * weight
    return P + inward * disp[..., None], float(disp.max())


def make_object(name, states, faces, uv, keyname):
    ob = mesh_from_arrays(name, states[0], faces)
    me = ob.data
    uvl = me.uv_layers.new(name='UVMap')
    li = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get('vertex_index', li)
    uvl.data.foreach_set('uv', uv[li].ravel())
    ob.shape_key_add(name='Basis')
    k = ob.shape_key_add(name=keyname)
    k.data.foreach_set('co', np.ascontiguousarray(states[1], np.float32).ravel())
    return ob


LAND = {}

# =====================================================================================
# Semilunar valves
# =====================================================================================
def semilunar(name, C, n, H, cusp_degs, cusp_names):
    e1, e2, n = root_frame(n)

    def dirv(deg):
        a = np.radians(np.asarray(deg, float))
        return np.cos(a)[..., None] * e1 + np.sin(a)[..., None] * e2

    nu, nv = 49, 21
    us = np.linspace(-0.975, 0.975, nu)
    vs = np.linspace(0, 1, nv)
    au = np.abs(us)
    Hc = 0.90 * H
    th = 0.046
    V_all = [[], []]
    F_all, UV_all = [], []
    off = 0
    for ci, deg0 in enumerate(cusp_degs):
        ang_a = deg0 + us * 58.5
        z_a = Hc * au ** 1.8
        da = dirv(ang_a)
        org = C + z_a[:, None] * n
        rw = ray_wall(org, da, rmax=2.2)
        rw = np.where(np.isnan(rw), np.nanmedian(rw), rw)
        rw = ndi.gaussian_filter1d(rw, 1.2, mode='nearest')
        A = org + (rw + 0.04)[:, None] * da
        Rc = 0.5 * (rw[0] + rw[-1])
        # ---- closed: free margins meet the neighbours along the three radial coaptation lines
        kk = 0.15
        ang_f = deg0 + 60.0 * np.tanh(us / kk) / np.tanh(1 / kk)
        s0 = 0.02
        rf = Rc * (np.sqrt(us ** 2 + s0) - np.sqrt(s0)) / (np.sqrt(1 + s0) - np.sqrt(s0))
        zc = 0.66 * Hc
        z_f = zc + (Hc - zc) * au ** 1.3
        o = (th / 2 + 0.006) / 0.866
        Fc = C + z_f[:, None] * n + rf[:, None] * dirv(ang_f) + o * dirv(deg0)[None, :]
        wb = smoothstep(0.90, 1.0, au)[:, None]
        Fc = Fc * (1 - wb) + A * wb
        cl = 0.30 * H * (1 - 0.7 * au)                       # lunula / coaptation height
        P2 = Fc - cl[:, None] * n
        P1 = 0.5 * (A + P2) - (0.11 * H * (1 - us ** 2))[:, None] * n
        closed = bezier(A, P1, P2, Fc, vs)
        L = arclen(closed)
        # ---- open: cusp swings up towards (but not onto) the sinus wall
        ang_o = deg0 + us * 57.0
        r_o = Rc * (1 - 0.09 * (1 - us ** 2))
        dr = (rw + 0.04) - r_o
        z_o = z_a + np.sqrt(np.maximum(L ** 2 - dr ** 2, (0.25 * L) ** 2)) * 0.97
        do = dirv(ang_o)
        Fo = C + z_o[:, None] * n + r_o[:, None] * do
        wb = smoothstep(0.93, 1.0, au)[:, None]
        Fo = Fo * (1 - wb) + A * wb
        bul = (1 - us ** 2)[:, None] * do
        O1 = A + (Fo - A) / 3 + 0.03 * bul
        O2 = A + 2 * (Fo - A) / 3 + 0.06 * bul
        opened = bezier(A, O1, O2, Fo, vs)
        U, Vv = np.meshgrid(us, vs, indexing='ij')
        wfree = smoothstep(0.08, 0.3, Vv) * smoothstep(1.0, 0.86, np.abs(U))      # hinge + commissures stay put
        inw = np.repeat(-do[:, None, :], nv, 1)
        opened, _ = unpenetrate(opened, inw, closed_u=False, weight=wfree, margin=0.05)
        closed, _ = unpenetrate(closed, np.repeat(-da[:, None, :], nv, 1), closed_u=False, weight=wfree, margin=0.03)
        thick = th * (1 + 0.9 * (1 - Vv) ** 4) * (1 - 0.35 * smoothstep(0.85, 1.0, np.abs(U)))
        thick = thick + 0.075 * np.exp(-(U / 0.12) ** 2) * smoothstep(0.78, 1.0, Vv)      # nodule
        thick = thick - 0.012 * smoothstep(0.6, 0.9, Vv) * smoothstep(0.2, 0.5, np.abs(U))  # lunulae
        (vc, vo), F, uv = shell([closed, opened], thick, False)
        V_all[0].append(vc)
        V_all[1].append(vo)
        F_all.append(F + off)
        uv[:, 0] = (uv[:, 0] + ci) / 3.0
        UV_all.append(uv)
        off += len(vc)
        LAND['%s_cusp_%s' % (name, cusp_names[ci])] = closed[nu // 2, nv // 2].tolist()
        LAND['%s_nodule_%s' % (name, cusp_names[ci])] = closed[nu // 2, -1].tolist()
    ob = make_object('Valve_' + name, [np.concatenate(V_all[0]), np.concatenate(V_all[1])],
                     np.concatenate(F_all), np.concatenate(UV_all), 'semilunar_valves_open')
    pen = [(solid(np.concatenate(v)) < -0.05).mean() for v in V_all]
    log('%s valve: %d verts, embedded fraction closed/open %.3f / %.3f' % (name, off, pen[0], pen[1]))
    return ob


semilunar('Aortic', AO_C, AO_N, AO_H, AO_CUSP_DEG, AO_CUSP_NAMES)
semilunar('Pulmonary', PV_C, PV_N, PV_H, PV_CUSP_DEG, PV_CUSP_NAMES)

# =====================================================================================
# Atrioventricular valves
# =====================================================================================
PMS = papillary_muscles()


def av_valve(name, C, n, e1, e2, leaflets, zc, smile, tilt_open, shift):
    """leaflets: list of (label, start_deg) ordered by angle; zc: coaptation centre (e1, e2) coords."""
    nth, nv = 300, 27
    thd = np.linspace(0, 360, nth, endpoint=False)
    a = np.radians(thd)
    d = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    C = np.asarray(C, float) - shift * n                  # hinge line inside the fibrous annulus
    for _ in range(3):                                   # recentre on the real orifice
        rw = ray_wall(np.repeat((C - 0.05 * n)[None, :], nth, 0), d, rmax=2.5)
        med = np.nanmedian(rw)
        rw = np.clip(np.where(np.isnan(rw), med, rw), 0.72 * med, 1.3 * med)
        cen = (rw[:, None] * d).mean(axis=0) * 2.0
        C = C + 0.8 * cen
    rw = ndi.gaussian_filter1d(rw, 3.0, mode='wrap')
    starts = np.array([l[1] for l in leaflets], float)
    ends = np.roll(starts, -1)
    ends[-1] += 360.0
    nl = len(leaflets)
    li = np.zeros(nth, int)
    u = np.zeros(nth)
    for i in range(nl):
        rel = np.mod(thd - starts[i], 360.0)
        m = rel < (ends[i] - starts[i])
        li[m] = i
        u[m] = rel[m] / (ends[i] - starts[i])
    saddle = -0.10 * np.cos(2 * (a - np.pi / 2))
    Pa = C + (rw + 0.045)[:, None] * d + saddle[:, None] * n

    def at_deg(deg):
        aa = math.radians(deg)
        r = np.interp(np.mod(deg, 360.0), np.append(thd, 360.0), np.append(rw, rw[0]))
        return C + r * (math.cos(aa) * e1 + math.sin(aa) * e2), r

    Z = C + zc[0] * e1 + zc[1] * e2
    K = [C + 0.80 * (at_deg(s)[0] - C) for s in starts]
    s = 1 - np.abs(2 * u - 1)                       # 0 at the commissures, 1 mid-leaflet
    cj = np.where(u < 0.5, li, (li + 1) % nl)       # commissure this half of the leaflet belongs to
    Q = np.zeros((nth, 3))
    se = s ** 0.85
    for j in range(nl):
        m = cj == j
        if smile:
            K0, K1 = K[0], K[1]
            ctl = 2 * Z - 0.5 * (K0 + K1)
            w = (se[m] * 0.5) if j == 0 else (1 - se[m] * 0.5)
            w = w[:, None]
            Q[m] = (1 - w) ** 2 * K0 + 2 * w * (1 - w) * ctl + w ** 2 * K1
        else:
            Q[m] = K[j][None, :] * (1 - se[m])[:, None] + Z[None, :] * se[m][:, None]
    h_t = 0.20 + 0.44 * s
    q = 0.17 + 0.34 * s
    scal = np.ones(nth)
    for i, lf in enumerate(leaflets):
        m = li == i
        for cu in lf[2]:                                  # clefts between scallops
            scal[m] -= 0.72 * np.exp(-((u[m] - cu) / 0.04) ** 2)
        q[m] *= lf[3]
    scal *= 1 + 0.06 * np.sin(a * 9 + 1.0) + 0.04 * np.sin(a * 17 + 0.3)
    q = q * scal
    h_t = h_t * (0.55 + 0.45 * np.clip(scal, 0, 1))
    T = Q + h_t[:, None] * n
    gdir = Pa - T
    gdir -= (gdir @ n)[:, None] * n
    gdir /= (np.linalg.norm(gdir, axis=1, keepdims=True) + 1e-9)
    th0 = 0.066
    T = T + (th0 / 2 + 0.014) * smoothstep(0.0, 0.14, s)[:, None] * gdir
    vs = np.linspace(0, 1, nv)
    Tin = T - ((T - Pa) @ n)[:, None] * n                 # T projected to the annular level of Pa
    B1 = Pa + 0.55 * (Tin - Pa) + 0.03 * n
    closed = bezier(Pa, B1, T, T + q[:, None] * n, vs)
    L = arclen(closed)
    # ---- open: leaflets hang into the ventricle as a funnel
    beta = np.radians(np.array([tilt_open[i] for i in li], float))
    beta = ndi.gaussian_filter1d(beta, 6.0, mode='wrap')
    ell = L[:, None] * vs[None, :]
    inward = -d
    opened = (Pa[:, None, :] + (ell * np.cos(beta)[:, None])[..., None] * n
              + (ell * np.sin(beta)[:, None] + 0.10 * L[:, None] * vs[None, :] ** 2)[..., None] * inward[:, None, :])
    inw = np.repeat(inward[:, None, :], nv, 1)
    opened, dmax_o = unpenetrate(opened, inw)
    closed, dmax_c = unpenetrate(closed, inw, margin=0.03)
    U, Vv = np.meshgrid(u, vs, indexing='ij')
    thick = th0 + 0.045 * (1 - Vv) ** 3 + 0.022 * smoothstep(0.7, 1.0, Vv)
    (vc, vo), F, uv = shell([closed, opened], thick, True)
    ob = make_object('Valve_' + name, [vc, vo], F, uv, 'av_valves_open')
    log('%s valve: annulus r %.2f-%.2f cm, leaflet length %.2f-%.2f cm, collision push closed/open %.2f / %.2f'
        % (name, rw.min(), rw.max(), L.min(), L.max(), dmax_c, dmax_o))
    LAND[name + '_annulus'] = Pa[::10].tolist()
    for i, lf in enumerate(leaflets):
        k = int(np.argmin(np.abs(np.where(li == i, u, 9) - 0.5)))
        LAND['%s_leaflet_%s' % (name, lf[0])] = closed[k, nv // 3].tolist()
    return dict(name=name, C=C, n=n, thd=thd, li=li, u=u, starts=starts, ends=ends, closed=closed, opened=opened,
                nv=nv, d=d, leaflets=leaflets)


# mitral: anterior (aortic) leaflet spans ~125 deg centred on +e2; posterior (mural) leaflet P1-P2-P3
mitral = av_valve('Mitral', MV_C, MV_N, MV_E1, MV_E2,
                  [('anterior', 27.0, (), 1.0), ('posterior', 153.0, (1 / 3, 2 / 3), 0.9)],
                  zc=(0.0, -0.36 * MV_RB), smile=True, tilt_open=(4.0, 20.0), shift=0.40)

# tricuspid: septal leaflet centred on +e2 (towards the septum); which side is anterior?
ant_sign = 1.0 if (TV_E1 @ (-D_AX)) > 0 else -1.0
if ant_sign > 0:        # +e1 (theta = 0) is anterior-superior
    tri_leaf = [('septal', 40.0, (), 0.8), ('posterior', 150.0, (), 0.95), ('anterior', 275.0, (), 1.0)]
else:
    tri_leaf = [('septal', 30.0, (), 0.8), ('anterior', 140.0, (), 1.0), ('posterior', 265.0, (), 0.95)]
tilt = {'septal': 14.0, 'anterior': 20.0, 'posterior': 20.0}
tricuspid = av_valve('Tricuspid', TV_C, TV_N, TV_E1, TV_E2, tri_leaf, zc=(0.0, 0.22), smile=False,
                     tilt_open=tuple(tilt[l[0]] for l in tri_leaf), shift=0.55)

# =====================================================================================
# Chordae tendineae
# =====================================================================================
class Tubes:
    def __init__(self):
        self.V = [[], []]
        self.F = []
        self.n = 0

    def add(self, A, B, rad, bows, sides=5, ns=9):
        """Segment given as end points per pose: A = (A_closed, A_open) ..."""
        ring_idx = []
        for st in range(2):
            a, b = np.asarray(A[st], float), np.asarray(B[st], float)
            mid = 0.5 * (a + b) + bows[st]
            sgrid = np.linspace(0, 1, ns)[:, None]
            P = (1 - sgrid) ** 2 * a + 2 * sgrid * (1 - sgrid) * mid + sgrid ** 2 * b
            tg = np.gradient(P, axis=0)
            tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
            ref = np.array([0.3, 0.5, 0.8])
            e1 = np.cross(tg, ref)
            e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
            e2 = np.cross(tg, e1)
            ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
            rr = np.linspace(rad[0], rad[1], ns)[:, None, None]
            ring = P[:, None, :] + rr * (np.cos(ang)[None, :, None] * e1[:, None, :] + np.sin(ang)[None, :, None] * e2[:, None, :])
            verts = np.concatenate([ring.reshape(-1, 3), P[:1], P[-1:]])
            self.V[st].append(verts)
        base = self.n
        for i in range(ns - 1):
            for k in range(sides):
                k2 = (k + 1) % sides
                self.F.append((base + i * sides + k, base + i * sides + k2, base + (i + 1) * sides + k2, base + (i + 1) * sides + k))
        c0, c1 = base + ns * sides, base + ns * sides + 1
        for k in range(sides):
            k2 = (k + 1) % sides
            self.F.append((c0, base + k2, base + k, base + k))
            self.F.append((c1, base + (ns - 1) * sides + k, base + (ns - 1) * sides + k2, base + (ns - 1) * sides + k2))
        self.n += ns * sides + 2

    def build(self, name):
        F = np.array(self.F, np.int32)
        tri = F[:, 2] == F[:, 3]
        quads = F[~tri]
        tris = F[tri][:, :3]
        # triangulate everything for a uniform face size
        allf = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]], tris])
        V0, V1 = np.concatenate(self.V[0]), np.concatenate(self.V[1])
        ob = mesh_from_arrays(name, V0, allf)
        ob.shape_key_add(name='Basis')
        k = ob.shape_key_add(name='av_valves_open')
        k.data.foreach_set('co', np.ascontiguousarray(V1, np.float32).ravel())
        return ob


def leaflet_point(valve, deg, vfrac):
    i = int(round(np.mod(deg, 360.0) / 360.0 * len(valve['thd']))) % len(valve['thd'])
    j = int(round(vfrac * (valve['nv'] - 1)))
    return valve['closed'][i, j], valve['opened'][i, j]


def chordae_for(valve, heads_by_commissure, tubes, rng, per_head=12):
    starts, ends = valve['starts'], valve['ends']
    nl = len(starts)
    axis_pt, n = valve['C'], valve['n']
    for j, heads in heads_by_commissure.items():
        prev_span = ends[(j - 1) % nl] - starts[(j - 1) % nl]
        next_span = ends[j] - starts[j]
        lo = starts[j] - 0.47 * prev_span
        hi = starts[j] + 0.47 * next_span
        nins = per_head * len(heads)
        degs = np.linspace(lo, hi, nins) + rng.uniform(-1.5, 1.5, nins)
        # split the fan between the heads (ordered so that fans do not cross)
        mids = [np.mean(degs[k * per_head:(k + 1) * per_head]) for k in range(len(heads))]
        order = list(range(len(heads)))
        if len(heads) == 2:
            p0 = leaflet_point(valve, mids[0], 1.0)[0]
            if np.linalg.norm(p0 - heads[0]) > np.linalg.norm(p0 - heads[1]):
                order = [1, 0]
        for hk, hidx in enumerate(order):
            tip = np.asarray(heads[hidx], float)
            hd = degs[hk * per_head:(hk + 1) * per_head]
            ins = []
            for m, dg in enumerate(hd):
                vf = 1.0 if m % 2 == 0 else rng.uniform(0.74, 0.86)
                ins.append(leaflet_point(valve, dg, vf))
            for tk in range(0, per_head, 4):          # one first-order trunk per 4 insertions
                grp = ins[tk:tk + 4]
                cen = [np.mean([p[st] for p in grp], axis=0) for st in range(2)]
                start = [tip - 0.06 * unit(cen[0] - tip)] * 2
                n1 = [tip + rng.uniform(0.42, 0.55) * (cen[st] - tip) for st in range(2)]
                slack = max(0.0, np.linalg.norm(cen[0] - tip) - np.linalg.norm(cen[1] - tip))
                out = unit(np.cross(cen[0] - tip, n)) * rng.choice([-1, 1])
                bow = lambda sc: [rng.normal(0, 0.012, 3), out * min(0.22, slack * 0.45) * sc]
                tubes.add(start, n1, (0.036, 0.030), bow(0.6))
                for pk in range(0, len(grp), 2):
                    pair = grp[pk:pk + 2]
                    pc = [np.mean([p[st] for p in pair], axis=0) for st in range(2)]
                    n2 = [n1[st] + rng.uniform(0.5, 0.62) * (pc[st] - n1[st]) for st in range(2)]
                    tubes.add(n1, n2, (0.028, 0.022), bow(0.5))
                    for p in pair:
                        tubes.add(n2, [p[0], p[1]], (0.019, 0.015), bow(0.35))


def commissure_heads(valve, pm_names):
    """Assign each papillary muscle to the nearest commissure."""
    out = {}
    used = set()
    for nm in pm_names:
        tips = PMS[nm]['tips']
        c = np.mean(tips, axis=0)
        best, bd = None, 1e9
        for j, sdeg in enumerate(valve['starts']):
            if j in used:
                continue
            a = math.radians(sdeg)
            i = int(round(np.mod(sdeg, 360.0) / 360.0 * len(valve['thd']))) % len(valve['thd'])
            p = valve['closed'][i, 0]
            dd = np.linalg.norm(p - c)
            if dd < bd:
                best, bd = j, dd
        used.add(best)
        out[best] = tips
    return out


rng = np.random.default_rng(3)
tb = Tubes()
chordae_for(mitral, commissure_heads(mitral, ['LV_PM_anterolateral', 'LV_PM_posteromedial']), tb, rng, per_head=12)
ch_m = tb.build('Chordae_Mitral')
tb = Tubes()
chordae_for(tricuspid, commissure_heads(tricuspid, ['RV_PM_anterior', 'RV_PM_posterior', 'RV_PM_septal']), tb, rng, per_head=12)
# direct septal chordae (characteristic of the tricuspid valve)
sep_i = [i for i, l in enumerate(tricuspid['leaflets']) if l[0] == 'septal'][0]
for f in np.linspace(0.2, 0.8, 5):
    dg = tricuspid['starts'][sep_i] + f * (tricuspid['ends'][sep_i] - tricuspid['starts'][sep_i])
    pc, po = leaflet_point(tricuspid, dg, 0.8)
    # nearest septal wall point: march from the leaflet towards the LV axis
    sgm, dlt, tt_ = W2L(pc)
    phi = math.atan2(dlt, sgm)
    root = lv_epi(phi, tt_ + 0.75, -0.05)
    tb.add([root, root], [pc, po], (0.024, 0.016), [np.zeros(3), np.zeros(3)])
ch_t = tb.build('Chordae_Tricuspid')
log('chordae: mitral %d verts, tricuspid %d verts' % (len(ch_m.data.vertices), len(ch_t.data.vertices)))

for nm, C, n in (('Mitral', MV_C, MV_N), ('Tricuspid', TV_C, TV_N), ('Aortic', AO_C, AO_N), ('Pulmonary', PV_C, PV_N)):
    LAND[nm + '_center'] = np.asarray(C).tolist()
    LAND[nm + '_normal'] = np.asarray(n).tolist()
for nm, d in PMS.items():
    LAND[nm] = np.mean([p for p, r in d['trunk']], axis=0).tolist()
save_json('landmarks03', LAND)

for ob in bpy.data.objects:
    for p in ob.data.polygons:
        p.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=bpath('stage03.blend'))
log('stage 03 complete')
