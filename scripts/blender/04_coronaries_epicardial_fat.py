"""
04_coronaries_epicardial_fat.py
Three-dimensional coronary circulation and subepicardial adipose tissue.
  * Every vessel is a tapered organic tube whose centreline is shrink-wrapped onto the epicardial
    SDF (Newton projection) with an anatomical offset, and — for the vessels that travel in the
    atrioventricular / interventricular sulci — relaxed into the bottom of the groove.
  * Arteries: left main -> LAD (diagonals, septal perforators) + LCx (obtuse marginals,
    posterolateral);  RCA (conus, sinoatrial-node branch, acute marginals, PDA at the crux).
  * Veins: great / middle / small cardiac veins and posterior vein of the LV draining into the
    coronary sinus.
  * Branch unions are made seamless (manifold) by voxel remeshing the swept tubes.
  * Fat: lobulated SDF layer riding on the epicardium along the sulci, partially burying the vessels.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heartlib import *      # noqa
from scipy import ndimage as ndi
from scipy.interpolate import CubicSpline
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
g = Grid()
fd = np.load(bpath('fields02.npz'))
ST = int(fd['stride'])
OUT = fd['outer'].astype(np.float32)
OB = ndi.gaussian_filter(OUT, 0.32 / (g.h * ST))          # groove potential (low inside concavities)
log('stage 04')
rng = np.random.default_rng(42)


def val(F, P):
    return g.sample(F, P, stride=ST)


def grad(F, P, e=0.06):
    G = np.zeros_like(P)
    for i in range(3):
        d = np.zeros(3)
        d[i] = e
        G[:, i] = (val(F, P + d) - val(F, P - d)) / (2 * e)
    return G


def normals(P):
    G = grad(OUT, P)
    return G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)


def project(P, off=0.0, it=7):
    P = np.array(P, float)
    for _ in range(it):
        G = grad(OUT, P)
        gn = np.linalg.norm(G, axis=1) + 1e-6
        step = np.clip((val(OUT, P) - off) / gn, -0.35, 0.35)       # clamped Newton step (robust far from the wall)
        P -= (step / gn)[:, None] * G
    return P


def settle(P, steps=26, lr=0.035):
    """Slide surface points towards the bottom of the local sulcus."""
    P = project(P)
    for _ in range(steps):
        n = normals(P)
        gb = grad(OB, P, 0.1)
        gt = gb - np.einsum('ij,ij->i', gb, n)[:, None] * n
        mag = np.linalg.norm(gt, axis=1, keepdims=True)
        P = project(P - lr * gt / (mag + 1e-6) * np.clip(mag * 6.0, 0, 1), it=3)
    return P


def resample(P, step=0.1):
    P = np.asarray(P, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    P, s = P[keep], s[keep]
    n = max(4, int(s[-1] / step) + 1)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, i]) for i in range(3)], 1)


def spline(way, step=0.1):
    way = np.asarray(way, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1))])
    cs = CubicSpline(s, way, bc_type='natural')
    return cs(np.linspace(0, s[-1], max(4, int(s[-1] / step) + 1)))


def smooth(P, sig=2.0):
    Q = ndi.gaussian_filter1d(P, sig, axis=0, mode='nearest')
    Q[0], Q[-1] = P[0], P[-1]
    return Q


VESSELS = {'art': [], 'vein': []}
NAMED = {}


def vessel(name, kind, way, r0, r1, groove=0, wiggle=0.0, wl=1.4, side=0.0, offk=0.62, start_free=0, power=1.0):
    """Shrink-wrapped tapered vessel.  way: waypoints; groove: sulcus relaxation steps;
    side: lateral shift (cm) along the surface; start_free: number of initial samples left unprojected."""
    P = spline(way)
    S = project(P)
    if groove:
        S = settle(S, groove)
    S = project(smooth(resample(S), 2.5))
    n = normals(S)
    tg = np.gradient(S, axis=0)
    tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
    lat = np.cross(n, tg)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))])
    u = arc / arc[-1]
    if wiggle or side:
        taper = smoothstep(0.0, 0.12, u)
        w = 0.7 * wiggle * (np.sin(2 * np.pi * arc / (wl * 1.5) + rng.uniform(0, 6.28)) + 0.35 * np.sin(2 * np.pi * arc / (wl * 0.62) + rng.uniform(0, 6.28)))
        S = project(S + lat * ((w + side) * taper)[:, None])
        n = normals(S)
    R = r1 + (r0 - r1) * (1 - u) ** power
    C = S + n * (R * offk)[:, None]
    if start_free:
        k = min(start_free, len(C) - 2)
        a = np.asarray(way[0], float)
        blend = smoothstep(0, 1, np.linspace(0, 1, k))[:, None]
        C[:k] = (a[None, :] * (1 - np.linspace(0, 1, k))[:, None] + C[k] * np.linspace(0, 1, k)[:, None]) * (1 - blend) + C[:k] * blend
    VESSELS[kind].append((C, R))
    NAMED[name] = dict(kind=kind, P=np.round(C[::3], 3).tolist(), R=np.round(R[::3], 3).tolist(),
                       mid=C[len(C) // 2].tolist())
    return dict(C=C, R=R, S=S, n=n)


def free_vessel(name, kind, way, r0, r1):
    C = spline(way, 0.08)
    R = np.linspace(r0, r1, len(C))
    VESSELS[kind].append((C, R))
    NAMED[name] = dict(kind=kind, P=np.round(C[::3], 3).tolist(), R=np.round(R[::3], 3).tolist(), mid=C[len(C) // 2].tolist())


def at(v, f):
    return v['S'][int(np.clip(f, 0, 1) * (len(v['S']) - 1))]


def iv_phi(t, sign):
    """Angle about the LV axis of the anterior (sign=-1) / posterior (+1) interventricular sulcus."""
    t = float(np.clip(t, 0.2, 6.3))
    R = float(lv_rc(t) + lv_wall(t))
    bow = 0.18 * math.sin(min(t / 7.3, 1) * math.pi)
    c = float(rv_c(t)) + bow
    q = float(rv_q(t)) + RV_WALL
    K = (R * R + c * c + RV_DELTA0 ** 2 - q * q) / (2 * R)
    h = math.hypot(c, RV_DELTA0)
    return math.atan2(RV_DELTA0, c) + sign * math.acos(np.clip(K / h, -1, 1))


rad = math.radians
PH_A0, PH_P0 = iv_phi(0.3, -1), iv_phi(0.3, 1)
log('IV sulci at phi = %.0f / %.0f deg' % (math.degrees(PH_A0), math.degrees(PH_P0)))

# ----------------------------------------------------------------------------------------
# Left coronary artery
# ----------------------------------------------------------------------------------------
OST_L = AO_C + 0.85 * AO_N + 1.50 * cusp_dir(AO_N, AO_CUSP_DEG[1])
OST_R = AO_C + 0.85 * AO_N + 1.50 * cusp_dir(AO_N, AO_CUSP_DEG[0])
BIF = lv_epi(PH_A0 - rad(14), -0.30, 0.15)
lm = vessel('left_main', 'art', [OST_L - 0.3 * cusp_dir(AO_N, AO_CUSP_DEG[1]), OST_L + 0.35 * unit(BIF - OST_L), (OST_L + BIF) / 2, BIF],
            0.235, 0.215, start_free=5)
ts = [-0.1, 0.6, 1.5, 2.5, 3.5, 4.5, 5.5, 6.3, 7.0, 7.6]
lad_way = [BIF] + [lv_epi(iv_phi(t, -1), t, 0.05) for t in ts] + [lv_epi(iv_phi(6.3, -1) + rad(35), 8.25, 0.0), lv_epi(PH_P0 - rad(10), 7.7, 0.0)]
lad = vessel('LAD', 'art', lad_way, 0.20, 0.065, groove=22, wiggle=0.07, wl=1.9)
for nm, f, dphi, dt, r in (('diagonal_1', 0.16, -62, 3.3, 0.105), ('diagonal_2', 0.33, -50, 2.9, 0.09), ('diagonal_3', 0.50, -40, 2.0, 0.07)):
    p0 = at(lad, f)
    s, d, t0 = W2L(p0)
    ph0 = math.atan2(d, s)
    way = [p0] + [lv_epi(ph0 + rad(dphi) * k, t0 + dt * k ** 0.85, 0.05) for k in (0.25, 0.5, 0.75, 1.0)]
    dg = vessel(nm, 'art', way, r, 0.04, wiggle=0.08, wl=1.2)
    if nm == 'diagonal_1':
        q0 = at(dg, 0.45)
        s, d, t0 = W2L(q0)
        ph0 = math.atan2(d, s)
        vessel('diagonal_1b', 'art', [q0] + [lv_epi(ph0 - rad(22) * k, t0 + 1.6 * k, 0.05) for k in (0.5, 1.0)], 0.055, 0.035, wiggle=0.05, wl=1.0)
for i, f in enumerate((0.12, 0.22, 0.33, 0.45)):                     # septal perforators dive into the septum
    p0 = at(lad, f)
    s, d, t0 = W2L(p0)
    tgt = L2W(s * 0.55 + 1.3, d * 0.42, t0 + 0.9)
    free_vessel('septal_%d' % (i + 1), 'art', [p0, (p0 * 0.55 + tgt * 0.45) + 0.1 * A_AX, tgt], 0.07, 0.035)
for nm, f, dphi, dt in (('rv_branch_lad_1', 0.2, 28, 1.3), ('rv_branch_lad_2', 0.42, 24, 1.1)):
    p0 = at(lad, f)
    s, d, t0 = W2L(p0)
    ph0 = math.atan2(d, s)
    vessel(nm, 'art', [p0, lv_epi(ph0 + rad(dphi) * 0.5, t0 + dt * 0.5, 0.6), lv_epi(ph0 + rad(dphi), t0 + dt, 0.9)], 0.06, 0.035, wiggle=0.05, wl=1.0)

lcx_phis = [PH_A0 - rad(a) for a in (14, 40, 68, 96, 124, 152, 180, 205)]
lcx = vessel('LCx', 'art', [BIF] + [lv_epi(p, -0.42, 0.1) for p in lcx_phis[1:]], 0.185, 0.085, groove=26, wiggle=0.03, wl=2.2)
for nm, f, dt, dphi, r in (('obtuse_marginal_1', 0.34, 4.9, 6, 0.115), ('obtuse_marginal_2', 0.60, 4.3, 14, 0.10), ('posterolateral_LCx', 0.9, 3.6, 12, 0.08)):
    p0 = at(lcx, f)
    s, d, t0 = W2L(p0)
    ph0 = math.atan2(d, s)
    way = [p0] + [lv_epi(ph0 + rad(dphi) * k, max(t0, 0.0) + 0.2 + dt * k, 0.05) for k in (0.2, 0.45, 0.7, 1.0)]
    om = vessel(nm, 'art', way, r, 0.04, wiggle=0.09, wl=1.3)
    if nm == 'obtuse_marginal_1':
        q0 = at(om, 0.4)
        s, d, t0 = W2L(q0)
        ph0 = math.atan2(d, s)
        vessel('obtuse_marginal_1b', 'art', [q0] + [lv_epi(ph0 - rad(20) * k, t0 + 2.0 * k, 0.05) for k in (0.5, 1.0)], 0.06, 0.035, wiggle=0.05, wl=1.0)
log('left coronary artery')

# ----------------------------------------------------------------------------------------
# Right coronary artery
# ----------------------------------------------------------------------------------------
CRUX = lv_epi(PH_P0 + rad(4), -0.38, 0.1)
TVc = TV_C - 0.5 * TV_N


def tv_ang(P):
    q = np.asarray(P) - TVc
    return math.atan2(q @ TV_E2, q @ TV_E1)


a_s, a_e = tv_ang(OST_R), tv_ang(CRUX)
# travel around the free-wall side (away from the septum, i.e. through -90 deg)
def unwrap_through(a0, a1, through=-math.pi / 2):
    cands = [a1 + 2 * math.pi * k for k in (-1, 0, 1)]
    best = None
    for c in cands:
        lo, hi = min(a0, c), max(a0, c)
        for th in (through - 2 * math.pi, through, through + 2 * math.pi):
            if lo < th < hi and (best is None or abs(c - a0) < abs(best - a0)):
                best = c
    return best if best is not None else a1


a_e = unwrap_through(a_s, a_e)
angs = np.linspace(a_s, a_e, 11)[1:-1]
ring = [TVc + 2.05 * (math.cos(a) * TV_E1 + math.sin(a) * TV_E2) for a in angs]
rca_way = [OST_R - 0.3 * cusp_dir(AO_N, AO_CUSP_DEG[0]), OST_R + 0.3 * unit(ring[0] - OST_R)] + ring + [CRUX]
rca = vessel('RCA', 'art', rca_way, 0.20, 0.13, groove=26, wiggle=0.03, wl=2.4, start_free=5)
pda_way = [CRUX] + [lv_epi(iv_phi(t, 1), t, 0.05) for t in (0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.8)]
pda = vessel('PDA', 'art', pda_way, 0.125, 0.05, groove=22, wiggle=0.05, wl=1.7)
plv0 = at(rca, 1.0)
vessel('posterolateral_RCA', 'art', [CRUX] + [lv_epi(PH_P0 + rad(a), -0.35 + tt_, 0.05) for a, tt_ in ((25, 0.1), (45, 0.9), (55, 2.4), (60, 3.8))],
       0.10, 0.04, wiggle=0.07, wl=1.3)
# conus branch and sinoatrial-node artery from the proximal RCA
p0 = at(rca, 0.07)
vessel('conus_branch', 'art', [p0, p0 + 0.6 * unit(PV_C - p0) + np.array([0, -0.4, 0.1]), PV_C - 1.1 * PV_N + np.array([-0.9, -1.0, 0.0]),
                                PV_C - 0.6 * PV_N + np.array([-0.1, -1.4, 0.0])], 0.07, 0.035, wiggle=0.04, wl=1.0)
p0 = at(rca, 0.10)
vessel('SA_node_artery', 'art', [p0, np.array([-2.2, -0.75, 3.55]), np.array([-2.75, -0.55, 4.25]), np.array([-3.35, -0.55, 4.75]),
                                  np.array([-3.95, -0.45, 4.65])], 0.065, 0.04, wiggle=0.04, wl=1.1)
def rca_near(P):
    return rca['S'][int(np.argmin(np.linalg.norm(rca['S'] - np.asarray(P)[None, :], axis=1)))]


# RV free-wall branches fan out obliquely towards the apex with a gentle S-shaped course
for nm, psi, r, t1, dpsi, bend in (('rv_anterior_branch', -20, 0.075, 3.2, -26, 9), ('acute_marginal', 30, 0.105, 5.7, -20, -7),
                                   ('acute_marginal_2', 54, 0.07, 3.6, 10, 6), ('rv_anterior_branch_2', 4, 0.06, 2.4, 16, -8)):
    ks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    way = [rv_free(rad(psi + dpsi * k ** 1.3 + bend * math.sin(math.pi * k * 1.4)), 0.25 + (t1 - 0.25) * k, -RV_WALL - 0.02) for k in ks]
    way[0] = rca_near(way[0])
    v_ = vessel(nm, 'art', way, r, 0.04, wiggle=0.085, wl=1.15)
    for f, sgn in ((0.42, 1), (0.66, -1)) if r > 0.065 else ((0.55, 1),):     # small second-order twigs
        q0 = at(v_, f)
        s_, d_, t_ = W2L(q0)
        ps = math.atan2(d_ - RV_DELTA0, s_ - float(rv_c(max(t_, 0.3))))
        tw = [q0] + [rv_free(ps + sgn * rad(18) * k, max(t_, 0.3) + 1.3 * k, -RV_WALL - 0.02) for k in (0.5, 1.0)]
        vessel(nm + '_twig_%d' % int(f * 100), 'art', tw, 0.042, 0.03, wiggle=0.05, wl=0.8)
log('right coronary artery')

# ----------------------------------------------------------------------------------------
# Cardiac veins
# ----------------------------------------------------------------------------------------
lad_S = lad['S'][: int(0.86 * len(lad['S']))][::-1]                  # apex -> base alongside the LAD
gcv1 = vessel('great_cardiac_vein', 'vein', lad_S[::6], 0.07, 0.17, side=0.30, power=1.0)
lcx_S = lcx['S'][int(0.08 * len(lcx['S'])):]
gcv2 = vessel('great_cardiac_vein_AV', 'vein', np.concatenate([gcv1['S'][-3:-1], lcx_S[::6]]), 0.17, 0.27, side=-0.33)
CS_END = project((CRUX + 1.15 * unit(RA_C - np.array([0.0, 0.0, 1.2]) - CRUX))[None, :], it=12)[0]
cs_way = [gcv2['S'][-1]] + [lv_epi(PH_A0 - rad(a), -0.5, 0.1) for a in (222, 240)] + [lv_epi(PH_P0 + rad(8), -0.55, 0.1), CS_END]
cs = vessel('coronary_sinus', 'vein', cs_way, 0.30, 0.44, groove=12, side=-0.12, power=0.6, offk=0.55)
pda_S = pda['S'][: int(0.9 * len(pda['S']))][::-1]
mcv = vessel('middle_cardiac_vein', 'vein', np.concatenate([pda_S[::6], [at(cs, 0.88)]]), 0.06, 0.20, side=0.26)
rca_S = rca['S'][int(0.45 * len(rca['S'])):]
vessel('small_cardiac_vein', 'vein', np.concatenate([rca_S[::6], [at(cs, 0.97)]]), 0.07, 0.15, side=0.28)
pv0 = at(cs, 0.35)
s, d, t0 = W2L(pv0)
ph0 = math.atan2(d, s)
vessel('posterior_vein_LV', 'vein', [lv_epi(ph0 + rad(6), 5.2, 0.05), lv_epi(ph0 + rad(10), 3.6, 0.05), lv_epi(ph0 + rad(4), 2.0, 0.05),
                                      lv_epi(ph0, 0.6, 0.05), pv0], 0.05, 0.15, wiggle=0.07, wl=1.5)
om_v = NAMED['obtuse_marginal_1']
omP = np.array(om_v['P'])[::-1]
vessel('left_marginal_vein', 'vein', np.concatenate([project(omP)[2::2], [at(gcv2, 0.45)]]), 0.05, 0.12, side=0.22)
for nm, psi in (('anterior_cardiac_vein_1', -6), ('anterior_cardiac_vein_2', 20)):
    way = [rv_free(rad(psi - 16), 3.8, -RV_WALL), rv_free(rad(psi + 4), 2.6, -RV_WALL), rv_free(rad(psi - 5), 1.5, -RV_WALL),
           rv_free(rad(psi), 0.7, -RV_WALL)]
    way.append(rca_near(rv_free(rad(psi), 0.25, -RV_WALL)))
    vessel(nm, 'vein', way, 0.045, 0.085, wiggle=0.07, wl=1.3, side=0.1)
log('cardiac veins')


# ----------------------------------------------------------------------------------------
# Tube sweep + seamless (voxel) union
# ----------------------------------------------------------------------------------------
def sweep(C, R, sides=12):
    n = len(C)
    tg = np.gradient(C, axis=0)
    tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
    e1 = np.cross(tg[0], [0.31, 0.55, 0.77])
    e1 /= np.linalg.norm(e1)
    E1 = [e1]
    for i in range(1, n):                                   # parallel transport
        e = E1[-1] - (E1[-1] @ tg[i]) * tg[i]
        E1.append(e / np.linalg.norm(e))
    E1 = np.array(E1)
    E2 = np.cross(tg, E1)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = C[:, None, :] + R[:, None, None] * (np.cos(ang)[None, :, None] * E1[:, None, :] + np.sin(ang)[None, :, None] * E2[:, None, :])
    V = np.concatenate([ring.reshape(-1, 3), [C[0] - tg[0] * R[0] * 0.8], [C[-1] + tg[-1] * R[-1] * 0.9]])
    F = []
    for i in range(n - 1):
        for k in range(sides):
            k2 = (k + 1) % sides
            F.append((i * sides + k, i * sides + k2, (i + 1) * sides + k2))
            F.append((i * sides + k, (i + 1) * sides + k2, (i + 1) * sides + k))
    c0, c1 = n * sides, n * sides + 1
    for k in range(sides):
        k2 = (k + 1) % sides
        F.append((c0, k2, k))
        F.append((c1, (n - 1) * sides + k, (n - 1) * sides + k2))
    return V, np.array(F, np.int32)


def build_tree(name, items, voxel, target):
    Vs, Fs, off = [], [], 0
    for C, R in items:
        V, F = sweep(np.asarray(C), np.asarray(R))
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    ob = mesh_from_arrays(name, np.concatenate(Vs), np.concatenate(Fs))
    bpy.context.view_layer.objects.active = ob
    for o in bpy.data.objects:
        o.select_set(o == ob)
    m = ob.modifiers.new('rm', 'REMESH')
    m.mode = 'VOXEL'
    m.voxel_size = voxel
    m.adaptivity = 0.0
    m.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier='rm')
    m = ob.modifiers.new('sm', 'SMOOTH')
    m.factor = 0.6
    m.iterations = 3
    bpy.ops.object.modifier_apply(modifier='sm')
    nf = sum(len(p.vertices) - 2 for p in ob.data.polygons)
    if nf > target:
        m = ob.modifiers.new('dec', 'DECIMATE')
        m.ratio = target / nf
        bpy.ops.object.modifier_apply(modifier='dec')
    log('%s: %d tris' % (name, sum(len(p.vertices) - 2 for p in ob.data.polygons)))
    return ob


VOXC = float(os.environ.get('HEART_CORO_VOXEL', '0.022'))
build_tree('Coronary_Arteries', VESSELS['art'], VOXC, 110000)
build_tree('Coronary_Veins', VESSELS['vein'], VOXC * 1.25, 80000)

# ----------------------------------------------------------------------------------------
# Epicardial fat
# ----------------------------------------------------------------------------------------
outer = load('outer02')
guides = []      # (surface polyline, half-width cm, thickness cm)
guides.append((lcx['S'], 1.00, 0.34))
guides.append((rca['S'], 1.05, 0.36))
guides.append((cs['S'], 0.9, 0.30))
guides.append((lad['S'][: int(0.72 * len(lad['S']))], 0.72, 0.27))
guides.append((pda['S'][: int(0.70 * len(pda['S']))], 0.62, 0.24))
guides.append((lm['S'], 0.8, 0.3))
am = np.array(NAMED['acute_marginal']['P'])
guides.append((project(am[: len(am) // 2]), 0.42, 0.16))
root = np.array([PV_C - 0.9 * PV_N + 1.55 * cusp_dir(PV_N, a) for a in np.linspace(-150, 60, 12)])
guides.append((project(root), 0.55, 0.2))
root2 = np.array([AO_C + 0.2 * AO_N + 1.75 * cusp_dir(AO_N, a) for a in np.linspace(-110, 60, 10)])
guides.append((project(root2), 0.6, 0.22))

pts = np.concatenate([gd[0] for gd in guides])
W = Win.around(g, pts, 1.6)
X, Y, Z = W.xyz()
wmap = np.zeros(W.shape, np.float32)       # normalised dome profile from every guide
for S, hw, th in guides:
    seed = np.ones(W.shape, bool)
    Sd = resample(S, g.h * 0.7)
    idx = np.round((Sd - g.lo) / g.h).astype(int) - np.array([s.start for s in W.sl])
    ok = np.all((idx >= 0) & (idx < np.array(W.shape)), axis=1)
    idx = idx[ok]
    seed[idx[:, 0], idx[:, 1], idx[:, 2]] = False
    # taper the amount of fat along the guide (less towards the distal end)
    amp = np.ones(W.shape, np.float32)
    dist, ind = ndi.distance_transform_edt(seed, sampling=g.h, return_indices=True)
    order = np.full(W.shape, 0, np.float32)
    order[idx[:, 0], idx[:, 1], idx[:, 2]] = np.linspace(0, 1, len(Sd))[ok]
    along = order[ind[0], ind[1], ind[2]]
    hw_l = hw * (1.0 - 0.45 * along)
    prof = np.sqrt(np.clip(1.0 - (dist / hw_l) ** 2, 0, 1)) * th * (1.0 - 0.35 * along)
    wmap = np.maximum(wmap, prof.astype(np.float32))
    del dist, ind, order, along, prof, seed
log('fat guides rasterised', W.shape)
o = outer[W.sl]
band = (wmap > 0.01) & (o > -0.2) & (o < 0.9)
idx = np.nonzero(band)
Pn = np.stack([np.broadcast_to(a_, W.shape)[idx] for a_ in (X, Y, Z)], axis=1)
big = pw_noise(Pn, 101, (1.6, 1.6, 1.6), K=14)                       # patchy coverage
lob1 = np.abs(pw_noise(Pn, 102, (0.55, 0.55, 0.55), K=18))           # billowy lobules
lob2 = np.abs(pw_noise(Pn, 103, (0.24, 0.24, 0.24), K=18))
h = wmap[idx] * (0.78 + 0.34 * big) + 0.075 * lob1 * smoothstep(0.03, 0.12, wmap[idx]) + 0.03 * lob2 * smoothstep(0.03, 0.1, wmap[idx])
h = h - 0.045
fat = np.full(W.shape, 0.5, np.float32)
fat[idx] = np.maximum(o[idx] - h, -(o[idx] + 0.06))
# never let fat spill into the lumens / cut stumps: stay close to the myocardial surface only
fat = ndi.gaussian_filter(fat, 0.55)
lab, nlab = ndi.label(fat < 0)
if nlab > 1:
    sizes = ndi.sum_labels(np.ones_like(lab, np.uint8), lab, np.arange(1, nlab + 1))
    small = np.nonzero(sizes < 0.02 / g.h ** 3)[0] + 1            # drop crumbs < 0.02 cm3
    fat[np.isin(lab, small)] = 0.2
gw = Grid(g.lo + np.array([s.start for s in W.sl]) * g.h, g.lo + (np.array([s.stop for s in W.sl]) - 1) * g.h, g.h)
vf, ff, vol = extract_surface(gw, fat[: gw.n[0], : gw.n[1], : gw.n[2]])
ob = mesh_from_arrays('Epicardial_Fat', vf, ff)
log('fat: %d tris, %.1f cm3' % (len(ff), vol))
target = int(os.environ.get('HEART_FAT_TRIS', '160000'))
if len(ff) > target:
    bpy.context.view_layer.objects.active = ob
    for o_ in bpy.data.objects:
        o_.select_set(o_ == ob)
    m = ob.modifiers.new('dec', 'DECIMATE')
    m.ratio = target / len(ff)
    bpy.ops.object.modifier_apply(modifier='dec')

NAMED['_ostia'] = dict(left=OST_L.tolist(), right=OST_R.tolist(), crux=CRUX.tolist())
save_json('landmarks04', NAMED)
bpy.ops.wm.save_as_mainfile(filepath=bpath('stage04.blend'))
log('stage 04 complete')
