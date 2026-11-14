"""
heartlib.py — shared anatomical frame, landmarks, SDF toolkit and physiological
displacement fields for the procedural adult human heart.

Units: centimetres.  World axes (anatomical position):
    +X = patient's left      +Y = posterior      +Z = superior
The ventricular frame (sigma, delta, t) is attached to the left-ventricular long axis:
    t     : base -> apex            (A_AX, pointing down / left / anterior)
    sigma : LV axis -> RV            (S_AX, pointing right / anterior)
    delta : towards the diaphragm    (D_AX, inferior / posterior interventricular groove)
"""
import os
import json
import math
import time
import numpy as np

BUILD = os.environ.get('HEART_BUILD', '/tmp/heart_build')
VOX = float(os.environ.get('HEART_VOXEL', '0.04'))      # voxel edge in cm
FAR = 3.0
os.makedirs(BUILD, exist_ok=True)

_T0 = time.time()


def log(*a):
    print('[heart %7.1fs]' % (time.time() - _T0), *a, flush=True)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------------------
# Anatomical frame
# --------------------------------------------------------------------------------------
A_AX = unit([0.58, -0.42, -0.70])
_s0 = np.array([-0.78, -0.62, 0.0])
S_AX = unit(_s0 - (_s0 @ A_AX) * A_AX)
D_AX = np.cross(A_AX, S_AX)
O_L = np.array([1.0, 1.0, 0.9])          # centre of the LV base (mitral / aortic plane)


def L2W(sig, dlt, t):
    return O_L + sig * S_AX + dlt * D_AX + t * A_AX


def W2L(P):
    q = np.asarray(P, float) - O_L
    return q @ S_AX, q @ D_AX, q @ A_AX


def perp_frame(n, hint=(0.0, -1.0, 0.0)):
    """Orthonormal frame (e1, e2, n) with e1 as close as possible to `hint`."""
    n = unit(n)
    h = np.asarray(hint, float)
    e1 = unit(h - (h @ n) * n)
    e2 = np.cross(n, e1)
    return e1, e2, n


# --------------------------------------------------------------------------------------
# Left ventricle (bullet-shaped canal along the long axis)
# --------------------------------------------------------------------------------------
LV_T = np.array([0.0, 1.0, 2.5, 4.0, 5.5, 6.6, 7.3])
LV_RC = np.array([1.75, 2.0, 2.05, 1.86, 1.42, 0.92, 0.42])        # cavity radius
LV_WALL = np.array([1.20, 1.28, 1.32, 1.30, 1.20, 1.0, 0.72])   # 12-13 mm free wall / septum
LV_APEX_T = 8.45


def lv_rc(t):
    return np.interp(t, LV_T, LV_RC)


def lv_wall(t):
    return np.interp(t, LV_T, LV_WALL)


def lv_axis(t):
    """Slightly curved LV centreline (the cavity bows a little towards the free wall)."""
    bow = 0.18 * np.sin(np.clip(np.asarray(t, float) / 7.3, 0, 1) * np.pi)
    return L2W(-bow, 0.0, t) if np.ndim(t) == 0 else O_L + np.outer(-bow, S_AX) + np.outer(t, A_AX)


# --------------------------------------------------------------------------------------
# Right ventricle (crescent wrapped around the septum): blob circle radius Q centred at c
# --------------------------------------------------------------------------------------
RV_T = np.array([0.2, 1.5, 3.0, 4.5, 5.6, 6.4])
RV_Q = np.array([3.40, 3.50, 3.40, 2.90, 2.30, 1.85])
RV_W = np.array([2.25, 2.40, 2.20, 1.80, 1.30, 0.95])           # septum -> free wall (cavity)
RV_APEX_T = 6.9
RV_WALL = 0.42                                                   # 4 mm free wall


def rv_c(t):
    return lv_rc(t) + lv_wall(t) + np.interp(t, RV_T, RV_W) - np.interp(t, RV_T, RV_Q)


def rv_q(t):
    return np.interp(t, RV_T, RV_Q)


RV_DELTA0 = -0.25      # crescent slightly rotated towards the anterior surface


# --------------------------------------------------------------------------------------
# Valve landmarks
# --------------------------------------------------------------------------------------
MV_C = L2W(-0.55, 0.45, 0.0)
MV_N = A_AX.copy()
MV_RA, MV_RB = 1.60, 1.30                 # semi-axes (commissural, antero-posterior)

TV_C = L2W(3.95, 1.15, 0.15)
TV_N = unit(A_AX + 0.10 * S_AX - 0.12 * D_AX)
TV_R = 1.50

AO_C = L2W(1.35, -1.15, -1.10)            # aortic annulus centre
AO_N = unit([-0.30, -0.15, 0.94])         # root axis (up, right, slightly anterior)
AO_R = 1.10
AO_H = 1.50                               # annulus -> sinotubular junction

PV_C = L2W(1.76, -4.30, -0.64)            # pulmonary annulus centre
PV_N = unit([0.14, 0.25, 0.96])
PV_R = 1.18
PV_H = 1.40

# --------------------------------------------------------------------------------------
# Atria
# --------------------------------------------------------------------------------------
RA_C = np.array([-3.45, -0.10, 1.10])
RA_RAD = np.array([1.95, 2.10, 2.50])
LA_C = np.array([0.30, 3.00, 2.70])
LA_RAD = np.array([2.50, 1.80, 2.00])

RAA_BASE = [((-3.6, -1.5, 3.2), 1.00), ((-3.3, -1.9, 1.9), 0.95), ((-2.6, -1.25, 3.3), 0.80)]
RAA_TIP = ((-1.55, -2.75, 3.85), 0.38)

LAA_PATH = [((2.3, 2.3, 3.5), 0.80), ((3.3, 1.3, 3.75), 0.70), ((3.72, 0.1, 3.62), 0.60),
            ((3.60, -1.1, 3.42), 0.48), ((3.32, -1.92, 3.28), 0.30)]

# --------------------------------------------------------------------------------------
# Great vessels: centreline control points (world, cm) with lumen radius
# --------------------------------------------------------------------------------------
def _ao(d):
    return tuple(AO_C + d * AO_N)


AORTA = [(_ao(-0.9), 1.05), (_ao(0.0), AO_R), (_ao(0.75), 1.16), (_ao(AO_H), 1.14), (_ao(3.2), 1.28),
         (_ao(5.2), 1.30), ((-1.25, -0.25, 9.15), 1.28), ((-0.25, 1.10, 10.05), 1.22),
         ((1.20, 2.90, 9.90), 1.15), ((2.30, 4.50, 8.95), 1.10), ((2.70, 5.35, 7.20), 1.07),
         ((2.80, 5.60, 5.20), 1.05), ((2.82, 5.65, 3.90), 1.05)]
AORTA_CUT = ((2.80, 5.60, 4.95), (0.02, 0.03, -1.0))

BCT = [((-0.80, 0.30, 9.4), 0.64), ((-1.20, 0.22, 10.6), 0.62), ((-1.50, 0.15, 11.6), 0.60)]
RSA = [((-1.50, 0.15, 11.5), 0.46), ((-2.25, 0.25, 12.3), 0.42), ((-3.30, 0.40, 13.0), 0.42)]
RCCA = [((-1.50, 0.15, 11.5), 0.42), ((-1.48, -0.05, 12.5), 0.38), ((-1.45, -0.15, 13.7), 0.38)]
LCCA = [((0.15, 1.45, 9.9), 0.42), ((0.33, 1.32, 11.5), 0.40), ((0.50, 1.22, 13.7), 0.40)]
LSA = [((1.10, 2.75, 9.8), 0.52), ((1.62, 2.92, 11.4), 0.50), ((2.45, 3.05, 13.7), 0.50)]
BRANCH_CUTS = [((-2.95, 0.35, 12.75), (-0.80, 0.10, 0.60)), ((-1.46, -0.12, 13.0), (0.0, -0.08, 1.0)),
               ((0.45, 1.25, 12.9), (0.08, -0.05, 1.0)), ((2.15, 3.0, 12.8), (0.42, 0.06, 0.9))]


def _pv(d):
    return tuple(PV_C + d * PV_N)


PT = [(_pv(-0.5), 1.20), (_pv(0.0), PV_R), (_pv(0.7), 1.22), ((1.58, -1.85, 5.35), 1.25),
      ((1.90, -0.90, 6.35), 1.25), ((2.20, 0.20, 6.90), 1.22)]
RPA = [((2.20, 0.20, 6.90), 1.02), ((0.90, 1.50, 6.75), 1.00), ((-1.00, 2.30, 6.60), 0.95),
       ((-3.00, 2.75, 6.50), 0.90), ((-5.00, 2.90, 6.40), 0.88), ((-6.2, 2.95, 6.35), 0.88)]
LPA = [((2.20, 0.20, 6.90), 0.98), ((3.30, 1.30, 7.10), 0.94), ((4.40, 2.50, 7.15), 0.90),
       ((5.40, 3.40, 7.00), 0.85), ((6.4, 4.2, 6.8), 0.85)]
RPA_CUT = ((-5.15, 2.92, 6.39), (-1.0, 0.05, -0.03))
LPA_CUT = ((5.45, 3.45, 6.99), (0.75, 0.63, -0.12))
LIGAMENT = [((2.85, 1.25, 7.85), 0.20), ((2.45, 2.40, 8.20), 0.17), ((1.95, 3.45, 8.62), 0.20)]

SVC = [((-3.90, 0.30, 11.0), 1.00), ((-3.90, 0.30, 7.0), 1.00), ((-3.78, 0.25, 4.6), 1.05),
       ((-3.60, 0.15, 3.0), 1.10)]
SVC_CUT = ((-3.90, 0.30, 9.7), (0.0, 0.0, 1.0))
IVC = [((-3.20, 1.75, -5.2), 1.15), ((-3.25, 1.60, -2.6), 1.15), ((-3.32, 1.05, -0.9), 1.22)]
IVC_CUT = ((-3.22, 1.72, -3.55), (0.0, 0.05, -1.0))

PVEINS = {
    'RSPV': [((-6.0, 3.65, 4.55), 0.74), ((-3.2, 3.50, 4.15), 0.76), ((-1.5, 3.30, 3.60), 0.82)],
    'RIPV': [((-6.0, 4.30, 1.95), 0.68), ((-3.2, 4.00, 2.10), 0.70), ((-1.5, 3.50, 2.20), 0.76)],
    'LSPV': [((6.4, 3.35, 4.85), 0.72), ((3.6, 3.30, 4.20), 0.75), ((2.2, 3.20, 3.60), 0.82)],
    'LIPV': [((6.4, 4.40, 2.45), 0.68), ((3.7, 4.10, 2.30), 0.70), ((2.2, 3.60, 2.30), 0.76)],
}
PVEIN_CUTS = {
    'RSPV': ((-4.95, 3.60, 4.42), (-1.0, 0.04, 0.12)), 'RIPV': ((-4.95, 4.20, 2.0), (-1.0, 0.1, -0.05)),
    'LSPV': ((5.10, 3.33, 4.6), (1.0, 0.02, 0.2)), 'LIPV': ((5.10, 4.28, 2.4), (1.0, 0.1, 0.05)),
}

WALLS = dict(aorta=0.20, pt=0.16, pa=0.14, cava=0.13, pvein=0.13, ra=0.22, la=0.25, raa=0.17, laa=0.17)

REGION = dict(LV=1, RV=2, RA=3, LA=4, AORTA=5, PA=6, CAVA=7, PVEIN=8, LIG=9)

GRID_LO = np.array([-6.6, -6.7, -6.5])
GRID_HI = np.array([7.5, 7.5, 13.6])


# --------------------------------------------------------------------------------------
# SDF toolkit
# --------------------------------------------------------------------------------------
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.minimum(a, b) - h * h * (k * 0.25)


def smax(a, b, k):
    if k <= 0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) * (1.0 / k)
    return np.maximum(a, b) + h * h * (k * 0.25)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sd_round_cone(X, Y, Z, a, b, r1, r2):
    """Exact SDF of a round cone (sphere-swept segment with linearly varying radius)."""
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    r1 = float(r1)
    r2 = float(r2)
    ba = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    l2 = ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2
    rr = r1 - r2
    a2 = l2 - rr * rr
    if l2 < 1e-10 or a2 <= 1e-9:
        c, r = (a, r1) if r1 >= r2 else (b, r2)
        return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) - r
    il2 = 1.0 / l2
    px, py, pz = X - a[0], Y - a[1], Z - a[2]
    y = px * ba[0] + py * ba[1] + pz * ba[2]
    z = y - l2
    qx = px * l2 - ba[0] * y
    qy = py * l2 - ba[1] * y
    qz = pz * l2 - ba[2] * y
    x2 = qx * qx + qy * qy + qz * qz
    del qx, qy, qz
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    d = (np.sqrt(x2 * (a2 * il2)) + y * rr) * il2 - r1
    c1 = np.sign(z) * a2 * z2 > k
    d = np.where(c1, np.sqrt(x2 + z2) * il2 - r2, d)
    c2 = (np.sign(y) * a2 * y2 < k) & ~c1
    d = np.where(c2, np.sqrt(x2 + y2) * il2 - r1, d)
    return d.astype(np.float32, copy=False)


def resample_path(ctrl, step=0.45):
    """ctrl: list of ((x,y,z), r).  Natural cubic spline, ~uniform arclength resampling."""
    from scipy.interpolate import CubicSpline, PchipInterpolator
    P = np.array([c[0] for c in ctrl], float)
    R = np.array([c[1] for c in ctrl], float)
    if len(P) == 2:
        n = max(2, int(np.ceil(np.linalg.norm(P[1] - P[0]) / step)) + 1)
        u = np.linspace(0, 1, n)[:, None]
        return P[0] * (1 - u) + P[1] * u, R[0] * (1 - u[:, 0]) + R[1] * u[:, 0]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    cs = CubicSpline(s, P, bc_type='natural')
    rs = PchipInterpolator(s, R)
    dense = np.linspace(0, s[-1], 400)
    Pd = cs(dense)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))])
    n = max(2, int(np.ceil(arc[-1] / step)) + 1)
    ua = np.linspace(0, arc[-1], n)
    sd = np.interp(ua, arc, dense)
    return cs(sd), rs(sd)


def path_length(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


class Grid:
    def __init__(self, lo=GRID_LO, hi=GRID_HI, h=VOX):
        self.lo = np.asarray(lo, float)
        self.h = float(h)
        self.n = np.ceil((np.asarray(hi, float) - self.lo) / h).astype(int) + 1
        self.ax = [(self.lo[i] + h * np.arange(self.n[i])).astype(np.float32) for i in range(3)]

    def new(self, val=FAR, dtype=np.float32):
        return np.full(tuple(self.n), val, dtype)

    def window(self, lo, hi):
        i0 = np.clip(np.floor((np.asarray(lo, float) - self.lo) / self.h).astype(int), 0, self.n - 1)
        i1 = np.clip(np.ceil((np.asarray(hi, float) - self.lo) / self.h).astype(int) + 1, 1, self.n)
        i1 = np.maximum(i1, i0 + 1)
        return tuple(slice(int(a), int(b)) for a, b in zip(i0, i1))

    def coords(self, sl):
        return (self.ax[0][sl[0]][:, None, None], self.ax[1][sl[1]][None, :, None],
                self.ax[2][sl[2]][None, None, :])

    def sample(self, F, P, order=1, stride=1):
        """Trilinear sample of grid field F (possibly strided) at world points P (N,3)."""
        from scipy.ndimage import map_coordinates
        idx = ((np.asarray(P, float) - self.lo) / (self.h * stride)).T
        return map_coordinates(F, idx, order=order, mode='nearest')


class Win:
    """A scalar field stored on a sub-window of the global grid."""

    def __init__(self, g, lo, hi, fill=FAR):
        self.g = g
        self.sl = g.window(lo, hi)
        self.shape = tuple(s.stop - s.start for s in self.sl)
        self.F = np.full(self.shape, fill, np.float32)

    @classmethod
    def around(cls, g, pts, pad):
        P = np.array([p for p in pts], float).reshape(-1, 3)
        return cls(g, P.min(0) - pad, P.max(0) + pad)

    @classmethod
    def like(cls, other, fill=FAR):
        w = cls.__new__(cls)
        w.g, w.sl, w.shape = other.g, other.sl, other.shape
        w.F = np.full(w.shape, fill, np.float32)
        return w

    def xyz(self):
        return self.g.coords(self.sl)

    def local(self):
        X, Y, Z = self.xyz()
        qx, qy, qz = X - float(O_L[0]), Y - float(O_L[1]), Z - float(O_L[2])
        out = []
        for ax in (S_AX, D_AX, A_AX):
            out.append((qx * float(ax[0]) + qy * float(ax[1]) + qz * float(ax[2])).astype(np.float32))
        return out

    def _sub(self, lo, hi):
        gs = self.g.window(lo, hi)
        loc, glob = [], []
        for s, m in zip(gs, self.sl):
            a, b = max(s.start, m.start), min(s.stop, m.stop)
            if b <= a:
                return None, None
            loc.append(slice(a - m.start, b - m.start))
            glob.append(slice(a, b))
        return tuple(loc), tuple(glob)

    def cone(self, a, b, r1, r2, pad=1.0):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        lo = np.minimum(a - r1, b - r2) - pad
        hi = np.maximum(a + r1, b + r2) + pad
        loc, glob = self._sub(lo, hi)
        if loc is None:
            return
        X, Y, Z = self.g.coords(glob)
        d = sd_round_cone(X, Y, Z, a, b, r1, r2)
        np.minimum(self.F[loc], d, out=self.F[loc])

    def canal(self, P, R, pad=1.0):
        for i in range(len(P) - 1):
            self.cone(P[i], P[i + 1], R[i], R[i + 1], pad)
        return self

    def path(self, ctrl, pad=1.0, step=0.45, dr=0.0):
        P, R = resample_path(ctrl, step)
        return self.canal(P, R + dr, pad)

    def sphere(self, c, r, pad=1.0):
        self.cone(c, c, r, r, pad)

    def ellipsoid(self, c, rad, rot=None):
        """Approximate SDF of an ellipsoid (accurate near the surface)."""
        X, Y, Z = self.xyz()
        px, py, pz = X - float(c[0]), Y - float(c[1]), Z - float(c[2])
        if rot is not None:
            Rm = np.asarray(rot, float)
            px, py, pz = [px * float(Rm[0, i]) + py * float(Rm[1, i]) + pz * float(Rm[2, i]) for i in range(3)]
        rx, ry, rz = [float(v) for v in rad]
        k0 = np.sqrt((px / rx) ** 2 + (py / ry) ** 2 + (pz / rz) ** 2)
        k1 = np.sqrt((px / rx ** 2) ** 2 + (py / ry ** 2) ** 2 + (pz / rz ** 2) ** 2)
        d = np.where(k1 > 1e-6, k0 * (k0 - 1.0) / np.maximum(k1, 1e-6), -min(rx, ry, rz))
        np.minimum(self.F, np.clip(d, -FAR, FAR).astype(np.float32), out=self.F)

    def get(self, other, fill=FAR):
        """Values of `other` (Win or full-grid array) resampled onto this window."""
        if isinstance(other, np.ndarray):
            return other[self.sl]
        out = np.full(self.shape, fill, np.float32)
        loc_me, loc_ot = [], []
        for m, o in zip(self.sl, other.sl):
            a, b = max(m.start, o.start), min(m.stop, o.stop)
            if b <= a:
                return out
            loc_me.append(slice(a - m.start, b - m.start))
            loc_ot.append(slice(a - o.start, b - o.start))
        out[tuple(loc_me)] = other.F[tuple(loc_ot)]
        return out

    def full(self, fill=FAR):
        out = self.g.new(fill)
        out[self.sl] = self.F
        return out


def half_space(X, Y, Z, p, n):
    """Signed distance to plane through p with unit normal n (positive on the +n side)."""
    n = unit(n)
    return ((X - float(p[0])) * float(n[0]) + (Y - float(p[1])) * float(n[1]) + (Z - float(p[2])) * float(n[2]))


def pw_noise(P, seed, lam=(1.0, 1.0, 1.0), K=22):
    """Band-limited Gaussian-like noise built from K random plane waves.  P: (N,3) array.
    `lam` are the characteristic wavelengths along each input axis."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(K, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    f = rng.uniform(0.75, 1.3, size=K)
    q = (2 * np.pi) * d * f[:, None] / np.asarray(lam, float)[None, :]
    ph = rng.uniform(0, 2 * np.pi, size=K)
    out = np.zeros(len(P), np.float32)
    P32 = np.asarray(P, np.float32)
    for k in range(K):
        out += np.cos(P32 @ q[k].astype(np.float32) + np.float32(ph[k]))
    return out * np.float32(math.sqrt(2.0 / K))


# --------------------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------------------
def bpath(name):
    return os.path.join(BUILD, name)


def save(name, arr):
    np.save(bpath(name + '.npy'), arr)


def load(name, mmap=False):
    return np.load(bpath(name + '.npy'), mmap_mode='r' if mmap else None)


def save_json(name, obj):
    with open(bpath(name + '.json'), 'w') as f:
        json.dump(obj, f)


def load_json(name):
    with open(bpath(name + '.json')) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Physiological displacement fields (shared by every mesh so nothing interpenetrates)
# --------------------------------------------------------------------------------------
class Motion:
    """Evaluates the shape-key displacement fields at arbitrary world points.

    ventricular_contraction : base descends towards the apex, the walls thicken inwards
                              (endocardium >> epicardium) and the apex twists counter-clockwise
                              against the base (helical torsion), RV free wall bellows towards
                              the septum.
    atrial_systole          : atria + appendages shrink towards the AV orifices, the ventricles
                              receive the atrial kick (slight end-diastolic expansion).
    """

    def __init__(self):
        d = np.load(bpath('motion_fields.npz'))
        self.stride = int(d['stride'])
        self.g = Grid()
        self.ra = d['ra']
        self.la = d['la']
        self.rvfree = d['rvfree']

    def _s(self, F, P):
        return self.g.sample(F, P, order=1, stride=self.stride)

    def local(self, P):
        q = np.asarray(P, float) - O_L
        return q @ S_AX, q @ D_AX, q @ A_AX

    def ventricular(self, P):
        P = np.asarray(P, float)
        sig, dlt, t = self.local(P)
        L = LV_APEX_T
        # --- longitudinal: AV plane pulled 1.25 cm towards the (almost fixed) apex
        tt = np.clip(t / L, -1.0, 1.0)
        long_w = np.where(t >= 0, (1 - tt) ** 1.15, smoothstep(-4.2, 0.0, t))
        d_long = 1.25 * long_w
        # --- radial thickening about the LV axis
        bow = 0.18 * np.sin(np.clip(t / 7.3, 0, 1) * np.pi)
        rs, rd = sig + bow, dlt
        rho = np.sqrt(rs * rs + rd * rd) + 1e-6
        tc = np.clip(t, 0, 7.3)
        rc, rw = lv_rc(tc), lv_wall(tc)
        x = np.clip((rho - rc * 0.55) / (rc * 0.45 + rw), 0.0, 1.0)     # 0 mid-cavity -> 1 epicardium
        c_rad = 0.30 - 0.235 * smoothstep(0.0, 1.0, x)
        vent_w = smoothstep(-0.9, 0.3, t) * (1 - 0.35 * smoothstep(6.6, 8.4, t))
        scale = -c_rad * vent_w
        # --- RV free wall bellows towards the septum
        rvw = self._s(self.rvfree, P) * vent_w
        scale = scale - 0.085 * rvw
        # --- torsion: apex CCW (seen from the apex), base CW
        ang = np.radians(13.0) * smoothstep(0.5, L, t) - np.radians(4.0) * smoothstep(-2.5, 0.2, t) * (
            1 - smoothstep(0.5, 3.5, t))
        ang = ang * (1 - 0.25 * rvw)
        ca, sa = np.cos(ang), np.sin(ang)
        ns = rs * (1 + scale)
        nd = rd * (1 + scale)
        rs2 = ca * ns - sa * nd
        rd2 = sa * ns + ca * nd
        disp = (np.outer(rs2 - rs, S_AX) + np.outer(rd2 - rd, D_AX) + np.outer(d_long, A_AX))
        # the atria are stretched (reservoir phase) rather than displaced rigidly
        return disp

    def atrial(self, P):
        P = np.asarray(P, float)
        disp = np.zeros_like(P)
        for F, c, anchor, amt in ((self.ra, RA_C, TV_C, 0.20), (self.la, LA_C, MV_C, 0.17)):
            w = self._s(F, P)
            # contract towards a point between the atrial centre and the AV orifice
            tgt = c * 0.6 + anchor * 0.4
            disp += (tgt - P) * (amt * w)[:, None]
        # atrial kick: ventricles expand a little, AV plane lifts
        sig, dlt, t = self.local(P)
        vent_w = smoothstep(-0.3, 0.8, t) * (1 - smoothstep(6.0, 8.4, t))
        rho = np.sqrt(sig * sig + dlt * dlt) + 1e-6
        k = 0.022 * vent_w
        disp += np.outer(sig * k, S_AX) + np.outer(dlt * k, D_AX)
        disp += np.outer(-0.12 * smoothstep(-3.0, 0.0, t) * (1 - smoothstep(0.0, 5.0, t)), A_AX)
        return disp


# --------------------------------------------------------------------------------------
# Derived landmarks shared by several stages
# --------------------------------------------------------------------------------------
_v = AO_C - MV_C
_v = _v - (_v @ MV_N) * MV_N
MV_E2 = unit(_v)                        # towards the aortic valve (anterior leaflet side)
MV_E1 = np.cross(MV_N, MV_E2)           # commissural axis

_v = L2W(0, 0, 0.15) - TV_C
_v = _v - (_v @ TV_N) * TV_N
TV_E2 = unit(_v)                        # towards the septum (septal leaflet side)
TV_E1 = np.cross(TV_N, TV_E2)


def lv_cyl(phi, t, inset):
    """Point inside the LV cavity: angle phi about the axis, `inset` cm from the compact wall."""
    c = lv_axis(float(t))
    rho = float(lv_rc(t)) - inset
    return c + rho * (math.cos(phi) * S_AX + math.sin(phi) * D_AX)


def lv_epi(phi, t, off=0.0):
    c = lv_axis(float(t))
    rho = float(lv_rc(t) + lv_wall(t)) + off
    return c + rho * (math.cos(phi) * S_AX + math.sin(phi) * D_AX)


def rv_free(psi, t, inset):
    """Point `inset` cm inside the RV free wall at blob angle psi (0 = straight away from the septum)."""
    rho = float(rv_q(t)) - inset
    return L2W(float(rv_c(t)) + rho * math.cos(psi), RV_DELTA0 + rho * math.sin(psi), float(t))


def _phi_of(P):
    s, d, _ = W2L(P)
    return math.atan2(d, s)


PHI_PM = _phi_of(MV_C + 1.3 * MV_E1)        # posteromedial papillary muscle
PHI_AL = _phi_of(MV_C - 1.3 * MV_E1)        # anterolateral papillary muscle


def papillary_muscles():
    """Returns {name: dict(ctrl=[(P, r)...] trunk, heads=[[(P, r)...], ...], tips=[P, ...])}."""
    out = {}
    for name, phi in (('LV_PM_anterolateral', PHI_AL), ('LV_PM_posteromedial', PHI_PM)):
        trunk = [(lv_cyl(phi, 6.05, 0.28), 0.50), (lv_cyl(phi, 5.2, 0.55), 0.58),
                 (lv_cyl(phi, 4.3, 0.76), 0.54), (lv_cyl(phi, 3.75, 0.86), 0.46)]
        heads, tips = [], []
        for dphi in (-0.24, 0.24):
            tip = lv_cyl(phi + dphi, 3.0, 0.92)
            heads.append([(lv_cyl(phi, 3.75, 0.86), 0.44), (lv_cyl(phi + 0.6 * dphi, 3.35, 0.90), 0.34),
                          (tip, 0.21)])
            tips.append(tip)
        out[name] = dict(trunk=trunk, heads=heads, tips=tips)
    r = math.radians
    # RV anterior papillary muscle (largest, receives the moderator band)
    tipA = rv_free(r(-26), 3.05, 1.02)
    out['RV_PM_anterior'] = dict(
        trunk=[(rv_free(r(-38), 4.65, 0.15), 0.46), (rv_free(r(-33), 3.9, 0.62), 0.42)],
        heads=[[(rv_free(r(-33), 3.9, 0.62), 0.40), (rv_free(r(-29), 3.4, 0.9), 0.32), (tipA, 0.2)]],
        tips=[tipA])
    tipP = rv_free(r(50), 2.75, 0.72)
    out['RV_PM_posterior'] = dict(
        trunk=[(rv_free(r(54), 4.1, 0.12), 0.36), (rv_free(r(52), 3.4, 0.5), 0.32)],
        heads=[[(rv_free(r(52), 3.4, 0.5), 0.30), (tipP, 0.18)]],
        tips=[tipP])
    tipS = lv_epi(r(-30), 1.55, 0.62)
    out['RV_PM_septal'] = dict(
        trunk=[(lv_epi(r(-28), 2.0, -0.15), 0.30), (lv_epi(r(-29), 1.75, 0.3), 0.25)],
        heads=[[(lv_epi(r(-29), 1.75, 0.3), 0.24), (tipS, 0.15)]],
        tips=[tipS])
    return out


MODERATOR_BAND = [(lv_epi(math.radians(-10), 4.25, -0.15), 0.24),
                  ((lv_epi(math.radians(-16), 4.4, 0.7) + rv_free(math.radians(-38), 4.6, 0.6)) / 2 + 0.25 * A_AX, 0.19),
                  (rv_free(math.radians(-38), 4.6, 0.3), 0.24)]


def merge_outer(outer, region, win, o, rid, k):
    """Smooth-union the shell `o` (window field) into the global outer envelope and tag the region."""
    cur = outer[win.sl]
    reg = region[win.sl]
    reg[o < cur] = rid
    outer[win.sl] = smin(cur, o, k)


def add_feature(feat, win, d):
    np.minimum(feat[win.sl], d, out=feat[win.sl])


AO_CUSP_DEG = (-15.0, 105.0, 225.0)      # right coronary, left coronary, non-coronary
PV_CUSP_DEG = (0.0, 120.0, 240.0)        # anterior, left, right
AO_CUSP_NAMES = ('right_coronary', 'left_coronary', 'non_coronary')
PV_CUSP_NAMES = ('anterior', 'left', 'right')


def root_frame(n):
    """(anterior, left, axis) frame of an arterial root."""
    return perp_frame(n, (0.0, -1.0, 0.0))


def cusp_dir(n, deg):
    e1, e2, _ = root_frame(n)
    a = math.radians(deg)
    return math.cos(a) * e1 + math.sin(a) * e2


def mesh_from_arrays(name, verts, faces, collection=None):
    import bpy
    me = bpy.data.meshes.new(name)
    verts = np.ascontiguousarray(verts, np.float32)
    faces = np.ascontiguousarray(faces, np.int32)
    me.vertices.add(len(verts))
    me.vertices.foreach_set('co', verts.ravel())
    nv = faces.shape[1]
    me.loops.add(faces.size)
    me.loops.foreach_set('vertex_index', faces.ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set('loop_start', np.arange(0, faces.size, nv, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(len(faces), nv, np.int32))
    me.polygons.foreach_set('use_smooth', np.ones(len(faces), bool))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def mesh_arrays(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', v)
    return v.reshape(-1, 3)


def extract_surface(g, S, level=0.0):
    """Marching cubes (the 'voxel remesh') of field S; returns world-space verts, tri faces (outward)."""
    from skimage.measure import marching_cubes
    v, f, _, _ = marching_cubes(S, level, spacing=(g.h, g.h, g.h), gradient_direction='ascent')
    v = v + g.lo[None, :]
    # make sure the winding is outward (positive signed volume)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0
    if vol < 0:
        f = f[:, ::-1]
    return v.astype(np.float32), np.ascontiguousarray(f.astype(np.int32)), abs(vol)


class Surf:
    """Queries on the epicardial envelope SDF (stage-02 `outer` field, strided)."""

    def __init__(self):
        fd = np.load(bpath('fields02.npz'))
        self.st = int(fd['stride'])
        self.g = Grid()
        self.OUT = fd['outer'].astype(np.float32)
        self.BLOOD = fd['blood'].astype(np.float32)

    def val(self, P, F=None):
        return self.g.sample(self.OUT if F is None else F, P, stride=self.st)

    def grad(self, P, e=0.06):
        G = np.zeros_like(P)
        for i in range(3):
            d = np.zeros(3)
            d[i] = e
            G[:, i] = (self.val(P + d) - self.val(P - d)) / (2 * e)
        return G

    def normals(self, P):
        G = self.grad(P)
        return G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)

    def project(self, P, off=0.0, it=10):
        P = np.array(P, float).reshape(-1, 3)
        for _ in range(it):
            G = self.grad(P)
            gn = np.linalg.norm(G, axis=1) + 1e-6
            step = np.clip((self.val(P) - off) / gn, -0.35, 0.35)
            P -= (step / gn)[:, None] * G
        return P


def add_motion_keys(ob, motion):
    """Adds the two myocardial shape keys (shared displacement field) to a mesh object."""
    P = mesh_arrays(ob).astype(np.float64)
    if ob.data.shape_keys is None:
        ob.shape_key_add(name='Basis')
    for name, fn in (('atrial_systole', motion.atrial), ('ventricular_contraction', motion.ventricular)):
        k = ob.shape_key_add(name=name)
        k.data.foreach_set('co', np.ascontiguousarray(P + fn(P), np.float32).ravel())
