"""
01_chambers_myocardium.py
Parametric / signed-distance construction of the four cardiac chambers:
  * LV: thick-walled (12-13 mm) bullet;  RV: thin-walled (4 mm) crescent wrapped around a
    septum that bulges into the RV;  atria with their appendages.
  * endocardial relief: trabeculae carneae (fine in the LV, coarse in the RV), papillary muscles,
    moderator band, pectinate muscles, crista terminalis, fossa ovalis + limbus.
The script only produces volumetric fields (cached in $HEART_BUILD); stage 02 adds the great
vessels, performs the boolean union / subtraction and extracts the organic manifold surface
(voxel remesh == marching cubes of the composed field).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heartlib import *      # noqa

g = Grid()
log('stage 01 — grid', g.n.tolist(), '= %.1f M voxels @ %.2f mm' % (np.prod(g.n) / 1e6, VOX * 10))

outer = g.new()
region = g.new(0, np.uint8)
feat = g.new()

# =====================================================================================
# Ventricles
# =====================================================================================
corners = [L2W(s, d, t) for s in (-3.7, 6.6) for d in (-6.4, 4.6) for t in (-2.3, 8.8)]
V = Win.around(g, corners, 0.1)
sig, dlt, t = V.local()
log('ventricular window', V.shape)

tt = np.linspace(0.0, 7.3, 19)
lvP = lv_axis(tt)
lv_c = Win.like(V).canal(lvP, lv_rc(tt), pad=1.4)
lv_o = Win.like(V).canal(lvP, lv_rc(tt) + lv_wall(tt), pad=1.0)

lvot_ctrl = [(L2W(0.75, -0.60, 1.3), 1.00), (L2W(1.10, -0.95, 0.0), 1.08), (AO_C, AO_R), (AO_C + 0.4 * AO_N, AO_R)]
lvot = Win.like(V).path(lvot_ctrl, pad=1.0)

lv_cav = smax(lv_c.F, -(t + 0.15), 0.5)
lum_LV = smin(lv_cav, lvot.F, 0.35)
out_LV = smax(lv_o.F, -(t + 0.55), 0.6)
out_LV = np.minimum(out_LV, lvot.F - 0.28)
log('LV done')

rt = np.array([0.2, 0.9, 1.5, 2.2, 3.0, 3.8, 4.5, 5.1, 5.6, 6.0, 6.4])
rvP = np.array([L2W(rv_c(x), RV_DELTA0, x) for x in rt])
rv_b = Win.like(V).canal(rvP, rv_q(rt), pad=1.0)
rv_blob = smax(smax(rv_b.F, -(t + 0.25), 0.5), t - RV_APEX_T, 0.9)
rv_body = smax(rv_blob, -lv_o.F, 0.12)                       # septum = LV wall, convex towards the RV

inl = Win.like(V).path([(TV_C - 0.3 * TV_N, 1.50), (TV_C + 1.7 * TV_N, 1.45)], pad=1.0)
inlet = smax(smax(inl.F, -(lv_c.F - 0.65), 0.1), -(t + 0.25), 0.3)

con_ctrl = [(L2W(2.9, -1.6, 2.6), 1.20), (L2W(2.8, -3.3, 0.9), 1.30), (PV_C - 0.9 * PV_N, 1.22),
            (PV_C, PV_R), (PV_C + 0.4 * PV_N, PV_R)]
cP, cR = resample_path(con_ctrl, 0.4)
con = Win.like(V).canal(cP, cR, pad=1.0)
cwall = np.interp(np.linspace(0, 1, len(cP)), [0, 0.55, 0.85, 1.0], [0.42, 0.42, 0.26, 0.18])
con_o = Win.like(V).canal(cP, cR + cwall, pad=0.8)
conus = smax(con.F, -(lum_LV - 0.6), 0.1)

lum_RV = smin(smin(rv_body, inlet, 0.4), conus, 0.45)
out_RVb = smax(rv_blob - RV_WALL, -(t + 0.6), 0.6)
out_RVb = smin(out_RVb, smax(inl.F - RV_WALL, -(t + 0.6), 0.4), 0.3)
out_RV = smin(out_RVb, con_o.F, 0.35)
log('RV done')

merge_outer(outer, region, V, out_LV, REGION['LV'], 0.0)
merge_outer(outer, region, V, out_RV, REGION['RV'], 0.30)

# free-wall weight for the RV bellows motion
rvfree_V = (smoothstep(0.0, 1.2, lv_o.F) * smoothstep(0.6, -0.2, out_RV)).astype(np.float32)

# =====================================================================================
# Atria
# =====================================================================================
raa_pts = [b[0] for b in RAA_BASE] + [RAA_TIP[0]]
RA = Win.around(g, [RA_C - RA_RAD, RA_C + RA_RAD] + raa_pts, 1.4)
ra_body = Win.like(RA)
ra_body.ellipsoid(RA_C, RA_RAD)
raa = Win.like(RA)
for (b, r) in RAA_BASE:
    raa.cone(b, RAA_TIP[0], r, RAA_TIP[1], pad=0.9)
mid = (np.array(RAA_BASE[0][0]) + np.array(RAA_BASE[1][0])) / 2
raa.cone(mid, (mid + np.array(RAA_TIP[0])) / 2, 0.95, 0.7, pad=0.9)
# crenellated, ear-like auricle
X, Y, Z = RA.xyz()
idx = np.nonzero(raa.F < 0.6)
Pn = np.stack([np.broadcast_to(a, RA.shape)[idx] for a in (X, Y, Z)], axis=1)
raa.F[idx] += 0.075 * pw_noise(Pn, 5, (0.62, 0.62, 0.62), K=14) * smoothstep(0.8, -0.2, ra_body.F[idx])
lum_RA = smin(ra_body.F, raa.F, 0.45)
out_RA = np.minimum(smin(ra_body.F - WALLS['ra'], raa.F - WALLS['raa'], 0.35), lum_RA - WALLS['raa'])

laa_pts = [p[0] for p in LAA_PATH]
LA = Win.around(g, [LA_C - LA_RAD, LA_C + LA_RAD] + laa_pts, 1.4)
la_body = Win.like(LA)
la_body.ellipsoid(LA_C, LA_RAD)
laa = Win.like(LA).path(LAA_PATH, pad=0.9, step=0.3)
# lobulated (crenellated) appendage: small side lobes along the path
lP, lR = resample_path(LAA_PATH, 0.42)
rng = np.random.default_rng(7)
for i in range(2, len(lP) - 1):
    tng = unit(lP[min(i + 1, len(lP) - 1)] - lP[i - 1])
    e1, e2, _ = perp_frame(tng, (0, 0, 1))
    for sgn in (-1, 1):
        ang = rng.uniform(-0.5, 0.5) + (0 if sgn > 0 else np.pi)
        c = lP[i] + (lR[i] * 0.62) * (np.cos(ang) * e1 + np.sin(ang) * e2) + rng.normal(0, 0.05, 3)
        laa.sphere(c, lR[i] * rng.uniform(0.5, 0.66), pad=0.6)
lum_LA = smin(la_body.F, laa.F, 0.4)
out_LA = np.minimum(smin(la_body.F - WALLS['la'], laa.F - WALLS['laa'], 0.3), lum_LA - WALLS['laa'])

# keep the chambers separated (interatrial septum, AV fibrous plane, infundibular septum)
LVw = Win.like(V); LVw.F = lum_LV
RVw = Win.like(V); RVw.F = lum_RV
lum_LA = smax(lum_LA, -(LA.get(LVw) - 0.5), 0.1)
lum_LA = smax(lum_LA, -(LA.get(RVw) - 0.5), 0.1)
LAw = Win.like(LA); LAw.F = lum_LA
lum_RA = smax(lum_RA, -(RA.get(RVw) - 0.5), 0.1)
lum_RA = smax(lum_RA, -(RA.get(LVw) - 0.45), 0.1)
lum_RA = smax(lum_RA, -(RA.get(LAw) - 0.32), 0.08)
log('atria done')

# ---- fossa ovalis (thinned oval floor in the interatrial septum) + limbus -----------
f_dir = unit([0.76, 0.60, 0.10])
ray = RA_C[None, :] + np.linspace(0.3, 3.2, 300)[:, None] * f_dir[None, :]
tmp = g.new(); tmp[RA.sl] = lum_RA
vals = g.sample(tmp, ray)
hit = ray[int(np.argmax(vals > 0))] if np.any(vals > 0) else RA_C + 1.9 * f_dir
del tmp
fe1, fe2, _ = perp_frame(f_dir, (0, 0, 1))
X, Y, Z = RA.xyz()
px, py, pz = X - float(hit[0]), Y - float(hit[1]), Z - float(hit[2])
u = px * float(f_dir[0]) + py * float(f_dir[1]) + pz * float(f_dir[2])
a1 = px * float(fe1[0]) + py * float(fe1[1]) + pz * float(fe1[2])
a2 = px * float(fe2[0]) + py * float(fe2[1]) + pz * float(fe2[2])
k0 = np.sqrt((u / 0.17) ** 2 + (a1 / 0.95) ** 2 + (a2 / 0.70) ** 2)
lum_RA = smin(lum_RA, ((k0 - 1.0) * 0.17).astype(np.float32), 0.08)
limb = Win.like(RA)
th = np.radians(np.linspace(-50, 230, 26))
lpts = [hit + 0.03 * f_dir + 1.02 * np.sin(a) * fe1 + 0.78 * np.cos(a) * fe2 for a in th]
limb.canal(lpts, [0.115] * len(lpts), pad=0.5)
add_feature(feat, RA, limb.F)
del px, py, pz, u, a1, a2, k0
FOSSA = dict(center=hit.tolist(), normal=f_dir.tolist())
log('fossa ovalis at', np.round(hit, 2))

merge_outer(outer, region, RA, out_RA, REGION['RA'], 0.25)
merge_outer(outer, region, LA, out_LA, REGION['LA'], 0.25)

# ---- AV orifices ---------------------------------------------------------------------
ORI = Win.around(g, [TV_C, MV_C], 3.6)
ori = Win.like(ORI)
ori.cone(TV_C - 1.5 * TV_N, TV_C + 0.6 * TV_N, 1.45, TV_R, pad=0.8)
for sgn in (-1, 1):
    ori.cone(MV_C - 1.4 * MV_N + sgn * 0.27 * MV_E1, MV_C + 0.6 * MV_N + sgn * 0.27 * MV_E1, 1.28, 1.32, pad=0.8)
# aorto-mitral curtain: above the annulus the mitral inflow never communicates with the LVOT
tO = ORI.local()[2]
LVOTw = Win.like(V); LVOTw.F = lvot.F
ori.F = smax(ori.F, -np.maximum(ORI.get(LVOTw) - 0.24, tO - 0.3), 0.06)
# fibrous annuli: make sure the orifices are always walled
merge_outer(outer, region, ORI, smax(ori.F - 0.32, tO - 0.3, 0.3), REGION['RA'], 0.3)
X, Y, Z = ORI.xyz()
region[ORI.sl][(region[ORI.sl] == REGION['RA']) & (half_space(X, Y, Z, (MV_C + TV_C) / 2, MV_C - TV_C) > 0)] = REGION['LA']

# =====================================================================================
# Endocardial relief
# =====================================================================================
# ---- papillary muscles + moderator band ---------------------------------------------
pm = Win.like(V)
for name, d in papillary_muscles().items():
    P, R = resample_path(d['trunk'], 0.3)
    pm.canal(P, R, pad=0.6)
    for h in d['heads']:
        P, R = resample_path(h, 0.25)
        pm.canal(P, R, pad=0.6)
mb = Win.like(V).path(MODERATOR_BAND, pad=0.5, step=0.3)
add_feature(feat, V, smin(pm.F, mb.F, 0.15))
log('papillary muscles + moderator band')


def trabeculae(c, m, hs, hr, lam, ws, lamr, wr, stretch, seed):
    """Trabecular meshwork inside the layer 0 > c > -h of cavity field c (window V)."""
    mask = (c > -(hs + 0.12)) & (c < 0.14) & (m > 0.02)
    idx = np.nonzero(mask)
    Pn = np.stack([sig[idx], dlt[idx], t[idx]], axis=1)
    lam3 = (lam, lam, lam * stretch)
    n1 = pw_noise(Pn, seed, lam3)
    n2 = pw_noise(Pn, seed + 1, lam3)
    n3 = pw_noise(Pn, seed + 2, (lamr, lamr, lamr * stretch * 1.3))
    cc = c[idx]
    mm = m[idx]
    d_str = np.sqrt(n1 * n1 + n2 * n2) * (lam / (2 * np.pi)) * 1.05 - ws * (0.75 + 0.5 * mm)
    d_rdg = np.abs(n3) * (lamr / (2 * np.pi)) * 1.05 - wr
    d = np.minimum(np.maximum(d_str, -(cc + hs * mm)), np.maximum(d_rdg, -(cc + hr * mm)))
    out = np.full(c.shape, FAR, np.float32)
    out[idx] = d
    log('   trabecular voxels: %.2f M' % (len(cc) / 1e6))
    return out


phi = np.arctan2(dlt, sig)
m_lv = smoothstep(0.9, 2.6, t) * smoothstep(0.0, 0.6, lvot.F)
m_lv = m_lv * (1 - 0.8 * smoothstep(0.45, 0.85, np.cos(phi)) * (1 - smoothstep(2.6, 4.2, t)))   # smooth upper septum
tr_lv = trabeculae(lum_LV, m_lv.astype(np.float32), 0.40, 0.22, 0.60, 0.088, 0.55, 0.062, 2.6, 11)
add_feature(feat, V, tr_lv)
m_rv = smoothstep(0.6, 1.9, t) * smoothstep(0.05, 0.9, con.F)       # smooth infundibulum
tr_rv = trabeculae(lum_RV, m_rv.astype(np.float32), 0.58, 0.32, 0.95, 0.12, 0.85, 0.09, 2.2, 23)
add_feature(feat, V, tr_rv)
del tr_lv, tr_rv, phi, m_lv, m_rv
log('trabeculae carneae')

# ---- pectinate muscles + crista terminalis (right atrium) ---------------------------
X, Y, Z = RA.xyz()
ux, uy, uz = (X - float(RA_C[0])) / float(RA_RAD[0]), (Y - float(RA_C[1])) / float(RA_RAD[1]), (Z - float(RA_C[2])) / float(RA_RAD[2])
un = np.sqrt(ux * ux + uy * uy + uz * uz) + 1e-5
dirv = (-0.6 * ux - 0.8 * uy) / un
reg = np.minimum((0.12 - dirv) * 2.0, raa.F - 0.25)                  # < 0 inside the pectinate territory
idx = np.nonzero((lum_RA > -0.45) & (lum_RA < 0.15))
Pn = np.stack([np.broadcast_to(a_, RA.shape)[idx] for a_ in (X, Y, Z)], axis=1)
warp = np.zeros(RA.shape, np.float32); warp[idx] = pw_noise(Pn, 31, (1.5, 1.5, 1.1), K=12)
wid = np.zeros(RA.shape, np.float32); wid[idx] = pw_noise(Pn, 32, (0.9, 0.9, 0.9), K=12)
gcoord = Z + 0.22 * X - 0.10 * Y + 0.16 * warp          # ridges fan out from the crista, slightly wavy
lam_p = 0.52
tri = np.abs(np.mod(gcoord / lam_p, 1.0) - 0.5) * lam_p
pect = np.maximum(np.maximum(tri - (0.085 + 0.03 * wid), -(lum_RA + 0.27 + 0.07 * wid)), reg).astype(np.float32)
del warp, wid
add_feature(feat, RA, pect)
al = np.radians(153)
crista = Win.like(RA)
cpts = [RA_C + 0.97 * RA_RAD * np.array([np.cos(al + 0.25 * b) * np.cos(b), np.sin(al + 0.25 * b) * np.cos(b), np.sin(b)])
        for b in np.radians(np.linspace(64, -60, 22))]
crista.canal(cpts, [0.2] * len(cpts), pad=0.5)
add_feature(feat, RA, crista.F)
del ux, uy, uz, un, dirv, reg, gcoord, tri, pect

# ---- pectinate ridges inside the left atrial appendage ------------------------------
X, Y, Z = LA.xyz()
gcoord = 0.92 * Y + 0.30 * X + 0.2 * Z
tri = np.abs(np.mod(gcoord / 0.36, 1.0) - 0.5) * 0.36
pect = np.maximum(np.maximum(tri - 0.062, -(lum_LA + 0.17)), laa.F - 0.12).astype(np.float32)
pect = np.maximum(pect, -(la_body.F - 0.35))                         # appendage only
add_feature(feat, LA, pect)
log('pectinate muscles, crista terminalis')

# =====================================================================================
# Cache
# =====================================================================================
for name, win, F in (('lum_LV', V, lum_LV), ('lum_RV', V, lum_RV), ('lum_RA', RA, lum_RA), ('lum_LA', LA, lum_LA),
                     ('orif', ORI, ori.F)):
    full = g.new()
    full[win.sl] = F
    save(name, full)
    del full
save('outer01', outer)
save('region01', region)
save('feat01', feat)

st = 3
ra_w = g.new(0.0); la_w = g.new(0.0); rv_w = g.new(0.0)
tR = RA.local()[2]
ra_w[RA.sl] = smoothstep(0.9, 0.1, out_RA) * smoothstep(-0.3, -1.6, tR)
tL = LA.local()[2]
la_w[LA.sl] = smoothstep(0.9, 0.1, out_LA) * smoothstep(-0.3, -1.6, tL)
rv_w[V.sl] = rvfree_V
np.savez_compressed(bpath('motion_fields.npz'), stride=st, ra=ra_w[::st, ::st, ::st], la=la_w[::st, ::st, ::st],
                    rvfree=rv_w[::st, ::st, ::st])
save_json('landmarks01', dict(fossa=FOSSA))
log('stage 01 complete')
