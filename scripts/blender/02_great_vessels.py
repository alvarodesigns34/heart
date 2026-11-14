"""
02_great_vessels.py
Great vessels with real wall thickness and their organic (seamless) union with the
atrio-ventricular block:
  aorta (3 sinuses of Valsalva, ascending, arch with brachiocephalic / left common carotid /
  left subclavian, descending), pulmonary trunk + right/left pulmonary arteries, ligamentum
  arteriosum, superior & inferior venae cavae (Eustachian valve), four pulmonary veins.
Then: boolean composition  solid = outer_envelope \ blood_pool, flat vessel stumps, gaussian
relax, connected-component clean-up, marching cubes -> manifold mesh, quadric decimation.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heartlib import *      # noqa
from scipy import ndimage as ndi

g = Grid()
log('stage 02 — grid', g.n.tolist())
outer = load('outer01')
region = load('region01')
feat = load('feat01')
lum = {k: load('lum_' + k) for k in ('LV', 'RV', 'RA', 'LA')}
orif = load('orif')
FLOW = {}


def vessel(group, ctrl, wall, rid, k_lum=0.3, k_out=0.2, extra=None, name=None, cut=None):
    P, R = resample_path(ctrl, 0.42)
    w = Win.around(g, P, float(R.max()) + wall + 1.3)
    w.canal(P, R, pad=wall + 1.0)
    if extra is not None:
        extra(w)
    if group is not None:
        newl = smin(lum[group][w.sl], w.F, k_lum)
        lum[group][w.sl] = newl
        # wall follows the blended lumen (never thins at junctions), but only next to this vessel
        o = np.maximum(newl - wall, w.F - wall - 0.35)
        if cut is not None:                       # flat anatomical stump: the wall ends on a plane
            X, Y, Z = w.xyz()
            pc = [float(v) for v in cut[0]]
            dist = np.sqrt((X - pc[0]) ** 2 + (Y - pc[1]) ** 2 + (Z - pc[2]) ** 2)
            rc = 2.3 * (float(R.max()) + wall)
            o = np.maximum(o, np.minimum(half_space(X, Y, Z, cut[0], cut[1]), rc - dist))
        merge_outer(outer, region, w, o, rid, k_out)
    else:
        merge_outer(outer, region, w, w.F, rid, k_out)
    if name:
        FLOW[name] = dict(P=np.round(P, 3).tolist(), R=np.round(R, 3).tolist())
    return w


def sinuses(c, n, degs, off, rad, h):
    def f(w):
        s = Win.like(w)
        for dg in degs:
            s.sphere(c + h * n + off * cusp_dir(n, dg), rad, pad=1.0)
        w.F = smin(w.F, s.F, 0.28)
    return f


# ---- arterial trunks -------------------------------------------------------------------
vessel('LV', AORTA, WALLS['aorta'], REGION['AORTA'], extra=sinuses(AO_C, AO_N, AO_CUSP_DEG, 0.66, 0.84, 0.72), name='aorta', cut=AORTA_CUT)
for nm, br, ct in (('bct', BCT, None), ('rsa', RSA, BRANCH_CUTS[0]), ('rcca', RCCA, BRANCH_CUTS[1]),
                   ('lcca', LCCA, BRANCH_CUTS[2]), ('lsa', LSA, BRANCH_CUTS[3])):
    vessel('LV', br, 0.13, REGION['AORTA'], k_lum=0.18, k_out=0.25, name=nm, cut=ct)
vessel('RV', PT, WALLS['pt'], REGION['PA'], extra=sinuses(PV_C, PV_N, PV_CUSP_DEG, 0.64, 0.74, 0.62), name='pt')
vessel('RV', RPA, WALLS['pa'], REGION['PA'], k_lum=0.35, k_out=0.3, name='rpa', cut=RPA_CUT)
vessel('RV', LPA, WALLS['pa'], REGION['PA'], k_lum=0.35, k_out=0.3, name='lpa', cut=LPA_CUT)
vessel(None, LIGAMENT, 0.0, REGION['LIG'], k_out=0.22)
log('arteries')

# ---- veins ---------------------------------------------------------------------------
vessel('RA', SVC, WALLS['cava'], REGION['CAVA'], k_lum=0.6, k_out=0.45, name='svc', cut=SVC_CUT)
vessel('RA', IVC, WALLS['cava'], REGION['CAVA'], k_lum=0.6, k_out=0.45, name='ivc', cut=IVC_CUT)
for nm, ctrl in PVEINS.items():
    vessel('LA', ctrl, WALLS['pvein'], REGION['PVEIN'], k_lum=0.45, k_out=0.35, name=nm.lower(), cut=PVEIN_CUTS[nm])
log('veins')

# ---- Eustachian valve: crescentic fold at the anterior rim of the IVC orifice -------
ic = np.array(IVC[-1][0]) + np.array([0.0, 0.0, 0.25])
iz = unit(np.array(IVC[-1][0]) - np.array(IVC[-2][0]))
ia, ib, _ = perp_frame(iz, unit(TV_C - ic))
ew = Win.around(g, [ic], 2.4)
prev_lo = prev_hi = None
for a in np.radians(np.linspace(-78, 78, 17)):
    s = math.cos(a * 1.15)
    lo_p = ic + 1.22 * (math.cos(a) * ia + math.sin(a) * ib)
    hi_p = ic + (1.22 - 0.38 * s) * (math.cos(a) * ia + math.sin(a) * ib) + (0.15 + 0.5 * s) * iz
    ew.cone(lo_p, hi_p, 0.11, 0.065, pad=0.4)
    if prev_lo is not None:
        ew.cone(prev_hi, hi_p, 0.065, 0.065, pad=0.4)
        ew.cone(prev_lo, hi_p, 0.10, 0.065, pad=0.4)
    prev_lo, prev_hi = lo_p, hi_p
add_feature(feat, ew, ew.F)

# ---- keep the two circulations (and non-communicating neighbours) apart -------------------
def carve(a, b, sep):
    lum[a] = smax(lum[a], -(lum[b] - sep), 0.1)


carve('RV', 'LV', 0.40)
carve('LA', 'LV', 0.45)
carve('LA', 'RV', 0.34)
carve('RA', 'LV', 0.40)
carve('RA', 'RV', 0.34)
carve('RA', 'LA', 0.30)
log('carved')

blood_R = np.minimum(lum['RA'], lum['RV'])
blood_L = np.minimum(lum['LA'], lum['LV'])
blood0 = np.minimum(np.minimum(blood_R, blood_L), orif)
blood = smax(blood0, -feat, 0.08)
S = np.maximum(outer, -blood)

bb = np.argwhere(S < 0)
assert bb.min() > 1 and np.all(bb.max(0) < g.n - 2), 'solid touches the grid boundary'
log('composed')

S = ndi.gaussian_filter(S, 0.6)
# keep the single largest solid component; fill voids that do not communicate with the outside
lab, n = ndi.label(S < 0)
if n > 1:
    sizes = ndi.sum_labels(np.ones_like(lab, np.uint8), lab, np.arange(1, n + 1))
    keep = 1 + int(np.argmax(sizes))
    S[(lab > 0) & (lab != keep)] = 0.2
    log('removed %d floating fragments' % (n - 1))
lab, n = ndi.label(S >= 0)
ext = lab[0, 0, 0]
bad = (lab != ext) & (lab > 0)
if bad.any():
    S[bad] = -0.2
    log('filled %d enclosed voids (%.0f voxels)' % (n - 1, bad.sum()))
del lab

verts, faces, vol = extract_surface(g, S)
log('marching cubes: %d verts, %d tris, tissue volume %.0f cm3 (~%.0f g)' % (len(verts), len(faces), vol, vol * 1.055))

st = 2
np.savez(bpath('fields02.npz'), stride=st, outer=outer[::st, ::st, ::st], blood=blood[::st, ::st, ::st],
         blood0=blood0[::st, ::st, ::st], bloodR=blood_R[::st, ::st, ::st], bloodL=blood_L[::st, ::st, ::st],
         solid=S[::st, ::st, ::st])
save('region02', region)
save('outer02', outer)
save('solid02', S.astype(np.float32))
save_json('flow_paths', FLOW)
np.savez(bpath('heart_raw.npz'), v=verts, f=faces)

# ---- Blender mesh ----------------------------------------------------------------------
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
ob = mesh_from_arrays('Heart', verts, faces)
target = int(os.environ.get('HEART_TRIS', '750000'))
if len(faces) > target:
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    m = ob.modifiers.new('dec', 'DECIMATE')
    m.ratio = target / len(faces)
    bpy.ops.object.modifier_apply(modifier='dec')
    log('decimated to', len(ob.data.polygons))
bpy.ops.wm.save_as_mainfile(filepath=bpath('stage02.blend'))
log('stage 02 complete')
