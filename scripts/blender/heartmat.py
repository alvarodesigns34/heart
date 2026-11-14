"""
heartmat.py — procedural wet-tissue PBR materials (Cycles node graphs), per-vertex anatomical
attributes (tissue-region weights, endocardial mask, Torrent-Guasp helical fibre coordinates) and
the studio light rig shared by the bake stage and the diagnostic renders.
"""
import math
import numpy as np
import bpy
from scipy import ndimage as ndi
from heartlib import *      # noqa

_cache = {}


def _fields():
    if 'f' not in _cache:
        g = Grid()
        reg = np.asarray(load('region02', mmap=True)[::2, ::2, ::2])
        fd = np.load(bpath('fields02.npz'))
        W = {}
        for name, ids in (('vent', (1, 2)), ('atr', (3, 4)), ('aorta', (5,)), ('pa', (6,)), ('cava', (7,)),
                          ('pvein', (8,)), ('lig', (9,)), ('rv', (2,)), ('la', (4,))):
            W[name] = ndi.gaussian_filter(np.isin(reg, ids).astype(np.float32), 1.3)
        _cache['f'] = (g, W, fd['blood'].astype(np.float32))
    return _cache['f']


def _set_attr(me, name, kind, arr):
    a = me.attributes.get(name)
    if a is None:
        a = me.attributes.new(name, kind, 'POINT')
    a.data.foreach_set('color' if kind == 'FLOAT_COLOR' else 'vector', np.ascontiguousarray(arr, np.float32).ravel())


def heart_attributes(ob, cut=None):
    """Tissue-region weights, endocardium mask and helical fibre coordinates for the myocardial mesh."""
    g, W, BL = _fields()
    P = mesh_arrays(ob).astype(np.float64)
    s = lambda F: g.sample(F, P, stride=2)
    w = {k: s(v) for k, v in W.items()}
    tot = sum(w[k] for k in ('vent', 'atr', 'aorta', 'pa', 'cava', 'pvein', 'lig')) + 1e-5
    endo = smoothstep(0.13, 0.035, s(BL))
    q = P - O_L
    sig, dlt, t = q @ S_AX, q @ D_AX, q @ A_AX
    bow = 0.18 * np.sin(np.clip(t / 7.3, 0, 1) * np.pi)
    rho = np.hypot(sig + bow, dlt)
    phi = np.arctan2(dlt, sig + bow)
    # Torrent-Guasp band: left-handed helix in the subepicardium, right-handed in the subendocardium
    psi = phi + np.where(endo > 0.5, 0.22, -0.19) * t
    fib = np.stack([3.0 * np.cos(psi), 3.0 * np.sin(psi), 0.10 * t + 0.22 * rho], axis=1)
    cutw = np.zeros(len(P))
    if cut is not None:
        p, n = np.asarray(cut[0], float), unit(cut[1])
        cutw = smoothstep(0.035, 0.012, np.abs((P - p) @ n))
    me = ob.data
    one = np.ones(len(P))
    _set_attr(me, 'wA', 'FLOAT_COLOR', np.stack([w['vent'] / tot, w['atr'] / tot, w['aorta'] / tot, w['pa'] / tot], 1))
    _set_attr(me, 'wB', 'FLOAT_COLOR', np.stack([w['cava'] / tot, w['pvein'] / tot, w['lig'] / tot, endo], 1))
    _set_attr(me, 'wC', 'FLOAT_COLOR', np.stack([w['rv'] / tot, w['la'] / tot, cutw, one], 1))
    _set_attr(me, 'fib', 'FLOAT_VECTOR', fib)


class NG:
    """Tiny helper to write shader graphs as expressions."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()

    def n(self, typ, **kw):
        nd = self.nt.nodes.new(typ)
        for k, v in kw.items():
            setattr(nd, k, v)
        return nd

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        elif isinstance(v, (tuple, list)) and len(v) == 3 and sock.type == 'RGBA':
            sock.default_value = (*v, 1.0)
        else:
            sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.n('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.set(nd.inputs[0], a)
        if b is not None:
            self.set(nd.inputs[1], b)
        if c is not None:
            self.set(nd.inputs[2], c)
        return nd.outputs[0]

    def add(self, a, b):
        return self.math('ADD', a, b)

    def mul(self, a, b):
        return self.math('MULTIPLY', a, b)

    def mix(self, f, a, b):
        nd = self.n('ShaderNodeMix', data_type='RGBA')
        self.set(nd.inputs[0], f)
        self.set(nd.inputs[6], a)
        self.set(nd.inputs[7], b)
        return nd.outputs[2]

    def scale(self, col, f):
        nd = self.n('ShaderNodeVectorMath', operation='SCALE')
        self.set(nd.inputs[0], col)
        self.set(nd.inputs[3], f)
        return nd.outputs[0]

    def vmul(self, vec, k):
        nd = self.n('ShaderNodeVectorMath', operation='MULTIPLY')
        self.set(nd.inputs[0], vec)
        nd.inputs[1].default_value = k
        return nd.outputs[0]

    def noise(self, vec, scale, detail=2.0, rough=0.5, distortion=0.0):
        nd = self.n('ShaderNodeTexNoise')
        self.set(nd.inputs['Vector'], vec)
        nd.inputs['Scale'].default_value = scale
        nd.inputs['Detail'].default_value = detail
        nd.inputs['Roughness'].default_value = rough
        nd.inputs['Distortion'].default_value = distortion
        return nd.outputs[0]

    def voronoi(self, vec, scale, feature='F1', rand=1.0):
        nd = self.n('ShaderNodeTexVoronoi', feature=feature)
        self.set(nd.inputs['Vector'], vec)
        nd.inputs['Scale'].default_value = scale
        nd.inputs['Randomness'].default_value = rand
        return nd.outputs['Distance']

    def ramp(self, x, lo, hi, a=0.0, b=1.0, smooth=False):
        nd = self.n('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        self.set(nd.inputs[0], x)
        nd.inputs[1].default_value = lo
        nd.inputs[2].default_value = hi
        nd.inputs[3].default_value = a
        nd.inputs[4].default_value = b
        return nd.outputs[0]

    def ridge(self, vec, scale, w0, w1):
        """Thin meandering lines (iso-contours of a noise field) - small vessels."""
        d = self.math('ABSOLUTE', self.math('SUBTRACT', self.noise(vec, scale, 3.0, 0.5), 0.5))
        return self.ramp(d, w0, w1, 1.0, 0.0, smooth=True)

    def attr(self, name):
        return self.n('ShaderNodeAttribute', attribute_name=name)

    def sep(self, col):
        nd = self.n('ShaderNodeSeparateColor')
        self.set(nd.inputs[0], col)
        return nd.outputs

    def warp(self, vec, amount, scale):
        nz = self.n('ShaderNodeTexNoise')
        self.set(nz.inputs['Vector'], vec)
        nz.inputs['Scale'].default_value = scale
        off = self.scale(nz.outputs['Color'], amount)
        nd = self.n('ShaderNodeVectorMath', operation='ADD')
        self.set(nd.inputs[0], vec)
        self.set(nd.inputs[1], off)
        return nd.outputs[0]

    def finish(self, col, rough, height, bump=0.5, dist=0.02, sss=0.14, sss_rad=(0.6, 0.15, 0.08), sss_scale=0.2,
               coat=0.55, coat_rough=0.13, sheen=0.0, extra=None):
        out = self.n('ShaderNodeOutputMaterial')
        b = self.n('ShaderNodeBsdfPrincipled')
        self.set(b.inputs['Base Color'], col)
        self.set(b.inputs['Roughness'], rough)
        if height is not None:
            bp = self.n('ShaderNodeBump')
            bp.inputs['Strength'].default_value = bump
            bp.inputs['Distance'].default_value = dist
            self.set(bp.inputs['Height'], height)
            self.nt.links.new(bp.outputs[0], b.inputs['Normal'])
        b.inputs['Subsurface Weight'].default_value = sss
        b.inputs['Subsurface Radius'].default_value = sss_rad
        b.inputs['Subsurface Scale'].default_value = sss_scale
        b.inputs['Coat Weight'].default_value = coat
        b.inputs['Coat Roughness'].default_value = coat_rough
        b.inputs['Sheen Weight'].default_value = sheen
        b.inputs['IOR'].default_value = 1.4
        self.nt.links.new(b.outputs[0], out.inputs[0])
        return b


EPI = dict(vent=(0.30, 0.045, 0.035), atr=(0.335, 0.075, 0.078), aorta=(0.60, 0.31, 0.235), pa=(0.47, 0.32, 0.35),
           cava=(0.19, 0.125, 0.25), pvein=(0.50, 0.19, 0.18), lig=(0.70, 0.58, 0.46))
ENDO = dict(vent=(0.42, 0.105, 0.085), atr=(0.47, 0.165, 0.15), aorta=(0.67, 0.41, 0.32), pa=(0.59, 0.41, 0.40),
            cava=(0.33, 0.20, 0.30), pvein=(0.58, 0.29, 0.26), lig=(0.70, 0.58, 0.46))


def mat_heart(name='Myocardium'):
    mat = bpy.data.materials.new(name)
    G = NG(mat)
    pos = G.n('ShaderNodeTexCoord').outputs['Object']
    aA, aB, aC = G.attr('wA'), G.attr('wB'), G.attr('wC')
    A, B, C = G.sep(aA.outputs['Color']), G.sep(aB.outputs['Color']), G.sep(aC.outputs['Color'])
    wts = dict(vent=A[0], atr=A[1], aorta=A[2], pa=aA.outputs['Alpha'], cava=B[0], pvein=B[1], lig=B[2])
    endo = aB.outputs['Alpha']
    rv, cutw = C[0], C[2]
    fib = G.attr('fib').outputs['Vector']

    def blend(tab):
        col = None
        for k in ('vent', 'atr', 'aorta', 'pa', 'cava', 'pvein', 'lig'):
            col = tab[k] if col is None else G.mix(wts[k], col, tab[k])
        return col

    col = G.mix(endo, blend(EPI), blend(ENDO))
    # RV / thin-walled regions are a touch more violaceous
    col = G.mix(G.mul(rv, G.math('SUBTRACT', 1.0, endo)), col, G.mix(0.5, col, (0.27, 0.045, 0.07)))
    # large-scale mottling
    col = G.scale(col, G.ramp(G.noise(pos, 0.55, 3.0, 0.55), 0.3, 0.7, 0.80, 1.17))
    col = G.mix(G.ramp(G.noise(pos, 1.7, 3.0, 0.6), 0.52, 0.75, 0.0, 0.35), col, G.scale(col, 0.72))
    # myocardial fibre striation (helical band)
    stri = G.add(G.mul(G.noise(fib, 7.0, 2.0, 0.5), 0.6), G.mul(G.noise(fib, 23.0, 2.0, 0.5), 0.4))
    sfac = G.ramp(stri, 0.36, 0.64, 0.80, 1.19)
    sfac = G.add(1.0, G.mul(wts['vent'], G.math('SUBTRACT', sfac, 1.0)))
    col = G.scale(col, sfac)
    # longitudinal streaks on the vessels
    vessel_w = G.math('ADD', G.add(wts['aorta'], wts['pa']), G.add(wts['cava'], wts['pvein']), clamp=True)
    vn = G.noise(G.vmul(pos, (6.0, 6.0, 0.7)), 1.0, 3.0, 0.55)
    col = G.scale(col, G.add(1.0, G.mul(vessel_w, G.ramp(vn, 0.35, 0.65, -0.11, 0.10))))
    # superficial capillary / vasa-vasorum network (epicardial side only)
    l1 = G.ridge(G.warp(pos, 0.25, 0.9), 0.75, 0.004, 0.013)
    l2 = G.mul(G.ridge(G.warp(pos, 0.15, 2.2), 1.9, 0.004, 0.012), 0.6)
    lines = G.mul(G.math('MAXIMUM', l1, l2), G.math('SUBTRACT', 1.0, endo))
    lines = G.mul(lines, G.ramp(G.noise(pos, 0.9, 2.0), 0.42, 0.62, 0.0, 1.0))        # patchy
    lines = G.mul(lines, G.add(1.0, G.mul(vessel_w, -0.55)))                          # fainter vasa vasorum
    col = G.mix(G.mul(lines, 0.6), col, (0.26, 0.012, 0.015))
    # crevices darker / crests paler (trabeculae, pectinate ridges)
    pt = G.n('ShaderNodeNewGeometry').outputs['Pointiness']
    pfac = G.ramp(pt, 0.43, 0.57, 0.62, 1.22)
    col = G.scale(col, G.add(1.0, G.mul(G.add(0.35, G.mul(endo, 0.65)), G.math('SUBTRACT', pfac, 1.0))))
    # freshly cut muscle (diagnostic cut-aways)
    cutc = G.scale((0.40, 0.065, 0.05), G.ramp(G.noise(fib, 9.0, 3.0, 0.6), 0.3, 0.7, 0.75, 1.25))
    col = G.mix(cutw, col, cutc)

    rough = G.add(0.37, G.ramp(G.noise(pos, 3.0, 3.0), 0.3, 0.7, -0.07, 0.09))
    rough = G.math('ADD', G.add(rough, G.mul(endo, -0.08)), G.mul(vessel_w, 0.06), clamp=True)
    height = G.add(G.mul(G.mul(stri, wts['vent']), 0.55), G.mul(G.noise(pos, 26.0, 2.0), 0.22))
    height = G.add(height, G.mul(G.noise(pos, 5.0, 3.0), 0.35))
    height = G.add(height, G.mul(lines, 0.3))
    height = G.add(height, G.mul(vessel_w, G.mul(vn, 0.3)))
    mat['bsdf'] = G.finish(col, rough, height, bump=0.55, dist=0.022).name
    return mat


def mat_fat(name='EpicardialFat'):
    mat = bpy.data.materials.new(name)
    G = NG(mat)
    pos = G.n('ShaderNodeTexCoord').outputs['Object']
    wp = G.warp(pos, 0.12, 3.0)
    d = G.voronoi(wp, 3.4, 'F1')
    de = G.voronoi(wp, 3.4, 'DISTANCE_TO_EDGE')
    d2 = G.voronoi(G.warp(pos, 0.05, 6.0), 9.0, 'F1')
    col = G.mix(G.ramp(G.noise(pos, 1.1, 3.0), 0.3, 0.7), (0.78, 0.50, 0.13), (0.88, 0.67, 0.27))
    col = G.scale(col, G.ramp(d, 0.0, 0.42, 1.12, 0.84))
    col = G.mix(G.ramp(de, 0.0, 0.05, 0.65, 0.0, smooth=True), col, (0.60, 0.27, 0.06))
    col = G.scale(col, G.ramp(d2, 0.0, 0.2, 1.05, 0.93))
    lines = G.mul(G.ridge(G.warp(pos, 0.2, 1.2), 1.1, 0.004, 0.014), G.ramp(G.noise(pos, 0.8, 2.0), 0.42, 0.6, 0.0, 1.0))
    col = G.mix(G.mul(lines, 0.6), col, (0.42, 0.03, 0.02))
    pt = G.n('ShaderNodeNewGeometry').outputs['Pointiness']
    col = G.scale(col, G.ramp(pt, 0.44, 0.56, 0.72, 1.14))
    rough = G.add(0.36, G.ramp(G.noise(pos, 4.0, 2.0), 0.3, 0.7, -0.06, 0.08))
    height = G.add(G.mul(G.ramp(d, 0.0, 0.4, 1.0, 0.0), 0.6), G.add(G.mul(G.ramp(d2, 0.0, 0.2, 1.0, 0.0), 0.25), G.mul(G.noise(pos, 30.0, 2.0), 0.12)))
    mat['bsdf'] = G.finish(col, rough, height, bump=0.6, dist=0.03, sss=0.5, sss_rad=(0.9, 0.55, 0.2), sss_scale=0.3,
                           coat=0.7, coat_rough=0.12).name
    return mat


def mat_vessel(name, c0, c1):
    mat = bpy.data.materials.new(name)
    G = NG(mat)
    pos = G.n('ShaderNodeTexCoord').outputs['Object']
    col = G.mix(G.ramp(G.noise(pos, 2.2, 3.0), 0.3, 0.7), c0, c1)
    rough = G.add(0.30, G.ramp(G.noise(pos, 6.0, 2.0), 0.3, 0.7, -0.05, 0.07))
    mat['bsdf'] = G.finish(col, rough, G.noise(pos, 22.0, 2.0), bump=0.25, dist=0.01, sss=0.3, sss_rad=(0.5, 0.1, 0.08),
                           sss_scale=0.1, coat=0.8, coat_rough=0.08).name
    return mat


def mat_valve(name='ValveLeaflet'):
    mat = bpy.data.materials.new(name)
    G = NG(mat)
    uv = G.n('ShaderNodeTexCoord').outputs['UV']
    vc = G.n('ShaderNodeVertexColor', layer_name='Col').outputs['Color']
    stre = G.noise(G.vmul(uv, (70.0, 2.5, 1.0)), 1.0, 3.0, 0.6)
    col = G.scale(vc, G.ramp(stre, 0.3, 0.7, 0.9, 1.1))
    rough = G.add(0.33, G.ramp(stre, 0.3, 0.7, -0.05, 0.05))
    mat['bsdf'] = G.finish(col, rough, stre, bump=0.2, dist=0.006, sss=0.7, sss_rad=(1.0, 0.5, 0.35), sss_scale=0.12,
                           coat=0.5, coat_rough=0.15, sheen=0.3).name
    return mat


def mat_plain(name, col, rough=0.3, sss=0.4):
    mat = bpy.data.materials.new(name)
    G = NG(mat)
    mat['bsdf'] = G.finish(col, rough, None, sss=sss, sss_rad=(1.0, 0.6, 0.4), sss_scale=0.05, coat=0.5).name
    return mat


def valve_vertex_colors(ob):
    """Thin translucent belly, thicker pale hinge and rough zone."""
    me = ob.data
    uvl = me.uv_layers.active
    n = len(me.vertices)
    v = np.zeros(n)
    if uvl is not None:
        uv = np.empty(len(me.loops) * 2, np.float32)
        uvl.data.foreach_get('uv', uv)
        li = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get('vertex_index', li)
        v[li] = uv.reshape(-1, 2)[:, 1]
    hinge = smoothstep(0.25, 0.0, v)
    edge = smoothstep(0.72, 1.0, v)
    base = np.array([0.80, 0.60, 0.50])
    pale = np.array([0.90, 0.80, 0.70])
    pink = np.array([0.74, 0.45, 0.38])
    col = base[None, :] * (1 - hinge - 0.7 * edge)[:, None] + pink[None, :] * hinge[:, None] + pale[None, :] * (0.7 * edge)[:, None]
    ca = me.color_attributes.get('Col') or me.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', np.concatenate([col, np.ones((n, 1))], 1).astype(np.float32).ravel())
    me.color_attributes.active_color = ca
    me.color_attributes.render_color_index = list(me.color_attributes).index(ca)


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    for p in ob.data.polygons:
        p.use_smooth = True


def procedural_materials():
    return dict(
        heart=mat_heart(), fat=mat_fat(),
        artery=mat_vessel('CoronaryArtery', (0.46, 0.03, 0.025), (0.60, 0.05, 0.04)),
        vein=mat_vessel('CoronaryVein', (0.085, 0.065, 0.22), (0.17, 0.10, 0.30)),
        valve=mat_valve(), chordae=mat_plain('ChordaeTendineae', (0.84, 0.76, 0.64), 0.32))


def material_for(name, mats):
    if name.startswith('Valve'):
        return mats['valve']
    if name.startswith('Chordae'):
        return mats['chordae']
    return {'Heart': mats['heart'], 'Epicardial_Fat': mats['fat'], 'Coronary_Arteries': mats['artery'],
            'Coronary_Veins': mats['vein']}.get(name, mats['heart'])


# --------------------------------------------------------------------------------------
# Studio rig for Cycles
# --------------------------------------------------------------------------------------
def setup_cycles(scene, samples=96, res=1024):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'


def studio_world(scene, bg=(0.0022, 0.0024, 0.0033), env_strength=0.28):
    """Dark surgical-theatre backdrop for the camera, soft studio HDRI for reflections."""
    w = bpy.data.worlds.new('Studio')
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputWorld')
    mixs = nt.nodes.new('ShaderNodeMixShader')
    lp = nt.nodes.new('ShaderNodeLightPath')
    bgc = nt.nodes.new('ShaderNodeBackground')
    bgc.inputs[0].default_value = (*bg, 1)
    env = nt.nodes.new('ShaderNodeBackground')
    env.inputs[1].default_value = env_strength
    import os
    hdr = os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'studiolights', 'world', 'studio.exr')
    if os.path.exists(hdr):
        tex = nt.nodes.new('ShaderNodeTexEnvironment')
        tex.image = bpy.data.images.load(hdr, check_existing=True)
        nt.links.new(tex.outputs[0], env.inputs[0])
    else:
        env.inputs[0].default_value = (0.5, 0.5, 0.55, 1)
    nt.links.new(lp.outputs['Is Camera Ray'], mixs.inputs[0])
    nt.links.new(env.outputs[0], mixs.inputs[1])
    nt.links.new(bgc.outputs[0], mixs.inputs[2])
    nt.links.new(mixs.outputs[0], out.inputs[0])


def area_light(scene, name, loc, target, energy, size, color):
    from mathutils import Vector
    ld = bpy.data.lights.new(name, 'AREA')
    ld.energy = energy
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return ob


def camera_rig(scene, view_dir, center, ortho_scale=None, dist=60.0, lens=85.0):
    """Camera looking along -view_dir at `center`, plus warm key / cold rim / soft fill lights."""
    from mathutils import Vector
    d = Vector(view_dir).normalized()
    c = Vector(center)
    cd = bpy.data.cameras.new('Cam')
    cam = bpy.data.objects.new('Cam', cd)
    scene.collection.objects.link(cam)
    cam.location = c + d * dist
    up = Vector((0, 0, 1)) if abs(d.z) < 0.92 else Vector((0, 1, 0))
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    if ortho_scale:
        cd.type = 'ORTHO'
        cd.ortho_scale = ortho_scale
    else:
        cd.lens = lens
    cd.clip_end = 500
    scene.camera = cam
    right = d.cross(up).normalized() * -1.0
    upv = right.cross(d).normalized() * -1.0
    lights = [
        area_light(scene, 'Key', c + d * 34 + right * -26 + upv * 24, c, 15000, 9, (1.0, 0.86, 0.72)),
        area_light(scene, 'Rim', c - d * 30 + right * 26 + upv * 16, c, 30000, 10, (0.60, 0.76, 1.0)),
        area_light(scene, 'Fill', c + d * 30 + right * 30 - upv * 8, c, 2600, 22, (0.85, 0.9, 1.0)),
        area_light(scene, 'Top', c + upv * 40 + d * 8, c, 5000, 7, (1.0, 0.97, 0.92)),
    ]
    return cam, lights
