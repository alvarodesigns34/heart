"""
05_materials_bake_export.py
  1. assembles every stage into one scene;
  2. procedural wet-tissue materials (helical myocardial fibre striation, capillary network, serosal
     moisture variation, SSS for fat / leaflets) — see heartmat.py;
  3. UV atlas + Cycles bake of Albedo / Normal / Roughness / AO (AO+Roughness packed as ORM);
  4. physiological shape keys on every mesh from the shared displacement field
     (`atrial_systole`, `ventricular_contraction`) next to the valve keys from stage 03;
  5. glTF export (real-world metres, Y-up) + viewer metadata (pins, blood-flow routes, cut planes).
The raw .glb is then quantised + Meshopt-compressed (textures -> WebP) by scripts/optimize-glb.mjs.
"""
import os
import sys
import subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heartlib import *      # noqa
import heartmat as hm
import bpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT_DIR = os.environ.get('HEART_OUT', os.path.join(ROOT, 'public', 'models'))
TEX = int(os.environ.get('HEART_TEX', '4096'))
SCALE = 0.0092            # cm -> m, and in-vivo (end-diastolic) -> textbook 12 x 8.5 x 6 cm heart
os.makedirs(OUT_DIR, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
for st in ('stage02', 'stage03', 'stage04'):
    with bpy.data.libraries.load(bpath(st + '.blend')) as (src, dst):
        dst.objects = list(src.objects)
    for o in dst.objects:
        sc.collection.objects.link(o)
heart = bpy.data.objects['Heart']
fat = bpy.data.objects['Epicardial_Fat']
log('stage 05 — objects:', [o.name for o in sc.objects])


def only(ob):
    for o in sc.objects:
        o.select_set(o == ob)
    bpy.context.view_layer.objects.active = ob


# ---------------------------------------------------------------------------------------
# materials + UVs
# ---------------------------------------------------------------------------------------
mats = hm.procedural_materials()
hm.heart_attributes(heart)
for o in sc.objects:
    if o.name.startswith('Valve'):
        hm.valve_vertex_colors(o)
    hm.assign(o, hm.material_for(o.name, mats))


def smart_uv(ob, angle, margin):
    only(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin, area_weight=0.0)
    bpy.ops.object.mode_set(mode='OBJECT')


smart_uv(heart, 72.0, 0.0005)
log('heart UV atlas')
smart_uv(fat, 72.0, 0.001)
log('fat UV atlas')

hm.setup_cycles(sc, 4, 512)
sc.render.bake.margin = 6
sc.render.bake.use_clear = True
sc.render.bake.target = 'IMAGE_TEXTURES'
w = bpy.data.worlds.new('bake')
sc.world = w
w.light_settings.distance = 0.55                     # local AO only: crevices, not whole chambers


def px(img):
    a = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def new_image(name, size, srgb, data=None):
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.colorspace_settings.name = 'sRGB' if srgb else 'Non-Color'
    if data is not None:
        img.pixels.foreach_set(np.ascontiguousarray(data, np.float32).ravel())
    return img


def save_png(img, name):
    img.filepath_raw = bpath(name + '.png')
    img.file_format = 'PNG'
    img.save()
    img.pack()


def bake(ob, prefix, size, ao_samples):
    mat = ob.data.materials[0]
    nt = mat.node_tree
    bsdf = nt.nodes[mat['bsdf']]
    sss = bsdf.inputs['Subsurface Weight'].default_value
    coat = bsdf.inputs['Coat Weight'].default_value
    bsdf.inputs['Subsurface Weight'].default_value = 0.0
    bsdf.inputs['Coat Weight'].default_value = 0.0
    tex = nt.nodes.new('ShaderNodeTexImage')
    nt.nodes.active = tex
    only(ob)
    out = {}
    for kind, typ, sz, smp in (('albedo', 'DIFFUSE', size, 3), ('normal', 'NORMAL', size, 2),
                               ('rough', 'ROUGHNESS', size // 2, 2), ('ao', 'AO', size // 2, ao_samples)):
        img = new_image('%s_%s' % (prefix, kind), sz, kind == 'albedo')
        tex.image = img
        sc.cycles.samples = smp
        sc.cycles.use_denoising = False
        if typ == 'DIFFUSE':
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})
        elif typ == 'NORMAL':
            bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT')
        else:
            bpy.ops.object.bake(type=typ)
        out[kind] = img
        log('   baked %s %s %d px' % (prefix, kind, sz))
    nt.nodes.remove(tex)
    bsdf.inputs['Subsurface Weight'].default_value = sss
    bsdf.inputs['Coat Weight'].default_value = coat
    from scipy import ndimage as ndi
    ao = px(out['ao'])[..., 0]
    ao = np.clip(ndi.gaussian_filter(ao, 1.0), 0, 1) ** 1.25
    ao = 0.12 + 0.88 * ao
    rg = px(out['rough'])[..., 0]
    orm = np.stack([ao, rg, np.zeros_like(ao), np.ones_like(ao)], axis=-1)
    out['orm'] = new_image(prefix + '_orm', size // 2, False, orm)
    for k in ('albedo', 'normal', 'orm'):
        save_png(out[k], '%s_%s' % (prefix, k))
    return out


def gltf_group():
    ng = bpy.data.node_groups.get('glTF Material Output')
    if ng is None:
        ng = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
        ng.interface.new_socket(name='Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
        ng.nodes.new('NodeGroupInput')
    return ng


def export_material(name, maps, coat=0.85, coat_rough=0.12, sheen=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes['Principled BSDF']
    ta = nt.nodes.new('ShaderNodeTexImage')
    ta.image = maps['albedo']
    nt.links.new(ta.outputs[0], b.inputs['Base Color'])
    tn = nt.nodes.new('ShaderNodeTexImage')
    tn.image = maps['normal']
    nm = nt.nodes.new('ShaderNodeNormalMap')
    nt.links.new(tn.outputs[0], nm.inputs['Color'])
    nt.links.new(nm.outputs[0], b.inputs['Normal'])
    to = nt.nodes.new('ShaderNodeTexImage')
    to.image = maps['orm']
    sp = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(to.outputs[0], sp.inputs[0])
    nt.links.new(sp.outputs[1], b.inputs['Roughness'])
    nt.links.new(sp.outputs[2], b.inputs['Metallic'])
    grp = nt.nodes.new('ShaderNodeGroup')
    grp.node_tree = gltf_group()
    nt.links.new(sp.outputs[0], grp.inputs['Occlusion'])
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Coat Roughness'].default_value = coat_rough
    b.inputs['Sheen Weight'].default_value = sheen
    return mat


def plain_material(name, col, rough, coat=0.85, vertex_color=False, sheen=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*col, 1)
    if vertex_color:
        vc = nt.nodes.new('ShaderNodeVertexColor')
        vc.layer_name = 'Col'
        nt.links.new(vc.outputs[0], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = rough
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Coat Roughness'].default_value = 0.12
    b.inputs['Sheen Weight'].default_value = sheen
    return mat


heart_maps = bake(heart, 'heart', TEX, int(os.environ.get('HEART_AO_SAMPLES', '24')))
fat_maps = bake(fat, 'fat', TEX // 2, 24)
bpy.ops.wm.save_as_mainfile(filepath=bpath('stage05_baked.blend'))

EXPORT = {
    'Heart': export_material('Myocardium', heart_maps),
    'Epicardial_Fat': export_material('EpicardialFat', fat_maps, sheen=0.3),
    'Coronary_Arteries': plain_material('CoronaryArtery', (0.50, 0.035, 0.03), 0.30),
    'Coronary_Veins': plain_material('CoronaryVein', (0.11, 0.075, 0.25), 0.32),
}
valve_mat = plain_material('ValveLeaflet', (1, 1, 1), 0.34, coat=0.5, vertex_color=True, sheen=0.4)
chordae_mat = plain_material('ChordaeTendineae', (0.84, 0.76, 0.64), 0.32, coat=0.5)
for o in sc.objects:
    if o.type != 'MESH':
        continue
    m = EXPORT.get(o.name) or (valve_mat if o.name.startswith('Valve') else chordae_mat)
    hm.assign(o, m)
for nm in ('wA', 'wB', 'wC', 'fib'):
    a = heart.data.attributes.get(nm)
    if a is not None:
        heart.data.attributes.remove(a)

# ---------------------------------------------------------------------------------------
# shape keys
# ---------------------------------------------------------------------------------------
motion = Motion()
for o in sc.objects:
    if o.type == 'MESH':
        add_motion_keys(o, motion)
        log('   shape keys %-20s %s' % (o.name, [k.name for k in o.data.shape_keys.key_blocks[1:]]))

root = bpy.data.objects.new('HumanHeart', None)
sc.collection.objects.link(root)
root.scale = (SCALE, SCALE, SCALE)
heart.name = 'Myocardium'
for o in sc.objects:
    if o.type == 'MESH':
        o.parent = root
bpy.context.view_layer.update()
raw = bpath('human_heart_raw.glb')
bpy.ops.export_scene.gltf(filepath=raw, export_format='GLB', export_yup=True, export_apply=False,
                          export_morph=True, export_morph_normal=False, export_morph_tangent=False,
                          export_tangents=False, export_normals=True, export_texcoords=True,
                          export_materials='EXPORT', export_image_format='AUTO', export_vertex_color='MATERIAL',
                          export_cameras=False, export_lights=False, export_animations=False, export_skins=False,
                          export_extras=False)
log('raw glb: %.1f MB' % (os.path.getsize(raw) / 1e6))

# ---------------------------------------------------------------------------------------
# viewer metadata
# ---------------------------------------------------------------------------------------
def gl(P):
    P = np.asarray(P, float).reshape(-1, 3)
    return np.round(SCALE * np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1), 5)


def glv(v):
    return gl(np.asarray(v, float)[None, :])[0].tolist()


surf = Surf()
L01, L03, L04 = load_json('landmarks01'), load_json('landmarks03'), load_json('landmarks04')
rad = math.radians
PINS = []


def pin(key, label, desc, P, layer='myocardium', interior=False, project=True, lift=0.06):
    P = np.asarray(P, float)
    if project and not interior:
        P = surf.project(P[None, :])[0]
        n = surf.normals(P[None, :])[0]
    else:
        n = unit(P - np.array([0.5, 1.0, 1.5]))
    Q = P + n * lift
    PINS.append(dict(id=key, label=label, desc=desc, layer=layer, interior=bool(interior), pos=glv(Q),
                     normal=[round(float(v), 4) for v in (n[0], n[2], -n[1])],
                     d_vc=glv(motion.ventricular(Q[None, :])[0]), d_as=glv(motion.atrial(Q[None, :])[0])))


A = np.array
pin('aorta_asc', 'Aorta ascendente', 'Nace del ventrículo izquierdo; ~3 cm de diámetro. Conduce la sangre oxigenada hacia la circulación sistémica.', AO_C + 3.6 * AO_N + 1.5 * cusp_dir(AO_N, -50))
pin('arco_aortico', 'Cayado aórtico', 'Arco que cruza sobre la bifurcación pulmonar y da origen a los tres troncos supraaórticos.', A([0.4, 1.6, 11.4]))
pin('aorta_desc', 'Aorta descendente', 'Continuación torácica de la aorta, posterior y a la izquierda de la aurícula izquierda.', A([3.9, 5.2, 6.4]))
pin('tronco_braquiocefalico', 'Tronco braquiocefálico', 'Primera rama del cayado; se divide en arteria subclavia derecha y carótida común derecha.', A([-1.7, -0.3, 10.8]))
pin('carotida_izq', 'A. carótida común izquierda', 'Segunda rama del cayado; irriga el lado izquierdo de cabeza y cuello.', A([0.35, 0.8, 11.8]))
pin('subclavia_izq', 'A. subclavia izquierda', 'Tercera rama del cayado; irriga el miembro superior izquierdo.', A([2.2, 2.7, 11.7]))
pin('tronco_pulmonar', 'Tronco pulmonar', 'Lleva sangre desoxigenada del ventrículo derecho a los pulmones; se bifurca bajo el cayado aórtico.', A([1.9, -3.2, 5.3]))
pin('ap_derecha', 'A. pulmonar derecha', 'Pasa por detrás de la aorta ascendente y la vena cava superior hacia el pulmón derecho.', A([-2.6, 2.6, 7.8]))
pin('ap_izquierda', 'A. pulmonar izquierda', 'Rama más corta; se dirige al hilio pulmonar izquierdo por delante de la aorta descendente.', A([4.5, 2.2, 8.2]))
pin('ligamento_arterioso', 'Ligamento arterioso', 'Vestigio fibroso del conducto arterioso fetal entre la arteria pulmonar izquierda y el cayado.', A(LIGAMENT[1][0]), lift=0.25, project=False)
pin('vcs', 'Vena cava superior', 'Drena la sangre venosa de cabeza, cuello y miembros superiores en la aurícula derecha.', A([-4.8, -0.6, 7.6]))
pin('vci', 'Vena cava inferior', 'Drena la sangre venosa de la mitad inferior del cuerpo; su orificio posee la válvula de Eustaquio.', A([-4.3, 2.4, -2.6]))
pin('vp_sup_der', 'V. pulmonar superior derecha', 'Una de las cuatro venas pulmonares: sangre oxigenada hacia la aurícula izquierda.', A([-3.6, 4.6, 4.6]))
pin('vp_inf_der', 'V. pulmonar inferior derecha', 'Desemboca en la pared posterior de la aurícula izquierda.', A([-3.6, 5.0, 1.9]))
pin('vp_sup_izq', 'V. pulmonar superior izquierda', 'Desemboca junto a la base de la orejuela izquierda.', A([4.2, 4.4, 4.6]))
pin('vp_inf_izq', 'V. pulmonar inferior izquierda', 'Desemboca en la pared posterior de la aurícula izquierda.', A([4.2, 5.0, 2.2]))
pin('auricula_der', 'Aurícula derecha', 'Recibe el retorno venoso sistémico (cavas y seno coronario).', RA_C + A([-2.3, -0.9, 0.3]))
pin('orejuela_der', 'Orejuela derecha', 'Apéndice triangular de base ancha, tapizado por músculos pectíneos; cubre la raíz aórtica.', A([-2.5, -2.9, 3.2]))
pin('auricula_izq', 'Aurícula izquierda', 'Cámara más posterior; recibe las cuatro venas pulmonares.', LA_C + A([0, 2.3, 0.3]))
pin('orejuela_izq', 'Orejuela izquierda', 'Apéndice largo, estrecho y festoneado que abraza el tronco pulmonar.', A([4.4, -0.1, 4.2]))
pin('ventriculo_der', 'Ventrículo derecho', 'Pared libre de 3-5 mm; sección en semiluna abrazando al tabique. Forma la cara esternocostal.', rv_free(rad(-12), 3.2, -0.6))
pin('ventriculo_izq', 'Ventrículo izquierdo', 'Pared de 12-15 mm, tres veces más gruesa que la derecha; genera la presión sistémica.', lv_epi(rad(-150), 3.6, 0.2))
pin('apex', 'Ápex cardíaco', 'Vértice formado por el ventrículo izquierdo; orientado abajo, adelante y a la izquierda.', L2W(0, 0, 8.7))
pin('cono_arterioso', 'Cono arterioso', 'Infundíbulo liso del ventrículo derecho que conduce a la válvula pulmonar.', L2W(3.3, -4.6, 1.0))
for key, label, desc, src, layer in (
        ('tci', 'Tronco coronario izquierdo', 'Nace del seno aórtico izquierdo; se bifurca en descendente anterior y circunfleja.', 'left_main', 'coronary'),
        ('lad', 'A. descendente anterior (LAD)', 'Recorre el surco interventricular anterior; da ramas diagonales y septales.', 'LAD', 'coronary'),
        ('diagonal', 'Rama diagonal', 'Rama de la LAD para la pared anterolateral del ventrículo izquierdo.', 'diagonal_1', 'coronary'),
        ('lcx', 'A. circunfleja (LCx)', 'Discurre por el surco auriculoventricular izquierdo; da las marginales obtusas.', 'LCx', 'coronary'),
        ('marginal_obtusa', 'Rama marginal obtusa', 'Irriga la pared lateral del ventrículo izquierdo.', 'obtuse_marginal_1', 'coronary'),
        ('rca', 'A. coronaria derecha (RCA)', 'Recorre el surco auriculoventricular derecho hasta la cruz cardíaca.', 'RCA', 'coronary'),
        ('nodo_sinusal', 'A. del nodo sinusal', 'Rama proximal de la RCA que asciende hacia la unión cava superior-aurícula.', 'SA_node_artery', 'coronary'),
        ('marginal_aguda', 'Rama marginal aguda', 'Irriga la pared libre del ventrículo derecho a lo largo del borde agudo.', 'acute_marginal', 'coronary'),
        ('pda', 'A. descendente posterior (PDA)', 'Nace en la cruz cardíaca y recorre el surco interventricular posterior.', 'PDA', 'coronary'),
        ('seno_coronario', 'Seno coronario', 'Colector venoso posterior que desemboca en la aurícula derecha.', 'coronary_sinus', 'coronary'),
        ('vena_magna', 'Vena cardíaca magna', 'Acompaña a la LAD y a la circunfleja hasta el seno coronario.', 'great_cardiac_vein', 'coronary'),
        ('vena_media', 'Vena cardíaca media', 'Acompaña a la descendente posterior.', 'middle_cardiac_vein', 'coronary'),
        ('vena_menor', 'Vena cardíaca menor', 'Recorre el surco auriculoventricular derecho junto a la RCA.', 'small_cardiac_vein', 'coronary')):
    C = np.array(L04[src]['P'])
    R = np.array(L04[src]['R'])
    i = len(C) // 2
    n = surf.normals(C[i:i + 1])[0]
    Q = C[i] + n * (R[i] + 0.05)
    PINS.append(dict(id=key, label=label, desc=desc, layer=layer, interior=False, pos=glv(Q),
                     normal=[round(float(v), 4) for v in (n[0], n[2], -n[1])],
                     d_vc=glv(motion.ventricular(Q[None, :])[0]), d_as=glv(motion.atrial(Q[None, :])[0])))
C = np.array(L04['RCA']['P'])
pin('grasa', 'Grasa epicárdica', 'Tejido adiposo subepicárdico lobulado que rellena los surcos y envuelve los vasos coronarios.', C[len(C) // 3], layer='fat', lift=0.42)
PMS = papillary_muscles()
pin('valvula_mitral', 'Válvula mitral', 'Dos valvas (anterior y posterior) ancladas por cuerdas tendinosas a dos músculos papilares.', MV_C - 0.2 * MV_N, 'valves', True)
pin('valvula_tricuspide', 'Válvula tricúspide', 'Tres valvas: anterior, posterior y septal.', TV_C - 0.3 * TV_N, 'valves', True)
pin('valvula_aortica', 'Válvula aórtica', 'Tres valvas semilunares con nódulos de Arancio, alojadas en los senos de Valsalva.', AO_C + 0.8 * AO_N, 'valves', True)
pin('valvula_pulmonar', 'Válvula pulmonar', 'Tres valvas semilunares (anterior, derecha e izquierda) con nódulos de Morgagni.', PV_C + 0.8 * PV_N, 'valves', True)
pin('cuerdas', 'Cuerdas tendinosas', 'Cordones fibrosos de 1.º, 2.º y 3.er orden entre los papilares y el borde libre de las valvas.', (PMS['LV_PM_posteromedial']['tips'][0] + MV_C) / 2, 'valves', True)
pin('papilar_al', 'M. papilar anterolateral', 'Músculo papilar del ventrículo izquierdo.', PMS['LV_PM_anterolateral']['trunk'][2][0], 'myocardium', True)
pin('papilar_pm', 'M. papilar posteromedial', 'Músculo papilar del ventrículo izquierdo.', PMS['LV_PM_posteromedial']['trunk'][2][0], 'myocardium', True)
pin('papilar_ant_vd', 'M. papilar anterior (VD)', 'El mayor del ventrículo derecho; recibe la banda moderadora.', PMS['RV_PM_anterior']['trunk'][1][0], 'myocardium', True)
pin('banda_moderadora', 'Banda moderadora', 'Trabécula septomarginal: puente muscular del tabique al papilar anterior, con la rama derecha del haz de His.', MODERATOR_BAND[1][0], 'myocardium', True)
pin('tabique_iv', 'Tabique interventricular', 'Se abomba hacia el ventrículo derecho; mismo grosor que la pared libre izquierda.', lv_epi(0.0, 3.6, -0.6), 'myocardium', True)
pin('trabeculas', 'Trabéculas carnosas', 'Relieve muscular esponjoso del endocardio ventricular (fino en VI, grueso en VD).', lv_cyl(rad(170), 5.3, 0.1), 'myocardium', True)
pin('fosa_oval', 'Fosa oval', 'Depresión del tabique interauricular, vestigio del foramen oval fetal.', A(L01['fossa']['center']) - 0.1 * A(L01['fossa']['normal']), 'myocardium', True)
al = rad(153)
pin('crista_terminalis', 'Crista terminalis', 'Cresta muscular que separa el seno venoso liso de la pared pectínea.', RA_C + 0.93 * RA_RAD * A([math.cos(al), math.sin(al), 0.0]), 'myocardium', True)
pin('pectineos', 'Músculos pectíneos', 'Crestas musculares paralelas de la pared lateral de la aurícula derecha y las orejuelas.', RA_C + 0.9 * RA_RAD * A([-0.62, -0.72, 0.25]), 'myocardium', True)
pin('eustaquio', 'Válvula de Eustaquio', 'Pliegue semilunar en el borde anterior del orificio de la vena cava inferior.', A(IVC[-1][0]) + A([0.3, -1.05, 0.55]), 'myocardium', True)
log('pins:', len(PINS))


# ---- blood-flow routes (centrelines through veins -> atrium -> AV valve -> ventricle -> artery)
def rv_mid(psi_deg, t):
    r = float(rv_q(t)) - 0.5 * float(np.interp(t, RV_T, RV_W))
    return L2W(float(rv_c(t)) + r * math.cos(rad(psi_deg)), RV_DELTA0 + r * math.sin(rad(psi_deg)), t)


def pth(ctrl, r=None, rev=False, upto=None, frm=None):
    P, R = resample_path(ctrl, 0.5)
    if upto is not None:                       # keep the part on the heart side of a cut plane
        keep = ((P - np.asarray(upto[0])) @ unit(upto[1])) < 0.0
        P, R = P[keep], R[keep]
    if rev:
        P, R = P[::-1], R[::-1]
    return [(p, (rr if r is None else r) * 0.72) for p, rr in zip(P, R)]


def chamber(points, r):
    return [(np.asarray(p, float), r) for p in points]


RIGHT_CORE = (chamber([TV_C - 1.0 * TV_N], 1.0), chamber([TV_C + 0.5 * TV_N], 1.0),
              chamber([rv_mid(22, 1.6), rv_mid(20, 3.1), rv_mid(8, 4.6), rv_mid(-12, 5.3), rv_mid(-30, 4.2), rv_mid(-42, 3.0)], 0.75)
              + chamber([L2W(2.9, -1.6, 2.6), L2W(2.8, -3.3, 0.9)], 0.85), chamber([PV_C], 0.85))
LEFT_CORE = (chamber([MV_C - 1.0 * MV_N], 0.95), chamber([MV_C + 0.6 * MV_N], 0.95),
             chamber([lv_axis(2.4) - 0.55 * S_AX + 0.25 * D_AX, lv_axis(4.4) - 0.4 * S_AX + 0.15 * D_AX, lv_axis(5.9),
                      lv_axis(4.6) + 0.7 * S_AX - 0.5 * D_AX, lv_axis(2.8) + 0.8 * S_AX - 0.65 * D_AX, L2W(0.75, -0.6, 1.3),
                      L2W(1.1, -0.95, 0.0)], 0.8), chamber([AO_C], 0.8))
ROUTES = []


def route(kind, vein, atrium_pts, core, artery, name):
    pre, av, vent, sl = core
    seq = vein + atrium_pts + pre
    i_av = len(seq)
    seq = seq + av + vent
    i_sl = len(seq)
    seq = seq + sl + artery
    P = np.array([p for p, r in seq], float)
    R = np.array([r for p, r in seq], float)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    ROUTES.append(dict(name=name, kind=kind, pts=gl(P).tolist(), radius=np.round(R * SCALE, 5).tolist(),
                       u_vein=float(arc[len(vein)] / arc[-1]), u_av=float(arc[i_av] / arc[-1]),
                       u_sl=float(arc[i_sl] / arc[-1]), length=float(arc[-1] * SCALE)))


svc_in = pth(SVC, upto=SVC_CUT)
ivc_in = pth(IVC, upto=IVC_CUT)
pt_out = pth(PT[1:])
ra_top = chamber([RA_C + A([0, 0.2, 0.9]), RA_C + A([0.3, -0.3, -0.4])], 1.15)
ra_bot = chamber([RA_C + A([0.1, 0.5, -1.0]), RA_C + A([0.4, -0.3, -0.3])], 1.15)
for vn, vp, ap in (('svc', svc_in, ra_top), ('ivc', ivc_in, ra_bot)):
    for an, art, cut in (('rpa', RPA, RPA_CUT), ('lpa', LPA, LPA_CUT)):
        route('deoxy', vp, ap, RIGHT_CORE, pt_out + pth(art[1:], upto=cut), vn + '_' + an)
ao_asc = pth(AORTA[2:8])
ao_desc = pth(AORTA[7:], upto=AORTA_CUT)
la_mid = chamber([LA_C + A([0.0, -0.1, -0.2])], 1.2)
outs = {'desc': ao_asc + ao_desc, 'bct': pth(AORTA[2:7]) + pth(BCT) + pth(RCCA[1:], upto=BRANCH_CUTS[1]),
        'lcca': ao_asc + pth(LCCA, upto=BRANCH_CUTS[2]), 'lsa': pth(AORTA[2:9]) + pth(LSA, upto=BRANCH_CUTS[3])}
for vn, an in (('RSPV', 'desc'), ('RIPV', 'bct'), ('LSPV', 'desc'), ('LIPV', 'lsa'), ('RSPV', 'lcca'), ('LIPV', 'desc')):
    route('oxy', pth(PVEINS[vn], upto=PVEIN_CUTS[vn]), la_mid, LEFT_CORE, outs[an], vn.lower() + '_' + an)
log('flow routes:', len(ROUTES))

meta = dict(
    units='metres (glTF, Y-up)', scale=SCALE,
    axes=dict(long_axis=[round(float(v), 5) for v in (A_AX[0], A_AX[2], -A_AX[1])],
              septal=[round(float(v), 5) for v in (S_AX[0], S_AX[2], -S_AX[1])],
              lv_base=glv(O_L), apex=glv(L2W(0, 0, LV_APEX_T)), center=glv(np.array([0.4, 0.6, 2.9])),
              four_chamber_normal=[round(float(v), 5) for v in (lambda d: (d[0], d[2], -d[1]))(unit(-D_AX * 0.6 + np.array([0.0, -0.75, 0.1])))],
              four_chamber_point=glv(L2W(0, -0.2, 0)), valve_plane_point=glv(L2W(0, 0, -0.78))),
    valves={k.lower(): dict(center=glv(c), normal=[round(float(v), 5) for v in (n[0], n[2], -n[1])])
            for k, c, n in (('mitral', MV_C, MV_N), ('tricuspid', TV_C, TV_N), ('aortic', AO_C, AO_N), ('pulmonary', PV_C, PV_N))},
    pins=PINS, routes=ROUTES)
with open(os.path.join(OUT_DIR, 'heart_meta.json'), 'w') as f:
    json.dump(meta, f, separators=(',', ':'))

opt = os.path.join(ROOT, 'scripts', 'optimize-glb.mjs')
final = os.path.join(OUT_DIR, 'human_heart.glb')
if os.path.exists(os.path.join(ROOT, 'node_modules', '@gltf-transform')) and os.path.exists(opt):
    subprocess.check_call(['node', opt, raw, final])
    log('final glb: %.1f MB' % (os.path.getsize(final) / 1e6))
else:
    log('node_modules missing: run `npm install && node scripts/optimize-glb.mjs %s %s`' % (raw, final))
log('stage 05 complete')
