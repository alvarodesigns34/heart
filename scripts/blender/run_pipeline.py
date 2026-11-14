"""
run_pipeline.py — master orchestrator.

    blender --background --python scripts/blender/run_pipeline.py -- [--stages 01,02,03,04,05]
                                                                    [--no-render] [--voxel 0.04]

Runs every modelling stage in its own headless Blender process (so a failure in one stage never
corrupts the others and each stage can be re-run in isolation), then renders the four diagnostic
views with Cycles into /tmp/renders:
    diag_01_anterior.png, diag_02_posterior.png, diag_03_cutaway_4chambers.png, diag_04_valves_closeup.png
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []


def opt(name, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
    return default


STAGES = [s for s in str(opt('--stages', '01,02,03,04,05')).split(',') if s and s != 'none']
RENDER = not opt('--no-render', False)
if opt('--voxel'):
    os.environ['HEART_VOXEL'] = str(opt('--voxel'))
RENDER_DIR = opt('--render-dir', '/tmp/renders')
VIEWS = str(opt('--views', '1,2,3,4')).split(',')
SAMPLES = int(opt('--samples', 96))
RES = int(opt('--res', 1024))


def ensure_deps():
    try:
        import scipy  # noqa
        import skimage  # noqa
    except ImportError:
        py = sys.executable
        subprocess.call([py, '-m', 'ensurepip'])
        subprocess.check_call([py, '-m', 'pip', 'install', 'scipy', 'scikit-image'])


def run_stages():
    import bpy
    scripts = sorted(f for f in os.listdir(HERE) if f[:2].isdigit() and f.endswith('.py'))
    for s in scripts:
        if s[:2] not in STAGES:
            continue
        print('=' * 100 + '\n>>> ' + s, flush=True)
        r = subprocess.run([bpy.app.binary_path, '--background', '--python', os.path.join(HERE, s)])
        if r.returncode != 0:
            raise SystemExit('stage %s failed' % s)


# ----------------------------------------------------------------------------------------
# Diagnostic renders
# ----------------------------------------------------------------------------------------
def render_diagnostics():
    import bpy
    import bmesh
    import numpy as np
    from heartlib import (Grid, load, bpath, half_space, extract_surface, mesh_from_arrays, L2W, unit, A_AX, D_AX,
                          log, mesh_arrays, AO_C, AO_N, PV_C, PV_N)
    import heartmat as hm

    os.makedirs(RENDER_DIR, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    for st in ('stage02', 'stage03', 'stage04'):
        with bpy.data.libraries.load(bpath(st + '.blend')) as (src, dst):
            dst.objects = list(src.objects)
        for o in dst.objects:
            sc.collection.objects.link(o)
    mats = hm.procedural_materials()
    heart = bpy.data.objects['Heart']
    hm.heart_attributes(heart)
    aux = [o for o in sc.objects if o.type == 'MESH' and o.name != 'Heart']
    for o in aux:
        if o.name.startswith('Valve'):
            hm.valve_vertex_colors(o)
        hm.assign(o, hm.material_for(o.name, mats))
    hm.assign(heart, mats['heart'])
    hm.setup_cycles(sc, SAMPLES, RES)
    hm.studio_world(sc)
    g = Grid()
    S = None

    def clear_rig():
        for o in list(sc.objects):
            if o.type in ('CAMERA', 'LIGHT') or o.name.startswith('CUT_'):
                bpy.data.objects.remove(o, do_unlink=True)

    def cut_copy(o, p, n):
        """Planar section of an auxiliary (closed, manifold) mesh, capped with an exact boolean so that
        band-shaped sections (fat sleeves, vessel walls) close correctly. Keeps the side opposite to n."""
        from mathutils import Vector
        # evaluated copy: keeps the current shape-key pose (e.g. open AV valves)
        me = bpy.data.meshes.new_from_object(o.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        c = bpy.data.objects.new('CUT_' + o.name, me)
        sc.collection.objects.link(c)
        if o.name.startswith(('Valve', 'Chordae')):
            # thin leaflets touch themselves along the coaptation line: an exact boolean would drop
            # them, a plain bisection is enough for sub-millimetre sheets
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=p, plane_no=n, clear_outer=True)
            bm.to_mesh(me)
            bm.free()
            return c
        box = bpy.data.objects.new('CUT_box', bpy.data.meshes.new('CUT_box'))
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(box.data)
        bm.free()
        sc.collection.objects.link(box)
        nn = Vector(unit(n))
        box.scale = (80.0, 80.0, 80.0)
        box.location = Vector(p) + nn * 40.0
        box.rotation_euler = nn.to_track_quat('Z', 'Y').to_euler()
        m = c.modifiers.new('cut', 'BOOLEAN')
        m.operation = 'DIFFERENCE'
        m.solver = 'EXACT'
        m.object = box
        for x in sc.objects:
            x.select_set(x == c)
        bpy.context.view_layer.objects.active = c
        bpy.ops.object.modifier_apply(modifier='cut')
        bpy.data.objects.remove(box, do_unlink=True)
        return c

    def flat_section(ob, p, n, tol=2e-3):
        """Flat-shade the section faces (on the cut plane): smooth normals there produce radial streaks."""
        V = mesh_arrays(ob)
        on = np.abs((V - np.asarray(p, float)) @ np.asarray(n, float)) < tol
        for poly in ob.data.polygons:
            if all(on[i] for i in poly.vertices):
                poly.use_smooth = False

    def root_keep(X, Y, Z, C, n, below, above, r):
        """SDF-ish field, negative inside a capped cylinder around an arterial root."""
        C = [float(v) for v in C]
        n = unit(n)
        qx, qy, qz = X - C[0], Y - C[1], Z - C[2]
        h = qx * float(n[0]) + qy * float(n[1]) + qz * float(n[2])
        rad = np.sqrt(np.maximum(qx * qx + qy * qy + qz * qz - h * h, 0))
        return np.maximum(np.maximum(rad - r, h - above), -h - below)

    def shot(name, view_dir, center, scale=None, cut=None, lens=85.0, av=0.0, sl=0.0, keep_roots=False):
        nonlocal S
        clear_rig()
        for o in aux:
            if o.data.shape_keys:
                for k in o.data.shape_keys.key_blocks[1:]:
                    k.value = av if k.name == 'av_valves_open' else (sl if k.name == 'semilunar_valves_open' else 0.0)
        if cut is None:
            heart.hide_render = False
            for o in aux:
                o.hide_render = False
        else:
            p, n = np.asarray(cut[0], float), unit(cut[1])
            if S is None:
                S = load('solid02')
            X, Y, Z = g.coords(tuple(slice(None) for _ in range(3)))
            R = half_space(X, Y, Z, p, n)
            if keep_roots:      # classic valve-plane preparation: atria removed, arterial roots preserved
                R = np.minimum(R, root_keep(X, Y, Z, AO_C, AO_N, 3.0, 1.42, 1.95))
                R = np.minimum(R, root_keep(X, Y, Z, PV_C, PV_N, 3.0, 1.32, 1.85))
            v, f, _ = extract_surface(g, np.maximum(S, R))
            del R
            c = mesh_from_arrays('CUT_Heart', v, f)
            hm.heart_attributes(c, cut=(p, n))
            hm.assign(c, mats['heart'])
            heart.hide_render = True
            for o in aux:
                if keep_roots and o.name.startswith(('Valve', 'Chordae')):
                    o.hide_render = False          # the whole valvular apparatus stays in place
                    continue
                o.hide_render = True
                # interpenetrating tissues (fat partially buries the vessels and sits 0.6 mm into the
                # myocardium): stagger their section planes so the caps never z-fight
                off = 0.012 if o.name.startswith('Coronary') else (-0.015 if o.name == 'Epicardial_Fat' else 0.0)
                pc = p + off * n
                cc = cut_copy(o, pc, n)
                hm.assign(cc, hm.material_for(o.name, mats))
                flat_section(cc, pc, n)
        hm.camera_rig(sc, view_dir, center, ortho_scale=scale, lens=lens)
        sc.render.filepath = os.path.join(RENDER_DIR, name + '.png')
        bpy.ops.render.render(write_still=True)
        log('rendered', sc.render.filepath)

    if '1' in VIEWS:
        shot('diag_01_anterior', (0.10, -1.0, 0.10), (0.35, 0.5, 3.3), lens=92)
    if '2' in VIEWS:
        shot('diag_02_posterior', (-0.08, 1.0, 0.16), (0.35, 0.5, 3.3), lens=92)
    if '3' in VIEWS:
        vd = unit(-D_AX * 0.6 + np.array([0.0, -0.75, 0.1]))
        shot('diag_03_cutaway_4chambers', tuple(vd), tuple(L2W(1.3, 0.0, 1.5) + np.array([0, 0, 0.6])), scale=17.0, cut=(L2W(0, -0.2, 0), vd), av=0.85)
    if '4' in VIEWS:
        vd = unit(-A_AX + 0.12 * np.array([0.0, -1.0, 0.0]))
        shot('diag_04_valves_closeup', tuple(vd), tuple(L2W(1.7, -1.4, -1.0)), scale=10.4, cut=(L2W(0, 0, -0.62), -A_AX),
             keep_roots=True)


if __name__ == '__main__':
    ensure_deps()
    run_stages()
    if RENDER:
        render_diagnostics()
