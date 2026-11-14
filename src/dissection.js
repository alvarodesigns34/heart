import * as THREE from 'three';

/**
 * Real dissection: interactive section planes whose cut surface is closed with a stencil-buffer
 * cap, so the walls read as solid sliced tissue (never hollow shells).
 *
 * For every (layer, plane) pair the layer's closed meshes are drawn twice into the stencil buffer
 * only (back faces ++, front faces --); wherever the count is non-zero the plane lies inside
 * tissue, and a lit, textured quad is drawn there.  Works with morph targets because the helper
 * meshes share geometry and morphTargetInfluences with the visible ones.
 */
export class Dissection {
  constructor(scene) {
    this.scene = scene;
    this.planes = [];          // {id, plane, base (Vector3), normal (unit, kept side), offset, invert, enabled}
    this.active = [];          // THREE.Plane[] shared by every clipped material
    this.layers = [];          // {name, meshes, visible, entries:[{planeDef, group, cap}]}
    this.capSize = 4.2;
  }

  addPlane(id, base, keepNormal) {
    const def = { id, base: base.clone(), normal: keepNormal.clone().normalize(), offset: 0, invert: false, enabled: false,
      plane: new THREE.Plane() };
    this.planes.push(def);
    this.updatePlane(def);
    return def;
  }

  updatePlane(def) {
    const n = def.normal.clone().multiplyScalar(def.invert ? -1 : 1);
    const p = def.base.clone().addScaledVector(def.normal, def.offset);
    def.plane.setFromNormalAndCoplanarPoint(n, p);
    def.point = p;
  }

  /** capMaterials: {beauty, normal} — both get the stencil test configured here */
  addLayer(name, meshes, capMaterials) {
    const layer = { name, meshes, visible: true, entries: [] };
    const li = this.layers.length;
    this.planes.forEach((def, pi) => {
      const order = 1 + li * this.planes.length + pi;
      const group = new THREE.Group();
      group.visible = false;
      const mk = (side, op) => new THREE.MeshBasicMaterial({
        side, depthWrite: false, depthTest: false, colorWrite: false, stencilWrite: true,
        stencilFunc: THREE.AlwaysStencilFunc, stencilFail: op, stencilZFail: op, stencilZPass: op,
        clippingPlanes: [def.plane],
      });
      const back = mk(THREE.BackSide, THREE.IncrementWrapStencilOp);
      const front = mk(THREE.FrontSide, THREE.DecrementWrapStencilOp);
      for (const m of meshes) {
        for (const mat of [back, front]) {
          const h = new THREE.Mesh(m.geometry, mat);
          h.morphTargetInfluences = m.morphTargetInfluences;
          h.morphTargetDictionary = m.morphTargetDictionary;
          h.renderOrder = order;
          h.frustumCulled = false;
          h.userData.stencilHelper = true;
          m.parent.add(h);
          h.position.copy(m.position);
          h.quaternion.copy(m.quaternion);
          h.scale.copy(m.scale);
          h.visible = false;
          group.userData.helpers = (group.userData.helpers || []).concat(h);
        }
      }
      const prep = (mat) => {
        mat.stencilWrite = true;
        mat.stencilRef = 0;
        mat.stencilFunc = THREE.NotEqualStencilFunc;
        mat.stencilFail = THREE.ReplaceStencilOp;
        mat.stencilZFail = THREE.ReplaceStencilOp;
        mat.stencilZPass = THREE.ReplaceStencilOp;
        mat.side = THREE.DoubleSide;
        return mat;
      };
      const beauty = prep(capMaterials.beauty.clone());
      const normal = prep(capMaterials.normal.clone());
      const cap = new THREE.Mesh(new THREE.PlaneGeometry(this.capSize, this.capSize), beauty);
      cap.renderOrder = order + 0.1;
      cap.visible = false;
      cap.userData.cap = true;
      cap.userData.beauty = beauty;
      cap.userData.normal = normal;
      cap.onAfterRender = (renderer) => renderer.clearStencil();
      this.scene.add(cap);
      layer.entries.push({ def, group, cap });
    });
    this.layers.push(layer);
    return layer;
  }

  setLayerVisible(name, v) {
    const l = this.layers.find((x) => x.name === name);
    if (l) l.visible = v;
    this.refresh();
  }

  refresh() {
    this.active.length = 0;
    for (const d of this.planes) {
      this.updatePlane(d);
      if (d.enabled) this.active.push(d.plane);
    }
    for (const layer of this.layers) {
      for (const e of layer.entries) {
        const on = e.def.enabled && layer.visible;
        for (const h of e.group.userData.helpers) h.visible = on;
        e.cap.visible = on;
        if (on) {
          const others = this.active.filter((p) => p !== e.def.plane);
          e.cap.userData.beauty.clippingPlanes = others;
          e.cap.userData.normal.clippingPlanes = others;
          e.cap.position.copy(e.def.point);
          const n = e.def.plane.normal;
          e.cap.lookAt(e.def.point.x - n.x, e.def.point.y - n.y, e.def.point.z - n.z);
        }
      }
    }
  }

  get anyEnabled() {
    return this.active.length > 0;
  }

  /** true if a world-space point survives every active plane */
  keeps(p, margin = 0) {
    for (const pl of this.active) if (pl.distanceToPoint(p) < margin) return false;
    return true;
  }
}

/** Procedural cross-section texture: sliced striated muscle (or lobulated fat). */
export function tissueTexture(kind = 'muscle') {
  const s = 512;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  let seed = kind === 'muscle' ? 7 : 19;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  if (kind === 'muscle') {
    g.fillStyle = '#6a1914';
    g.fillRect(0, 0, s, s);
    for (let i = 0; i < 5200; i++) {            // fascicles cut obliquely: short darker / lighter strokes
      const x = rnd() * s, y = rnd() * s, l = 6 + rnd() * 26, a = 0.55 + (rnd() - 0.5) * 0.9;
      const light = rnd() > 0.5;
      g.strokeStyle = light ? `rgba(160,62,50,${0.10 + rnd() * 0.16})` : `rgba(48,8,8,${0.12 + rnd() * 0.2})`;
      g.lineWidth = 0.6 + rnd() * 1.6;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l);
      g.stroke();
    }
    for (let i = 0; i < 260; i++) {             // perimysium / small vessels
      g.fillStyle = `rgba(226,170,150,${0.08 + rnd() * 0.12})`;
      g.beginPath();
      g.arc(rnd() * s, rnd() * s, 0.6 + rnd() * 1.8, 0, 6.3);
      g.fill();
    }
  } else {
    g.fillStyle = '#d9a640';
    g.fillRect(0, 0, s, s);
    for (let i = 0; i < 900; i++) {
      const x = rnd() * s, y = rnd() * s, r = 5 + rnd() * 13;
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, `rgba(244,214,120,${0.5 + rnd() * 0.3})`);
      gr.addColorStop(1, 'rgba(180,110,30,0.0)');
      g.fillStyle = gr;
      g.beginPath();
      g.arc(x, y, r, 0, 6.3);
      g.fill();
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(7, 7);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
