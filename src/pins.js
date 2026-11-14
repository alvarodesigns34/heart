import * as THREE from 'three';

/** Interactive 3D anatomical pins: world-anchored, follow the beating wall, decluttered labels. */
export class Pins {
  constructor(container, pins, toWorld, worldScale, onSelect) {
    this.container = container;
    this.onSelect = onSelect;
    this.selected = null;
    this.enabled = true;
    this.labelMode = 'all';
    this.items = pins.map((p) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'pin' + (p.interior ? ' pin--interior' : '');
      el.dataset.layer = p.layer;
      el.innerHTML = '<i class="pin__dot"></i><span class="pin__label"></span>';
      el.querySelector('.pin__label').textContent = p.label;
      el.setAttribute('aria-label', p.label);
      container.appendChild(el);
      const item = {
        ...p, el, label: p.label, base: toWorld(p.pos),
        dvc: new THREE.Vector3(...p.d_vc).multiplyScalar(worldScale),
        das: new THREE.Vector3(...p.d_as).multiplyScalar(worldScale),
        n: new THREE.Vector3(...p.normal), world: new THREE.Vector3(), w: 0, shown: false,
      };
      el.addEventListener('click', (e) => {
        e.stopPropagation();
        this.select(item);
      });
      return item;
    });
    this._v = new THREE.Vector3();
    this._c = new THREE.Vector3();
  }

  select(item) {
    if (this.selected) this.selected.el.classList.remove('is-selected');
    this.selected = item;
    if (item) item.el.classList.add('is-selected');
    this.onSelect(item);
  }

  /** ctx: {camera, width, height, state, dissection, layers:{myocardium,...}, interiorVisible} */
  update(ctx) {
    const { camera, width, height, state, dissection, layers } = ctx;
    camera.getWorldPosition(this._c);
    const placed = [];
    const list = [];
    const maxLabels = width < 640 ? 7 : 40;
    for (const it of this.items) {
      it.world.copy(it.base).addScaledVector(it.dvc, state.vc).addScaledVector(it.das, state.as);
      let vis = this.enabled && layers[it.layer] !== false;
      if (vis && !dissection.keeps(it.world, -0.004)) vis = false;
      if (vis) {
        if (it.interior) vis = ctx.interiorVisible;
        else {
          this._v.copy(this._c).sub(it.world).normalize();
          vis = this._v.dot(it.n) > 0.08;
        }
      }
      if (vis) {
        this._v.copy(it.world).project(camera);
        vis = this._v.z < 1 && Math.abs(this._v.x) < 1.02 && Math.abs(this._v.y) < 1.02;
        it.sx = (this._v.x * 0.5 + 0.5) * width;
        it.sy = (-this._v.y * 0.5 + 0.5) * height;
        it.depth = this._v.z;
      }
      if (vis !== it.shown) {
        it.el.classList.toggle('is-visible', vis);
        it.shown = vis;
      }
      if (vis) list.push(it);
    }
    // nearest first; greedily keep labels that do not collide
    list.sort((a, b) => (a === this.selected ? -1 : b === this.selected ? 1 : a.depth - b.depth));
    for (const it of list) {
      if (!it.w) it.w = it.el.querySelector('.pin__label').offsetWidth + 22;
      let right = it.sx >= width * 0.5;                     // labels point outwards...
      if (right && it.sx + it.w + 8 > width) right = false;  // ...unless they would leave the screen
      else if (!right && it.sx - it.w - 8 < 0) right = true;
      const x0 = right ? it.sx + 6 : it.sx - it.w - 6;
      const rect = [x0, it.sy - 11, x0 + it.w, it.sy + 11];
      let ok = (this.labelMode === 'all' && placed.length < maxLabels) || it === this.selected;
      if (ok) {
        for (const r of placed) {
          if (rect[0] < r[2] && rect[2] > r[0] && rect[1] < r[3] && rect[3] > r[1]) {
            ok = it === this.selected;
            break;
          }
        }
      }
      if (ok) placed.push(rect);
      it.el.classList.toggle('is-labelled', ok);
      it.el.classList.toggle('is-left', !right);
      it.el.style.transform = `translate3d(${it.sx.toFixed(1)}px, ${it.sy.toFixed(1)}px, 0)`;
    }
  }
}
