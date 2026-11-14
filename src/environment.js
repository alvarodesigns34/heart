import * as THREE from 'three';

/**
 * Procedural "surgical theatre" HDR environment: a dark room with a twin-head operating lamp
 * overhead, a cool side soft-box and a cold back strip.  Pre-filtered with PMREM so that the wet
 * clearcoat picks up crisp, believable highlights from every direction.
 */
export function surgicalEnvironment(renderer) {
  const scene = new THREE.Scene();
  const room = new THREE.Mesh(
    new THREE.BoxGeometry(24, 14, 24),
    new THREE.MeshBasicMaterial({ color: new THREE.Color(0.012, 0.013, 0.017), side: THREE.BackSide }),
  );
  scene.add(room);

  const panel = (geo, pos, color, intensity) => {
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: new THREE.Color(...color).multiplyScalar(intensity), side: THREE.DoubleSide }));
    m.position.set(...pos);
    m.lookAt(0, 0, 0);
    scene.add(m);
    return m;
  };
  // operating lamp: two heads, warm white, each a ring of LED cells
  for (const [cx, cy, cz, r, k] of [[-2.4, 5.6, 3.6, 1.25, 26], [2.8, 5.2, 1.6, 0.95, 17]]) {
    panel(new THREE.CircleGeometry(r * 0.42, 24), [cx, cy, cz], [1.0, 0.93, 0.84], k);
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2;
      panel(new THREE.CircleGeometry(r * 0.24, 16), [cx + Math.cos(a) * r * 0.8, cy, cz + Math.sin(a) * r * 0.8], [1.0, 0.94, 0.86], k * 0.8);
    }
  }
  panel(new THREE.PlaneGeometry(3.2, 6.5), [-9, 1.2, 1.5], [0.78, 0.88, 1.0], 2.4);     // cool key-side soft-box
  panel(new THREE.PlaneGeometry(7.5, 1.3), [5.5, 2.6, -8.5], [0.55, 0.75, 1.0], 9.0);    // cold rim strip
  panel(new THREE.PlaneGeometry(2.2, 4.0), [9.5, 0.4, 2.5], [1.0, 0.82, 0.7], 1.0);      // faint warm bounce
  panel(new THREE.PlaneGeometry(10, 10), [0, -6.5, 0], [0.6, 0.45, 0.4], 0.12);          // drape below

  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(scene, 0.035).texture;
  pmrem.dispose();
  scene.traverse((o) => {
    if (o.isMesh) {
      o.geometry.dispose();
      o.material.dispose();
    }
  });
  return tex;
}
