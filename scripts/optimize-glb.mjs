// Quantises + Meshopt-compresses the raw Blender export (geometry AND morph targets) and
// re-encodes the baked PBR maps as WebP.   node scripts/optimize-glb.mjs raw.glb public/models/human_heart.glb
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, prune, textureCompress, meshopt } from '@gltf-transform/functions';
import { MeshoptEncoder } from 'meshoptimizer';
import sharp from 'sharp';
import fs from 'node:fs';

const [, , input, output] = process.argv;
if (!input || !output) {
  console.error('usage: node scripts/optimize-glb.mjs <raw.glb> <out.glb>');
  process.exit(1);
}
await MeshoptEncoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder });
const doc = await io.read(input);

await doc.transform(
  dedup(),
  prune({ keepAttributes: true, keepLeaves: true }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^baseColor/, quality: 88, effort: 80 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^normal/, quality: 93, effort: 80 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', slots: /^(occlusion|metallicRoughness)/, quality: 86, effort: 80 }),
  meshopt({ encoder: MeshoptEncoder, level: 'medium', quantizePosition: 14, quantizeNormal: 10, quantizeTexcoord: 13 }),
);

fs.mkdirSync(new URL('.', 'file://' + (output.startsWith('/') ? output : process.cwd() + '/' + output)).pathname, { recursive: true });
await io.write(output, doc);
const mb = (p) => (fs.statSync(p).size / 1e6).toFixed(2) + ' MB';
let tris = 0;
for (const mesh of doc.getRoot().listMeshes()) {
  for (const prim of mesh.listPrimitives()) tris += prim.getIndices().getCount() / 3;
}
console.log(`[optimize-glb] ${input} (${mb(input)}) -> ${output} (${mb(output)}), ${Math.round(tris)} triangles`);
