import { expect, test } from 'bun:test';
import { inspectPrivateGlb } from './inspection-display';

function fixture(mutate: (document: any, binary: DataView) => void = () => {}) {
  const binary = new DataView(new ArrayBuffer(48));
  [0, 0, 0, 1, 0, 0, 0, 1, 0].forEach((value, index) => binary.setFloat32(index * 4, value, true));
  [0, 1, 2].forEach((value, index) => binary.setUint32(36 + index * 4, value, true));
  const document = {
    asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0, translation: [0, 0, 0] }],
    buffers: [{ byteLength: 48 }], bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 36 }, { buffer: 0, byteOffset: 36, byteLength: 12 }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }, { bufferView: 1, componentType: 5125, count: 3, type: 'SCALAR' }],
    meshes: [{ extras: { instanceId: 'front-shell', templateId: 'front', role: 'shell' }, primitives: [{ attributes: { POSITION: 0 }, indices: 1 }] }],
    extras: { classification: 'placement-inspection', patternDigest: 'a'.repeat(64) },
  };
  mutate(document, binary);
  const metadata = new TextEncoder().encode(JSON.stringify(document));
  const paddedLength = Math.ceil(metadata.length / 4) * 4;
  const bytes = new ArrayBuffer(28 + paddedLength + binary.byteLength);
  const header = new DataView(bytes);
  [0x46546c67, 2, bytes.byteLength, paddedLength, 0x4e4f534a].forEach((value, index) => header.setUint32(index * 4, value, true));
  new Uint8Array(bytes, 20, paddedLength).fill(32);
  new Uint8Array(bytes, 20, metadata.length).set(metadata);
  header.setUint32(20 + paddedLength, binary.byteLength, true);
  header.setUint32(24 + paddedLength, 0x004e4942, true);
  new Uint8Array(bytes, 28 + paddedLength).set(new Uint8Array(binary.buffer));
  return bytes;
}

test('private display accepts self-contained pattern geometry and retains source identity', () => {
  expect(inspectPrivateGlb(fixture()).meshes[0]!.extras.templateId).toBe('front');
});

test('private display rejects external resources, scene cycles, unbounded accessors and corrupt geometry', () => {
  const attacks: ((document: any, binary: DataView) => void)[] = [
    document => { document.buffers[0].uri = 'https://untrusted.invalid/leak'; },
    document => { document.meshes[0].extensions = { unknown: {} }; },
    document => { document.nodes[0].children = [0]; },
    document => { document.accessors[0].count = 1000000000; },
    document => { document.accessors[0].sparse = {}; },
    document => { document.bufferViews[0].byteOffset = 999999; },
    document => { document.nodes[0].scale = [2, 2, 2]; },
    document => { document.extras.classification = 'simulated-drape'; },
    document => { delete document.meshes[0].extras.templateId; },
    (_document, binary) => { binary.setFloat32(0, NaN, true); },
    (_document, binary) => { binary.setUint32(36, 100, true); },
  ];
  for (const attack of attacks) expect(() => inspectPrivateGlb(fixture(attack))).toThrow();
  const truncated = fixture().slice(0, -4);
  expect(() => inspectPrivateGlb(truncated)).toThrow();
});
