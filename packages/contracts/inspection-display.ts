export function inspectPrivateGlb(bytes: ArrayBuffer) {
  if (bytes.byteLength < 28 || bytes.byteLength > 16 * 1024 * 1024)
    throw new Error('The 3D file exceeds the supported display budget.');
  const header = new DataView(bytes);
  if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2 || header.getUint32(8, true) !== bytes.byteLength)
    throw new Error('Invalid 3D file header.');
  const metadataLength = header.getUint32(12, true);
  if (metadataLength % 4 || metadataLength > 2 * 1024 * 1024 || 28 + metadataLength > bytes.byteLength || header.getUint32(16, true) !== 0x4e4f534a)
    throw new Error('Invalid 3D metadata.');
  const document = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 20, metadataLength)));
  const binaryOffset = 20 + metadataLength;
  if (header.getUint32(binaryOffset + 4, true) !== 0x004e4942 || binaryOffset + 8 + header.getUint32(binaryOffset, true) !== bytes.byteLength)
    throw new Error('The 3D file must contain one embedded geometry buffer.');
  const visit = (value: unknown, depth = 0) => {
    if (depth > 24) throw new Error('3D metadata nesting exceeds the display budget.');
    if (!value || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
      if (['uri', 'extensions', 'extensionsUsed', 'extensionsRequired'].includes(key))
        throw new Error('External resources and extensions are not supported in private 3D files.');
      visit(child, depth + 1);
    }
  };
  visit(document);
  if (document.asset?.version !== '2.0' || document.buffers?.length !== 1 || document.buffers[0].byteLength > header.getUint32(binaryOffset, true) || document.images?.length || document.animations?.length || document.skins?.length)
    throw new Error('Unsupported 3D display content.');
  if (!Array.isArray(document.meshes) || !document.meshes.length || document.meshes.length > 100 || document.nodes?.length !== document.meshes.length)
    throw new Error('Invalid physical piece inventory.');
  if (document.scene !== 0 || document.scenes?.length !== 1 || document.scenes[0].nodes?.length !== document.nodes.length || document.scenes[0].nodes.some((node: unknown, index: number) => node !== index))
    throw new Error('Invalid 3D scene.');
  if (!Array.isArray(document.accessors) || document.accessors.length !== document.meshes.length * 2 || !Array.isArray(document.bufferViews) || document.bufferViews.length !== document.accessors.length)
    throw new Error('Invalid 3D geometry buffers.');
  let totalVertices = 0;
  for (const [index, node] of document.nodes.entries()) {
    if (node.mesh !== index || node.children || node.matrix || node.rotation || node.scale || !Array.isArray(node.translation) || node.translation.length !== 3 || node.translation.some((value: unknown) => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 100))
      throw new Error('Unsupported piece placement.');
  }
  for (const [index, accessor] of document.accessors.entries()) {
    const positions = index % 2 === 0;
    if (accessor.bufferView !== index || accessor.sparse || accessor.byteOffset || accessor.normalized || accessor.componentType !== (positions ? 5126 : 5125) || accessor.type !== (positions ? 'VEC3' : 'SCALAR') || !Number.isInteger(accessor.count) || accessor.count < 3 || accessor.count > (positions ? 100000 : 450000))
      throw new Error('Unsupported geometry accessor.');
    const buffer = document.bufferViews[index];
    const length = accessor.count * (positions ? 12 : 4);
    if (buffer.buffer !== 0 || buffer.byteStride || !Number.isInteger(buffer.byteOffset) || buffer.byteOffset < 0 || buffer.byteOffset % 4 || buffer.byteLength !== length || buffer.byteOffset + length > document.buffers[0].byteLength)
      throw new Error('3D geometry exceeds its buffer.');
    const start = binaryOffset + 8 + buffer.byteOffset;
    if (positions) {
      totalVertices += accessor.count;
      if (totalVertices > 250000) throw new Error('3D vertex budget exceeded.');
      for (let offset = 0; offset < length; offset += 4) {
        const value = header.getFloat32(start + offset, true);
        if (!Number.isFinite(value) || Math.abs(value) > 100) throw new Error('Invalid vertex coordinate.');
      }
    } else {
      if (accessor.count % 3) throw new Error('Invalid triangle index count.');
      for (let offset = 0; offset < length; offset += 4)
        if (header.getUint32(start + offset, true) >= document.accessors[index - 1].count) throw new Error('Invalid triangle vertex index.');
    }
  }
  const identities = new Set<string>();
  for (const [index, mesh] of document.meshes.entries()) {
    const primitive = mesh.primitives?.[0];
    if (mesh.primitives?.length !== 1 || primitive.indices !== index * 2 + 1 || primitive.attributes?.POSITION !== index * 2 || Object.keys(primitive.attributes).length !== 1 || primitive.targets || (primitive.mode !== undefined && primitive.mode !== 4))
      throw new Error('Unsupported physical piece mesh.');
    const { instanceId, templateId, role } = mesh.extras ?? {};
    if (typeof instanceId !== 'string' || typeof templateId !== 'string' || typeof role !== 'string' || !instanceId || !templateId || identities.has(instanceId))
      throw new Error('Missing or duplicate source piece identity.');
    identities.add(instanceId);
  }
  if (document.extras?.classification !== 'placement-inspection')
    throw new Error('This 3D result classification is not supported by this viewer.');
  return document as { extras: { classification: 'placement-inspection'; patternDigest: string }; meshes: { extras: { instanceId: string; templateId: string; role: string } }[] };
}
