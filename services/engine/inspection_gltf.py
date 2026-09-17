import json
import struct

from assembly import expand_instance_meshes


def inspection_glb(inspection):
    instances = expand_instance_meshes(inspection)
    buffer = bytearray()
    views = []
    accessors = []
    meshes = []
    nodes = []
    row_x = 0
    row_y = 0
    row_height = 0
    for instance in instances:
        positions = [list(struct.unpack("<3f", struct.pack("<3f", point[0] / 1000, -point[1] / 1000, point[2] / 1000))) for point in instance["restPositions"]]
        indices = [index for face in instance["triangles"] for index in reversed(face)]
        vertex_offset = len(buffer)
        for point in positions:
            buffer.extend(struct.pack("<3f", *point))
        views.append({"buffer": 0, "byteOffset": vertex_offset, "byteLength": len(positions) * 12, "target": 34962})
        minimum = [min(point[axis] for point in positions) for axis in range(3)]
        maximum = [max(point[axis] for point in positions) for axis in range(3)]
        accessors.append({"bufferView": len(views) - 1, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": minimum, "max": maximum})
        position_accessor = len(accessors) - 1
        index_offset = len(buffer)
        for index in indices:
            buffer.extend(struct.pack("<I", index))
        views.append({"buffer": 0, "byteOffset": index_offset, "byteLength": len(indices) * 4, "target": 34963})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5125, "count": len(indices), "type": "SCALAR"})
        meshes.append({"name": instance["instanceId"], "primitives": [{"attributes": {"POSITION": position_accessor}, "indices": len(accessors) - 1, "material": 0}], "extras": {"instanceId": instance["instanceId"], "templateId": instance["templateId"], "role": instance["role"], "sourceVertexOrderPreserved": True}})
        width, height = maximum[0] - minimum[0], maximum[1] - minimum[1]
        if row_x and row_x + width > 3:
            row_x = 0
            row_y -= row_height + 0.1
            row_height = 0
        nodes.append({"mesh": len(meshes) - 1, "name": instance["instanceId"], "translation": [row_x - minimum[0], row_y - maximum[1], 0]})
        row_x += width + 0.1
        row_height = max(row_height, height)
    document = {
        "asset": {"version": "2.0", "generator": "Sew pattern inspection / 1"},
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
        "materials": [{"doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [0.72, 0.79, 0.85, 1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "buffers": [{"byteLength": len(buffer)}], "bufferViews": views, "accessors": accessors,
        "extras": {"classification": "placement-inspection", "patternDigest": inspection["patternDigest"], "constructionDigest": inspection["constructionDigest"], "mesher": inspection["mesher"], "maxEdgeMm": inspection["maxEdgeMm"], "units": "m", "layout": "Rigid flat arrangement; not assembled or simulated", "allowances": "omitted", "unresolvedPhysicalRoles": inspection["unresolvedPhysicalRoles"], "capabilityGaps": inspection["capabilityGaps"], "privacy": "Dimension-revealing pattern derivative; private by default"},
    }
    metadata = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    buffer.extend(b"\0" * (-len(buffer) % 4))
    total = 12 + 8 + len(metadata) + 8 + len(buffer)
    if total > 16 * 1024 * 1024:
        raise ValueError("Display artifact budget exceeded")
    return struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(metadata), 0x4E4F534A) + metadata + struct.pack("<II", len(buffer), 0x004E4942) + buffer
