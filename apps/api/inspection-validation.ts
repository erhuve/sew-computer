import { z } from 'zod';
import { canonical, type PatternGeometry } from '../../packages/contracts';
import { InspectionSchema, type Inspection } from '../../packages/contracts/assembly';
import { ApiError, hash } from './validation';

const natural=z.number().int().nonnegative().max(16000000);
const vector=z.tuple([z.number().finite(),z.number().finite(),z.number().finite()]);
function sourceSupportInside(points:[number,number][],support:[number,number][]):boolean {
  const inside=(point:[number,number])=>{
    let contained=false;
    for(let index=0;index<points.length-1;index++) {
      const start=points[index]!,end=points[index+1]!;
      const cross=(point[0]-start[0])*(end[1]-start[1])-(point[1]-start[1])*(end[0]-start[0]);
      if(Math.abs(cross)<1e-7&&point[0]>=Math.min(start[0],end[0])-1e-8&&point[0]<=Math.max(start[0],end[0])+1e-8&&point[1]>=Math.min(start[1],end[1])-1e-8&&point[1]<=Math.max(start[1],end[1])+1e-8)return true;
      if((start[1]>point[1])!==(end[1]>point[1])&&point[0]<(end[0]-start[0])*(point[1]-start[1])/(end[1]-start[1])+start[0])contained=!contained;
    }
    return contained;
  };
  for(let index=0;index<support.length;index++) {
    const start=support[index]!,end=support[(index+1)%support.length]!;
    const horizontal=end[0]-start[0],vertical=end[1]-start[1],length=horizontal*horizontal+vertical*vertical;
    if(length===0)continue;
    const divisions=[0,1];
    for(let boundary=0;boundary<points.length-1;boundary++) {
      const first=points[boundary]!,second=points[boundary+1]!;
      const deltaX=second[0]-first[0],deltaY=second[1]-first[1],denominator=horizontal*deltaY-vertical*deltaX;
      if(Math.abs(denominator)<1e-10) {
        for(const point of [first,second])if(Math.abs((point[0]-start[0])*vertical-(point[1]-start[1])*horizontal)<1e-7)divisions.push(Math.max(0,Math.min(1,((point[0]-start[0])*horizontal+(point[1]-start[1])*vertical)/length)));
      } else {
        const along=((first[0]-start[0])*deltaY-(first[1]-start[1])*deltaX)/denominator;
        const other=((first[0]-start[0])*vertical-(first[1]-start[1])*horizontal)/denominator;
        if(along>0&&along<1&&other>=-1e-8&&other<=1+1e-8)divisions.push(along);
      }
    }
    divisions.sort((first,second)=>first-second);
    for(let part=1;part<divisions.length;part++) {
      const middle=(divisions[part-1]!+divisions[part]!)/2;
      if(!inside([start[0]+horizontal*middle,start[1]+vertical*middle]))return false;
    }
  }
  return true;
}
const gltfSchema=z.object({
  asset:z.object({version:z.literal('2.0'),generator:z.string().max(200)}).strict(),scene:z.literal(0),
  scenes:z.array(z.object({nodes:z.array(natural).max(64)}).strict()).length(1),
  nodes:z.array(z.object({mesh:natural,name:z.string().max(160),translation:vector}).strict()).min(1).max(64),
  meshes:z.array(z.object({name:z.string().max(160),primitives:z.array(z.object({attributes:z.object({POSITION:natural}).strict(),indices:natural,material:z.literal(0)}).strict()).length(1),extras:z.object({instanceId:z.string(),templateId:z.string(),role:z.enum(['shell','facing']),sourceVertexOrderPreserved:z.literal(true)}).strict()}).strict()).min(1).max(64),
  materials:z.array(z.object({doubleSided:z.literal(true),pbrMetallicRoughness:z.object({baseColorFactor:z.tuple([z.number(),z.number(),z.number(),z.literal(1)]),metallicFactor:z.literal(0),roughnessFactor:z.literal(1)}).strict()}).strict()).length(1),
  buffers:z.array(z.object({byteLength:natural}).strict()).length(1),
  bufferViews:z.array(z.object({buffer:z.literal(0),byteOffset:natural,byteLength:natural,target:z.union([z.literal(34962),z.literal(34963)])}).strict()).max(128),
  accessors:z.array(z.object({bufferView:natural,componentType:z.union([z.literal(5126),z.literal(5125)]),count:natural,type:z.enum(['VEC3','SCALAR']),min:vector.optional(),max:vector.optional()}).strict()).max(128),
  extras:z.object({classification:z.literal('placement-inspection'),patternDigest:z.string(),constructionDigest:z.string(),mesher:z.string(),maxEdgeMm:z.number(),units:z.literal('m'),layout:z.string(),allowances:z.literal('omitted'),unresolvedPhysicalRoles:z.array(z.unknown()),capabilityGaps:z.array(z.string()),privacy:z.string()}).strict(),
}).strict();

export function validateInspection(report:Uint8Array,mesh:Uint8Array,pattern:PatternGeometry,patternDigest:string,constructionDigest:string):Inspection {
  if(!(report instanceof Uint8Array)||!(mesh instanceof Uint8Array)||report.length>16*1024*1024||mesh.length>16*1024*1024||mesh.length<32)throw new ApiError(422,'3D artifact budget exceeded');
  const value=InspectionSchema.parse(JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(report)));
  if(value.patternDigest!==patternDigest||value.constructionDigest!==constructionDigest||value.displayArtifact.sha256!==hash(mesh)||value.displayArtifact.bytes!==mesh.length)throw new ApiError(422,'3D source or artifact digest mismatch');
  validateInspectionRest(value,pattern);
  const templates=new Map(value.templates.map(template=>[template.templateId,template]));
  const bytes=Buffer.from(mesh);
  const metadataSize=bytes.readUInt32LE(12),binaryHeader=20+metadataSize;
  if(bytes.readUInt32LE(0)!==0x46546c67||bytes.readUInt32LE(4)!==2||bytes.readUInt32LE(8)!==mesh.length||bytes.readUInt32LE(16)!==0x4e4f534a||metadataSize%4||binaryHeader+8>mesh.length)throw new ApiError(422,'Invalid GLB envelope');
  const binarySize=bytes.readUInt32LE(binaryHeader);
  if(bytes.readUInt32LE(binaryHeader+4)!==0x004e4942||binarySize%4||binaryHeader+8+binarySize!==mesh.length)throw new ApiError(422,'Invalid GLB binary');
  const model=gltfSchema.parse(JSON.parse(bytes.subarray(20,binaryHeader).toString('utf8')));
  if(model.extras.patternDigest!==patternDigest||model.extras.constructionDigest!==constructionDigest||model.extras.mesher!==value.mesher||model.extras.maxEdgeMm!==value.maxEdgeMm||canonical(model.extras.capabilityGaps)!==canonical(value.capabilityGaps)||canonical(model.extras.unresolvedPhysicalRoles)!==canonical(value.unresolvedPhysicalRoles)||model.nodes.length!==value.instances.length||model.meshes.length!==value.instances.length||model.scenes[0]!.nodes.length!==model.nodes.length||model.buffers[0]!.byteLength!==binarySize||model.accessors.length!==value.instances.length*2||model.bufferViews.length!==value.instances.length*2)throw new ApiError(422,'GLB source inventory mismatch');
  const binary=bytes.subarray(binaryHeader+8);
  let binaryCursor=0;
  value.instances.forEach((instance,index)=>{
    const node=model.nodes[index]!,display=model.meshes[index]!,template=templates.get(instance.templateId)!;
    if(model.scenes[0]!.nodes[index]!==index||node.mesh!==index||node.name!==instance.id||display.name!==instance.id||display.extras.instanceId!==instance.id||display.extras.templateId!==instance.templateId||display.extras.role!==instance.role||node.translation.some(number=>Math.abs(number)>100))throw new ApiError(422,'GLB instance mapping mismatch');
    const primitive=display.primitives[0]!;
    if(primitive.attributes.POSITION!==index*2||primitive.indices!==index*2+1)throw new ApiError(422,'GLB accessor inventory mismatch');
    const positions=model.accessors[primitive.attributes.POSITION],indices=model.accessors[primitive.indices];
    if(!positions||!indices||positions.componentType!==5126||positions.type!=='VEC3'||positions.count!==template.restPositions.length||indices.componentType!==5125||indices.type!=='SCALAR'||indices.count!==template.triangles.length*3)throw new ApiError(422,'GLB accessor mismatch');
    const positionView=model.bufferViews[positions.bufferView],indexView=model.bufferViews[indices.bufferView];
    if(!positionView||!indexView||positions.bufferView!==index*2||indices.bufferView!==index*2+1||positionView.byteOffset!==binaryCursor||indexView.byteOffset!==binaryCursor+positionView.byteLength||positionView.target!==34962||indexView.target!==34963||positionView.byteOffset%4||indexView.byteOffset%4||positionView.byteLength!==positions.count*12||indexView.byteLength!==indices.count*4||positionView.byteOffset+positionView.byteLength>binary.length||indexView.byteOffset+indexView.byteLength>binary.length)throw new ApiError(422,'GLB buffer bounds invalid');
    binaryCursor=indexView.byteOffset+indexView.byteLength;
    const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
    template.restPositions.forEach((point,vertexIndex)=>{
      const expected=[(instance.mirrorX?-point[0]:point[0])/1000,-point[1]/1000,0];
      expected.forEach((coordinate,axis)=>{
        const actual=binary.readFloatLE(positionView.byteOffset+vertexIndex*12+axis*4);
        if(!Number.isFinite(actual)||Math.abs(actual-coordinate)>1e-6)throw new ApiError(422,'GLB geometry differs from source mesh');
        minimum[axis]=Math.min(minimum[axis]!,actual);maximum[axis]=Math.max(maximum[axis]!,actual);
      });
    });
    if(canonical(positions.min)!==canonical(minimum)||canonical(positions.max)!==canonical(maximum))throw new ApiError(422,'GLB display bounds mismatch');
    template.triangles.forEach((face,faceIndex)=>{
      const expected=instance.mirrorX?face:[...face].reverse();
      expected.forEach((vertex,offset)=>{if(binary.readUInt32LE(indexView.byteOffset+(faceIndex*3+offset)*4)!==vertex)throw new ApiError(422,'GLB topology differs from source mesh');});
    });
  });
  if(binaryCursor!==binary.length)throw new ApiError(422,'GLB contains unreferenced binary data');
  return value;
}

/** Validate source correspondence independently of any display pose. */
export function validateInspectionRest(value:Pick<Inspection,'instances'|'templates'>,pattern:PatternGeometry):void {
  const panels=new Map(pattern.panels.map(panel=>[panel.id,panel]));
  const templates=new Map(value.templates.map(template=>[template.templateId,template]));
  if(templates.size!==value.templates.length||templates.size!==panels.size||new Set(value.instances.map(instance=>instance.id)).size!==value.instances.length)throw new ApiError(422,'3D physical inventory mismatch');
  let vertices=0,triangles=0;
  for(const [templateId,panel] of panels) {
    const template=templates.get(templateId);
    const instances=value.instances.filter(instance=>instance.templateId===templateId);
    if(!template||instances.length!==panel.cutQuantity||instances.filter(instance=>instance.role==='shell').length!==1||instances.some(instance=>instance.id!==`${templateId}:${instance.role}`))throw new ApiError(422,'3D physical inventory mismatch');
    const mirror=templateId.startsWith('opening_binding_')?templateId.startsWith('opening_binding_right_'):templateId.endsWith('_right');
    if(new Set(instances.map(instance=>instance.role)).size!==instances.length||instances.some(instance=>instance.mirrorX!==mirror||canonical(instance.sourceGrainline)!==canonical(panel.draft?.grainline)))throw new ApiError(422,'3D handedness or grain mismatch');
    if(template.sourceWeights.length!==template.restPositions.length)throw new ApiError(422,'3D source mapping missing');
    const checkedSupport=new Set<string>();
    template.restPositions.forEach((position,index)=>{
      const weights=template.sourceWeights[index]!;
      let sum=0,horizontal=0,vertical=0;
      for(const item of weights) {
        const source=panel.points[item.point];
        if(!source||item.point>=panel.points.length-1)throw new ApiError(422,'3D source mapping invalid');
        sum+=item.weight;horizontal+=source[0]*item.weight;vertical+=source[1]*item.weight;
      }
      if(Math.abs(sum-1)>1e-9||Math.hypot(horizontal-position[0],vertical-position[1])>1e-6)throw new ApiError(422,'3D rest geometry changed');
      const supportKey=weights.map(item=>item.point).sort((first,second)=>first-second).join(':');
      if(new Set(weights.map(item=>item.point)).size!==weights.length)throw new ApiError(422,'3D duplicate source correspondence');
      if(!checkedSupport.has(supportKey)) {
        if(!sourceSupportInside(panel.points,weights.map(item=>panel.points[item.point]!)))throw new ApiError(422,'3D source interpolation crosses outside pattern');
        checkedSupport.add(supportKey);
      }
      if(index<panel.points.length-1&&(weights.length!==1||weights[0]!.point!==index||weights[0]!.weight!==1))throw new ApiError(422,'3D original boundary identity lost');
    });
    if(template.triangles.some(face=>new Set(face).size!==3||face.some(index=>index>=template.restPositions.length)))throw new ApiError(422,'3D triangle indices invalid');
    const edges=new Map<string,{first:number;second:number;count:number}>(),used=new Set<number>();
    let meshArea=0;
    for(const face of template.triangles) {
      const [first,second,third]=face.map(index=>template.restPositions[index]!) as [[number,number],[number,number],[number,number]];
      const area=((second[0]-first[0])*(third[1]-first[1])-(second[1]-first[1])*(third[0]-first[0]))/2;
      if(area<=1e-10)throw new ApiError(422,'3D degenerate or inverted triangle');
      meshArea+=area;
      face.forEach((start,index)=>{
        used.add(start);
        const end=face[(index+1)%3]!,key=[start,end].sort((first,second)=>first-second).join(':');
        const existing=edges.get(key);
        if(existing) {
          if(existing.count!==1||existing.first!==end||existing.second!==start)throw new ApiError(422,'3D overlapping or nonmanifold topology');
          existing.count++;
        } else edges.set(key,{first:start,second:end,count:1});
      });
    }
    const sourceArea=Math.abs(panel.points.slice(0,-1).reduce((total,point,index)=>total+point[0]*panel.points[index+1]![1]-point[1]*panel.points[index+1]![0],0))/2;
    if(used.size!==template.restPositions.length||Math.abs(meshArea-sourceArea)>Math.max(1e-5,sourceArea*1e-9))throw new ApiError(422,'3D surface coverage mismatch');
    const intervals:number[][][]=panel.points.slice(0,-1).map(()=>[]);
    for(const edge of edges.values())if(edge.count===1) {
      const start=template.restPositions[edge.first]!,end=template.restPositions[edge.second]!;
      let matched=false;
      for(let sourceIndex=0;sourceIndex<panel.points.length-1;sourceIndex++) {
        const sourceStart=panel.points[sourceIndex]!,sourceEnd=panel.points[sourceIndex+1]!;
        const horizontal=sourceEnd[0]-sourceStart[0],vertical=sourceEnd[1]-sourceStart[1],lengthSquared=horizontal*horizontal+vertical*vertical;
        const parameter=(point:[number,number])=>((point[0]-sourceStart[0])*horizontal+(point[1]-sourceStart[1])*vertical)/lengthSquared;
        const distance=(point:[number,number])=>Math.abs((point[0]-sourceStart[0])*vertical-(point[1]-sourceStart[1])*horizontal)/Math.sqrt(lengthSquared);
        const from=parameter(start),to=parameter(end);
        if(distance(start)<1e-6&&distance(end)<1e-6&&Math.min(from,to)>=-1e-8&&Math.max(from,to)<=1+1e-8) {
          intervals[sourceIndex]!.push([Math.min(from,to),Math.max(from,to)]);matched=true;break;
        }
      }
      if(!matched)throw new ApiError(422,'3D unexplained mesh boundary');
    }
    for(const segments of intervals) {
      segments.sort((first,second)=>first[0]!-second[0]!);
      let cursor=0;
      for(const segment of segments) {
        if(Math.abs(segment[0]!-cursor)>1e-7)throw new ApiError(422,'3D boundary gaps or overlaps');
        cursor=segment[1]!;
      }
      if(Math.abs(cursor-1)>1e-7)throw new ApiError(422,'3D incomplete source boundary');
    }
    vertices+=template.restPositions.length*instances.length;triangles+=template.triangles.length*instances.length;
  }
  if(vertices>150000||triangles>250000||value.instances.some(instance=>!panels.has(instance.templateId)))throw new ApiError(422,'3D mesh budget exceeded');
}
