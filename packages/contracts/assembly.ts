import { z } from 'zod';

const digest = z.string().regex(/^[a-f0-9]{64}$/);
const name = z.string().min(1).max(160);
const point = z.tuple([z.number().finite(), z.number().finite()]);
export const InspectionSchema = z.object({
  schemaVersion: z.literal(1), classification: z.literal('placement-inspection'), units: z.literal('mm'),
  patternDigest: digest, constructionDigest: digest, compiler: name, mesher: name,
  maxEdgeMm: z.number().min(5).max(100),
  instances: z.array(z.object({ id: name, templateId: name, role: z.enum(['shell','facing']), mirrorX: z.boolean(), sourceGrainline: z.tuple([point,point]) }).strict()).min(1).max(64),
  templates: z.array(z.object({ templateId: name, restPositions: z.array(point).min(3).max(100000), triangles: z.array(z.tuple([z.number().int().nonnegative(),z.number().int().nonnegative(),z.number().int().nonnegative()])).min(1).max(150000), sourceWeights:z.array(z.array(z.object({point:z.number().int().nonnegative(),weight:z.number().positive().max(1)}).strict()).min(1).max(3)).min(3).max(100000) }).passthrough()).min(1).max(32),
  unresolvedPhysicalRoles: z.array(z.object({templateId:name,role:name,reason:z.string().max(2000)}).strict()).max(64),
  capabilityGaps: z.array(z.string().max(2000)).max(100),
  validation:z.array(z.object({templateId:name}).passthrough()).max(32),
  displayArtifact:z.object({filename:z.literal('inspection.glb'),sha256:digest,bytes:z.number().int().positive().max(16*1024*1024)}).strict(),
}).passthrough();
export type Inspection = z.infer<typeof InspectionSchema>;
export type ThreeDJob = {
  id:string; projectId:string; revisionId:string; requestId:string; patternDigest:string; inputDigest:string;
  status:'queued'|'running'|'succeeded'|'failed'|'cancelled'|'stale'; error:string|null; createdAt:string; updatedAt:string;
  result?:{classification:'placement-inspection';fabricInstances:number;unresolvedPhysicalRoles:number;capabilityGaps:string[]};
  sourceCurrent?:boolean;
};
