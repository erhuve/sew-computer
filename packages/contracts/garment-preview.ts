import { z } from 'zod';
const coordinate=z.number().finite().min(-5).max(5);
const point=z.tuple([coordinate,coordinate]);
const vector=z.tuple([coordinate,coordinate,coordinate]);
const index=z.number().int().nonnegative().max(60000);
const face=z.tuple([index,index,index]);
const digest=z.string().regex(/^[a-f0-9]{64}$/);
const name=z.string().min(1).max(160);
export const GarmentPreviewSchema=z.object({
  profile:z.literal('sew-guided-cloth-preview/2'),classification:z.literal('guided-shape-approximation'),
  units:z.literal('m'),acceptedSimulation:z.literal(false),patternDigest:digest,constructionDigest:digest,
  generatorSha256:digest,assemblyDigest:digest,materialDigest:digest,
  sourceVertices:z.number().int().positive().max(60000),
  pieces:z.array(z.object({
    instanceId:name,templateId:name,role:z.enum(['shell','facing']),mirrorX:z.boolean(),
    restXY:z.array(point).min(3).max(60000),positions:z.array(vector).min(3).max(60000),
    triangles:z.array(face).min(1).max(120000),restTriangles:z.array(face).min(1).max(120000),
    sourceWeights:z.array(z.array(z.object({point:index,weight:z.number().positive().max(1)}).strict()).min(1).max(3)).min(3).max(60000),
    boundaryLoops:z.array(z.array(index).min(2).max(60000)).max(128),
  }).strict()).min(1).max(64),
  buttons:z.array(z.object({position:vector,normal:vector,templateId:name}).strict()).max(1000),
  iterations:z.array(z.object({iteration:z.number().int().positive().max(1000),edgeStrainP95:z.number().finite().nonnegative(),edgeStrainMax:z.number().finite().nonnegative(),seamGapMaxMm:z.number().finite().nonnegative()}).strict()).min(1).max(1000),
}).passthrough();
export type GarmentPreview=z.infer<typeof GarmentPreviewSchema>;
