import { z } from 'zod';
import { BomSchema, canonical, ConstructionSchema, DocumentSchema, MeasurementSchema, PomSchema, RequirementSchema, type GarmentDocument } from './index';
import { FeatureSchema, ShirtDesignSchema } from './design';

const SuggestedMeasurement=z.union([MeasurementSchema.options[1],MeasurementSchema.options[2],MeasurementSchema.options[3]]);

export const InterpretationSchema = z.object({
  summary: z.string().min(1).max(2000),
  garment: DocumentSchema.shape.garment.extend({length:SuggestedMeasurement,ease:SuggestedMeasurement,design:ShirtDesignSchema.nullable()}),
  requirements: z.array(RequirementSchema.omit({id:true}).extend({feature:FeatureSchema})).min(1).max(30),
  bom: z.array(BomSchema.omit({id:true})).max(20),
  poms: z.array(PomSchema.omit({id:true}).extend({target:SuggestedMeasurement,tolerance:SuggestedMeasurement})).max(20),
  construction: z.array(ConstructionSchema.omit({id:true})).max(20),
  questions: z.array(z.string().min(1).max(1000)).max(20),
}).strict();
export type Interpretation = z.infer<typeof InterpretationSchema>;
export function interpretationJsonSchema():Record<string,unknown> {
  const schema=z.toJSONSchema(InterpretationSchema);
  delete schema.$schema;
  const convert=(value:unknown):unknown=>Array.isArray(value)?value.map(convert):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).map(([key,child])=>[key==='oneOf'?'anyOf':key,convert(child)])):value;
  return convert(schema) as Record<string,unknown>;
}
export type DesignProposal = {
  id:string; projectId:string; baseVersion:number; baseRevisionId:string|null;
  summary:string; questions:string[]; document:GarmentDocument;
  provider:string; model:string; createdAt:string; inputTokens:number|null; outputTokens:number|null;
};
export type InterpretationStatus = {available:boolean;provider:string;model:string;maxOutputTokens:number|null;timeoutSeconds:number;referenceLimit:number};

export function rebaseAcceptedDesign(accepted:GarmentDocument,submitted:GarmentDocument,latest:GarmentDocument):GarmentDocument {
  const next=structuredClone(accepted);
  for(const key of ['title','brief','sizeLabel'] as const)if(latest[key]!==submitted[key])next[key]=latest[key];
  for(const key of ['body','garment'] as const) {
    for(const field of Object.keys(latest[key])) {
      const before=(submitted[key] as Record<string,unknown>)[field],after=(latest[key] as Record<string,unknown>)[field];
      if(canonical(before)!==canonical(after))(next[key] as Record<string,unknown>)[field]=structuredClone(after);
    }
  }
  for(const key of ['requirements','bom','poms','construction','callouts','views'] as const) {
    const oldIds=new Set(submitted[key].map(row=>row.id));
    const localRows=new Map(latest[key].map(row=>[row.id,row]));
    const acceptedIds=new Set(accepted[key].map(row=>row.id));
    const rows=accepted[key].filter(row=>!oldIds.has(row.id)||localRows.has(row.id)).map(row=>{
      const before=submitted[key].find(prior=>prior.id===row.id),local=localRows.get(row.id);
      return local&&canonical(local)!==canonical(before)?structuredClone(local):row;
    });
    rows.push(...latest[key].filter(row=>!acceptedIds.has(row.id)).map(row=>structuredClone(row)));
    (next as Record<string,unknown>)[key]=rows;
  }
  return next;
}
