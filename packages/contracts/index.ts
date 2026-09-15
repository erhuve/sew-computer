import { z } from 'zod';

export const text = z.string().max(8000);
export const Id = z.string().regex(/^[a-zA-Z0-9_-]{1,100}$/);
export const Unit = z.enum(['mm', 'cm', 'in']);
export const MeasurementSchema = z.discriminatedUnion('state', [
  z.object({ state: z.literal('known'), value: z.number().finite().min(0).max(20000), unit: Unit, source: z.string().min(1).max(300) }).strict(),
  z.object({ state: z.literal('assumed'), value: z.number().finite().min(0).max(20000), unit: Unit, source: z.string().min(1).max(300) }).strict(),
  z.object({ state: z.literal('unknown') }).strict(),
  z.object({ state: z.literal('not-applicable') }).strict(),
]);
export type Measurement = z.infer<typeof MeasurementSchema>;
export function mm(m: Measurement): number | null { return 'value' in m ? m.value * ({mm:1,cm:10,in:25.4}[m.unit]) : null; }
export function convert(m: Measurement, unit: z.infer<typeof Unit>): Measurement { return 'value' in m ? {...m, unit, value: Number((mm(m)! / ({mm:1,cm:10,in:25.4}[unit])).toFixed(8))} : m; }
export function assumed(value: number, unit: 'mm'|'cm'|'in' = 'mm', source = 'Explicit synthetic example; not the wearer’s measurements'): Extract<Measurement,{state:'assumed'}> { return {state:'assumed',value,unit,source}; }
export const RequirementSchema = z.object({id:Id,text, status:z.enum(['unresolved','supported','unsupported']), note:text}).strict();
export const BomSchema = z.object({id:Id,name:z.string().max(240),category:z.enum(['fabric','lining','trim','other']),specification:text,placement:text,quantity:z.string().max(240),source:z.string().max(500)}).strict();
export const PomSchema = z.object({id:Id,name:z.string().max(240),method:text,target:MeasurementSchema,tolerance:MeasurementSchema,size:z.string().max(100),note:text}).strict();
export const ConstructionSchema = z.object({id:Id,operation:text,note:text}).strict();
export const CalloutSchema = z.object({id:Id,anchor:z.string().max(300),text}).strict();
export const ViewSchema = z.object({id:Id,assetId:Id,role:z.enum(['front','back','detail']),kind:z.enum(['reference','technical-flat','sketch']),caption:text}).strict();
export const DocumentSchema = z.object({
  schemaVersion:z.literal(1), title:z.string().min(1).max(160), brief:text, sizeLabel:z.string().max(100),
  garment:z.object({family:z.enum(['none','shirt','skirt','trousers']),length:MeasurementSchema,ease:MeasurementSchema,flare:z.number().finite().min(0).max(3)}).strict(),
  body:z.object({height:MeasurementSchema,bust:MeasurementSchema,waist:MeasurementSchema,hip:MeasurementSchema,shoulder:MeasurementSchema}).strict(),
  requirements:z.array(RequirementSchema).max(80), bom:z.array(BomSchema).max(80), poms:z.array(PomSchema).max(80),
  construction:z.array(ConstructionSchema).max(80),callouts:z.array(CalloutSchema).max(80),views:z.array(ViewSchema).max(20),
  interpretation:z.object({provider:z.string().max(200),model:z.string().max(200),adapter:z.literal('sew-interpretation/1'),proposalId:Id,createdAt:z.string().max(40)}).strict().optional(),
}).strict();
export type GarmentDocument = z.infer<typeof DocumentSchema>;
export function emptyDocument(title='Untitled garment',brief=''):GarmentDocument { return {schemaVersion:1,title,brief,sizeLabel:'Not specified',garment:{family:'none',length:{state:'unknown'},ease:{state:'unknown'},flare:1},body:{height:{state:'unknown'},bust:{state:'unknown'},waist:{state:'unknown'},hip:{state:'unknown'},shoulder:{state:'unknown'}},requirements:[],bom:[],poms:[],construction:[],callouts:[],views:[]}; }
export type Project = {id:string,title:string,createdAt:string,updatedAt:string,headRevisionId:string|null};
export type Draft = {projectId:string,version:number,baseRevisionId:string|null,document:GarmentDocument,updatedAt:string};
export type Revision = {id:string,projectId:string,number:number,parentRevisionId:string|null,document:GarmentDocument,digest:string,createdAt:string};
export type Job = {id:string,projectId:string,revisionId:string,requestId:string,status:'queued'|'running'|'succeeded'|'failed'|'cancelled'|'stale',error:string|null,createdAt:string,updatedAt:string};
export type Artifact = {id:string,projectId:string,revisionId:string,jobId:string|null,kind:'pattern-json'|'pattern-svg'|'pattern-pdf'|'reference',filename:string,mime:string,digest:string,bytes:number,classification:'screen-preview'|'printable-reference',createdAt:string};
export type ReviewComment = {id:string,projectId:string,revisionId:string,anchor:string,text:string,reportedReviewer:string,recordedBy:'owner',createdAt:string};
export type ProjectState = {project:Project,draft:Draft,revisions:Revision[],jobs:Job[],artifacts:Artifact[],comments:ReviewComment[]};
export type PatternPanel = {id:string,name:string,points:[number,number][],widthMm:number,heightMm:number,cutQuantity?:number};
export type PatternGeometry = {schemaVersion:1,units:'mm',inputDigest:string,engineVersion:string,family:string,panels:PatternPanel[],stitches:{panelA:string,edgeA:number,panelB:string,edgeB:number}[],warnings:string[],classification:'printable-reference',assumptions:string[]};
export const DisclosureSchema = z.object({includeBody:z.boolean(),includeReferences:z.boolean(),includePatterns:z.boolean()}).strict();
export type Disclosure = z.infer<typeof DisclosureSchema>;
export type ExportSnapshot = {id:string,projectId:string,revision:Revision,document:GarmentDocument,artifacts:Artifact[],comments:ReviewComment[],disclosure:Disclosure,createdAt:string};
export type ExportFile = {filename:string,mime:string,bytes:Uint8Array};
export type ExportResult = {manifest:Record<string,unknown>,files:ExportFile[]};
export type ImportChange = {path:string,before:unknown,after:unknown,conflict:boolean,baseline?:unknown};
export type ImportPreview = {id:string,projectId:string,baseRevisionId:string|null,baseVersion:number,changes:ImportChange[],warnings:string[]};
export function canonical(value:unknown):string { if (value === null || typeof value !== 'object') return JSON.stringify(value); if(Array.isArray(value)) return '['+value.map(canonical).join(',')+']'; return '{'+Object.keys(value as object).sort().map(k=>JSON.stringify(k)+':'+canonical((value as Record<string,unknown>)[k])).join(',')+'}'; }
export async function digest(value: unknown):Promise<string> { const bytes = new TextEncoder().encode(canonical(value));return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join(''); }
