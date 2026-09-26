import { z } from 'zod';
const color=z.string().regex(/^#[0-9a-fA-F]{6}$/);
export const FabricPrintSchema=z.object({
  kind:z.enum(['stripes','checks','dots','image']),
  inkColor:color,
  tileMm:z.number().finite().min(5).max(1000),
  rotationDeg:z.number().finite().min(-180).max(180),
  assetId:z.string().regex(/^[a-zA-Z0-9_-]{1,100}$/).nullable(),
}).strict();
export const FabricAppearanceSchema = z.object({color,print:FabricPrintSchema.nullable().optional()}).strict();
export type FabricAppearance = z.infer<typeof FabricAppearanceSchema>;
export type FabricPrint = z.infer<typeof FabricPrintSchema>;
export const defaultFabricColor = '#eeeae1';
