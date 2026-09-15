import { expect, test } from 'bun:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { codexInterpreter } from './codex-interpreter';
import { interpretationFixture } from '../../packages/test-fixtures/interpretation';

async function withStream(events:unknown[],check:(interpreter:ReturnType<typeof codexInterpreter>,captured:()=>any)=>Promise<void>) {
  const directory=await mkdtemp(join(import.meta.dir,'.test-codex-'));
  const path=join(directory,'auth.json');
  await writeFile(path,JSON.stringify({tokens:{access_token:'synthetic-auth-token',account_id:'synthetic-account'}}));
  const original=globalThis.fetch;
  let request:any;
  globalThis.fetch=(async(_url,options)=>{
    request=JSON.parse(String(options?.body));
    return new Response(events.map(event=>'data: '+JSON.stringify(event)+'\n\n').join(''),{headers:{'Content-Type':'text/event-stream'}});
  }) as typeof fetch;
  try {await check(codexInterpreter(path,'test-model'),()=>request);}finally{globalThis.fetch=original;await rm(directory,{recursive:true,force:true});}
}
test('Codex completion uses streamed items when the final output array is empty and sends no tools',async()=>{
  const item={type:'message',content:[{type:'output_text',text:JSON.stringify(interpretationFixture)}]};
  await withStream([{type:'response.output_item.done',item},{type:'response.completed',response:{status:'completed',output:[],usage:{input_tokens:100,output_tokens:200}}}],async(interpreter,captured)=>{
    const result=await interpreter.run({document:{brief:'A top'},images:[],signal:new AbortController().signal});
    expect(result.value).toEqual(interpretationFixture);expect(result.outputTokens).toBe(200);
    expect(captured().tools).toEqual([]);expect(captured().tool_choice).toBe('none');expect(captured().store).toBe(false);
    expect(JSON.stringify(captured())).not.toContain('synthetic-auth-token');
    expect(JSON.stringify(captured().text.format.schema)).not.toContain('oneOf');
  });
});
test('Codex adapter rejects tool attempts, incomplete streams, oversized answers and invalid JSON',async()=>{
  const cases=[
    [{type:'response.output_item.added',item:{type:'function_call',name:'shell'}}],
    [{type:'response.output_text.delta',delta:'unfinished'}],
    [{type:'response.output_text.delta',delta:'x'.repeat(32001)}],
    [{type:'response.completed',response:{status:'completed',output:[{type:'message',content:[{type:'output_text',text:'not json'}]}]}}],
  ];
  for(const events of cases)await withStream(events,async interpreter=>{
    await expect(interpreter.run({document:{brief:'A top'},images:[],signal:new AbortController().signal})).rejects.toThrow();
  });
});
