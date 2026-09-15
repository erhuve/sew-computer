import { readFile } from 'node:fs/promises';
import { interpretationJsonSchema } from '../../packages/contracts/interpretation';
import { ApiError } from './validation';
import { interpretationPrompt, type Interpreter } from './interpretation';

export function codexInterpreter(authFile:string,model:string):Interpreter {
  return {
    status:{available:true,provider:'OpenAI · existing Codex login',model,maxOutputTokens:null,timeoutSeconds:120,referenceLimit:3},
    async run(input) {
      let auth:{tokens?:{access_token?:string;account_id?:string}};
      try {auth=JSON.parse(await readFile(authFile,'utf8'));}catch{throw new ApiError(503,'Codex login is unavailable; reconnect Codex in Zo settings');}
      if(!auth.tokens?.access_token)throw new ApiError(503,'Codex login is unavailable; reconnect Codex in Zo settings');
      const content:any[]=[{type:'input_text',text:JSON.stringify(input.document)}];
      for(const image of input.images)content.push({type:'input_image',image_url:image,detail:'low'});
      const schema=interpretationJsonSchema();
      const response=await fetch('https://chatgpt.com/backend-api/codex/responses',{
        method:'POST',redirect:'error',signal:input.signal,
        headers:{Authorization:`Bearer ${auth.tokens.access_token}`,'Content-Type':'application/json',...(auth.tokens.account_id?{'ChatGPT-Account-Id':auth.tokens.account_id}:{})},
        body:JSON.stringify({model,instructions:interpretationPrompt,input:[{role:'user',content}],tools:[],tool_choice:'none',store:false,stream:true,reasoning:{effort:'low'},
          text:{format:{type:'json_schema',name:'garment_proposal',strict:true,schema}}}),
      });
      if(!response.ok) {
        await response.body?.cancel();
        throw new ApiError(502,response.status===401?'Codex login expired; reconnect Codex in Zo settings':`Codex design request failed (HTTP ${response.status}); your draft is unchanged`);
      }
      if(!response.body)throw new ApiError(502,'Codex returned no proposal stream');
      const reader=response.body.getReader(),decoder=new TextDecoder();
      let buffer='',bytes=0,outputBytes=0;
      const completedItems:any[]=[];
      try {
        for(;;) {
          const part=await reader.read();
          if(part.done)break;
          bytes+=part.value.length;
          if(bytes>4*1024*1024)throw new ApiError(502,'Design response exceeded its stream budget');
          buffer+=decoder.decode(part.value,{stream:true});
          let newline:number;
          while((newline=buffer.indexOf('\n'))>=0) {
            const line=buffer.slice(0,newline).trim();buffer=buffer.slice(newline+1);
            if(!line.startsWith('data: ')||line==='data: [DONE]')continue;
            const event=JSON.parse(line.slice(6));
            if(event.type==='response.output_item.added'&&!['message','reasoning'].includes(event.item?.type))throw new ApiError(502,'Unexpected model operation rejected');
            if(event.type==='response.output_item.done')completedItems.push(event.item);
            if(event.type==='response.output_text.delta') {
              outputBytes+=Buffer.byteLength(event.delta??'');
              if(outputBytes>32000)throw new ApiError(502,'Design proposal exceeded its 32 KB output budget');
            }
            if(event.type==='response.failed'||event.type==='error')throw new ApiError(502,'Codex could not complete this design proposal');
            if(event.type==='response.completed') {
              const result=event.response;
              if(result.status!=='completed')throw new ApiError(502,'Codex returned an incomplete proposal');
              const output=result.output?.length?result.output:completedItems;
              if(output.some((item:any)=>!['message','reasoning'].includes(item.type)))throw new ApiError(502,'Unexpected model operation rejected');
              const text=output.filter((item:any)=>item.type==='message').flatMap((item:any)=>item.content??[]).filter((item:any)=>item.type==='output_text').map((item:any)=>item.text).join('');
              if(Buffer.byteLength(text)>32000)throw new ApiError(502,'Design proposal exceeded its 32 KB output budget');
              return {value:JSON.parse(text),inputTokens:result.usage?.input_tokens??null,outputTokens:result.usage?.output_tokens??null};
            }
          }
        }
        throw new ApiError(502,'Codex stream ended before the design was complete');
      } finally {await reader.cancel();reader.releaseLock();}
    },
  };
}
