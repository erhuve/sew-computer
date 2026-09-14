import {test as base,expect,type Page} from '@playwright/test';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {mkdtemp,rm} from 'node:fs/promises';
import {resolve} from 'node:path';

const root=resolve(import.meta.dirname,'../..');
const key='browser-tests-only-not-a-production-credential';
type Studio={page:Page;login:()=>Promise<void>;call:(method:string,path:string,body?:unknown)=>Promise<any>};
export const test=base.extend<{studio:Studio}>({studio:async({page},use)=>{
  const directory=await mkdtemp(resolve(root,'.planning/browser-'));
  const child=spawn('bun',[resolve(root,'tests/browser/server.ts')],{cwd:root,env:{...process.env,NODE_ENV:'production',SEW_DATA_DIR:directory,SEW_ACCESS_KEY:key},stdio:['ignore','pipe','pipe']});
  let diagnostics='';
  child.stderr.on('data',chunk=>{diagnostics+=chunk;});
  const exited=new Promise<void>(resolve=>child.once('exit',()=>resolve()));
  try {
    const origin=await new Promise<string>((resolve,reject)=>{
      const timer=setTimeout(()=>reject(new Error(`Browser server startup timeout: ${diagnostics.slice(-2000)}`)),15000);
      createInterface({input:child.stdout}).on('line',line=>{if(/^http:\/\/127\.0\.0\.1:\d+$/.test(line)){clearTimeout(timer);resolve(line);}});
      child.once('error',error=>{clearTimeout(timer);reject(error);});
      child.once('exit',code=>{clearTimeout(timer);reject(new Error(`Browser server exited ${code}: ${diagnostics.slice(-2000)}`));});
    });
    const login=async()=>{
      await page.goto(origin);
      await page.getByLabel('Owner access key',{exact:true}).fill(key);
      await page.getByRole('button',{name:'Open studio'}).click();
      await expect(page.getByRole('heading',{name:'What will you make?'})).toBeVisible();
    };
    const call=async(method:string,path:string,body?:unknown)=>page.evaluate(async args=>{
      const response=await fetch('/api'+args.path,{method:args.method,headers:{Accept:'application/json','X-Sew-Session':sessionStorage.getItem('sew-session')??'',...(args.body===undefined?{}:{'Content-Type':'application/json'})},body:args.body===undefined?undefined:JSON.stringify(args.body)});
      const data=response.status===204?null:await response.json();
      if(!response.ok)throw new Error(`${response.status} ${JSON.stringify(data)}`);
      return data;
    },{method,path,body});
    await use({page,login,call});
  } finally {
    await page.unrouteAll({behavior:'ignoreErrors'});
    child.kill('SIGTERM');
    await exited;
    await rm(directory,{recursive:true,force:true});
  }
}});
export {expect};
