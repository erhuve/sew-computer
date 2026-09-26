import {test,expect} from './fixture';
import {mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {shirtDocument} from '../../packages/test-fixtures/shirt';

test('extreme seam allowances stay inside the pattern canvas without overlapping pieces',async({studio})=>{
  await studio.login();
  const document=shirtDocument();
  document.title='Extreme allowance shirt';
  document.garment.flare=1.5;
  document.garment.length={state:'assumed',value:400,unit:'mm',source:'Synthetic canvas regression'};
  document.body.hip={state:'assumed',value:1600,unit:'mm',source:'Synthetic canvas regression'};
  document.garment.design!.seamAllowanceMm=20;
  const created=await studio.call('POST','/projects',{title:document.title,brief:document.brief});
  const path=`/projects/${created.project.id}`;
  const draft=await studio.call('PUT',`${path}/draft`,{expectedVersion:created.draft.version,expectedRevisionId:created.draft.baseRevisionId,document});
  const published=await studio.call('POST',`${path}/revisions`,{expectedVersion:draft.version,expectedRevisionId:draft.baseRevisionId});
  const job=await studio.call('POST',`${path}/jobs`,{revisionId:published.project.headRevisionId,requestId:'extreme-canvas'});
  await expect.poll(async()=>{
    const state=await studio.call('GET',path);
    return state.jobs.find((item:{id:string})=>item.id===job.id)?.status;
  },{timeout:45000}).toBe('succeeded');
  await studio.page.reload();
  await studio.page.getByRole('button',{name:/Extreme allowance shirt/}).click();await studio.page.getByRole('tab',{name:'Pattern',exact:true}).click();
  const canvas=studio.page.locator('.pattern-stage svg[role="img"]');
  await expect(canvas).toBeVisible();
  const layout=await canvas.evaluate(element=>{
    const svg=element as SVGSVGElement;
    const rootInverse=svg.getCTM()!.inverse();
    const pieces=[...svg.querySelectorAll<SVGGElement>('g[role="button"]')].map(group=>{
      const points=[...group.querySelectorAll<SVGGraphicsElement>('polygon, polyline, circle')].flatMap(shape=>{
        const bounds=shape.getBBox();
        const transform=rootInverse.multiply(shape.getCTM()!);
        return [[bounds.x,bounds.y],[bounds.x+bounds.width,bounds.y],[bounds.x,bounds.y+bounds.height],[bounds.x+bounds.width,bounds.y+bounds.height]].map(([horizontal,vertical])=>new DOMPoint(horizontal,vertical).matrixTransform(transform));
      });
      return {name:group.getAttribute('aria-label'),left:Math.min(...points.map(point=>point.x)),right:Math.max(...points.map(point=>point.x)),top:Math.min(...points.map(point=>point.y)),bottom:Math.max(...points.map(point=>point.y))};
    });
    return {width:svg.viewBox.baseVal.width,height:svg.viewBox.baseVal.height,pieces};
  });
  expect(layout.pieces).toHaveLength(18);
  for(const [index,piece] of layout.pieces.entries()) {
    expect(piece.left,piece.name!).toBeGreaterThan(0);
    expect(piece.top,piece.name!).toBeGreaterThan(0);
    expect(piece.right,piece.name!).toBeLessThan(layout.width);
    expect(piece.bottom,piece.name!).toBeLessThan(layout.height);
    for(const other of layout.pieces.slice(index+1)) {
      const separate=piece.right<other.left||other.right<piece.left||piece.bottom<other.top||other.bottom<piece.top;
      expect(separate,`${piece.name} must not overlap ${other.name}`).toBe(true);
    }
  }
});

test('brief to reviewed component design, real pieces and revision-specific export',async({studio})=>{
  await studio.login();
  const page=studio.page;
  await page.getByRole('button',{name:'New garment',exact:true}).first().click();
  await page.getByLabel('Garment name',{exact:true}).fill('Component shirt');
  await page.getByLabel('Your idea',{exact:true}).fill('complete-shirt-fixture: White button-up with practical tails and tasteful frills.');
  await page.getByRole('button',{name:'Create garment',exact:false}).click();
  await page.getByLabel('Send these inputs to the design model').check();
  await page.getByRole('button',{name:'Interpret my design',exact:true}).click();
  await expect(page.getByRole('img',{name:'front construction schematic'})).toBeVisible();
  await page.getByLabel('Preview with sample M estimates. Keeps measurements you’ve entered.').uncheck();await page.getByRole('button',{name:'Use design & set size',exact:true}).click();
  await page.getByRole('button',{name:/Apply sample/}).click();
  await page.getByRole('button',{name:/Generate garment|Update garment/,exact:true}).click();
  await page.getByRole('tab',{name:'Pattern',exact:true}).click();
  await expect(page.locator('.pattern-stage svg[role="img"]')).toBeVisible({timeout:45000});
  await expect(page.locator('.pattern-stage g[role="button"]')).toHaveCount(18);
  const projects=await studio.call('GET','/projects');
  const project=projects.projects[0];
  const state=await studio.call('GET',`/projects/${project.id}`);
  const generated=await studio.call('GET',`/projects/${project.id}/geometry/${state.project.headRevisionId}`);expect(generated.drafting.components).toContain('frills');
  const exported=await studio.call('POST',`/projects/${project.id}/exports`,{revisionId:state.project.headRevisionId,disclosure:{includeBody:false,includeReferences:false,includePatterns:true}});
  expect(exported.files.length).toBe(8);
  expect(exported.files.some((file:{filename:string})=>file.filename.startsWith('pattern-a4-tiled-'))).toBe(true);
  expect(exported.files.some((file:{filename:string})=>file.filename.startsWith('pattern-letter-tiled-'))).toBe(true);
  const evidence=resolve(import.meta.dirname,'../../docs/verification/garment-pipeline');
  await mkdir(evidence,{recursive:true});
  await page.screenshot({path:resolve(evidence,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect(page.locator('.pattern-stage')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await page.screenshot({path:resolve(evidence,'mobile.png'),fullPage:true});
});

test('generation waits for in-flight autosave instead of silently dropping the click',async({studio})=>{
  await studio.login();
  const page=studio.page;
  await page.getByRole('button',{name:'New garment',exact:true}).first().click();
  await page.getByLabel('Garment name',{exact:true}).fill('Autosave shirt');
  await page.getByLabel('Your idea',{exact:true}).fill('Simple top');
  await page.getByRole('button',{name:'Create garment',exact:false}).click();
  await page.getByRole('button',{name:'Customize',exact:true}).click();
  await page.getByRole('combobox',{name:'Geometry family',exact:true}).selectOption('shirt');
  await page.getByRole('button',{name:/Apply sample/}).click();
  let release!:()=>void;
  const gate=new Promise<void>(resolve=>{release=resolve;});
  let observed!:()=>void;
  const started=new Promise<void>(resolve=>{observed=resolve;});
  await page.route('**/api/projects/*/draft',async route=>{
    if(route.request().method()==='PUT') {observed();await gate;}
    await route.continue();
  });
  await started;
  await page.getByRole('button',{name:/Generate garment|Update garment/,exact:true}).click();
  release();
  await page.getByRole('tab',{name:'Pattern',exact:true}).click();
  await expect(page.locator('.pattern-stage')).toBeVisible({timeout:45000});
});
