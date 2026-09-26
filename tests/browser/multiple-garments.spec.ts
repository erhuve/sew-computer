import {test,expect} from './fixture';
import {mkdir} from 'node:fs/promises';
import {resolve} from 'node:path';
const evidence=resolve(import.meta.dirname,'../../docs/verification/multiple-garments');

for(const family of ['shirt','dress','skirt'] as const)test(`${family}: one-click start, edit, automatic preview, download and reopen`,async({studio})=>{
  test.setTimeout(180000);const page=studio.page;
  await studio.login();await mkdir(evidence,{recursive:true});
  if(family==='shirt')await page.screenshot({path:resolve(evidence,'home-desktop.png'),fullPage:true});
  await page.getByRole('button',{name:`Start a ${family}`,exact:true}).click();
  await expect(page.locator('.demo-shape-stage canvas')).toBeVisible({timeout:90000});
  await page.getByText('Size & measurements',{exact:true}).click();
  await page.getByText('Enter or adjust my measurements',{exact:true}).click();
  await expect(page.getByLabel('Hip value',{exact:true})).toBeVisible();
  await expect(page.getByLabel(family==='skirt'?'Waist value':'Bust value',{exact:true})).toBeVisible();
  await expect(page.getByLabel('Height value',{exact:true})).toHaveCount(0);
  await expect(page.getByLabel(family==='skirt'?'Shoulder value':'Waist value',{exact:true})).toHaveCount(0);
  await page.getByText('Size & measurements',{exact:true}).click();
  const project=(await studio.call('GET','/projects')).projects[0],path=`/projects/${project.id}`;
  const original=await studio.call('GET',path),revision=original.project.headRevisionId;
  const first=await studio.call('GET',`${path}/geometry/${revision}`);
  expect(first.family).toBe(family);
  await page.getByRole('button',{name:'Back',exact:true}).click();await page.getByRole('button',{name:'Front',exact:true}).click();
  await page.getByLabel('Length (cm)',{exact:true}).fill(family==='dress'?'111':'72');
  if(family==='dress') {
    // Autosaving a draft must not imply that already-generated downloads changed.
    await expect.poll(async()=>(await studio.call('GET',path)).draft.document.garment.length.value).toBe(1110);
    await page.getByRole('button',{name:'Download',exact:true}).click();
    await expect(page.getByText('These files use the selected saved version. Update the garment to include your latest edits.',{exact:true})).toBeVisible();
    const savedFile=page.getByRole('link',{name:/^pattern-.+\.json$/,exact:true});await expect(savedFile).toBeVisible();
    expect((await studio.call('GET',(await savedFile.getAttribute('href'))!.replace(/^\/api/,''))).inputDigest).toBe(first.inputDigest);
    await page.getByRole('button',{name:'Close dialog',exact:true}).click();
  }
  await page.getByRole('button',{name:'Update garment',exact:true}).click();
  await expect.poll(async()=>(await studio.call('GET',path)).project.headRevisionId).not.toBe(revision);
  const edited=await studio.call('GET',path);
  await expect.poll(async()=>(await studio.call('GET',`${path}/three-d/latest?revisionId=${edited.project.headRevisionId}`))?.status,{timeout:90000}).toBe('succeeded');
  await expect(page.getByRole('button',{name:'Back',exact:true})).toBeEnabled();
  const second=await studio.call('GET',`${path}/geometry/${edited.project.headRevisionId}`);
  expect(second.inputDigest).not.toBe(first.inputDigest);expect(second.panels[0].heightMm).not.toBe(first.panels[0].heightMm);
  await page.getByRole('button',{name:'Download',exact:true}).click();
  const link=page.getByRole('link',{name:/^pattern-.+\.json$/,exact:true});await expect(link).toBeVisible();
  const download=await studio.call('GET',(await link.getAttribute('href'))!.replace(/^\/api/,''));expect(download.inputDigest).toBe(second.inputDigest);expect(download.family).toBe(family);
  await page.getByRole('button',{name:'Close dialog',exact:true}).click();await page.reload();
  // Home remains a useful place to reopen a saved garment without a deep link.
  await page.getByRole('button',{name:new RegExp(project.title)}).click();await expect(page.locator('.demo-shape-stage canvas')).toBeVisible();
  await expect(page.getByRole('button',{name:'Back',exact:true})).toBeEnabled();
  await page.screenshot({path:resolve(evidence,`${family}-desktop.png`),fullPage:true});
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(391);
  await page.screenshot({path:resolve(evidence,`${family}-mobile.png`),fullPage:true});
});

for(const family of ['dress','skirt'] as const)test(`${family}: home description through reviewed model proposal and real generation`,async({studio})=>{
  test.setTimeout(150000);await studio.login();const page=studio.page;
  await page.getByLabel('What are you imagining?',{exact:true}).fill(`complete-${family}-fixture — a relaxed woven ${family}.`);
  await page.getByRole('button',{name:'Design with AI',exact:true}).click();
  await page.getByRole('button',{name:'Use design & preview',exact:true}).click();
  await expect(page.locator('.demo-shape-stage canvas')).toBeVisible({timeout:90000});
  const project=(await studio.call('GET','/projects')).projects[0],state=await studio.call('GET',`/projects/${project.id}`);
  const pattern=await studio.call('GET',`/projects/${project.id}/geometry/${state.project.headRevisionId}`);
  expect(pattern.family).toBe(family);expect(state.draft.document.interpretation.model).toBe('browser-fixture');
});
