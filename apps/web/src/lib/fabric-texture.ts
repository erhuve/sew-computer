import * as THREE from 'three';
import type {FabricPrint} from '../../../../packages/contracts/appearance';
import {apiResponse} from './api';
export async function fabricTexture(color:string,print:FabricPrint,projectId:string|undefined,signal:AbortSignal) {
  const canvas=document.createElement('canvas');canvas.width=canvas.height=512;
  const context=canvas.getContext('2d')!;context.fillStyle=color;context.fillRect(0,0,512,512);context.fillStyle=print.inkColor;
  if(print.kind==='image') {
    if(!print.assetId||!projectId)return null;
    const response=await apiResponse(`/projects/${projectId}/references/${print.assetId}`,{signal});
    const bitmap=await createImageBitmap(await response.blob());
    if(signal.aborted){bitmap.close();return null;}
    const scale=Math.min(1,1024/Math.max(bitmap.width,bitmap.height));canvas.width=Math.max(1,Math.round(bitmap.width*scale));canvas.height=Math.max(1,Math.round(bitmap.height*scale));context.fillStyle=color;context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(bitmap,0,0,canvas.width,canvas.height);bitmap.close();
  }else if(print.kind==='stripes')context.fillRect(0,0,256,512);
  else if(print.kind==='checks'){context.fillRect(0,0,256,256);context.fillRect(256,256,256,256);}
  else {context.beginPath();context.arc(256,256,95,0,Math.PI*2);context.fill();}
  const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;
  texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.repeat.set(1000/print.tileMm,1000/(print.tileMm*canvas.height/canvas.width));texture.rotation=print.rotationDeg*Math.PI/180;
  texture.anisotropy=4;return texture;
}
