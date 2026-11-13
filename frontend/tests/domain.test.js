import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { totals, recalculate, normalize, errors, errorMessage, polygonPoints, escapeHtml } from '../src/domain.js';

const fixture = async name => JSON.parse(await readFile(new URL(`../public/mock/${name}.json`, import.meta.url), 'utf8'));
const [foods, model, scan] = await Promise.all(['classes','gram-model','scan'].map(fixture));

test('fixture has all 45 classes and consistent calorie totals', () => {
  assert.equal(foods.length, 45);
  assert.equal(new Set(foods.map(f=>f.cls)).size, 45);
  assert.deepEqual(totals(scan.items), scan.total);
  assert.deepEqual(recalculate(scan.items,foods,model).map(i=>i.kcal), scan.items.map(i=>i.kcal));
});
test('rice portion edits, deletion and empty trays update all totals', () => {
  const items=recalculate(scan.items.map(i=>i.cls==='rice'?{...i,grams:300}:i),foods,model);
  assert.equal(items[0].kcal,390);
  assert.equal(totals(items).kcal-scan.total.kcal,94);
  const deleted=items.filter(i=>i.cls!=='rice');
  assert.equal(totals(items).kcal-totals(deleted).kcal,390);
  assert.deepEqual(totals([]),{grams:0,kcal:0,error_pct:30,daily_pct:0});
});
test('renaming uses the new class area formula and manual items use default portions', () => {
  const [renamed]=recalculate([{cls:'potato',source:'renamed',area_ratio:.15}],foods,model);
  const expected=Math.exp(model.area.intercept.potato+model.area.slope*Math.log(.15))*model.area.smearing;
  assert.equal(renamed.grams,Math.round(expected));
  assert.equal(renamed.kcal,Math.round(expected*.77));
  const [manual]=recalculate([{cls:'rice',source:'manual'}],foods,model);
  assert.equal(manual.grams,228);
  assert.throws(()=>recalculate([{cls:'missing'}],foods,model));
  assert.throws(()=>recalculate([{cls:'rice',grams:NaN}],foods,model));
});
test('search supports Vietnamese words entered without accents', () => {
  assert.equal(normalize('Cá viên'),'ca vien');
  assert.equal(normalize('đậu phụ rán'),'dau phu ran');
});
test('every documented API error has a human readable message', () => {
  for(const [code,message] of Object.entries(errors)) {
    assert.equal(errorMessage({code},400),message);
    assert.equal(errorMessage({detail:{code}},400),message);
    assert.equal(errorMessage({detail:code},400),message);
  }
  assert.equal(errorMessage({},413),errors.image_too_large);
  assert.equal(errorMessage({},503),errors.busy);
  assert.match(errorMessage({},500),/máy chủ/);
});
test('SVG coordinates remain in source-image space, preserve aspect ratio and reject invalid points', () => {
  const polygon=[[96,128],[480,640],[NaN,1],[10]];
  assert.equal(polygonPoints(polygon),'96,128 480,640');
  // Browser SVG viewBox scales both axes with the same factor as the full-size image.
  for(const width of [320,720,1440]) {
    const scale=width/scan.image.w;
    const height=scan.image.h*scale;
    assert.equal(96*scale/width,.1);
    assert.equal(128*scale/height,.1);
  }
});
test('untrusted names are escaped before insertion into templates', () => {
  assert.equal(escapeHtml('<img src="x" onerror=\'x\'>&'),'&lt;img src=&quot;x&quot; onerror=&#39;x&#39;&gt;&amp;');
});
