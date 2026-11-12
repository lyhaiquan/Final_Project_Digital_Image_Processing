import './style.css';
import { api, isMock } from './api.js';
import { escapeHtml as e, normalize, localDate, polygonPoints } from './domain.js';
import { prepareImage } from './image.js';

const label = value => e(String(value ?? '').replace(/[()]/g, ''));

const paths = {
  scan:'M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5M8 12h8',
  book:'M4 4h6a3 3 0 0 1 3 3v14a4 4 0 0 0-4-2H4zm9 3a3 3 0 0 1 3-3h5v15h-4a4 4 0 0 0-4 2',
  grid:'M4 4h6v6H4zm10 0h6v6h-6zM4 14h6v6H4zm10 0h6v6h-6z',
  chart:'M4 20h17M7 16v-5m5 5V5m5 11V8',
  arrow:'M5 12h14m-5-5 5 5-5 5',
  upload:'M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4',
  camera:'M4 7h4l2-3h4l2 3h4v13H4zM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
  check:'m5 12 4 4L19 6', plus:'M12 5v14M5 12h14', close:'m6 6 12 12M6 18 18 6',
  leaf:'M19 4c-13-3-16 15-7 15 8 0 8-10 7-15ZM6 21 16 9',
  search:'M16 10a6 6 0 1 1-12 0 6 6 0 0 1 12 0Zm-1 5 6 6',
  clock:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0ZM12 7v5l3 2',
  trash:'M4 7h16M9 7V4h6v3M6 7l1 14h10l1-14M10 10v7m4-7v7',
  edit:'m4 16 12-12 4 4L8 20H4zm10-10 4 4',
  info:'M12 11v6m0-10v1M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0'
};
const icon = (name) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] ? `<path d="${paths[name]}"/>` : ''}</svg>`;
const colors = ['#d3d9cb','#c8f06a','#efb779','#ad9aee','#f0d78e','#6fcaba'];
const foodColor = (item, i) => item.needs_confirm ? '#efad68' : item.cls === 'rice' ? colors[0] : colors[(i % 5)+1];
const app = document.querySelector('#app');
const modal = document.querySelector('#modal');
let foods = [], benchmark, result = null, imageUrl = '', file, sample = false;
let page = 'scan', busy = false, editing = false, controller, selectedId, removed, saved = false, depth = false;
let showRegions = true;
let diaryDate = localDate(), diary = {meals:[],total:{kcal:0,daily_pct:0}}, pageError = '', query = '', sort = 'name';
let modalAction, modalTarget, toastTimer, navigationVersion = 0;
const draftKey = `calokhay-draft-${isMock?'mock':'api'}`;

function toast(text) {
  const node = document.querySelector('#toast'); node.textContent = text; node.classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove('visible'), 4500);
}
function storeDraft() {
  try {
    if (result) sessionStorage.setItem(draftKey, JSON.stringify({result,sample,saved}));
    else sessionStorage.removeItem(draftKey);
  } catch { toast('Trình duyệt không lưu được bản nháp. Hãy giữ trang này mở.'); }
}
function releaseImage() { if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl); }
function navLink(id, name, glyph) { return `<a href="#${id}" aria-label="${name}" title="${name}" class="nav-link ${page===id?'active':''}" ${page===id?'aria-current="page"':''}>${icon(glyph)}<span>${name}</span>${page===id?'<i></i>':''}</a>`; }
function render() {
  app.innerHTML = `<aside class="sidebar"><a href="#scan" class="brand" aria-label="Calo Khay"><span class="brand-mark">${icon('scan')}</span><span>Calo Khay</span></a>
    <nav aria-label="Điều hướng chính">${navLink('scan','Phân tích ảnh','scan')}${navLink('diary','Nhật ký','book')}${navLink('foods','Bảng calo','grid')}${navLink('benchmark','Độ chính xác','chart')}</nav>
    <button class="sidebar-help" data-action="help">${icon('info')} Hướng dẫn chụp ảnh</button></aside>
    <div class="workspace"><header class="topbar"><span class="breadcrumb">${{scan:'Phân tích ảnh',diary:'Nhật ký',foods:'Bảng calo',benchmark:'Độ chính xác'}[page]}</span>${isMock?'<span class="status">Bản demo</span>':''}</header>
    <main>${pageError?`<div class="error" role="alert">${e(pageError)} <button data-action="retry-page">Thử lại</button></div>`:''}${page==='scan'?scanPage():page==='diary'?diaryPage():page==='foods'?foodsPage():benchmarkPage()}</main>
    </div>`;
  bind();
}
function heading(title, description='', extra='') {
  return `<div class="page-heading"><div><h1>${title}</h1>${description?`<p>${description}</p>`:''}</div>${extra}</div>`;
}
function scanPage() {
  return `${heading('Phân tích bữa ăn', 'Nhận diện món ăn và ước tính calo từ ảnh.', `<button class="secondary help-button" aria-label="Cách chụp ảnh" data-action="help">${icon('info')} Cách chụp ảnh</button>`)}
  <div class="scan-layout"><section class="panel image-panel"><div class="panel-title"><h2>Ảnh bữa ăn</h2>${imageUrl?`<button class="text-button" data-action="upload" ${busy||editing?'disabled':''}>${icon('upload')} Đổi ảnh</button>`:''}</div>
  ${imageUrl ? `<div class="image-stage ${busy?'scanning':''}"><div class="image-canvas"><img src="${e(imageUrl)}" alt="Ảnh khay cơm đã chọn"/>${result&&showRegions?overlay():''}</div>${busy?'<div class="scan-beam"></div><div class="scan-status">Đang phân tích…</div>':''}</div>${result&&(!isMock||sample)?`<div class="viewer-controls"><div class="view-switch" role="group" aria-label="Chế độ xem ảnh"><button data-view="original" aria-pressed="${!showRegions}">Ảnh gốc</button><button data-view="regions" aria-pressed="${showRegions}">Vùng nhận diện</button></div><span>${result.items.filter(i=>i.polygons.length).length} vùng</span></div>`:''}` : `<div class="dropzone" id="dropzone"><span class="upload-icon">${icon('upload')}</span><h3>Thêm ảnh bữa ăn</h3><p>Kéo thả ảnh hoặc chọn từ thiết bị</p><button class="primary" data-action="upload">${icon('plus')} Chọn ảnh</button><button class="text-button mobile-camera" data-action="camera">${icon('camera')} Chụp ảnh</button><small>JPG, PNG, WebP · Tối đa 10 MB</small></div>`}
  <input type="file" id="file-input" accept="image/jpeg,image/png,image/webp" hidden/><input type="file" id="camera-input" accept="image/*" capture="environment" hidden/>
  <div class="image-toolbar"><label class="switch-label"><input type="checkbox" id="depth" ${depth?'checked':''} ${busy?'disabled':''}/><span class="switch"></span>Đo độ sâu</label>${imageUrl?`<button class="${busy||result?'secondary':'primary'}" data-action="${busy?'cancel':'analyze'}" ${editing?'disabled':''}>${busy?'Hủy':result?'Phân tích lại':'Phân tích ảnh'}${!busy?icon('arrow'):''}</button>`:''}</div>
  ${!result?`<button class="sample-strip" data-action="sample" ${busy||editing?'disabled':''}><img src="/mock/sample.jpg" alt=""/><span>Dùng ảnh mẫu</span>${icon('arrow')}</button>`:''}</section>
  <section class="results-column ${result?'has-results':''}" aria-live="polite">${result?resultPanel():`<div class="panel empty-result"><div class="panel-title"><h2>Kết quả phân tích</h2></div><div class="empty-content">${icon('scan')}<h3>Chưa có kết quả</h3><p>Tải ảnh lên để bắt đầu phân tích.</p></div></div>`}</section></div>
  ${isMock?'<p class="demo-note">Dữ liệu minh họa. Chưa kết nối nhận diện ảnh.</p>':''}`;
}
function overlay() {
  if (isMock && !sample) return '';
  return `<svg class="food-overlay" viewBox="0 0 ${result.image.w} ${result.image.h}" aria-label="Vùng món ăn">${result.items.map((item,i) => item.polygons.map((polygon,j) => {
    const x = polygon.reduce((sum,p)=>sum+p[0],0)/polygon.length, y = polygon.reduce((sum,p)=>sum+p[1],0)/polygon.length;
    return `<g class="food-region ${selectedId===item.id?'selected':''}" style="--food-color:${foodColor(item,i)}" tabindex="0" role="button" aria-label="${i+1}. ${label(item.name)}" data-food="${item.id}"><title>${label(item.name)} · ${item.kcal} kcal${item.needs_confirm?' · Cần xác nhận':''}</title><polygon points="${polygonPoints(polygon)}"/>${j===0?`<circle cx="${x}" cy="${y}" r="26"/><text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${i+1}</text>`:''}</g>`;
  }).join('')).join('')}</svg>`;
}
function resultPanel() {
  const {items,total} = result, pending = items.filter(i=>i.needs_confirm), current=pending[0];
  const warningText = {no_food:'Chưa tìm thấy món ăn. Thử ảnh khác hoặc thêm món thủ công.',no_plate:'Chưa thấy rõ đĩa. Khối lượng dùng khẩu phần trung bình.',depth_plane_failed:'Chưa đo được độ sâu. Cơm dùng khẩu phần trung bình.',blurry:'Ảnh hơi mờ. Chụp rõ hơn để cải thiện kết quả.'};
  return `<div class="panel summary"><div class="panel-title"><h2>Tổng calo</h2><span class="small-tag green">${items.length} món</span></div><div class="calorie-total">${total.kcal}<span>kcal</span></div><div class="total-meta"><span>${total.grams} g</span><span>Sai số ±${total.error_pct}% mỗi món</span></div><div class="calorie-bar">${items.map((item,i)=>`<span style="width:${total.kcal?item.kcal/total.kcal*100:0}%;background:${foodColor(item,i)}" title="${label(item.name)}: ${item.kcal} kcal"></span>`).join('')}</div><p class="daily-note"><b>${total.daily_pct}%</b> nhu cầu tham chiếu <span>2.000 kcal / ngày</span></p></div>
  ${(result.warnings||[]).map(w=>`<div class="warning">${e(warningText[w]||w)}</div>`).join('')}
  ${current?`<div class="confirmation"><span class="eyebrow">${pending.length} món cần xác nhận</span><h3>Đây có phải là ${label(current.name)}?</h3><div class="confirmation-actions"><button data-confirm="${current.id}" ${editing?'disabled':''}>${icon('check')} Xác nhận</button>${current.suggestions.slice(0,2).map(cls=>`<button data-suggestion="${e(cls)}" data-id="${current.id}" ${editing?'disabled':''}>${label(foods.find(f=>f.cls===cls)?.name||cls)}</button>`).join('')}<button data-edit="${current.id}" ${editing?'disabled':''}>Món khác</button></div></div>`:''}
  <div class="panel food-list"><div class="panel-title"><h2>Danh sách món</h2><span class="muted">${editing?'Đang cập nhật…':''}</span></div>${items.map((item,i)=>`<div class="food-row ${selectedId===item.id?'highlight':''}" id="food-${item.id}"><span class="food-index" style="--food-color:${foodColor(item,i)}">${i+1}</span><button class="food-name" data-edit="${item.id}" aria-label="Đổi món ${label(item.name)}" ${editing?'disabled':''}>${label(item.name)}<small>${item.needs_confirm?'? Cần xác nhận':item.source==='manual'?'Thêm thủ công':item.source==='renamed'?'Đã chỉnh sửa':''}</small></button><button class="grams" data-grams="${item.id}" aria-label="Chỉnh khối lượng ${label(item.name)}" ${editing?'disabled':''}>${item.grams}<small> g</small></button><strong>${item.kcal}<small> kcal</small></strong><button class="icon-button" data-remove="${item.id}" aria-label="Xóa ${label(item.name)}" ${editing?'disabled':''}>${icon('close')}</button>${item.cls==='rice'?`<div class="rice-portions"><span>Khẩu phần</span>${[150,227,300].map(g=>`<button class="${item.grams===g?'chosen':''}" data-portion="${g}" data-id="${item.id}" aria-pressed="${item.grams===g}" ${editing?'disabled':''}>${g} g</button>`).join('')}</div>`:''}</div>`).join('')}
  <div class="list-actions"><button class="text-button" data-action="add" ${editing?'disabled':''}>${icon('plus')} Thêm món bị sót</button>${removed?'<button class="text-button" data-action="undo">Hoàn tác xóa</button>':''}</div></div>
  <button class="primary save-button" data-action="save" ${editing||busy||!items.length||saved?'disabled':''}>${icon(saved?'check':'book')}${saved?'Đã lưu bữa ăn':'Lưu vào nhật ký'}${icon('arrow')}</button>`;
}
function diaryPage() {
  return `${heading('Nhật ký', '',`<label class="date-picker">${icon('clock')}<input type="date" id="diary-date" value="${diaryDate}" aria-label="Chọn ngày xem nhật ký"/></label>`)}<div class="diary-summary panel"><span>Tổng calo trong ngày<strong>${diary.total.kcal}<small> kcal</small></strong></span><div><b>${diary.meals.length} bữa ăn đã lưu</b><p>${diary.total.daily_pct}% nhu cầu tham chiếu 2.000 kcal</p></div></div>
  ${diary.meals.length?`<div class="meal-grid">${diary.meals.map(meal=>`<article class="panel meal-card"><div class="panel-title"><h2>${icon('clock')} ${new Date(meal.eaten_at).toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'})}</h2><button class="icon-button" data-delete-meal="${e(meal.meal_id)}" aria-label="Xóa bữa ăn">${icon('trash')}</button></div><strong class="meal-kcal">${meal.total.kcal}<small> kcal</small></strong><p>${meal.items.map(i=>label(i.name)).join(' · ')}</p><span class="small-tag">${meal.items.length} MÓN · ${meal.total.grams} G</span>${meal.items.some(i=>i.needs_confirm)?'<p class="pending-note">Có món chưa được xác nhận</p>':''}</article>`).join('')}</div>`:`<div class="panel empty-page">${icon('book')}<h2>Chưa có bữa ăn</h2><p>Các bữa ăn đã lưu sẽ xuất hiện ở đây.</p><a href="#scan" class="primary">Quét khay cơm ${icon('arrow')}</a></div>`}<p class="subtle-note">${isMock?'Lưu trên thiết bị này. Không lưu ảnh.':'Nhật ký trên thiết bị này.'} Sai số ±30% mỗi món.</p>`;
}
function sourceLink(source) {
  if (/^https?:\/\//.test(source)) {
    try {
      const url = new URL(source);
      return `<a href="${e(url.href)}" target="_blank" rel="noopener noreferrer">${e(url.hostname.replace(/^www\./,''))} ↗</a>`;
    } catch { return e(source); }
  }
  return e((source || 'Không tính calo đồ uống').replace(/[()]/g, ''));
}
function foodResults() {
  const filtered = foods.filter(f=>normalize(f.name).includes(normalize(query))).sort((a,b)=>sort==='name'?a.name.localeCompare(b.name,'vi'):sort==='asc'?a.kcal_per_100g-b.kcal_per_100g:b.kcal_per_100g-a.kcal_per_100g);
  return filtered.length?filtered.map((food,i)=>`<tr><td><span class="row-number">${String(i+1).padStart(2,'0')}</span><strong>${label(food.name)}</strong><details class="food-details"><summary>Chi tiết</summary>${food.note?`<p>${label(food.note)}</p>`:''}<p>Nguồn: ${sourceLink(food.source)}</p></details></td><td class="numeric">${Math.round(food.kcal_per_100g)}<span class="table-unit"> kcal</span></td><td class="source-cell">${sourceLink(food.source)}</td></tr>`).join(''):'<tr><td colspan="3" class="no-results">Không tìm thấy món ăn.</td></tr>';
}
function foodsPage() {
  return `${heading('Bảng calo','45 món ăn · Năng lượng trên 100 g')}<div class="table-controls"><label class="search-field">${icon('search')}<input id="food-search" placeholder="Tìm món ăn…" value="${e(query)}" aria-label="Tìm món ăn"/></label><select id="food-sort" aria-label="Sắp xếp món ăn"><option value="name" ${sort==='name'?'selected':''}>Tên món A – Z</option><option value="asc" ${sort==='asc'?'selected':''}>Calo thấp → cao</option><option value="desc" ${sort==='desc'?'selected':''}>Calo cao → thấp</option></select></div><div class="panel table-wrap"><table><thead><tr><th>Món ăn</th><th class="numeric">kcal / 100 g</th><th>Nguồn</th></tr></thead><tbody id="food-table">${foodResults()}</tbody></table></div><p class="subtle-note">Một số giá trị chưa tính dầu và gia vị. Xem chi tiết từng món.</p>`;
}
function benchmarkPage() {
  if (!benchmark) return heading('Độ chính xác','Đang tải…');
  return `${heading('Độ chính xác',benchmark.sample)}<div class="metric-grid">${benchmark.metrics.map(m=>`<div class="panel metric"><span>${label(m.label)}</span><strong>${e(m.value)}</strong><small>${e(m.detail)}</small></div>`).join('')}</div><div class="benchmark-grid"><section class="panel"><div class="panel-title"><h2>So sánh mô hình</h2></div><div class="compare-legend"><span>Chỉ YOLO</span><span>+ DINOv2</span></div>${benchmark.comparisons.map(c=>`<div class="comparison"><strong>${label(c.label)}</strong><small>${e(c.better)}</small><div class="compare-bar"><i style="width:${c.before/Math.max(c.before,c.after)*85}%"></i><span>${c.before}</span></div><div class="compare-bar improved"><i style="width:${c.after/Math.max(c.before,c.after)*85}%"></i><span>${c.after}</span></div></div>`).join('')}</section><section class="panel limitations"><h2>Giới hạn mô hình</h2><p>9% vùng món bị bỏ sót, 20% bị nhận nhầm.</p><p>Sai số ±30% bao phủ 60% số món; ±50% bao phủ khoảng 75%.</p><p>Dữ liệu cân: 38 khay.</p><hr/><h3>Những món dễ nhầm</h3>${Object.entries(benchmark.confusions).slice(0,6).map(([cls,alts])=>`<p class="confusion-pair">${label(foods.find(f=>f.cls===cls)?.name||cls)} <span>↔</span> ${label(foods.find(f=>f.cls===alts[0])?.name||alts[0])}</p>`).join('')}</section></div><section class="panel ap-panel"><div class="panel-title"><h2>AP theo món · model seed 1</h2><span class="muted">Mask AP50–95</span></div><div class="ap-grid">${benchmark.ap.map(a=>`<div><span>${label(foods.find(f=>f.cls===a.cls)?.name||a.cls)}</span><b>${a.value}%</b><progress max="100" value="${a.value}" aria-label="AP ${e(a.cls)}"></progress></div>`).join('')}</div></section><p class="subtle-note">Nguồn: ${e(benchmark.source)}. Dự án học tập; YOLO: AGPL-3.0 · DINOv2: Apache-2.0 · Depth-Anything-V2 Large: CC-BY-NC-4.0.</p>`;
}
function openModal(title, body, action, target) {
  modalAction=action; modalTarget=target;
  modal.setAttribute('aria-labelledby','modal-title');
  modal.innerHTML=`<div class="modal-heading"><h2 id="modal-title">${title}</h2><button class="icon-button" id="close-modal" aria-label="Đóng">${icon('close')}</button></div>${body}`;
  modal.showModal(); modal.querySelector('#close-modal').onclick=()=>modal.close();
  modal.querySelector('form')?.addEventListener('submit', submitModal);
}
function openPicker(id) {
  openModal(id?'Đổi món ăn':'Thêm món bị sót',`<label class="search-field">${icon('search')}<input id="picker-search" placeholder="Tìm trong 45 món ăn…" aria-label="Tìm món"/></label><div id="picker-results" class="picker-results"></div>`, 'pick',id);
  const filter=()=>{
    const value=modal.querySelector('input').value;
    const list=foods.filter(f=>normalize(f.name).includes(normalize(value)) && !result.items.some(item=>item.cls===f.cls && item.id!==id));
    modal.querySelector('#picker-results').innerHTML=list.length?list.map(f=>`<button data-pick="${e(f.cls)}"><span>${label(f.name)}</span><small>${Math.round(f.kcal_per_100g)} kcal / 100 g</small>${icon('plus')}</button>`).join(''):'<p class="no-results">Không có món phù hợp hoặc món đã có trong khay.</p>';
    modal.querySelectorAll('[data-pick]').forEach(button=>button.onclick=async()=>{modal.close();await changeFood(id,button.dataset.pick);});
  };
  modal.querySelector('input').oninput=filter;filter();modal.querySelector('input').focus();
}
async function updateItems(items) {
  if (editing) return false;
  editing=true;render();
  try { const updated=await api.recalc(result.scan_id,items); result={...result,...updated}; saved=false;storeDraft(); return true; }
  catch(error) {toast(error.message);return false;}
  finally {editing=false;render();}
}
async function changeFood(id, cls) {
  const food=foods.find(f=>f.cls===cls);
  if(result.items.some(i=>i.cls===cls&&i.id!==id)) {toast('Món này đã có trong danh sách. Hãy chỉnh khối lượng của món đó.');return;}
  const items=id?result.items.map(item=>item.id===id?{...item,cls,name:food.name,grams:undefined,source:'renamed',needs_confirm:false}:item):[...result.items,{id:Math.max(0,removed?.id||0,...result.items.map(i=>i.id))+1,cls,name:food.name,grams:food.default_g,conf:1,needs_confirm:false,suggestions:[],polygons:[],area_ratio:null,source:'manual',gram_method:'class_mean'}];
  await updateItems(items);
}
async function chooseFile(chosen) {
  if(!chosen||busy||editing)return;
  try {const prepared=await prepareImage(chosen);releaseImage();imageUrl=prepared.url;file=prepared.file;sample=false;result=null;saved=false;removed=null;storeDraft();render();}
  catch(error){toast(error.message);}
}
async function analyze() {
  if(busy||editing||!imageUrl)return;
  busy=true;controller=new AbortController();render();
  try {
    if(sample&&!file) {const response=await fetch('/mock/sample.jpg');file=new File([await response.blob()],'sample.jpg',{type:'image/jpeg'});}
    const next=await api.scan(file,{depth,signal:controller.signal});
    if(controller.signal.aborted)return;
    result=next;saved=false;removed=null;selectedId=null;storeDraft();
    toast(isMock?'Đã tải kết quả mẫu.':'Phân tích hoàn tất.');
  }catch(error){if(error.name!=='AbortError')toast(error.message);}
  finally{busy=false;render();}
}
async function submitModal(event) {
  event.preventDefault(); const form=event.currentTarget, submit=form.querySelector('[type="submit"]');submit.disabled=true;
  try {
    if(modalAction==='grams'){
      const grams=Number(new FormData(form).get('grams'));
      if(!Number.isFinite(grams)||grams<10||grams>1000)throw new Error('Nhập khối lượng từ 10 đến 1.000 g.');
      if(await updateItems(result.items.map(i=>i.id===modalTarget?{...i,grams}:i)))modal.close();
    }
    if(modalAction==='save'){
      const data=new FormData(form);
      await api.saveMeal({scan_id:result.scan_id,items:result.items,eaten_at:new Date(data.get('eaten_at')).toISOString(),share_corrections:data.get('share')==='on'});
      saved=true;storeDraft();modal.close();toast('Đã lưu bữa ăn vào nhật ký.');render();
    }
    if(modalAction==='delete'){
      await api.deleteMeal(modalTarget);modal.close();await loadPage();toast('Đã xóa bữa ăn.');
    }
  }catch(error){const node=modal.querySelector('.form-error');if(node)node.textContent=error.message;else toast(error.message);}
  finally{submit.disabled=false;}
}
function bind() {
  const actions={
    upload:()=>document.querySelector('#file-input').click(),camera:()=>document.querySelector('#camera-input').click(),
    sample:()=>{releaseImage();sample=true;imageUrl='/mock/sample.jpg';file=null;result=null;saved=false;removed=null;storeDraft();analyze();},
    analyze,cancel:()=>controller?.abort(),add:()=>openPicker(),
    undo:async()=>{if(removed&&result.items.some(i=>i.cls===removed.cls)){toast('Món này đã được thêm lại. Hãy chỉnh khối lượng của món hiện tại.');return;}if(removed&&await updateItems([...result.items,removed].sort((a,b)=>a.id-b.id))){removed=null;render();}},
    save:()=>{const now=new Date();openModal('Lưu bữa ăn',`<form><label class="field-label">Thời gian ăn<input type="datetime-local" name="eaten_at" value="${localDate(now)}T${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}" required/></label><label class="consent"><input type="checkbox" name="share" ${isMock?'disabled':''}/><span>Góp ảnh để cải thiện mô hình<small>${isMock?'Chưa thu thập ảnh trong chế độ trải nghiệm.':'Cho phép nhóm lưu ảnh và chỉnh sửa để cải thiện nhận diện. Mặc định tắt.'}</small></span></label><p class="form-error" role="alert"></p><button class="primary" type="submit">${icon('book')} Lưu bữa ăn</button></form>`,'save');},
    help:()=>openModal('Cách chụp ảnh', '<ol class="help-list"><li>Chụp thẳng từ trên xuống.</li><li>Để trọn khay hoặc đĩa trong ảnh.</li><li>Tách các món, tránh chồng lên nhau.</li></ol><p class="help-note">Chọn ảnh đủ sáng, rõ nét và có cạnh ngắn từ 320 px.</p>', 'help'),
    'retry-page':()=>loadPage()
  };
  app.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>actions[button.dataset.action]?.());
  app.querySelectorAll('input[type=file]').forEach(input=>input.onchange=()=>chooseFile(input.files[0]));
  const drop=app.querySelector('#dropzone');
  if(drop){drop.ondragover=event=>{event.preventDefault();drop.classList.add('dragging');};drop.ondragleave=()=>drop.classList.remove('dragging');drop.ondrop=event=>{event.preventDefault();chooseFile(event.dataTransfer.files[0]);};}
  const depthInput=app.querySelector('#depth');if(depthInput)depthInput.onchange=()=>{depth=depthInput.checked;};
  app.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{showRegions=button.dataset.view==='regions';render();app.querySelector(`[data-view="${showRegions?'regions':'original'}"]`)?.focus();});
  app.querySelectorAll('[data-edit]').forEach(button=>button.onclick=()=>openPicker(Number(button.dataset.edit)));
  app.querySelectorAll('[data-grams]').forEach(button=>button.onclick=()=>{const item=result.items.find(i=>i.id===Number(button.dataset.grams));openModal('Chỉnh khối lượng',`<form><p>${label(item.name)}</p><label class="field-label">Khối lượng · g<input type="number" min="10" max="1000" step="1" required name="grams" value="${item.grams}"/></label><p class="form-error" role="alert"></p><button type="submit" class="primary">Cập nhật</button></form>`,'grams',item.id);});
  app.querySelectorAll('[data-remove]').forEach(button=>button.onclick=async()=>{const item=result.items.find(i=>i.id===Number(button.dataset.remove));if(await updateItems(result.items.filter(i=>i.id!==item.id))){removed=item;render();toast('Đã xóa món. Bạn có thể hoàn tác.');}});
  app.querySelectorAll('[data-confirm]').forEach(button=>button.onclick=()=>updateItems(result.items.map(i=>i.id===Number(button.dataset.confirm)?{...i,needs_confirm:false}:i)));
  app.querySelectorAll('[data-suggestion]').forEach(button=>button.onclick=()=>changeFood(Number(button.dataset.id),button.dataset.suggestion));
  app.querySelectorAll('[data-portion]').forEach(button=>button.onclick=()=>updateItems(result.items.map(i=>i.id===Number(button.dataset.id)?{...i,grams:Number(button.dataset.portion)}:i)));
  app.querySelectorAll('[data-food]').forEach(node=>{const select=()=>{selectedId=Number(node.dataset.food);render();const row=document.querySelector(`#food-${selectedId}`);row?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'nearest'});row?.querySelector('.food-name')?.focus({preventScroll:true});};node.onclick=select;node.onkeydown=event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();select();}};});
  const search=app.querySelector('#food-search');if(search)search.oninput=()=>{query=search.value;document.querySelector('#food-table').innerHTML=foodResults();};
  const sortInput=app.querySelector('#food-sort');if(sortInput)sortInput.onchange=()=>{sort=sortInput.value;document.querySelector('#food-table').innerHTML=foodResults();};
  const date=app.querySelector('#diary-date');if(date)date.onchange=()=>{if(date.value){diaryDate=date.value;loadPage();}};
  app.querySelectorAll('[data-delete-meal]').forEach(button=>button.onclick=()=>openModal('Xóa bữa ăn này?',`<form><p>Bữa ăn sẽ bị xóa khỏi nhật ký. Thao tác này không thể hoàn tác.</p><p class="form-error" role="alert"></p><button class="danger" type="submit">Xóa bữa ăn</button></form>`,'delete',button.dataset.deleteMeal));
}
async function loadPage() {
  const version=++navigationVersion;
  page=['scan','diary','foods','benchmark'].includes(location.hash.slice(1))?location.hash.slice(1):'scan';
  pageError='';render();
  try{
    if(!foods.length){const data=await api.classes();foods=Array.isArray(data)?data:data.classes;}
    if(page==='diary'){const data=await api.meals(diaryDate);if(version!==navigationVersion)return;diary=data;}
    if(page==='benchmark'&&!benchmark){const data=await api.benchmark();if(version!==navigationVersion)return;benchmark=data;}
  }catch(error){if(version===navigationVersion)pageError=error.message;}
  if(version===navigationVersion)render();
}
try {const draft=JSON.parse(sessionStorage.getItem(draftKey)||'null');if(draft){result=draft.result;sample=draft.sample;saved=draft.saved;imageUrl=sample?'/mock/sample.jpg':'';}}catch{ /* Storage may be unavailable or the draft may be invalid. */ }
window.addEventListener('hashchange',loadPage);
window.addEventListener('beforeunload',releaseImage);
loadPage();
