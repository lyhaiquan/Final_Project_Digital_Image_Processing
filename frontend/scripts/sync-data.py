"""Refresh frontend fixtures from the versioned model metadata (no ML dependencies)."""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'frontend/public/mock'
OUT.mkdir(parents=True, exist_ok=True)
model = json.loads((ROOT / 'models/gram_model.json').read_text(encoding='utf-8'))
foods = []
with (ROOT / 'models/calories.csv').open(encoding='utf-8', newline='') as f:
    for row in csv.DictReader(f):
        cls = row['class']
        foods.append(dict(cls=cls, name=row['name_vi'], kcal_per_100g=float(row['kcal_per_100g'] or 0),
                          default_g=round(model['class_mean_g'].get(cls, 50)),
                          suggestions=model['confusions'].get(cls, []), source=row['source'], note=row['note']))

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

write('classes.json', foods)
write('gram-model.json', model)
shutil.copyfile(ROOT / 'examples/B_212.jpg', OUT / 'sample.jpg')
# Hand-drawn polygons for an explicitly labelled UI fixture, not model inference.
regions = [
    ('rice', 228, .91, [[275,170],[490,135],[730,320],[725,550],[610,770],[420,775],[270,690],[360,510]]),
    ('potato', 84, .97, [[475,100],[600,135],[695,275],[692,440],[675,590],[617,588],[580,530],[475,548],[440,410],[455,275]]),
    ('pork_broccoli', 91, .68, [[180,260],[315,270],[372,355],[388,540],[455,660],[365,746],[220,715],[120,620],[140,360]]),
    ('fried_tofu', 46, .89, [[722,353],[866,377],[881,431],[800,493],[840,518],[827,597],[712,621],[683,539],[694,437]]),
    ('boiled_egg', 58, .94, [[305,158],[365,114],[429,118],[467,166],[470,230],[422,282],[351,285],[300,237]]),
    ('fish_ball', 55, .61, [[558,609],[665,588],[706,615],[698,653],[621,680],[625,736],[580,768],[496,738],[480,692],[533,656]])
]
items = []
for i, (cls, grams, conf, polygon) in enumerate(regions):
    food = next(f for f in foods if f['cls'] == cls)
    items.append(dict(id=i+1, cls=cls, name=food['name'], conf=conf, needs_confirm=conf < .75,
                      suggestions=food['suggestions'], area_px=0, area_ratio=.12,
                      grams=grams, gram_method='rice_mean' if cls == 'rice' else 'area',
                      kcal=round(grams*food['kcal_per_100g']/100), polygons=[polygon], source='model'))
kcal = sum(i['kcal'] for i in items)
write('scan.json', dict(scan_id='demo-b212', image=dict(w=960,h=1280), plate_found=True,
                       plate_px=566895, rice_volume=None, items=items,
                       total=dict(grams=sum(i['grams'] for i in items),kcal=kcal,error_pct=30,daily_pct=round(kcal/20)),
                       warnings=[], model_version='demo', timing_ms=850))
with (ROOT / 'docs/results/seg_v4_final_per_class_ap_s1.csv').open(encoding='utf-8') as f:
    ap = [dict(cls=r['lop'], value=round(float(r['full_s1'])*100,1)) for r in csv.DictReader(f)]
write('benchmark.json', dict(
    source='README.md mục 6; docs/results/seg_v4_final_per_class_ap_s1.csv',
    sample='36 ảnh kiểm thử · 10 khay có cân',
    metrics=[dict(label='Mask mAP50–95', value='56,6%', detail='± 1,4 điểm % · 3 seed'),
             dict(label='F1 danh sách món', value='84,5%', detail='YOLO + DINOv2'),
             dict(label='Sai số mỗi món', value='24,5 g', detail='MAE · có độ sâu'),
             dict(label='Sai số calo / khay', value='98 kcal', detail='MAE · có độ sâu')],
    comparisons=[dict(label='F1 danh sách món (%)', before=79.5, after=84.5, better='cao hơn tốt hơn'),
                 dict(label='Vùng món bị bỏ sót', before=41, after=33, better='thấp hơn tốt hơn'),
                 dict(label='Sai số gram / món', before=26.8, after=26.5, better='thấp hơn tốt hơn'),
                 dict(label='Sai số kcal / khay', before=122, after=111, better='thấp hơn tốt hơn')],
    ap=ap, confusions=model['confusions']))
print(f'Synced {len(foods)} foods, scan fixture and benchmark to {OUT}')
