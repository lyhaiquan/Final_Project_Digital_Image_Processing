export const normalize = (value) => value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').toLowerCase();
export const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const localDate = (date = new Date()) => `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`;
export function totals(items) {
  const grams = items.reduce((sum, item) => sum + item.grams, 0);
  const kcal = items.reduce((sum, item) => sum + item.kcal, 0);
  return { grams: Math.round(grams), kcal: Math.round(kcal), error_pct: 30, daily_pct: Math.round(kcal / 20) };
}
export function recalculate(items, foods, model) {
  return items.map(item => {
    const food = foods.find(food => food.cls === item.cls);
    if (!food) throw new Error('Món ăn không có trong danh mục.');
    let grams = item.grams;
    if (grams == null) {
      grams = item.source === 'renamed' && item.cls !== 'rice' && item.area_ratio > 0
        ? Math.exp((model.area.intercept[item.cls] ?? model.area.intercept._other) + model.area.slope * Math.log(item.area_ratio)) * model.area.smearing
        : food.default_g;
    }
    if (!Number.isFinite(grams) || grams < 0 || grams > 1000) throw new Error('Khối lượng không hợp lệ.');
    return {...item, name: food.name, grams: Math.round(grams), kcal: Math.round(grams * food.kcal_per_100g / 100)};
  });
}
export function polygonPoints(polygon) {
  return polygon.filter(p => p.length === 2 && p.every(Number.isFinite)).map(p => p.join(',')).join(' ');
}
export const errors = {
  invalid_image: 'Ảnh không hợp lệ. Hãy chọn JPG, PNG hoặc WebP.',
  image_too_small: 'Ảnh quá nhỏ. Cạnh ngắn cần ít nhất 320 px.',
  image_too_large: 'Ảnh quá lớn. Dung lượng tối đa là 10 MB.',
  scan_expired: 'Kết quả đã hết hạn. Hãy quét lại ảnh.',
  unknown_class: 'Món ăn không có trong danh mục. Hãy chọn lại.',
  busy: 'Máy đang bận. Hãy thử lại sau vài giây.',
  model_loading: 'Mô hình đang khởi động. Hãy thử lại sau vài giây.'
};
export function errorMessage(body, status) {
  const code = body?.code ?? body?.detail?.code ?? body?.detail;
  return errors[code] ?? (status === 413 ? errors.image_too_large : status === 503 ? errors.busy : 'Không thể kết nối máy chủ. Vui lòng thử lại.');
}
