import { errorMessage, localDate, recalculate, totals } from './domain.js';

export const isMock = import.meta.env.VITE_USE_MOCK !== 'false';
const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');
const readMock = async name => {
  const response = await fetch(`${import.meta.env.BASE_URL}mock/${name}.json`);
  if (!response.ok) throw new Error('Không tải được dữ liệu mẫu.');
  return response.json();
};
let deviceId = crypto.randomUUID();
try {
  deviceId = localStorage.getItem('calokhay-device') || deviceId;
  localStorage.setItem('calokhay-device', deviceId);
} catch { /* Private browsers may deny persistence; the current session still works. */ }
async function request(path, { signal, ...options } = {}) {
  try {
    const response = await fetch(`${base}${path}`, {
      ...options,
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
      headers: { 'X-Device-Id': deviceId, ...(options.body instanceof FormData ? {} : {'Content-Type': 'application/json'}), ...options.headers }
    });
    if (!response.ok) throw new Error(errorMessage(await response.json().catch(() => ({})), response.status));
    return response.status === 204 ? null : response.json();
  } catch (error) {
    if (error.name === 'TimeoutError') throw new Error('Quá 15 giây chưa có phản hồi. Hãy thử lại.');
    if (error instanceof TypeError) throw new Error('Không thể kết nối máy chủ. Bản nháp vẫn được giữ lại.');
    throw error;
  }
}
const post = (path, body) => request(path, { method: 'POST', body: JSON.stringify(body) });
function readMeals() { return JSON.parse(localStorage.getItem('calokhay-meals') || '[]'); }
export const api = {
  classes: () => isMock ? readMock('classes') : request('/classes'),
  benchmark: () => isMock ? readMock('benchmark') : request('/benchmark'),
  async scan(file, {depth, signal}) {
    if (isMock) {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, 1000);
        signal.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Đã hủy', 'AbortError')); }, {once:true});
      });
      return {...await readMock('scan'), scan_id: crypto.randomUUID()};
    }
    const body = new FormData(); body.append('image', file);
    return request(`/scan?depth=${depth}`, {method:'POST', body, signal});
  },
  async recalc(scanId, items) {
    if (!isMock) return post('/recalc', {scan_id: scanId, items});
    const [foods, model] = await Promise.all([readMock('classes'), readMock('gram-model')]);
    const next = recalculate(items, foods, model);
    return {items: next, total: totals(next)};
  },
  async saveMeal(meal) {
    if (!isMock) return post('/meals', meal);
    const entry = {...meal, meal_id: crypto.randomUUID(), total: totals(meal.items)};
    // Store nutrition only. Uploaded images are never persisted in mock mode.
    localStorage.setItem('calokhay-meals', JSON.stringify([...readMeals(), entry]));
    return {meal_id: entry.meal_id};
  },
  async meals(date) {
    if (!isMock) return request(`/meals?date=${encodeURIComponent(date)}`);
    const meals = readMeals().filter(meal => localDate(new Date(meal.eaten_at)) === date);
    return {meals, total: totals(meals.flatMap(meal => meal.items))};
  },
  async deleteMeal(id) {
    if (!isMock) return request(`/meals/${encodeURIComponent(id)}`, {method:'DELETE'});
    localStorage.setItem('calokhay-meals', JSON.stringify(readMeals().filter(meal => meal.meal_id !== id)));
  }
};
