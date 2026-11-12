import { errors } from './domain.js';

export async function prepareImage(file) {
  if (/\.heic|\.heif/i.test(file.name)) throw new Error('Hãy chuyển ảnh HEIC sang JPG trước khi tải lên.');
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error(errors.invalid_image);
  if (file.size > 10 * 1024 * 1024) throw new Error(errors.image_too_large);
  let bitmap;
  try { bitmap = await createImageBitmap(file, {imageOrientation:'from-image'}); }
  catch { throw new Error(errors.invalid_image); }
  try {
    if (Math.min(bitmap.width, bitmap.height) < 320) throw new Error(errors.image_too_small);
    const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale); canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .9));
    if (!blob) throw new Error(errors.invalid_image);
    return {file: new File([blob], 'meal.jpg', {type:'image/jpeg'}), url: URL.createObjectURL(blob), image:{w:canvas.width,h:canvas.height}};
  } finally { bitmap.close(); }
}
