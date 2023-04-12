import { EXTENSION } from "./extensions";

export const getMimeType = (ext: string) => {
  if (ext === EXTENSION.PNG) {
    return 'image/png';
  }
  if (ext === EXTENSION.JPG || ext === EXTENSION.JPEG) {
    return 'image/jpeg';
  }
  return '';
}
