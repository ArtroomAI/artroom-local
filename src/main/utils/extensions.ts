import { extname } from 'path';

export const IMAGE_EXTENSIONS = ['.jpg', '.png', '.jpeg'];
export const JSON_EXTENSIONS = ['.json'];
export const MODELS_EXTENSIONS = ['.ckpt', '.safetensor', '.pt', '.pth'];

export const getExtname = (name: string) => {
    return extname(name).toLowerCase();
}
