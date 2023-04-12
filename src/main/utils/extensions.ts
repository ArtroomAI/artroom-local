import { extname } from 'path';

export enum EXTENSION {
    JPG = 'jpg',
    PNG = 'png',
    JPEG = 'jpeg',

    JSON = 'json',

    CKPT = 'ckpt',
    SAFETENSORS = 'safetensors',
    PT = 'pt',
    PTH = 'pth',
}

export const IMAGE_EXTENSIONS = [EXTENSION.JPG, EXTENSION.PNG, EXTENSION.JPEG];
export const JSON_EXTENSIONS = [EXTENSION.JSON];
export const MODELS_EXTENSIONS = [EXTENSION.CKPT, EXTENSION.SAFETENSORS, EXTENSION.PT, EXTENSION.PTH];

export const getExtname = (name: string) => {
    return extname(name).toLowerCase().replace('.', '');
}
