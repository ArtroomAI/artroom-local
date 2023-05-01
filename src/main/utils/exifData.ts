import { exifValidator } from "../../interfaces/generated/exifDataValidator";
import { EXTENSION } from "./extensions";

import path from 'path';

const ExifParser = require('exif-parser');

const getPNGEXIF = (png: Buffer) => {
  const png_string = png.toString('utf-8');

  const start = png_string.indexOf(`{"text`);

  if(start === -1) return '';

  let count = 1;

  for(let i = start + 5; i < png_string.length; ++i) {
    if(png_string[i] === '{') {
      count++;
    } else if (png_string[i] === '}') {
      count--;
    }

    if(count === 0) {
      return png_string.substring(start, i + 1);
    }
  }
  return '';
}

const parseExifData = (buffer: string | Buffer, ext: string = ''): Partial<ExifDataType> => {
  try {
    if(typeof buffer === 'string') {
      return JSON.parse(buffer);
    }
    if (ext === EXTENSION.PNG) {
      return JSON.parse(getPNGEXIF(buffer));
    }
    if (ext === EXTENSION.JSON) {
      return JSON.parse(buffer.toString('utf-8'))
    }
    if (ext === EXTENSION.JPG || ext === EXTENSION.JPEG) {
      const parser = ExifParser.create(buffer);
      const exifData = parser.parse();
      return JSON.parse(exifData.tags.UserComment);
    }
  } catch {
    return {};
  }

  return {};
}

export const getExifData = (buffer: string | Buffer, ext?: string) => {
  const exif = parseExifData(buffer, ext);

  exif.width = parseWidth(exif);
  exif.height = parseHeigth(exif);
  exif.loras = parseLoras(exif.loras);
  exif.controlnet = exif.controlnet ?? 'none';
  exif.clip_skip = exif.clip_skip ?? 1;

  return exifValidator(exif);
}

// @DEPRECATE: LEGACY LORAS DON'T HAVE NAME ONLY PATH, IT WILL BE REMOVED 
const parseLoras = (loras: Lora[] = []): Lora[] => {
  return loras.map(el => {
    el.name = el.name ?? path.basename((el as any).path);
    return el;
  });
}
// @DEPRECATE: CHANGE 'W' INTO 'WIDTH' AND 'H' INTO 'HEIGHT'
const parseWidth = (exif: Partial<ExifDataType>) => {
  if('W' in exif) {
    return exif.W as number;
  } else if ('width' in exif) {
    return exif.width as number;
  }
  return NaN;
}
// @DEPRECATE: CHANGE 'W' INTO 'WIDTH' AND 'H' INTO 'HEIGHT'
const parseHeigth = (exif: Partial<ExifDataType>) => {
  if('H' in exif) {
    return exif.H as number;
  } else if ('height' in exif) {
    return exif.height as number;
  }
  return NaN;
}
