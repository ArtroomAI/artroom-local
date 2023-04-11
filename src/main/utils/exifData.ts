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

export const getExifData = (buffer: Buffer, ext: string): string => {
  if (ext === '.png') {
    return getPNGEXIF(buffer);
  }
  if (ext === '.jpg' || ext === '.jpeg') {
    try {
      const parser = ExifParser.create(buffer);
      const exifData = parser.parse();
      return exifData.tags.UserComment;
    } catch (error) {
      console.log("No exif data found")  
    }
  }

  return '';
}
