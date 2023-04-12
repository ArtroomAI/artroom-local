import { ipcMain } from 'electron';
import fs from 'fs';
import { electronDialog } from '../utils/electronDialog';
import { getExifData } from '../utils/exifData';
import { EXTENSION, getExtname, IMAGE_EXTENSIONS, JSON_EXTENSIONS } from '../utils/extensions';

export const settingsLoading = () => {
  ipcMain.handle("uploadSettings", async () => {
    const results = await electronDialog('File', [...IMAGE_EXTENSIONS, ...JSON_EXTENSIONS]);
  
    if (results.length) {
      const file = results[0];
      const content = fs.readFileSync(file);
      const ext = getExtname(file);
  
      return ext === EXTENSION.JSON ? content.toString('utf-8') : getExifData(content, ext);
    }
  
    return "";
  });
}
