import { clipboard, nativeImage, ipcMain } from "electron";

export const clipboardHandles = () => {
  ipcMain.handle('readFromClipboard', () => {
    return clipboard.readText('clipboard');
  });
  
  ipcMain.handle('copyToClipboard', async (_, text, type = 'image') => {
    if(type === 'image') {
      clipboard.writeImage(nativeImage.createFromDataURL(text));
    } else {
      clipboard.writeText(text);
    }
  });
}
