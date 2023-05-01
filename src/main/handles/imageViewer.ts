import { app, ipcMain } from 'electron';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { EXTENSION, getExtname, IMAGE_EXTENSIONS } from '../utils/extensions';
import { FileWatcher } from '../utils/fileWatcher';

export const setupImageViewer = (mainWindow: Electron.BrowserWindow) => {
  const readFiles = (folder_path: string): ImageViewerResultType => {
    const results = fs.readdirSync(folder_path, { withFileTypes: true })
      .map(dirent => {
        const name = dirent.name;
        const fullPath = path.join(folder_path, name);
        return {
          name,
          fullPath,
          isFolder: dirent.isDirectory(),
          // Array<EXTENSION>.includes(searchElement: EXTENSION, fromIndex?: number): boolean
          //           ^                                  ^
          // yes this is stupid type conversion, but typescript `includes` is stupid  v
          isImage: dirent.isFile() && IMAGE_EXTENSIONS.includes(getExtname(name) as EXTENSION)
        }
      })
      .filter(dirent => dirent.isFolder || dirent.isImage)
      .sort((a, b) => a.isFolder ? -1 : b.isFolder ? 1 : 0)

    return {
      error: null,
      results
    };
  }

  ipcMain.handle('getDisks', () => {
    return execSync('wmic logicaldisk get deviceid /value', { encoding: 'utf-8' })
      .split('\r\r\n')
      .filter(Boolean)
      .map(s => s.replace('DeviceID=', ''));
  });

  const watcher = new FileWatcher();

  const imageViewerWatcherCallback = (folder_path: string) => () => {
    mainWindow.webContents.send('imageViewerChange', readFiles(folder_path));
  }
  
  ipcMain.handle('imageViewer', (_, folder_path: string, batch_path: string): ImageViewerResultType => {
    if(!fs.existsSync(folder_path)) {
      if(fs.existsSync(batch_path)) {
        watcher.reassignWatcher(batch_path, imageViewerWatcherCallback(batch_path));
        return { error: { error: '', path: batch_path }, results: [] }
      }
      watcher.reassignWatcher(batch_path, imageViewerWatcherCallback(app.getPath('home')));
      return { error: { error: '', path: app.getPath('home') }, results: [] }
    }
    
    watcher.reassignWatcher(batch_path, imageViewerWatcherCallback(folder_path));
  
    return readFiles(folder_path);
  });  
}
