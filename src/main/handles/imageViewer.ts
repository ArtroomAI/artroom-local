import { app, ipcMain } from 'electron';
import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { EXTENSION, getExtname, IMAGE_EXTENSIONS } from '../utils/extensions';

export const setupImageViewer = (mainWindow: Electron.BrowserWindow) => {
  let current_folder_path = '';

  let watcher: fs.FSWatcher | null = null;

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

  const reassignWatcher = (folder_path: string) => {
    if(current_folder_path === folder_path) return;

    current_folder_path = folder_path;
    watcher?.close();
    watcher = fs.watch(folder_path, (eventType, filename) => {
      mainWindow.webContents.send('imageViewerChange', readFiles(folder_path));
    })
  }

  ipcMain.handle('getDisks', () => {
    return execSync('wmic logicaldisk get deviceid /value', { encoding: 'utf-8' })
      .split('\r\r\n')
      .filter(Boolean)
      .map(s => s.replace('DeviceID=', ''));
  });
  
  ipcMain.handle('imageViewer', (_, folder_path: string, batch_path: string): ImageViewerResultType => {
    if(!fs.existsSync(folder_path)) {
      if(fs.existsSync(batch_path)) {
        reassignWatcher(batch_path);
        return { error: { error: '', path: batch_path }, results: [] }
      }
      reassignWatcher(app.getPath('home'));
      return { error: { error: '', path: app.getPath('home') }, results: [] }
    }
    
    reassignWatcher(folder_path);
  
    return readFiles(folder_path);
  });  
}
