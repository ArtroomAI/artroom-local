import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';

export const setupImageViewer = (app: Electron.App, ipcMain: Electron.IpcMain) => {

  ipcMain.handle('getDisks', () => {
    return execSync('wmic logicaldisk get deviceid /value', { encoding: 'utf-8' })
        .split('\r\r\n')
        .filter(Boolean)
        .map(s => s.replace('DeviceID=', ''));
  });

  ipcMain.handle('imageViewer', (_, folder_path: string, batch_path: string): ImageViewerResultType => {
    if(!fs.existsSync(folder_path)) {
        if(fs.existsSync(batch_path)) {
            return {
                error: {
                    error: '',
                    path: batch_path
                },
                results: []
            }
        } else {
            return {
                error: {
                    error: '',
                    path: app.getPath('home')
                },
                results: []
            }
        }
    }

    const imageExtensions = ['.jpg', '.png', '.jpeg'];
    const results = fs.readdirSync(folder_path, { withFileTypes: true })
        .map(dirent => {
            const name = dirent.name;
            const fullPath = path.join(folder_path, name);
            return {
                name,
                fullPath,
                isFolder: dirent.isDirectory(),
                isImage: dirent.isFile() && imageExtensions.includes(path.extname(name).toLowerCase())
            }
        })
        .filter(dirent => dirent.isFolder || dirent.isImage)
        .sort((a, b) => a.isFolder ? -1 : b.isFolder ? 1 : 0)

    return {
        error: null,
        results
    };
  });

}
