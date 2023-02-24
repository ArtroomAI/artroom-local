import fs from 'fs';
import path from 'path';

export const setupQueueHandles = (ipcMain: Electron.IpcMain, artroom_path: string) => {
  const queue_path = path.join(artroom_path, 'artroom', 'settings', 'queue.json');

  ipcMain.handle('saveQueue', (_, queue) => {
    fs.writeFileSync(queue_path, queue, 'utf-8');
  });
  
  ipcMain.handle('readQueue', () => {
    return fs.readFileSync(queue_path, 'utf-8');
  });
}
