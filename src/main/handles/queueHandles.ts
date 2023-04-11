import { ipcMain } from 'electron';
import fs from 'fs';
import path from 'path';

export const setupQueueHandles = () => {
  ipcMain.handle('saveQueue', (_, queue, artroom_path) => {
    const queue_path = path.join(artroom_path, 'artroom', 'settings', 'queue.json');
    fs.writeFileSync(queue_path, queue, 'utf-8');
  });
  
  ipcMain.handle('readQueue', (_, artroom_path) => {
    const queue_path = path.join(artroom_path, 'artroom', 'settings', 'queue.json');
    return fs.readFileSync(queue_path, 'utf-8');
  });
}
