import { ipcMain } from 'electron';
import fs from 'fs';
import path from 'path';

export const setupQueueHandles = () => {
  ipcMain.handle('saveQueue', (_, queue, artroom_path) => {
    const queue_path = path.join(artroom_path, 'artroom', 'settings');
    fs.mkdirSync(queue_path, { recursive: true });
    fs.writeFileSync(path.join(queue_path, 'queue.json'), queue, 'utf-8');
  });
  
  ipcMain.handle('readQueue', (_, artroom_path) => {
    const queue_path = path.join(artroom_path, 'artroom', 'settings');
    fs.mkdirSync(queue_path, { recursive: true });
    return fs.readFileSync(path.join(queue_path, 'queue.json'), 'utf-8');
  });
}
