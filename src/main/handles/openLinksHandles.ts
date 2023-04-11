import { shell, ipcMain } from 'electron';

export const openLinksHandles = () => {
  ipcMain.handle("openDiscord", () => {
    shell.openExternal(`https://discord.gg/XNEmesgTFy`);
  });
  
  ipcMain.handle("openCivitai", () => {
    shell.openExternal(`https://civitai.com/`);
  });
  
  ipcMain.handle("openEquilibrium", () => {
    shell.openExternal(`https://equilibriumai.com/`);
  });
}
