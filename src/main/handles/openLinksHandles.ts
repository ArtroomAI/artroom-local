import { shell, ipcMain } from 'electron';

export const openLinksHandles = () => {
  ipcMain.handle("openDiscord", () => {
    shell.openExternal(`https://discord.gg/XNEmesgTFy`);
  });
  
  ipcMain.handle("openWebsite", () => {
    shell.openExternal(`https://artroom.ai`);
  });

  ipcMain.handle("openCivitai", () => {
    shell.openExternal(`https://civitai.com/`);
  });
  
  ipcMain.handle("openTutorial", () => {
    shell.openExternal(`https://artroomai.gitbook.io/tutorials/artroom-basics/overview/`);
  });
}
