import { shell, ipcMain } from 'electron'

export const openLinksHandles = () => {
  ipcMain.handle('openDiscord', () => {
    shell.openExternal(`https://discord.gg/XNEmesgTFy`)
  })

  ipcMain.handle('openPatreon', () => {
    shell.openExternal(`https://www.patreon.com/ArtroomAI`)
  })

  ipcMain.handle('openWebsite', () => {
    shell.openExternal(`https://artroom.ai`)
  })

  ipcMain.handle('openCivitai', () => {
    shell.openExternal(`https://civitai.com/`)
  })

  ipcMain.handle('openTutorial', () => {
    shell.openExternal(`https://artroomai.gitbook.io/tutorials/artroom-basics/overview/`)
  })

  ipcMain.handle('openInstallTutorial', () => {
    shell.openExternal(
      `https://artroomai.gitbook.io/local_tutorials/artroom-basics/common-issues-and-solutions/debugging-installation/`
    )
  })
}
