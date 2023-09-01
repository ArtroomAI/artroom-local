import { ipcMain, app } from 'electron'

export const exposeMenuFunctions = (browserWindow: Electron.BrowserWindow) => {
  ipcMain.handle('minimizeWindow', async () => {
    if (browserWindow.minimizable) {
      browserWindow.minimize()
    }
  })
  ipcMain.handle('maximizeWindow', async () => {
    if (browserWindow.maximizable) {
      browserWindow.maximize()
    }
  })
  ipcMain.handle('unmaximizeWindow', async () => {
    browserWindow.unmaximize()
  })
  ipcMain.handle('maxUnmaxWindow', async () => {
    if (browserWindow.isMaximized()) {
      browserWindow.unmaximize()
    } else {
      browserWindow.maximize()
    }
  })
  ipcMain.handle('closeWindow', async () => {
    browserWindow.close()
  })
  ipcMain.handle('getVersion', async () => {
    return app.getVersion()
  })
}
