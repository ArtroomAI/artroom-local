export const exposeMenuFunctions = (ipcMain, browserWindow, app) => {
    ipcMain.handle('minimizeWindow', async (event) => {
        if (browserWindow.minimizable) {
            browserWindow.minimize();
        }
    });
    ipcMain.handle('maximizeWindow', async (event) => {
        if (browserWindow.maximizable) {
            browserWindow.maximize();
        }
    });
    ipcMain.handle('unmaximizeWindow', async (event) => {
        browserWindow.unmaximize();
    });
    ipcMain.handle('maxUnmaxWindow', async (event) => {
        if (browserWindow.isMaximized()) {
            browserWindow.unmaximize();
        } else {
            browserWindow.maximize();
        }
    });
    ipcMain.handle('closeWindow', async (event) => {
        browserWindow.close();
    });
    ipcMain.handle('getVersion', async (event) => {
        return app.getVersion();
    });
}
