import { app, BrowserWindow, ipcMain, dialog, MessageBoxOptions, session } from 'electron'
import { autoUpdater } from 'electron-updater'
autoUpdater.autoDownload = false
import path from 'path'
import isDev from 'electron-is-dev'

require('dotenv').config()
import { setupHandles } from './ipcMainHandles'
import { Server } from './handles/serverHandles'

let win: BrowserWindow

const server = new Server()

function createWindow() {
  // Create the browser window.
  win = new BrowserWindow({
    width: 1350,
    height: 1000,
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false,
      devTools: isDev,
    },
  })

  setupHandles(win)
  server.serverHandles(ipcMain)

  win.setTitle('ArtroomAI v' + app.getVersion())

  if (!isDev) {
    win.removeMenu()
  }

  win.loadURL(
    isDev ? 'http://localhost:3000' : `file://${path.join(__dirname, '../build/index.html')}`
  )

  //TODO: In frontend, map out a button that lets users do an auto update and update the "don't remind me" or "check for auto update", check version number and press "check for updates or update" in the app itself
  // When an update is available, show a message to the user
  autoUpdater.on('update-available', () => {
    const message = 'An update is available. Would you like to download it now?'
    // const buttons = ['Install', 'Cancel', 'Don\'t ask again'];
    const buttons = ['Download', 'Cancel']

    // Show the message to the user
    dialog.showMessageBox({ message, buttons }).then(({ response }) => {
      // If the user clicks the "Install" button, install the update
      if (response === 0) {
        autoUpdater.downloadUpdate()
        alert('Artroom is downloading the latest update in the background.')
      }
      // if (response === 2) {
      //   //Set don't ask again to be true in state
      //   //Save state of don't ask again later in memory
      // }
    })
  })

  autoUpdater.on('update-downloaded', (event) => {
    const dialogOpts: MessageBoxOptions = {
      type: 'info',
      buttons: ['Restart', 'Later'],
      title: 'Application Update',
      message: event.releaseName,
      detail: 'A new version has been downloaded. Restart the application to apply the updates.',
    }
    dialog.showMessageBox(dialogOpts).then((returnValue) => {
      if (returnValue.response === 0) {
        if (process.platform !== 'darwin') {
          server.kill()
        }
        autoUpdater.quitAndInstall()
      }
    })
  })

  autoUpdater.on('download-progress', (info) => {
    win.webContents.send('downloadProgress', info)
  })

  win.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify()
    if (isDev) {
      console.log(path.resolve('scripts/fmkadmapgofadopljbjfkapdkoienihi'))
      session.defaultSession
        .loadExtension(path.resolve('scripts/fmkadmapgofadopljbjfkapdkoienihi'))
        .then(({ name }) => console.log(`Added Extension:  ${name}`))
        .catch((err) => console.log('An error occurred: ', err))
    }
  })
}

app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    server.kill()
    app.quit()
  }
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})
