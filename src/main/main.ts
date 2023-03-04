import { app, BrowserWindow, ipcMain, clipboard, shell, dialog, nativeImage, OpenDialogOptions, MessageBoxOptions, session } from 'electron';
import { autoUpdater } from "electron-updater";
autoUpdater.autoDownload = false;
import os from 'os';
import path from 'path';
import isDev from 'electron-is-dev';
import fs from "fs";
import glob from 'glob';
import axios from 'axios';
import kill from 'tree-kill';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
const ExifParser = require('exif-parser');

require("dotenv").config();
import { exposeMenuFunctions } from './menu-functions';
import { handlers } from './ipcHandles';
import { setupQueueHandles } from './ipcFileHandles';

let win: BrowserWindow;
let hd = os.homedir();
let LOCAL_URL = process.env.LOCAL_URL;
//Start with cleanup
axios.get(`${LOCAL_URL}/shutdown`)

//read installation log to find artroom path
// const artroom_install_log = userDataPath + "/artroom_install.log";
const artroom_install_log = hd + "\\AppData\\Local\\artroom_install.log";
let artroom_path = hd;
if (fs.existsSync(artroom_install_log)) {
  // Do something
  let temp = fs.readFileSync(artroom_install_log, 'utf-8');
  let lines = temp.split(/\r?\n/);
  artroom_path = lines[0];
}

const getPNGEXIF = (png: Buffer) => {
  const png_string = png.toString('utf-8');

  const start = png_string.indexOf(`{"text_prompts"`);

  if(start === -1) return '';

  const end = png_string.indexOf('}', start);

  const exif = png_string.substring(start, end + 1);
  return exif;
}

async function getImage(image_path: string) {
  return fs.promises.readFile(image_path).then(buffer => {
    let userComment = '';

    const ext = path.extname(image_path).toLowerCase();
    let mimeType;
    if (ext === '.png') {
      mimeType = 'image/png';
      userComment = getPNGEXIF(buffer);
    } else if (ext === '.jpg' || ext === '.jpeg') {
      mimeType = 'image/jpeg';
      try {
        const parser = ExifParser.create(buffer);
        const exifData = parser.parse();
        userComment = exifData.tags.UserComment;
      } catch (error) {
        console.log("No exif data found")  
      }
    } else {
      mimeType = '';
    }
    const b64 = `data:${mimeType};base64,${buffer.toString('base64')}`;

    return [b64, userComment] as [string, any];
  });
}

//pyTestCmd = "\"" + artroom_path +"\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python pytest.py";
//pyTestCmd = "\"" + artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python" + "\"";
const pyTestCmd = artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe";

const serverCommand = `"${artroom_path}\\artroom\\miniconda3\\Scripts\\conda" run --no-capture-output -p "${artroom_path}/artroom/miniconda3/envs/artroom-ldm" python server.py`;

const mergeModelsCommand = `"${artroom_path}\\artroom\\miniconda3\\Scripts\\conda" run --no-capture-output -p "${artroom_path}/artroom/miniconda3/envs/artroom-ldm" python model_merger.py`;

let server: ChildProcessWithoutNullStreams;

const getFiles = (folder_path: string, ext: string) => {
  return new Promise((resolve, reject) => {
    glob(`${folder_path}/**/*.{${ext}}`, {}, (err, files) => {
      if (err) {
        console.log("ERROR");
        resolve([]);
      }
      resolve(files?.map((match) => path.relative(folder_path, match)) ?? []);
    })
  });
}

console.log("Artroom Log: " + artroom_install_log);

function createWindow() {
  server = spawn(serverCommand, { detached: true, shell: true });

  ipcMain.handle('saveFromDataURL', async (event, data) => {
    const json = JSON.parse(data);
    const dataUrl = json.dataURL;
    const imagePath = json.imagePath;

    let imagePathDirName = path.dirname(imagePath);

    fs.mkdirSync(imagePathDirName, { recursive: true });

    // convert dataURL to a buffer
    try{
      const buffer = Buffer.from(dataUrl.split(',')[1], 'base64');
      try {
        // write the buffer to the specified image path
        fs.writeFileSync(imagePath, buffer);
      } catch (err) {
        console.error(err);
        return;
      }
    } catch (err) {
      axios.get(dataUrl, {
        responseType: "arraybuffer",
      }).then((raw) => {
        // create a base64 encoded string
        try {
          // write the buffer to the specified image path
          const buffer = Buffer.from(raw.data, "binary").toString("base64");
          //fs.writeFileSync(imagePath, buffer);
          fs.writeFile(imagePath, buffer, 'base64', function(err) {
            console.log(err);
          });
        } catch (err) {
          console.error(err);
          return;
        }
      });
    }
  });

  ipcMain.handle('getCkpts', async (event, data) => {
    return getFiles(data, 'ckpt,safetensors');
  });

  ipcMain.handle('getLoras', async (event, data) => {
    return getFiles(path.join(data, 'Loras'), 'ckpt,safetensors');
  });

  ipcMain.handle('getVaes', async (event, data) => {
    return getFiles(path.join(data, 'Vaes'), 'vae.pt,vae.ckpt,vae.safetensors');
  });

  ipcMain.handle('getImages', async (event, data) => {
    return getFiles(data, 'jpg,png,jpeg');
  });

  ipcMain.handle('showInExplorer', async (event, data) => {
    const p = path.resolve(data);
    shell.showItemInFolder(p);
  });

  ipcMain.handle('getImageFromPath', async (event, data) => {
    return new Promise((resolve, reject) => {
      if (data && data.length > 0) {
        getImage(data).then(([b64, metadata]) => {
          resolve({b64, metadata});
        }).catch(err => {
          console.log(err);
          reject(err);
        });
      } else {
        resolve("");
      }
    });
  });

  ipcMain.handle('copyToClipboard', async (event, b64) => {
    clipboard.writeImage(nativeImage.createFromDataURL(b64));
  });

  ipcMain.handle("reinstallArtroom", () => {
    return new Promise((resolve, reject) => {
      console.log("Reinstalling Artroom...");
      // Define the path to the external .exe file
      const exePath = 'py_cuda_install_debug.exe';

      // Spawn a new child process to run the .exe file
      const exeProcess = spawn('runas', ['/user:Administrator', exePath], {
        detached: true
      });

      // Listen for the 'close' event, which is emitted when the child process finishes
      exeProcess.on('close', (code: string) => {
        console.log(`Process exited with code ${code}`);
        resolve('Success');
      });
    });
  });

  ipcMain.handle("getSettings", () => {
    return new Promise((resolve, reject) => {
      fs.readFile(artroom_path + "\\artroom\\settings\\sd_settings.json", "utf8", function (err, data) {
        if (err) {
          console.log("Error!")
          reject(err);
          return;
        }
        resolve(data);
      });
    });
  });

  ipcMain.handle("uploadSettings", () => {
    return new Promise((resolve, reject) => {
      let properties: OpenDialogOptions['properties'];
      if (os.platform() === 'linux' || os.platform() === 'win32') {
        properties = ['openFile', 'multiSelections'];
      }
      else {
        properties = ['openFile', 'openDirectory', 'multiSelections'];
      }
      dialog.showOpenDialog({
        properties: properties,
        filters: [
          { name: 'Settings', extensions: ['json'] },
        ]
      }).then(result => {
        if (result.filePaths.length > 0) {
          fs.readFile(result.filePaths[0], "utf8", function (err, data) {
            if (err) {
              reject(err);
              return;
            }
            resolve(data);
          });
        }
        else {
          resolve("");
        }
      }).catch(err => {
        resolve("");
      })
    });
  });

  ipcMain.handle("chooseImages", async () => {
    let properties: OpenDialogOptions['properties'];
    if (os.platform() === 'linux' || os.platform() === 'win32') {
      properties = ['openFile', 'multiSelections'];
    } else {
      properties = ['openFile', 'openDirectory', 'multiSelections'];
    }

    const results = await dialog.showOpenDialog({
      properties: properties,
      filters: [
        { name: 'Images', extensions: ['jpg', 'png', 'jpeg'] },
      ]
    });

    return results.filePaths;
  });

  //Opens file explorer
  ipcMain.handle("getImageDir", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      fs.readFile(artroom_path + "\\artroom\\settings\\sd_settings.json", "utf8", function (err, data) {
        if (err) {
          reject(err);
          return;
        }
        const json = JSON.parse(data);
        let imgPath = path.join(json['image_save_path'], json['batch_name']);
        if (fs.existsSync(imgPath.split(path.sep).join(path.posix.sep))) {
          shell.openPath(imgPath.split(path.sep).join(path.posix.sep))
        }
        else {
          imgPath = JSON.parse(data)['image_save_path'];
          imgPath = imgPath.replace('%UserProfile%', artroom_path);
          shell.openPath(imgPath.split(path.sep).join(path.posix.sep))
        }
        resolve([data]);
      });
    });
  });

  ipcMain.handle("openDiscord", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      shell.openExternal(`https://discord.gg/XNEmesgTFy`);
    });
  });

  ipcMain.handle("openEquilibrium", (event) => {
    return new Promise((resolve, reject) => {
      shell.openExternal(`https://equilibriumai.com/`);
    });
  });


  ipcMain.handle('chooseUploadPath', (event) => {
    return new Promise((resolve, reject) => {
      dialog.showOpenDialog({
        properties: ['openDirectory'],
      }).then(result => {
        if (result.filePaths.length > 0) {
          resolve(result.filePaths[0]);
        }
        else {
          resolve("");
        }
      }).catch(err => {
        resolve("");
      })

    });
  });

  //startup test logic
  function runPyTests() {
    return new Promise((resolve, reject) => {
      let childPython = spawn(pyTestCmd, ['pytest.py']);
      var result = '';
      childPython.stdout.on(`data`, (data) => {
        result += data.toString();
      });

      childPython.on('close', function (code) {
        resolve(result)
      });
      childPython.on('error', function (err) {
        reject(err)
      });

    })
  };

  async function runTest() {
    try {
      const res = await runPyTests();
      return res
    } catch (err) {
      return err
    }
  }

  ipcMain.handle('runPyTests', () => {
    let python_path = artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe";
    if (!(fs.existsSync(artroom_install_log))) {
      return "cannot find artroom_install_log in " + artroom_install_log;
    } else if (!(fs.existsSync(python_path))) {
      return "cannot find python in " + python_path;
    } else {
      let res = runTest();
      return res;
      //return "success";
    }
  });

  ipcMain.handle('restartServer', async (event, isDebug) => {
    return new Promise(() => {
      console.log(`debug mode: ${isDebug}`)
      kill(server.pid);
      spawn("taskkill", ["/pid", `${server.pid}`, '/f', '/t']);
      server = spawn(serverCommand, { detached: isDebug, shell: true });
    });
  });

  ipcMain.handle('mergeModels',(event, data) => new Promise((resolve, reject) => {
        const json = JSON.parse(data);
        const parameters = [
                json.modelA,
                json.modelB  
        ];
        if (json.modelC.length > 0) {
            parameters.push(
                '--model_2',
                  json.modelC
            );
            parameters.push(
              '--alphaRange',
                json.alphaRange
          );
        }
        if (json.alpha) {
            parameters.push(
                '--alpha',
                json.alpha
            );
        }
        if (json.method) {
            parameters.push(
                '--method',
                json.method
            );
        }
        if (json.steps > 0) {
            parameters.push(
                '--steps',
                json.steps
            );
        }
        if (json.start_steps > 0) {
          parameters.push(
              '--start_steps',
              json.start_steps
          );
        }
        if (json.end_steps > 0) {
            parameters.push(
                '--end_steps',
                json.end_steps
            );
        }
        if (json.filename.length > 0) {
            parameters.push(
                '--output',
                json.filename
            );
        }
        const modelMergeServer = spawn(
            mergeModelsCommand,
            parameters,
            {
                detached: true,
                shell: true
            }
        );
        modelMergeServer.on(
            'message',
            (code, signal) => {
                console.log(`mergeModels message ${code} ${signal}`);
            }
        );
        modelMergeServer.on(
            'close',
            (code, signal) => {
                console.log(`mergeModels closed ${code} ${signal}`);
                resolve(code);
            }
        );
    })
  );

  // Create the browser window.
  win = new BrowserWindow({
    width: 1350,
    height: 1000,
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: true,
      contextIsolation: process.env.NODE_ENV === 'production',
      webSecurity: false,
      devTools: isDev
    }
  })
  
  exposeMenuFunctions(ipcMain, win, app);
  setupQueueHandles(ipcMain, artroom_path);
  handlers(win);

  win.setTitle("ArtroomAI v" + app.getVersion());
  
  if(!isDev) {
    win.removeMenu()
  }

  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  )

  

  //TODO: In frontend, map out a button that lets users do an auto update and update the "don't remind me" or "check for auto update", check version number and press "check for updates or update" in the app itself
  // When an update is available, show a message to the user
  autoUpdater.on('update-available', () => {
    const message = 'An update is available. Would you like to download it now?';
    // const buttons = ['Install', 'Cancel', 'Don\'t ask again'];
    const buttons = ['Download', 'Cancel'];

    // Show the message to the user
    dialog.showMessageBox({ message, buttons }).then(({ response }) => {
      // If the user clicks the "Install" button, install the update
      if (response === 0) {
        autoUpdater.downloadUpdate();
        alert('Artroom is downloading the latest update in the background.')
      }
      // if (response === 2) {
      //   //Set don't ask again to be true in state
      //   //Save state of don't ask again later in memory
      // }
    });
  });

  autoUpdater.on("update-downloaded", (event) => {
    const dialogOpts: MessageBoxOptions = {
      type: 'info',
      buttons: ['Restart', 'Later'],
      title: 'Application Update',
      message: event.releaseName,
      detail: 'A new version has been downloaded. Restart the application to apply the updates.'
    };
    dialog.showMessageBox(dialogOpts).then((returnValue) => {
      if (returnValue.response === 0) {
        if (process.platform !== 'darwin') {
          kill(server.pid);
          spawn("taskkill", ["/pid", `${server.pid}`, '/f', '/t']);
          axios.get(`${LOCAL_URL}/shutdown`)
        }  
        autoUpdater.quitAndInstall();
      }
    })
  });

  autoUpdater.on('download-progress', (info) => {
    win.webContents.send('downloadProgress', info);
  });

  win.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify();
    if(isDev) {
      console.log(path.resolve('scripts/fmkadmapgofadopljbjfkapdkoienihi'))
      session.defaultSession.loadExtension(path.resolve('scripts/fmkadmapgofadopljbjfkapdkoienihi'))
        .then(({ name }) => console.log(`Added Extension:  ${name}`))
        .catch((err) => console.log('An error occurred: ', err));
    }
  });
}

app.on('ready', createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    kill(server.pid);
    spawn("taskkill", ["/pid", `${server.pid}`, '/f', '/t']);
    axios.get(`${LOCAL_URL}/shutdown`)
    app.quit()
  }
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})