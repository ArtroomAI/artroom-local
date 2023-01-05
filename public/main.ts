import { app, BrowserWindow, ipcMain, clipboard, shell, dialog, nativeImage, OpenDialogOptions, MessageBoxOptions } from 'electron';
import { autoUpdater } from "electron-updater";
import { io } from 'socket.io-client';
autoUpdater.autoDownload = false;
import os from 'os';
import path from 'path';
import isDev from 'electron-is-dev';
import fs from "fs";
import glob from 'glob';
import axios from 'axios';
import kill from 'tree-kill';
import { spawn } from 'child_process';
const ExifParser = require('exif-parser');

require("dotenv").config();
import { exposeMenuFunctions } from './menu-functions';

// const getUserDataPath = () => {
//   const platform = os.platform();
//   if (platform === 'win32') {
//     return path.join(process.env.APPDATA, appName);
//   } else if (platform === 'darwin') {
//     return path.join(process.env.HOME, 'Library', 'Application Support', appName);
//   } else {
//     return path.join('var', 'local', appName);
//   }
// }

const imageGenerationSettings = "sd_settings.json"
const imageSettings = "image_settings.json"
const applicationSettings = "app_settings.json"

// const userDataPath = getUserDataPath();

let win;
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

//replace placeholders with actual paths
let sd_data;
let sd_data_json;
try {
  sd_data = fs.readFileSync(artroom_path + "\\artroom\\settings\\sd_settings.json", 'utf-8');
}
catch (e) {
  sd_data = fs.readFileSync("sd_settings.json", 'utf-8');
}
sd_data_json = JSON.parse(sd_data);
sd_data_json['image_save_path'] = sd_data_json['image_save_path'].replace('%UserProfile%', hd);
sd_data_json['image_save_path'] = path.normalize(sd_data_json['image_save_path']);
sd_data_json['ckpt'] = sd_data_json['ckpt'].replace('%InstallPath%', artroom_path);
sd_data_json['ckpt'] = path.normalize(sd_data_json['ckpt']);
sd_data_json['ckpt_dir'] = sd_data_json['ckpt_dir'].replace('%InstallPath%', artroom_path);
sd_data_json['ckpt_dir'] = path.normalize(sd_data_json['ckpt_dir']);
let debugModeInit = sd_data_json['debug_mode'] as boolean;
sd_data = JSON.stringify(sd_data_json, null, 2);
fs.writeFileSync(artroom_path + "\\artroom\\settings\\sd_settings.json", sd_data);

async function getB64(data: string) {
  return new Promise((resolve, reject) => {
    fs.promises.readFile(data, 'base64').then((result: string) => {
      const ext = path.extname(data).toLowerCase();
      let mimeType;
      if (ext === '.png') {
        mimeType = 'image/png';
      } else if (ext === '.jpg' || ext === '.jpeg') {
        mimeType = 'image/jpeg';
      } else {
        mimeType = '';
      }
      resolve(`data:${mimeType};base64,${result}`);
    }).catch((err: any) => {
      console.log(err);
      resolve("");
    });
  });
}

async function getMetadata(path: string) {
  return new Promise((resolve, reject) => {
    fs.promises.readFile(path).then(imageBuffer => {
      const parser = ExifParser.create(imageBuffer);
      const exifData = parser.parse();
      const userComment = exifData.tags.UserComment;
      resolve(userComment);
    }).catch(err => {
      console.log(err);
      resolve('');
    });
  });
}


//pyTestCmd = "\"" + artroom_path +"\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python pytest.py";
//pyTestCmd = "\"" + artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python" + "\"";
let pyTestCmd = artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe";

const serverCommand = "\"" + artroom_path + "\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run --no-capture-output -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python server.py";


const mergeModelsCommand = "\"" + artroom_path + "\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run --no-capture-output -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python model_merger.py";

let server = spawn(serverCommand, { detached: debugModeInit, shell: true })

function createWindow() {
  //Connect to server
  axios.get(`${LOCAL_URL}/start`);
  console.log("Artroom Log: " + artroom_install_log);

  const socket = io('http://localhost:5300');
  
  socket.on('connect', function() {
    console.log('Connected to Socket.IO server');
  });
  
  socket.on('message', function(message) {
    console.log('Received message: ' + message);
  });
  
  // Send a message to the server
  socket.emit('message', 'Hello, server!');

  ipcMain.handle('login', async(event,data) =>{
    return new Promise((resolve, reject) => {
      //axios.post('login')
      //store jwt token in safe storage
      //store url
    });
  })


  ipcMain.handle('getCkpts', async (event, data) => {
    return new Promise((resolve, reject) => {
      glob(`${data}/**/*.{ckpt,safetensors}`, {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        }
        files = files?.map(function (match) {
          return path.relative(data, match);
        });
        resolve(files);
      })
    });
  });

  ipcMain.handle('getVaes', async (event, data) => {
    return new Promise((resolve, reject) => {
      glob(`${data}/**/*.{.vae.pt}`, {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        }
        files = files?.map(function (match) {
          return path.relative(data, match);
        });
        resolve(files);
      })
    });
  });

  ipcMain.handle('getImages', async (event, data) => {
    return new Promise((resolve, reject) => {
      glob(`${data}/**/*.{jpg,png,jpeg}`, {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        }
        files = files?.map(function (match) {
          return path.relative(data, match);
        });
        resolve(files);
      })
    });
  });

  ipcMain.handle('getImageFromPath', async (event, data) => {
    return new Promise((resolve, reject) => {
      if (data && data.length > 0) {
        Promise.all([getB64(data), getMetadata(data)]).then(([b64, metadata]) => {
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
    return new Promise((resolve, reject) => {
      clipboard.writeImage(nativeImage.createFromDataURL(b64));
    });
  });

  ipcMain.handle("reinstallArtroom", (event, argx = 0) => {
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

  ipcMain.handle("getSettings", (event, argx = 0) => {
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

  ipcMain.handle("uploadSettings", (event, argx = 0) => {
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

  ipcMain.handle("chooseImages", (event, argx = 0) => {
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
          { name: 'Images', extensions: ['jpg', 'png', 'jpeg'] },
        ]
      }).then(result => {
        if (result.filePaths.length > 0) {
          resolve(result.filePaths);
        }
        else {
          resolve("");
        }
      }).catch(err => {
        resolve("");
      })
    });
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
        let imgPath = path.resolve(json['image_save_path'], json['batch_name']);
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

  ipcMain.handle('runPyTests', (event) => {
    let sd_path = artroom_path + "\\artroom\\settings\\sd_settings.json";
    let python_path = artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe";
    if (!(fs.existsSync(artroom_install_log))) {
      return "cannot find artroom_install_log in " + artroom_install_log;
    } else if (!(fs.existsSync(sd_path))) {
      return "cannot find sd_settings.json in " + sd_path;
    } else if (!(fs.existsSync(python_path))) {
      return "cannot find python in " + python_path;
    } else {
      let res = runTest();
      return res;
      //return "success";
    }
  });

  ipcMain.handle('restartServer', async (event, isDebug) => {
    return new Promise((resolve, reject) => {
      kill(server.pid);
      spawn("taskkill", ["/pid", `${server.pid}`, '/f', '/t']);
      if (isDebug) {
        server = spawn(serverCommand, { detached: true, shell: true });
      } else {
        server = spawn(serverCommand, { detached: false, shell: true });
      }
      return axios.get(`${LOCAL_URL}/get_progress`,
        { headers: { 'Content-Type': 'application/json' } }).then((result) => {
          resolve(result.status);
        });
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

  win.setTitle("ArtroomAI v" + app.getVersion());
  
  if(isDev) {
    win.webContents.openDevTools();
  } else {
    win.removeMenu()
  }

  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  )

  win.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify();
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


//TODO: In frontend, map out a button that lets users do an auto update and update the "don't remind me" or "check for auto update", check version number and press "check for updates or update" in the app itself
// When an update is available, show a message to the user
autoUpdater.on('update-available', () => {
  const message = 'An update is available. Would you like to install it now?';
  // const buttons = ['Install', 'Cancel', 'Don\'t ask again'];
  const buttons = ['Install', 'Cancel'];

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
      autoUpdater.quitAndInstall()
    }
  })
});
