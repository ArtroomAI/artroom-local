const { app, remote, BrowserWindow, ipcMain, clipboard, shell, dialog, nativeImage } = require('electron');
const { autoUpdater } = require("electron-updater");
autoUpdater.autoDownload = false;
const log = require('electron-log');
const os = require('os');
const path = require('path');
const isDev = require('electron-is-dev');
const fs = require("fs");
const glob = require('glob');
const axios = require('axios');
const kill = require('tree-kill');
require('@electron/remote/main').initialize()

const getUserDataPath = () => {
  const platform = os.platform();
  if (platform === 'win32') {
    return path.join(process.env.APPDATA, appName);
  } else if (platform === 'darwin') {
    return path.join(process.env.HOME, 'Library', 'Application Support', appName);
  } else {
    return path.join('var', 'local', appName);
  }
}

const imageGenerationSettings = "sd_settings.json"
const imageSettings = "image_settings.json"
const applicationSettings = "app_settings.json"

const userDataPath = getUserDataPath();

let win;
let hd = os.homedir();

//Start with cleanup
axios.get('http://127.0.0.1:5300/shutdown')

// axios.get('http://127.0.0.1:8000/').then(res => res.json()).then((result)=>{
//   console.log("Test: "+ result);
// })

var spawn = require('child_process').spawn;

//read installation log to find artroom path
// const artroom_install_log = userDataPath + "/artroom_install.log";
const artroom_install_log = hd + "\\AppData\\Local\\artroom_install.log";
let artroom_path = hd;
if (fs.existsSync(artroom_install_log)) {
  // Do something
  let temp = fs.readFileSync(artroom_install_log, 'UTF-8');
  let lines = temp.split(/\r?\n/);
  artroom_path = lines[0];
}

//replace placeholders with actual paths
let sd_data;
let sd_data_json;
try {
  sd_data = fs.readFileSync(artroom_path + "\\artroom\\settings\\sd_settings.json", 'UTF-8');
}
catch (e) {
  sd_data = fs.readFileSync("sd_settings.json", 'UTF-8');
}
sd_data_json = JSON.parse(sd_data);
sd_data_json['image_save_path'] = sd_data_json['image_save_path'].replace('%UserProfile%', hd);
sd_data_json['image_save_path'] = path.normalize(sd_data_json['image_save_path']);
sd_data_json['ckpt'] = sd_data_json['ckpt'].replace('%InstallPath%', artroom_path);
sd_data_json['ckpt'] = path.normalize(sd_data_json['ckpt']);
sd_data_json['ckpt_dir'] = sd_data_json['ckpt_dir'].replace('%InstallPath%', artroom_path);
sd_data_json['ckpt_dir'] = path.normalize(sd_data_json['ckpt_dir']);
let debugModeInit = sd_data_json['debug_mode'];
sd_data = JSON.stringify(sd_data_json, null, 2);
fs.writeFileSync(artroom_path + "\\artroom\\settings\\sd_settings.json", sd_data);

async function getB64(path) {
  return new Promise((resolve, reject) => {
    fs.promises.readFile(path, 'base64').then(result => {
      resolve("data:image/png;base64," + result);
    }).catch(err => {
      console.log("Not found");
      resolve("");
    });
  });
}

//pyTestCmd = "\"" + artroom_path +"\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python pytest.py";
//pyTestCmd = "\"" + artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python" + "\"";
let pyTestCmd = artroom_path + "\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe";

let command = "\"" + artroom_path + "\\artroom\\miniconda3\\Scripts\\conda" + "\"" + " run --no-capture-output -p " + "\"" + artroom_path + "/artroom/miniconda3/envs/artroom-ldm" + "\"" + " python server.py";

let server = spawn(command, { detached: debugModeInit, encoding: 'utf8', shell: true })

function createWindow() {
  //Connect to server
  axios.get('http://127.0.0.1::5300/start');
  console.log("Artroom Log: " + artroom_install_log);

  ipcMain.handle('login', async(event,data) =>{
    return new Promise((resolve, reject) => {
      //axios.post('login')
      //store jwt token in safe storage
      //store url
    });
  })


  ipcMain.handle('getCkpts', async (event, data) => {
    return new Promise((resolve, reject) => {
      glob(data + '/*.ckpt', {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        }
        files = files.map(function (match) {
          return path.relative(data, match);
        });
        resolve(files);
      })
    });
  });

  ipcMain.handle('getImageFromPath', async (event, data) => {
    return new Promise((resolve, reject) => {
      if (data && data.length > 0) {
        resolve(getB64(data));
      }
      else {
        resolve("");
      }
    });
  });

  ipcMain.handle('copyToClipboard', async (event, B64) => {
    return new Promise((resolve, reject) => {
      clipboard.writeImage(nativeImage.createFromDataURL(B64));
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
      exeProcess.on('close', (code) => {
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
      let properties;
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
      let properties;
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
        let imgPath = JSON.parse(data)['image_save_path'] + "/" + JSON.parse(data)['batch_name'];
        imgPath = imgPath.replace('%UserProfile%', artroom_path);
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

  ipcMain.handle("openEquilibrium", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      shell.openExternal(`https://equilibriumai.com/`);
    });
  });



  // ipcMain.handle('uploadInitImage', (event, argx = 0) => {
  //   return new Promise((resolve, reject) => {
  //     let properties;
  //     if(os.platform() === 'linux' || os.platform() === 'win32'){
  //       properties = ['openFile'];
  //     }
  //     else{
  //       properties = ['openFile', 'openDirectory']
  //     }
  //       dialog.showOpenDialog({
  //           properties: properties,
  //           filters: [
  //             { name: 'Images', extensions: ['jpg', 'png', 'jpeg'] },
  //           ]
  //       }).then(result => {
  //         if (result.filePaths.length > 0){
  //           resolve(getB64(result.filePaths[0]));
  //         }
  //         else{
  //           resolve("");
  //         }
  //       }).catch(err => {
  //         resolve("");
  //       })
  //     });
  //   });

  ipcMain.handle('chooseUploadPath', (event, argx = 0) => {
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

  ipcMain.handle('runPyTests', (event, argx = 0) => {
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
      spawn("taskkill", ["/pid", server.pid, '/f', '/t']);
      if (isDebug) {
        server = spawn(command, { detached: true, encoding: 'utf8', shell: true });
      } else {
        server = spawn(command, { detached: false, encoding: 'utf8', shell: true });
      }
      return axios.get('http://127.0.0.1:5300/get_progress',
        { headers: { 'Content-Type': 'application/json' } }).then((result) => {
          resolve(result.status);
        });
    });
  });



  // Create the browser window.
  win = new BrowserWindow({
    width: 1350,
    height: 1000,
    frame: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: true,
      enableRemoteModule: true,
      contextIsolation: true,
      webSecurity: false,
      devTools: true,
      autoHideMenuBar: false,
    }
  })
  win.setTitle("ArtroomAI v" + app.getVersion());
  if(!isDev){
    win.removeMenu()
  }
  win.webContents.openDevTools();
  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  )
  win.webContents.on("new-window", function (event, url) {
    event.preventDefault();
    shell.openExternal(url);
  });
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
    spawn("taskkill", ["/pid", server.pid, '/f', '/t']);
    axios.get('http://127.0.0.1:5300/shutdown')
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

autoUpdater.on("update-downloaded", (_event, releaseNotes, releaseName) => {
  const dialogOpts = {
    type: 'info',
    buttons: ['Restart', 'Later'],
    title: 'Application Update',
    message: process.platform === 'win32' ? releaseNotes : releaseName,
    detail: 'A new version has been downloaded. Restart the application to apply the updates.'
  };
  dialog.showMessageBox(dialogOpts).then((returnValue) => {
    if (returnValue.response === 0) {
      if (process.platform !== 'darwin') {
        kill(server.pid);
        spawn("taskkill", ["/pid", server.pid, '/f', '/t']);
        axios.get('http://127.0.0.1:5300/shutdown')
      }  
      autoUpdater.quitAndInstall()
    }
  })
});
