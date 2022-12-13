const { app, BrowserWindow,ipcMain, ipcRenderer, shell, dialog} = require('electron');
const { autoUpdater } = require("electron-updater");
const log = require('electron-log');
const os = require('os');
const path = require('path');
const isDev = require('electron-is-dev');
const fs = require("fs");
const glob = require('glob');
const axios = require('axios');
var kill = require('tree-kill');

require('@electron/remote/main').initialize()
var {PythonShell} = require('python-shell');

let win;
let hd = os.homedir();

var spawn = require('child_process').spawn;
server = spawn("\"" + hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python\" server.py", {detached: true, encoding: 'utf8', shell: true});

async function getB64(path){
  return new Promise((resolve, reject) => {
    fs.promises.readFile(path, 'base64').then(result => {
      resolve("data:image/png;base64,"+result);
    }).catch(err => {
      console.log("Not found");
      resolve("");
    });
  });
}


let runner;

function createWindow() {
  //Connect to server
  axios.post('http://127.0.0.1::5300/start',{}); 
       
  // ipcMain.handle('startSD', async (event) => {
  //   return new Promise((resolve, reject) => {
  //       let options = {
  //         pythonPath: hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe",
  //       };
  //       var addToQueue = new PythonShell('artroom_helpers/add_to_queue.py', options)
  //       addToQueue.on("message", function(message) {
  //         fs.readFile(hd+"\\artroom\\settings\\queue.json", "utf8", function (err, data) {
  //           let queue_json = JSON.parse(data);
  //           if (queue_json["Running"] === false){
  //             runner = spawn("\"" + hd+"\\artroom\\miniconda3\\envps\\artroom-ldm\\python\" runner.py", {detached: true, encoding: 'utf8', shell: true});
  //             runner.on('exit', function (code, signal) {
  //               fs.readFile(hd+"\\artroom\\settings\\queue.json", "utf8", function (err, data) {
  //                 let queue_json = JSON.parse(data);
  //                 //reset error_code
  //                 queue_json["Running"] = false;
  //                 fs.writeFileSync(hd+"\\artroom\\settings\\queue.json", JSON.stringify(queue_json));
  //               });

  //               fs.readFile(hd+"\\artroom\\settings\\error_mode.json", "utf8", function (err, data) {
  //                 if (err) {
  //                   resolve({"error_code": ""});
  //                 }
  //                 let err_data = JSON.parse(data);
  //                 //reset error_code
  //                 let err_data_reset = {"error_code": ""};
  //                 fs.writeFileSync(hd+"\\artroom\\settings\\error_mode.json", JSON.stringify(err_data_reset));
  //                 resolve(err_data);
  //                 });
  //               });
  //             }
  //         });
  //       });
  //     });
  //   });//End of ipcMain

    // ipcMain.handle('paintSD', async (event, data) => {
    //   return new Promise((resolve, reject) => {
    //         var base64Data = data.replace(/^data:image\/png;base64,/, "");
    //         fs.writeFile(hd+"\\artroom\\settings\\out.png", base64Data, 'base64', function(err) {
              
    //           let options = {
    //             pythonPath: hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe",
    //           };
    //           var addToQueue = new PythonShell('artroom_helpers/add_to_queue.py', options)
    //           addToQueue.on("message", function(message) {
    //             fs.readFile(hd+"\\artroom\\settings\\queue.json", "utf8", function (err, data) {
    //               let queue_json = JSON.parse(data);
    //               if (queue_json["Running"] === false){
    //                 runner = spawn("\"" + hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python\" runner.py", {detached: true, encoding: 'utf8', shell: true});
    //                 runner.on('exit', function (code, signal) {
    //                   fs.readFile(hd+"\\artroom\\settings\\queue.json", "utf8", function (err, data) {
    //                     let queue_json = JSON.parse(data);
    //                     //reset error_code
    //                     queue_json["Running"] = false;
    //                     fs.writeFileSync(hd+"\\artroom\\settings\\queue.json", JSON.stringify(queue_json));
    //                   });
    
    //                   fs.readFile(hd+"\\artroom\\settings\\error_mode.json", "utf8", function (err, data) {
    //                     if (err) {
    //                       resolve({"error_code": ""});
    //                     }
    //                     let err_data = JSON.parse(data);
    //                     //reset error_code
    //                     let err_data_reset = {"error_code": ""};
    //                     fs.writeFileSync(hd+"\\artroom\\settings\\error_mode.json", JSON.stringify(err_data_reset));
    //                     resolve(err_data);
    //                   });
    //                 });
    //                 }
    //               });
    //             });
    //         });
    //       });
    //     });//End of ipcMain

    ipcMain.handle('updateSettings', async (event, data) => {
      return new Promise((resolve, reject) => {
        // console.log("Starting");
        let options = {
          pythonPath: hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe",
          args: [JSON.stringify(data)]
        };
        var updateJSON = new PythonShell('artroom_helpers/updateSettings.py', options)
        updateJSON.on("message", function(message) {
          resolve("Finished");
       });
      });//End of ipcMain
    });

    // ipcMain.handle('upscale', async (event, data) => {
    //   return new Promise((resolve, reject) => {
    //         let options = {
    //         pythonPath: hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe",
    //         args: [JSON.stringify(data)]
    //       };
    //       var upscalerJSON = new PythonShell('artroom_helpers/upscaleJSON.py', options)
    //       // console.log("Update Settings");
    //       upscalerJSON.on("message", function(message) {
    //         var upscaler = spawn("\"" + hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python\" artroom_helpers/upscale.py", {detached: true, encoding: 'utf8', shell: true});
    //         upscaler.on('exit', function (code, signal) {
    //           let imgPath;
    //           if (data['upscale_dest'] === ""){
    //             imgPath = path.dirname(data['upscale_images'][0])
    //           }
    //           else{
    //             imgPath = data['upscale_dest'];
    //           }
    //           imgPath = imgPath.replace('%UserProfile%', hd);
    //           shell.openPath(imgPath.split(path.sep).join(path.posix.sep))  
    //           resolve("Finished");
    //         });
  
    //       });
    //     });
    //   });//End of ipcMain

    ipcMain.handle("getImageFromPath", (event, path) => {
      return new Promise((resolve, reject) => {
        getB64(path).then((b64Output)=>{
          resolve({"ImagePath": path, "B64": b64Output});
        })
      });
    });

    // ipcMain.handle("getImage", (event, argx = 0) => {
    //   return new Promise((resolve, reject) => {
    //     fs.readFile(hd+"\\artroom\\settings\\sd_settings.json", "utf8", function (err, data) {
    //       if (err) {  
    //         console.log("Error getting image!")
    //         reject(err);
    //         return;
    //       }         
    //       let imgPath = JSON.parse(data)['image_save_path'] + "/" + JSON.parse(data)['Settings']['batch_name'];
    //       imgPath = imgPath.replace('%UserProfile%', hd);
    //       let promptName = JSON.parse(data)['Settings']['text_prompts'].replace(/[^a-z0-9\s]/gi, '').split(/\s+/).join('_');
    //       let latestImagePath = imgPath + "/"+promptName;
    //       latestImagePath = latestImagePath.slice(0,150)+ "/latest.png";
    //       if (fs.existsSync(latestImagePath)){
    //         getB64(latestImagePath).then((b64Output)=>{
    //           resolve({"ImagePath": latestImagePath, "B64":b64Output});
    //         })  
    //       }
    //       else{
    //         resolve({"ImagePath": "", "B64": ""});
    //       } 
    //     });
    //   });
    // });

  // ipcMain.handle('updateConfig', async (event, data) => {
  //   let options = {
  //     pythonPath: hd+"\\artroom\\miniconda3\\envs\\artroom-ldm\\python.exe",
  //     args: [JSON.stringify(data)]
  //   };
  //   PythonShell.run('artroom_helpers/updateConfig.py',options,function(err,results) {
  //     if(err) throw err;
  //   })
  //   return;
  // });

  ipcMain.handle('getModelCkpts', async (event, data) => {
    return new Promise((resolve, reject) => {
      glob(data+'/*.ckpt', {}, (err,files)=>{
        if(err){
          console.log("ERROR");
          resolve([]);
        }
        resolve(files);
      })
    });
  });

  ipcMain.handle('getImages', async (event, data) => {
    return new Promise((resolve, reject) => {
        imagesPath = data.replace('%UserProfile%', hd).replaceAll("\\","/");
        glob(imagesPath+'/**/*{.png,.jpg,.jpeg}', {}, (err,files)=>{
        if(err){
          console.log(err);
          resolve([]);
        }
          resolve(files);
        })
      });
  });

  ipcMain.handle("getSettings", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      fs.readFile(hd+"\\artroom\\settings\\sd_settings.json", "utf8", function (err, data) {
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
      if(os.platform() === 'linux' || os.platform() === 'win32'){
        properties = ['openFile', 'multiSelections'];
      }
      else{
        properties = ['openFile', 'openDirectory', 'multiSelections'];
      }
        dialog.showOpenDialog({
            properties: properties,
            filters: [
              { name: 'Settings', extensions: ['json'] },
            ]
        }).then(result => {
          if (result.filePaths.length > 0){
            fs.readFile(result.filePaths[0], "utf8", function (err, data) {
              if (err) {
                reject(err);
                return;
              }
              resolve([data]);
            });           
          }
          else{
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
      if(os.platform() === 'linux' || os.platform() === 'win32'){
        properties = ['openFile', 'multiSelections'];
      }
      else{
        properties = ['openFile', 'openDirectory', 'multiSelections'];
      }
        dialog.showOpenDialog({
            properties: properties,
            filters: [
              { name: 'Images', extensions: ['jpg', 'png', 'jpeg'] },
            ]
        }).then(result => {
          if (result.filePaths.length > 0){
              resolve(result.filePaths);           
          }
          else{
            resolve("");
          }
        }).catch(err => {
          resolve("");
        })
    });
  });

  ipcMain.handle("getQueue", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      fs.readFile(hd+"\\artroom\\settings\\queue.json", "utf8", function (err, data) {
        if (err) {
          resolve({"Queue": [],"Running": false, "Keep_Warm": false})
        }
        resolve(data);
      });
    });
  });

  // ipcMain.handle("writeQueue", (event, data) => {
  //   return new Promise((resolve, reject) => {
  //     fs.writeFileSync(hd+"\\artroom\\settings\\queue.json", JSON.stringify(data));
  //       resolve("Finished!");
  //   });
  // });

  ipcMain.handle("getImageDir", (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      fs.readFile(hd+"\\artroom\\settings\\sd_settings.json", "utf8", function (err, data) {
        if (err) {
          reject(err);
          return;
        }         
        let imgPath = JSON.parse(data)['Config']['image_save_path'] + "/" + JSON.parse(data)['Settings']['batch_name'];
        imgPath = imgPath.replace('%UserProfile%', hd);
        if (fs.existsSync(imgPath.split(path.sep).join(path.posix.sep))){
          shell.openPath(imgPath.split(path.sep).join(path.posix.sep))
        }
        else{
          imgPath = JSON.parse(data)['Config']['image_save_path'];
          imgPath = imgPath.replace('%UserProfile%', hd);
          shell.openPath(imgPath.split(path.sep).join(path.posix.sep))
        }
        resolve([data]);
      });
    });
  });

  ipcMain.handle('uploadInitImage', (event, argx = 0) => {
    return new Promise((resolve, reject) => {
      let properties;
      if(os.platform() === 'linux' || os.platform() === 'win32'){
        properties = ['openFile'];
      }
      else{
        properties = ['openFile', 'openDirectory']
      }
        dialog.showOpenDialog({
            properties: properties,
            filters: [
              { name: 'Images', extensions: ['jpg', 'png', 'jpeg'] },
            ]
        }).then(result => {
          if (result.filePaths.length > 0){
            resolve(result.filePaths[0]);
          }
          else{
            resolve("");
          }
        }).catch(err => {
          resolve("");
        })
      });
    });

    ipcMain.handle('chooseUploadPath', (event, argx = 0) => {
      return new Promise((resolve, reject) => {
          dialog.showOpenDialog({
              properties: ['openDirectory'],
          }).then(result => {
            if (result.filePaths.length > 0){
              resolve(result.filePaths[0]);
            }
            else{
              resolve("");
            }
          }).catch(err => {
            resolve("");
          })
          
        });
      });

  // Create the browser window.
  win = new BrowserWindow({
    width: 1350,
    height: 950,
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      contextIsolation: true,
      webSecurity: false,
      devTools: true,
      autoHideMenuBar: false,
      preload: path.join(__dirname, "preload.js") 
    }
  })
  win.setTitle("ArtroomAI v" + app.getVersion());
  {isDev ? {} : win.removeMenu()}
  //win.webContents.openDevTools();
  win.loadURL(
    isDev
      ? 'http://localhost:3000'
      : `file://${path.join(__dirname, '../build/index.html')}`
  )
  win.webContents.on("new-window", function(event, url) {
    event.preventDefault();
    shell.openExternal(url);
  });
  win.once('ready-to-show', () => {
    autoUpdater.checkForUpdatesAndNotify();
  });
}

app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function () {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    kill(server.pid);
    app.quit()
  }
})

app.on('activate', function () {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})


autoUpdater.on("update-available", (_event, releaseNotes, releaseName) => {
  const dialogOpts = {
      type: 'info',
      buttons: ['Ok'],
      title: 'Application Update',
      message: process.platform === 'win32' ? releaseNotes : releaseName,
      detail: 'A new version is being downloaded.'
  }
  dialog.showMessageBox(dialogOpts, (response) => {

  });
})

autoUpdater.on("update-downloaded", (_event, releaseNotes, releaseName) => {
  const dialogOpts = {
      type: 'info',
      buttons: ['Restart', 'Later'],
      title: 'Application Update',
      message: process.platform === 'win32' ? releaseNotes : releaseName,
      detail: 'A new version has been downloaded. Restart the application to apply the updates.'
  };
  dialog.showMessageBox(dialogOpts).then((returnValue) => {
      if (returnValue.response === 0) autoUpdater.quitAndInstall()
  })
});

autoUpdater.on('download-progress', (progressObj) => {
  let log_message = "Download speed: " + progressObj.bytesPerSecond;
  log_message = log_message + ' - Downloaded ' + progressObj.percent + '%';
  log_message = log_message + ' (' + progressObj.transferred + "/" + progressObj.total + ')';
  sendStatusToWindow(log_message);
})

function sendStatusToWindow(text) {
    log.info(text);
    win.webContents.send('message', text);
}