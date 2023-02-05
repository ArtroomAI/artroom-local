import https from 'https';
import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ipcMain } from "electron";
import StreamZip from 'node-stream-zip';
import { deleteSync } from 'del';
import os from 'os';

let installationProcess: ChildProcessWithoutNullStreams;
let hd = os.homedir();
//Start with cleanup
// const artroom_install_log = userDataPath + "/artroom_install.log";
const artroom_install_log = hd + "\\AppData\\Local\\artroom_install.log";
let artroom_path = hd;
if (fs.existsSync(artroom_install_log)) {
  let temp = fs.readFileSync(artroom_install_log, 'utf-8');
  let lines = temp.split(/\r?\n/);
  artroom_path = lines[0];
  console.log(`NEW ARTROOM PATH: ${artroom_path}`)
}

async function removeDirectoryIfExists(PATH: fs.PathLike) {
    try {
      const exists = fs.existsSync(PATH);
      if (exists) {
        console.log(`${PATH} exists! Deleting...`)
        await fs.promises.rmdir(PATH, { recursive: true });
        console.log(`${PATH} deleted`)
      }
    } catch (error) {
      console.error(error);
    }
  }
  

const backupPythonInstallation = (mainWindow: Electron.BrowserWindow) => () => {
    console.log("REINSTALL BACKING")
    console.log(`VANILLA PATH: ${artroom_path}`)

    const URL = 'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/miniconda3.zip';
    const PATH = path.join(artroom_path, "\\artroom\\miniconda3");
    console.log(`ARTROOM PATH: ${PATH}`)
    const PATH_requirements = path.resolve('stable-diffusion/requirements.txt');
    console.log(`ARTROOM REQUIREMENTS PATH: ${PATH_requirements}`)

    const PATH_zip = path.join(artroom_path, "\\artroom\\file.zip")
    console.log(`ARTROOM ZIP PATH: ${PATH_zip}`)

    const installationCommand = `"${PATH}/Scripts/conda" run --no-capture-output -p "${PATH}/envs/artroom-ldm" python -m pip install -r "${PATH_requirements}" && set /p choice= "Finished! Please exit out of this window or press enter to close"`;

    removeDirectoryIfExists(PATH).then(()=>{
        const request = https.get(URL, (response) => {
            const len = parseInt(response.headers['content-length'], 10);
            let cur = 0;
            const toMB = (n: number) => (n / 1048576).toFixed(2);
    
            const file = fs.createWriteStream(PATH_zip);
            response.pipe(file);
    
            const total = toMB(len); //1048576 - bytes in 1 MB
    
            let chunk_counter = 0;
    
            response.on("data", (chunk) => {
                cur += chunk.length;
                ++chunk_counter;
                if(chunk_counter === 5000) {
                    console.log(`Downloading ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${total}mb`);
                    mainWindow.webContents.send('fixButtonProgress', `Downloading ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${total}mb`);
                    chunk_counter = 0;
                }
           });
    
            file.on("finish", () => {
                file.close();
                console.log('Downloading complete. Decompressing...');
                console.log(PATH_zip)
                mainWindow.webContents.send('fixButtonProgress', 'Downloading complete. Decompressing...');
                const zip = new StreamZip({ file: PATH_zip});
    
                zip.on('ready', () => {
                    fs.mkdirSync(PATH, { recursive: true });
                    zip.extract(null, path.join(path.join(artroom_path, "\\artroom\\")), (err, count) => {
                        mainWindow.webContents.send('fixButtonProgress', err ? 'Extract error' : `Extracted ${count} entries`);
                        console.log(err ? 'Extract error' : `Extracted ${count} entries`);
                        installationProcess = spawn(installationCommand, { shell: true, detached: true });
                        installationProcess.stdout.on("data", (data) => {
                            console.log(`stdout: ${data}`);
                        });
                        installationProcess.stderr.on("data", (data) => {
                            console.error(`stderr: ${data}`);
                        });
                        installationProcess.on("close", (code) => {
                            console.log(`child process exited with code ${code}`);
                            mainWindow.webContents.send('fixButtonProgress', `Finished! Please try reopening the app`);
                            removeDirectoryIfExists(PATH_zip)
                        });
                        zip.close();
                    });
                });
            });
    
            request.on("error", (e) => {
                mainWindow.webContents.send('fixButtonProgress', `Error: ${e.message}`);
                file.close();
            });
        });
    })

};

const reinstallPythonDependencies = () => () => {
    console.log("RESINSTALLING DEPENDENCIES")
    console.log(artroom_path);
    const PATH = path.join(artroom_path, "artroom\\miniconda3");
    console.log(PATH);
    const PATH_requirements = path.resolve('stable-diffusion/requirements.txt');
    console.log(PATH_requirements)
    const installationCommand = `"${PATH}/Scripts/conda" run --no-capture-output -p "${PATH}/envs/artroom-ldm" python -m pip install -r "${PATH_requirements}" && set /p choice= "Finished! Please exit out of this window or press enter to close"`;
    
    console.log(installationCommand)
    installationProcess = spawn(installationCommand, { shell: true, detached: true });

    installationProcess.stdout.on('data', function(data) {
        console.log("Child data: " + data);
      });
      installationProcess.on('error', function () {
        console.log("Failed to start child.");
      });
      installationProcess.on('close', function (code) {
        console.log('Child process exited with code ' + code);
      });
      installationProcess.stderr.on('data', function (err) {
        console.log(`error: ${err}`);
      });
      installationProcess.on('message', (msg) => {
        console.log(`msg ${msg}`)
      })
      installationProcess.stderr.on('message', (msg) => {
        console.log(`ermsg ${msg}`)
      })
      installationProcess.stdout.on('end', function () {
        console.log('Finished collecting data chunks.');
      });
}

export const handlers = (mainWindow: Electron.BrowserWindow) => {
    ipcMain.handle('pythonInstall', backupPythonInstallation(mainWindow));
    ipcMain.handle('pythonInstallDependencies', reinstallPythonDependencies());
}
