import https from 'https';
import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ipcMain } from "electron";
import yauzl from "yauzl";

let installationProcess: ChildProcessWithoutNullStreams;

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
  
const backupPythonInstallation = (mainWindow: Electron.BrowserWindow, artroomPath: string, gpuType: string) => () => {
    console.log("REINSTALL BACKING")
    console.log(`VANILLA PATH: ${artroomPath}`)
    const URL = gpuType === 'AMD' ? 
      'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/miniconda3_amd.zip' 
      : 
      'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/miniconda3.zip';

    const PATH = path.join(artroomPath, "\\artroom\\miniconda3");
    console.log(`ARTROOM PATH: ${PATH}`)
    const PATH_requirements = path.resolve('stable-diffusion/requirements.txt');
    console.log(`ARTROOM REQUIREMENTS PATH: ${PATH_requirements}`)

    const PATH_zip = path.join(artroomPath, "\\artroom\\file.zip")
    console.log(`ARTROOM ZIP PATH: ${PATH_zip}`)

    const installationCommand = `"${PATH}/Scripts/conda" run --no-capture-output -p "${PATH}/envs/artroom-ldm" python -m pip install -r "${PATH_requirements}" && set /p choice= "Finished! Please exit out of this window or press enter to close"`;

    removeDirectoryIfExists(PATH).then(()=>{
        const request = https.get(URL, (response) => {
            if (fs.existsSync(PATH_zip)) {
              fs.unlinkSync(PATH_zip);
            }

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
                yauzl.open(PATH_zip, { lazyEntries: true }, (error, zipFile) => {
                  if (error) {
                    console.error(`Error opening ZIP archive: ${error}`);
                    return;
                  }
                
                  const totalEntries = zipFile.entryCount;
                  let extractedEntries = 0;
                
                  zipFile.readEntry();
                
                  zipFile.on("entry", (entry) => {
                    if (/\/$/.test(entry.fileName)) {
                      // Directory entry
                      fs.mkdirSync(`${artroomPath}/artroom/${entry.fileName}`);
                      zipFile.readEntry();
                    } else {
                      // File entry
                      zipFile.openReadStream(entry, (error, readStream) => {
                        if (error) {
                          console.error(`Error opening read stream for ${entry.fileName}: ${error}`);
                          zipFile.readEntry();
                          return;
                        }
                
                        const writeStream = fs.createWriteStream(`${artroomPath}/artroom/${entry.fileName}`);
                
                        writeStream.on("close", () => {
                          extractedEntries++;
                
                          const progress = Math.round((extractedEntries / totalEntries) * 100);
                          console.log();
                          mainWindow.webContents.send('fixButtonProgress', `Extracting... ${progress}%`);
                          zipFile.readEntry();
                        });
                
                        readStream.pipe(writeStream);
                      });
                    }
                  });
                
                  zipFile.on("error", (error) => {
                    mainWindow.webContents.send('fixButtonProgress', `Error reading ZIP archive: ${error}`);
                  });
                
                  zipFile.on("end", () => {
                    mainWindow.webContents.send('fixButtonProgress', "Extraction complete. Updating packages...");
                    // Delete the ZIP file
                    fs.unlinkSync(PATH_zip);
                    installationProcess = spawn(installationCommand, { shell: true, detached: true });
                    installationProcess.stdout.on("data", (data) => {
                        console.log(`stdout: ${data}`);
                        mainWindow.webContents.send('fixButtonProgress', `Finished! Please try reopening the app`);
                    });
                    installationProcess.stderr.on("data", (data) => {
                        console.error(`stderr: ${data}`);
                    });
                    installationProcess.on("close", (code) => {
                        console.log(`child process exited with code ${code}`);
                        mainWindow.webContents.send('fixButtonProgress', `Finished! Please try reopening the app`);
                    });
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

const reinstallPythonDependencies = (artroomPath: string) => () => {
    console.log("RESINSTALLING DEPENDENCIES")
    console.log(artroomPath);
    const PATH = path.join(artroomPath, "artroom\\miniconda3");
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
  ipcMain.handle('pythonInstall', (event, artroomPath, gpuType) => {
    backupPythonInstallation(mainWindow, artroomPath, gpuType)();
  });    
  ipcMain.handle('pythonInstallDependencies', (event, artroomPath) => {
    reinstallPythonDependencies(artroomPath)();
  });    
}
