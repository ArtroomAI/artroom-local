import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn, exec } from 'child_process';
import { ipcMain } from "electron";
import yauzl from "yauzl";
import axios from 'axios';
import { pipeline as _pipeline } from 'stream';
import { promisify } from 'util';
import gpuInfo from 'gpu-info';
const pipeline = promisify(_pipeline);

let installationProcess: ChildProcessWithoutNullStreams;

const detectGPU = async () => {
  try {
    // Check if it's a Mac
    if (process.platform === 'darwin') {
      return 'Mac';
    }

    // Check if it's a NVIDIA GPU on Windows
    if (process.platform === 'win32') {
      const gpus = await gpuInfo();
      const nvidiaGPU = gpus.find(
        (gpu : any) => gpu.vendor.toLowerCase() === 'nvidia'
      );
      if (nvidiaGPU) {
        return 'NVIDIA';
      }
    }

    // Check if it's an AMD GPU on Windows
    if (process.platform === 'win32') {
      const gpus = await gpuInfo();
      const amdGPU = gpus.find(
        (gpu : any) => gpu.vendor.toLowerCase() === 'amd'
      );
      if (amdGPU) {
        return 'AMD';
      }
    }

    // Add detection logic for other GPU vendors if needed

    // No GPU detected
    return 'NVIDIA';
  } catch (error) {
    console.error('GPU detection failed:', error);
    return 'NVIDIA';
  }
};

const toMB = (n: number) => (n / 1048576).toFixed(2); //1048576 - bytes in 1 MB

function removeDirectoryIfExists(PATH: fs.PathLike) {
  try {
    const exists = fs.existsSync(PATH);
    if (exists) {
      console.log(`${PATH} exists! Deleting...`)
      fs.rmdirSync(PATH, { recursive: true });
      console.log(`${PATH} deleted`)
    }
  } catch (error) {
    console.error(error);
  }
}

function unzipFile(PATH_zip: string, artroomPath: string, mainWindow: Electron.BrowserWindow) {
  return new Promise<string>((resolve) => {
    yauzl.open(PATH_zip, { lazyEntries: true }, (error, zipFile) => {
      if (error) {
        console.error(`Error opening ZIP archive: ${error}`);
        mainWindow.webContents.send('fixButtonProgress', `Error opening ZIP archive: ${error}`);
        resolve('');
        return;
      }
    
      const totalEntries = zipFile.entryCount;
      let extractedEntries = 0;
    
      zipFile.readEntry();
    
      zipFile.on("entry", (entry) => {
        if (/\/$/.test(entry.fileName)) {
          // Directory entry
          fs.mkdirSync(`${artroomPath}/artroom/${entry.fileName}`, { recursive: true });
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
        resolve('');
      });
    
      zipFile.on("end", () => {
        mainWindow.webContents.send('fixButtonProgress', "Extraction complete. Updating packages...");
        // Delete the ZIP file
        fs.unlinkSync(PATH_zip);
        resolve('');
      });
    });
  })
}

const backupPythonInstallation = async (mainWindow: Electron.BrowserWindow, artroomPath: string) => {
    const gpuType = await detectGPU();
    console.log("REINSTALL BACKEND")
    console.log(`VANILLA PATH: ${artroomPath}`)
    const URL = gpuType === 'AMD' ? 
      'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/artroom_backend_amd.zip' 
      : 
      'https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/artroom_backend_nvidia.zip';

    const PATH = path.join(artroomPath, "\\artroom\\artroom_backend");
    console.log(`ARTROOM PATH: ${PATH}`)
    const PATH_requirements = gpuType === 'AMD' ? 
      path.resolve('requirements_amd.txt') 
      : 
      path.resolve('requirements_nvidia.txt');
    console.log(`ARTROOM REQUIREMENTS PATH: ${PATH_requirements}`);

    const PATH_zip = path.join(artroomPath, "\\artroom\\file.zip")
    console.log(`ARTROOM ZIP PATH: ${PATH_zip}`)

    removeDirectoryIfExists(PATH);

    if (fs.existsSync(PATH_zip)) {
      fs.unlinkSync(PATH_zip);
    }
  
    fs.mkdirSync(path.join(artroomPath, "artroom", "settings"), { recursive: true });

    const success = await download_via_https('Artroom', URL, PATH_zip, mainWindow);

    if(!success) return;

    await unzipFile(PATH_zip, artroomPath, mainWindow);
    await reinstallPythonDependencies(artroomPath, mainWindow);
    mainWindow.webContents.send('fixButtonProgress', `Finished downloading and installing required files!`);
    console.log('DONE')
    return
};

async function download_via_https(name: string, URL: string, file_path: string, mainWindow: Electron.BrowserWindow, retries = 5): Promise<boolean> {
  const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

  async function downloadWithRetry(retryCount: number): Promise<boolean> {
    if (retryCount === 0) {
      throw new Error(`Failed to download ${name} after ${retries} retries`);
    }

    try {
      const response = await axios.get(URL, {
        responseType: "stream",
        timeout: 10000, // 10 seconds timeout
        headers: {
          "Cache-Control": "no-cache",
        },
      });

      const len = parseInt(response.headers["content-length"], 10);
      let cur = 0;
      let chunk_counter = 0;

      response.data.on("data", (chunk: any) => {
        cur += chunk.length;
        ++chunk_counter;
        if (chunk_counter >= 2000) {
          console.log(`Downloading ${name} ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${toMB(len)}mb`);
          mainWindow.webContents.send(
            "fixButtonProgress",
            `Downloading ${name} ${(100 * cur / len).toFixed(2)}% - ${toMB(cur)}mb / ${toMB(len)}mb`
          );
          chunk_counter = 0;
        }
      });
      

      await pipeline(
        response.data,
        (function () {
          const file = fs.createWriteStream(file_path);
          file.on("finish", () => {
            mainWindow.webContents.send("fixButtonProgress", `Downloaded ${name} successfully.`);
          });
          file.on("error", (e) => {
            throw e;
          });
          return file;
        })()
      );

      return true;
    } catch (error) {
      mainWindow.webContents.send("fixButtonProgress", `Error: ${error.message}`);
      console.error(`Error downloading ${name}: ${error.message}`);
      await delay(1000 * Math.pow(2, retries - retryCount)); // Exponential backoff
      return downloadWithRetry(retryCount - 1);
    }
  }

  return downloadWithRetry(retries);
}

const MAX_INSTALL_RETRIES = 3;

const reinstallPythonDependencies = async (artroomPath: string, mainWindow?: Electron.BrowserWindow) => {
  const gpuType = await detectGPU();
  console.log("REINSTALLING DEPENDENCIES");
  const PATH = path.join(artroomPath, "artroom", "artroom_backend");
  const PATH_requirements = gpuType === 'AMD' ? 
  path.resolve('requirements_amd.txt') 
  : 
  path.resolve('requirements_nvidia.txt');

  const installationCommand = `"${PATH}/python" -m pip install --upgrade pip && "${PATH}/python" -m pip install -r "${PATH_requirements}"`;

  try {
    await executeInstallationCommand(installationCommand);
    mainWindow?.webContents.send('fixButtonProgress', `Finished downloading and installing required files!`);
    return;
  } catch (error) {
    console.error(`Error reinstalling Python dependencies: ${error}`);
    mainWindow?.webContents.send('fixButtonProgress', `Error: ${error.message}`);
  }
  mainWindow?.webContents.send('fixButtonProgress', `Failed to reinstall Python dependencies after ${MAX_INSTALL_RETRIES} retries.`);
};

async function executeInstallationCommand(command: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    installationProcess = spawn(command, { shell: true, detached: true });

    installationProcess.on('error', (error) => {
      reject(error);
    });

    installationProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Installation command failed with exit code ${code}`));
      }
    });
  });
}

const backupDownload = async (artroomPath?: string, mainWindow?: Electron.BrowserWindow) => {
    const options = {
      detached: true,
      shell: true,
  };

  exec('start cmd.exe /K "installer.bat"', (error, stdout, stderr) => {
    
    if (error) {
        console.error(`exec error: ${error}`);
        return;
    }
    console.log(`stdout: ${stdout}`);
    console.error(`stderr: ${stderr}`);
});
}


const downloadStarterModels = async (mainWindow: Electron.BrowserWindow, dir: string, realisticStarter: boolean, animeStarter: boolean, landscapesStarter: boolean) => {
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(path.join(dir, 'Vae'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'ControlNet'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'Lora'), { recursive: true });

  const realisticModel = "ChilloutMix.safetensors"
  const animeModel = "Counterfeit.safetensors"
  const landscapesModel = "Dreamshaper.safetensors"

  const realisticURL = "https://civitai.com/api/download/models/11745";
  const animeURL = "https://civitai.com/api/download/models/57618";
  const landscapesURL = "https://civitai.com/api/download/models/109123";

  const realisticPath = path.join(dir, realisticModel);
  const animePath = path.join(dir, animeModel);
  const landscapesPath = path.join(dir, landscapesModel);

  if (realisticStarter) {
    console.log(`DOWNLOADING FROM ${realisticURL}`)
    await download_via_https(realisticModel, realisticURL, realisticPath, mainWindow);
  }
  if (animeStarter) {
    console.log(`DOWNLOADING FROM ${animeURL}`)
    await download_via_https(animeModel, animeURL, animePath, mainWindow);
  }
  if (landscapesStarter) {
    console.log(`DOWNLOADING FROM ${landscapesURL}`)
    await download_via_https(landscapesModel, landscapesURL, landscapesPath, mainWindow);
  }
  console.log("All downloads complete!");
}

export const installerHandles = (mainWindow: Electron.BrowserWindow) => {
  ipcMain.handle('pythonInstall', (_, artroomPath) => {
    return backupPythonInstallation(mainWindow, artroomPath);
  });    
  ipcMain.handle('pythonInstallDependencies', (_, artroomPath) => {
    return reinstallPythonDependencies(artroomPath, mainWindow);
  });    
  ipcMain.handle('backupDownload', (_, artroomPath) => {
    return backupDownload(artroomPath, mainWindow);
  });    
  ipcMain.handle('downloadStarterModels', (_, dir, realisticStarter, animeStarter, landscapesStarter) => {
    return downloadStarterModels(mainWindow, dir, realisticStarter, animeStarter, landscapesStarter);
  });    
}


