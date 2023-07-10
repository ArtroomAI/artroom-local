import fs from 'fs';
import path from 'path';
import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ipcMain } from "electron";
import yauzl from "yauzl";
import axios from 'axios';
import { pipeline as _pipeline } from 'stream';
import { promisify } from 'util';
const pipeline = promisify(_pipeline);

let installationProcess: ChildProcessWithoutNullStreams;

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
        if (chunk_counter === 5000) {
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

const backupPythonInstallation = async (mainWindow: Electron.BrowserWindow, artroomPath: string, gpuType: string) => {
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
};

const reinstallPythonDependencies = (artroomPath: string, mainWindow?: Electron.BrowserWindow) => {
  return new Promise<string>((resolve) => {
    console.log("REINSTALLING DEPENDENCIES");
    const PATH = path.join(artroomPath, "artroom\\artroom_backend");
    const PATH_requirements = path.resolve('requirements.txt');
    const installationCommand = `"${PATH}/python.exe" -m pip install --upgrade pip && "${PATH}/python.exe" -m pip install -r "${PATH_requirements}" && pause && set /p choice= "Finished! Please exit out of this window or press enter to close"`;

    let installationProcess = spawn(installationCommand, { shell: true, detached: true });

    installationProcess.stdout.on('data', function(data) {
      console.log("Child data: " + data);
      resolve('');
    });

    installationProcess.on('error', function () {
      console.log("Failed to start child.");
      resolve('');
    });

    installationProcess.on('close', function (code) {
      console.log('Child process exited with code ' + code);
      mainWindow?.webContents.send('fixButtonProgress', `Finished downloading and installing required files!`);
      resolve('');
    });

    installationProcess.stderr.on('data', function (err) {
      console.log(`error: ${err}`);
      resolve('');
    });

    installationProcess.on('message', (msg) => {
      console.log(`msg ${msg}`);
      resolve('');
    });

    installationProcess.stderr.on('message', (msg) => {
      console.log(`ermsg ${msg}`);
      resolve('');
    });

    installationProcess.stdout.on('end', function () {
      console.log('Finished collecting data chunks.');
      mainWindow?.webContents.send('fixButtonProgress', `Finished downloading and installing required files!`);
      resolve('');
    });
  });
}


const downloadStarterModels = async (mainWindow: Electron.BrowserWindow, dir: string, realisticStarter: boolean, animeStarter: boolean, landscapesStarter: boolean) => {
  fs.mkdirSync(dir, { recursive: true });
  fs.mkdirSync(path.join(dir, 'Vae'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'ControlNet'), { recursive: true });
  fs.mkdirSync(path.join(dir, 'Lora'), { recursive: true });

  const bucketPath = "https://pub-060d7c8cf5e64af8b884ebb86d34de1a.r2.dev/models/models/"
  const realisticModel = "UmiAIMythologyAndBabes_aphroditeRealisticV1.safetensors"
  const animeModel = "UmiAIMythologyAndBabes_macrossAnimeUltimate1.safetensors"
  const landscapesModel = "UmiAIMythologyAndBabes_olympusLandscapesV10.safetensors"

  const realisticURL = path.join(bucketPath, realisticModel);
  const animeURL = path.join(bucketPath, animeModel);
  const landscapesURL = path.join(bucketPath, landscapesModel);

  const realisticPath = path.join(dir, realisticModel);
  const animePath = path.join(dir, animeModel);
  const landscapesPath = path.join(dir, landscapesModel);

  if (realisticStarter) {
    console.log(`DOWNLOADING FROM ${realisticURL}`)
    await download_via_https('Realistic Model', realisticURL, realisticPath, mainWindow);
  }
  if (animeStarter) {
    console.log(`DOWNLOADING FROM ${animeURL}`)
    await download_via_https('Anime Model', animeURL, animePath, mainWindow);
  }
  if (landscapesStarter) {
    console.log(`DOWNLOADING FROM ${landscapesURL}`)
    await download_via_https('Landscape Model', landscapesURL, landscapesPath, mainWindow);
  }
  console.log("All downloads complete!");
}

export const installerHandles = (mainWindow: Electron.BrowserWindow) => {
  ipcMain.handle('pythonInstall', (_, artroomPath, gpuType) => {
    return backupPythonInstallation(mainWindow, artroomPath, gpuType);
  });    
  ipcMain.handle('pythonInstallDependencies', (_, artroomPath) => {
    return reinstallPythonDependencies(artroomPath);
  });    
  ipcMain.handle('downloadStarterModels', (_, dir, realisticStarter, animeStarter, landscapesStarter) => {
    return downloadStarterModels(mainWindow, dir, realisticStarter, animeStarter, landscapesStarter);
  });    
}
