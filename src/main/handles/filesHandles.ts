import axios from "axios";
import { app, ipcMain, shell } from "electron";
import glob from 'glob';
import path from 'path';
import fs from 'fs';
import { electronDialog } from "../utils/electronDialog";
import { MODELS_EXTENSIONS, IMAGE_EXTENSIONS, getExtname } from "../utils/extensions";
import { getExifData } from "../utils/exifData";
import { getMimeType } from "../utils/getMimeType";
import { FileWatcher } from "../utils/fileWatcher";

const getFiles = async (folder_path: string, ext: string[], excludeFolders?: string[]) => {
  const exts = ext.join(',');

  if (!fs.existsSync(folder_path)) {
    fs.mkdirSync(folder_path, { recursive: true });
  }

  return new Promise<string[]>((resolve) => {
    if (folder_path.length) {
      glob(`${folder_path}/**/*.{${exts}}`, {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        } else if (!files || !files.length) {
          resolve([]);
        } else {
          resolve(
            files
              .filter((match) =>
                excludeFolders ? !excludeFolders.some((folder) => match.includes(folder)) : true
              )
              .map((match) => path.relative(folder_path, match))
          );
        }
      });
    } else {
      resolve([]);
    }
  });
};


async function getImage(image_path: string) {
  return fs.promises.readFile(image_path).then(buffer => {
    const ext = getExtname(image_path);
    const mimeType = getMimeType(ext);
    const exifData = getExifData(buffer, ext);

    const b64 = `data:${mimeType};base64,${buffer.toString('base64')}`;

    return { b64, metadata: exifData };
  });
}

export const filesHandles = (mainWindow: Electron.BrowserWindow) => {
  const modelsWatcher = new FileWatcher();

  const getModels = async (folder: string) => ({
    ckpts: await getFiles(folder, MODELS_EXTENSIONS, ['Lora', 'ControlNet', 'Vae', 'upscalers', 'Embeddings']),
    loras: await getFiles(path.join(folder, 'Lora'), MODELS_EXTENSIONS),
    vaes: await getFiles(path.join(folder, 'Vae'), MODELS_EXTENSIONS)
  })

  const modelsWatcherCallback = (folder: string) => async () => {
    mainWindow.webContents.send('modelsChange', await getModels(folder));
  }

  ipcMain.handle('modelsFolder', (_, folder_path: string) => {
    if(!fs.existsSync(folder_path)) {
      folder_path = app.getPath('home');
    }
    
    modelsWatcher.reassignWatcher(folder_path, modelsWatcherCallback(folder_path));

    return getModels(folder_path);
  });

  ipcMain.handle('saveFromDataURL', async (_, data) => {
    const json = JSON.parse(data);
    const dataUrl = json.dataURL;
    const imagePath = json.imagePath;

    fs.mkdirSync(path.dirname(imagePath), { recursive: true });

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

  ipcMain.handle('showInExplorer', async (_, data) => {
    shell.openPath(path.resolve(data));
  });

  ipcMain.handle('getImageFromPath', async (_, image_path: string) => {
    if (image_path) {
      return getImage(image_path);
    }

    return { b64: '', metadata: null };
  });

  ipcMain.handle("chooseImages", () => electronDialog('Files', IMAGE_EXTENSIONS));

  ipcMain.handle('chooseUploadPath', async () => {
    const dirs = await electronDialog('Directory');
    return dirs[0];
  });
}
