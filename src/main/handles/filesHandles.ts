import axios from "axios";
import { ipcMain, shell } from "electron";
import glob from 'glob';
import path from 'path';
import fs from 'fs';
import { electronDialog } from "../utils/electronDialog";
import { MODELS_EXTENSIONS, IMAGE_EXTENSIONS, getExtname } from "../utils/extensions";
import { getExifData } from "../utils/exifData";
import { getMimeType } from "../utils/getMimeType";

const getFiles = async (folder_path: string, ext: string[], excludeFolders?: string[]) => {
  const exts = ext.join(',');

  return new Promise<string[]>((resolve) => {
    if (folder_path.length) {
      glob(`${folder_path}/**/*.{${exts}}`, {}, (err, files) => {
        if (err) {
          console.log("ERROR");
          resolve([]);
        }
        resolve(
          files.filter((match) =>
            excludeFolders ? !excludeFolders.some((folder) => match.includes(folder)) : true
          ).map((match) => path.relative(folder_path, match))
        );
      });
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

export const filesHandles = () => {
  ipcMain.handle('getCkpts', async (_, data) => {
    return getFiles(data, MODELS_EXTENSIONS, ['Loras', 'ControlNet', 'Vaes', 'upscalers']);
  });

  ipcMain.handle('getLoras', async (_, data) => {
    return getFiles(path.join(data, 'Loras'), MODELS_EXTENSIONS);
  });

  ipcMain.handle('getVaes', async (_, data) => {
    return getFiles(path.join(data, 'Vaes'), MODELS_EXTENSIONS);
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
    shell.showItemInFolder(path.resolve(data));
  });

  ipcMain.handle('getImageFromPath', async (_, data) => {
    if (data && data.length > 0) {
      return await getImage(data);
    }

    return { b64: '', metadata: '' };
  });

  ipcMain.handle("chooseImages", () => electronDialog('Files', IMAGE_EXTENSIONS));

  ipcMain.handle('chooseUploadPath', () => electronDialog('Directory'));
}
