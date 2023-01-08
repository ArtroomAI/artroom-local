import { v4 as uuidv4 } from 'uuid';
import { ImageUploadResponse, Image } from '../painter';
import { setInitialCanvasImageAction, addImageToStagingAreaAction } from '../atoms/canvas.actions';
import { useSetRecoilState } from 'recoil';

const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;

type UploadImageConfig = {
  imageFile: File;
  setInitialCanvasImage: (arg0: Image) => void
};

export const uploadImage = (config: UploadImageConfig) => {

    console.log("START")
    const { imageFile, setInitialCanvasImage } = config;

    const formData = new FormData();
    formData.append('file', imageFile, imageFile.name);
    formData.append(
      'data',
      JSON.stringify({
        kind: 'init',
      })
    );
    
    fetch(`${LOCAL_URL}/invoke_upload`, {
      method: 'POST',
      body: formData
    }).then(async response => {
        const image = await response.json() as unknown as ImageUploadResponse;
        const newImage: Image = {
          uuid: uuidv4(),
          category: 'user',
          ...image,
        };
    
        setInitialCanvasImage(newImage);
    });

   
}
