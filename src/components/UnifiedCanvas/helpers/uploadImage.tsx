import React from 'react';
import _ from 'lodash';
import { v4 as uuidv4 } from 'uuid';
import { ImageUploadResponse, Image } from '../painter';

const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;

export const uploadImage = ({
              imageFile, 
              boundingBoxCoordinates, 
              boundingBoxDimensions,
              setPastLayerStates,
              pastLayerStates,
              layerState,
              maxHistory,
              setLayerState,
              setFutureLayerStates,
              setInitialCanvasImage}) => {

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
        if (layerState.objects.length == 0){
          setInitialCanvasImage(newImage)
        }
        else{
          const boundingBox = {
            ...boundingBoxCoordinates,
            ...boundingBoxDimensions,
          };        
          setPastLayerStates([
            ...pastLayerStates,
            _.cloneDeep(layerState),
          ]);
      
          if (pastLayerStates.length > maxHistory) {
            setPastLayerStates(pastLayerStates.slice(1));
          }
          setLayerState({
            ...layerState,
            objects: [
                ...layerState.objects,                   
                {
                  kind: 'image',
                  layer: 'base',
                  ...boundingBox,
                  image: newImage,
                },
              ],
          });
          setFutureLayerStates([]);        
        }
       
    });

   
}

