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
    console.log(imageFile);

    const boundingBox = {
      ...boundingBoxCoordinates,
      ...boundingBoxDimensions,
      };
        const image = {
          category: "user",
          height: 512,
          width: 512,
          mtime: 1673399421.3987432,
          url: imageFile.path,
          kind: "image",
          layer: "base",
          x: 0,
          y: 0
        }

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
}

