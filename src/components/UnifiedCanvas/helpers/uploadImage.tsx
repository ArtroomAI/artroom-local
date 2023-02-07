import React from 'react';
import _ from 'lodash';
import { v4 as uuidv4 } from 'uuid';

const loadImage = async (b64: string) => {
  const image = new Image();
  image.src = b64;
  return new Promise((resolve) => {
    image.onload = () => {
      resolve({width: image.width, height: image.height});
    }
  });
}

export const uploadImage = async ({
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

      const boundingBox = {
        ...boundingBoxCoordinates,
        ...boundingBoxDimensions,
      };
      const imageData = await loadImage(imageFile.b64);
      const image = {
        category: "user",
        height: imageData['height'],
        width: imageData['width'],
        mtime: 1673399421.3987432,
        url: URL.createObjectURL(imageFile),
        kind: "image",
        layer: "base",
        ...boundingBoxCoordinates,
        x: 0,
        y: 0
      }

      const newImage = {
        uuid: uuidv4(),
        category: 'user',
        ...image,
      };
      if (layerState.objects.length === 0){
        setInitialCanvasImage(newImage)
      }
      else{
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

