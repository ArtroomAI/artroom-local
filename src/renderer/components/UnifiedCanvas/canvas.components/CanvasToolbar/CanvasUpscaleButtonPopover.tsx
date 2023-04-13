import React, { FC, useCallback, useContext, useState } from 'react';
import { Button, Flex, useToast } from '@chakra-ui/react';
import { useRecoilValue } from 'recoil';
import { Popover, IconButton, Slider, Select } from '../../components';
import { GiResize } from 'react-icons/gi';
import { getCanvasBaseLayer, layerToDataURL } from '../../util';
import {stageScaleAtom, stageCoordinatesAtom } from '../../atoms/canvas.atoms';
import { SocketContext } from '../../../../socket';
import { imageSavePathState, modelsDirState } from '../../../../SettingsManager';

export const CanvasUpscaleButtonPopover: FC = () => {
  const toast = useToast({});
  const [upscaler, setUpscaler] = useState('RealESRGAN');
  const [upscale_factor, setUpscaleFactor] = useState(2);
  const stageScale = useRecoilValue(stageScaleAtom);   
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom);
  const socket = useContext(SocketContext);
  const imageSavePath = useRecoilValue(imageSavePathState);
  const modelsDir = useRecoilValue(modelsDirState);

  const sendUpscale = useCallback(() => {
    toast({
      title: 'Recieved!',
      description: 'You\'ll get a notification 🔔 when your upscale is ready!',
      status: 'success',
      position: 'top',
      duration: 1500,
      isClosable: false,
      containerStyle: {
        pointerEvents: 'none'
      }
    });
    const canvasBaseLayer = getCanvasBaseLayer();

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates
    );
    
    const output = {
      upscale_images: [dataURL],
      upscaler,
      upscale_factor,
      upscale_strength: 0.5,
      upscale_dest: "",
      image_save_path: imageSavePath,
      models_dir: modelsDir
    };

    socket.emit('upscale', output);
    
  }, [imageSavePath, modelsDir, socket, stageCoordinates, stageScale, toast, upscale_factor, upscaler]);

  return (
    <Popover
      trigger="hover"
      triggerComponent={
        <IconButton
          tooltip="Upscale Canvas"
          aria-label="Upscale Canvas"
          icon={<GiResize />}
        />
      }
    >
        <Flex minWidth="15rem" direction="column" gap="1rem" width="100%">
            <Slider
              step={0.5}
              label="Size"
              value={upscale_factor}
              withInput
              max={4}
              onChange={(newSize) => setUpscaleFactor(newSize)}
              sliderNumberInputProps={{ max: 4 }}
              inputReadOnly={false}
            />
            <Select
                label="Upscaler"
                id="upscaler"
                name="upscaler"
                onChange={(event) => setUpscaler(event.target.value)}
                value={upscaler}
                variant="outline"
                w="300px" 
                validValues={["RealESRGAN","RealESRGAN-Anime","GFPGANv1.3","GFPGANv1.4","RestoreFormer"]}                  
              />                 
          <Button onClick={sendUpscale}>Upscale</Button>
        </Flex>
    </Popover>
  );
};
