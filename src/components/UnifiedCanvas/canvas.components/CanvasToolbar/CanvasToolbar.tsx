import React from 'react';
import { ButtonGroup, createStandaloneToast } from '@chakra-ui/react';
import {
  FaArrowsAlt,
  FaCopy,
  FaCrosshairs,
  FaPlay,
  FaSave,
  FaTrash,
  FaUpload,
} from 'react-icons/fa';
import { useHotkeys } from 'react-hotkeys-hook';
import { FC, ChangeEvent } from 'react';
import { useRecoilState, useSetRecoilState, useRecoilValue } from 'recoil';
import { useImageUploader } from '../../hooks';
import {
  toolSelector,
  layerAtom,
  setIsMaskEnabledAction,
  resetCanvasAction,
  resetCanvasViewAction,
  resizeAndScaleCanvasAction,
  shouldCropToBoundingBoxOnSaveAtom,
  isStagingSelector,
  stageScaleAtom,
  boundingBoxCoordinatesAtom,
  boundingBoxDimensionsAtom,
  layerStateAtom,
  stageCoordinatesAtom,
} from '../../atoms/canvas.atoms';
import { imageSettingsState } from '../../../../atoms/atoms';
import { isProcessingAtom } from '../../atoms/system.atoms';
import { Select, IconButton } from '../../components';
import { CanvasLayer, isCanvasMaskLine, LAYER_NAMES_DICT } from '../../atoms/canvasTypes';
import { CanvasToolChooserOptions } from './CanvasToolChooserOptions';
import {copyImage, generateMask, getCanvasBaseLayer, layerToDataURL } from '../../util';
import { CanvasMaskOptions } from './CanvasMaskOptions';
import { CanvasSettingsButtonPopover } from './CanvasSettingsButtonPopover';
import { CanvasRedoButton } from './CanvasRedoButton';
import { CanvasUndoButton } from './CanvasUndoButton';
import axios from 'axios';
import path from 'path';

const LOCAL_URL = process.env.REACT_APP_LOCAL_URL;
const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;
const baseURL = LOCAL_URL;

export const CanvasOutpaintingControls: FC = () => {
  const canvasBaseLayer = getCanvasBaseLayer();

  const [tool, setTool] = useRecoilState(toolSelector);
  const [layer, setLayer] = useRecoilState(layerAtom);
  const [isMaskEnabled, setIsMaskEnabled] = useRecoilState(
    setIsMaskEnabledAction
  );
  const resetCanvas = useSetRecoilState(resetCanvasAction);
  const resetCanvasView = useSetRecoilState(resetCanvasViewAction);
  const resizeAndScaleCanvas = useSetRecoilState(resizeAndScaleCanvasAction);
  const shouldCropToBoundingBoxOnSave = useRecoilValue(
    shouldCropToBoundingBoxOnSaveAtom
  );
  const isProcessing = useRecoilValue(isProcessingAtom);
  const isStaging = useRecoilValue(isStagingSelector);

  const { openUploader } = useImageUploader();

  //For Generation
  const stageScale = useRecoilValue(stageScaleAtom);  
  const boundingBoxCoordinates = useRecoilValue(boundingBoxCoordinatesAtom);  
  const boundingBoxDimensions = useRecoilValue(boundingBoxDimensionsAtom);  
  const layerState = useRecoilValue(layerStateAtom);  
  const stageCoordinates = useRecoilValue(stageCoordinatesAtom);
  const [imageSettings, setImageSettings] = useRecoilState(imageSettingsState)
  const { ToastContainer, toast } = createStandaloneToast();

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  useHotkeys(
    ['r'],
    () => {
      handleResetCanvasView();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  useHotkeys(
    ['shift+s'],
    () => {
      handleSaveToGallery();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  useHotkeys(
    ['meta+c', 'ctrl+c'],
    () => {
      handleCopyImageToClipboard();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
  );

  const handleSelectMoveTool = () => setTool('move');
  
  const handleResetCanvasView = () => {
    const canvasBaseLayer = getCanvasBaseLayer();
    if (!canvasBaseLayer) return;
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    });
    console.log(clientRect, 'clientRect');
    resetCanvasView({
      contentRect: clientRect,
    });
  };

  const handleResetCanvas = () => {
    resetCanvas();
    resizeAndScaleCanvas();
  };

  const handleSaveToGallery = () => {
    const canvasBaseLayer = getCanvasBaseLayer();

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates
    );
    const timestamp = new Date().getTime();
    const imagePath = path.join(imageSettings.image_save_path, imageSettings.batch_name, timestamp + ".jpg");
    console.log(imagePath);
    window.api.saveFromDataURL(JSON.stringify({dataURL, imagePath}));
    toast({
      title: 'Image Saved',
      status: 'success',
      duration: 1500,
      isClosable: true,
    })
  };

  const handleCopyImageToClipboard = () => {
    const canvasBaseLayer = getCanvasBaseLayer();

    const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
      canvasBaseLayer,
      stageScale,
      stageCoordinates
    );
    window.api.copyToClipboard(dataURL);
    toast({
      title: 'Image Copied',
      status: 'success',
      duration: 1500,
      isClosable: true,
    })
  };

  const handleRunInpainting = () => {
    const canvasBaseLayer = getCanvasBaseLayer();
    const boundingBox = {
      ...boundingBoxCoordinates,
      ...boundingBoxDimensions,
    };
    const maskDataURL = generateMask(
      isMaskEnabled ? layerState.objects.filter(isCanvasMaskLine) : [],
      boundingBox
    );
  
    const tempScale = canvasBaseLayer.scale();
  
    canvasBaseLayer.scale({
      x: 1 / stageScale,
      y: 1 / stageScale,
    });
  
    const absPos = canvasBaseLayer.getAbsolutePosition();
  
    const imageDataURL = canvasBaseLayer.toDataURL({
      x: boundingBox.x + absPos.x,
      y: boundingBox.y + absPos.y,
      width: boundingBox.width,
      height: boundingBox.height,
    });

    canvasBaseLayer.scale(tempScale);
    const body = {
      ...imageSettings,
      init_image: imageDataURL,
      mask_image: maskDataURL
    };
    
    axios.post(
      `${baseURL}/add_to_queue`,
      body,
      {
        headers: { 'Content-Type': 'application/json' }
      }
      ).then(result =>{
        console.log(result);
      })
  };

  const handleChangeLayer = (e: ChangeEvent<HTMLSelectElement>) => {
    const newLayer = e.target.value as CanvasLayer;
    setLayer(newLayer);
    if (newLayer === 'mask' && !isMaskEnabled) {
      setIsMaskEnabled(true);
    }
  };

  return (
    <div className="inpainting-settings">
      <Select
        tooltip="Layer (Q)"
        tooltipProps={{ hasArrow: true, placement: 'top' }}
        value={layer}
        validValues={LAYER_NAMES_DICT}
        onChange={handleChangeLayer}
        isDisabled={isStaging}
      />

      <CanvasMaskOptions />
      <CanvasSettingsButtonPopover />

      <CanvasToolChooserOptions />

      <ButtonGroup isAttached>
        <IconButton
          aria-label="Move Tool (V)"
          tooltip="Move Tool (V)"
          icon={<FaArrowsAlt />}
          data-selected={tool === 'move' || isStaging}
          onClick={handleSelectMoveTool}
        />
        <IconButton
          aria-label="Reset View (R)"
          tooltip="Reset View (R)"
          icon={<FaCrosshairs />}
          onClick={handleResetCanvasView}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
      <IconButton
          aria-label="Save to Gallery (Shift+S)"
          tooltip="Save to Gallery (Shift+S)"
          icon={<FaSave />}
          onClick={handleSaveToGallery}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Copy to Clipboard (Cmd/Ctrl+C)"
          tooltip="Copy to Clipboard (Cmd/Ctrl+C)"
          icon={<FaCopy />}
          onClick={handleCopyImageToClipboard}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Upload"
          tooltip="Upload"
          icon={<FaUpload />}
          onClick={openUploader}
          isDisabled={isStaging}
        />
        <IconButton
          aria-label="Clear Canvas"
          tooltip="Clear Canvas"
          icon={<FaTrash />}
          onClick={handleResetCanvas}
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
          isDisabled={isStaging}
        />
      </ButtonGroup>

      <ButtonGroup isAttached>
        <CanvasUndoButton />
        <CanvasRedoButton />
      </ButtonGroup>



        <ButtonGroup isAttached>
        <IconButton
            aria-label="Run (Alt+R)"
            tooltip="Run (Alt+R)"
            icon={<FaPlay />}
            onClick={handleRunInpainting}
            isDisabled={isStaging}
          />
        </ButtonGroup>
    </div>
  );
};
