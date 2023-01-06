import React from 'react';
import { ButtonGroup } from '@chakra-ui/react';
import {
  FaArrowsAlt,
  FaCopy,
  FaCrosshairs,
  FaDownload,
  FaLayerGroup,
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
} from '../../atoms/canvas.atoms';
import { isProcessingAtom } from '../../atoms/system.atoms';
import { Select, IconButton } from '../../components';
import { CanvasLayer, LAYER_NAMES_DICT } from '../../atoms/canvasTypes';
import { CanvasToolChooserOptions } from './CanvasToolChooserOptions';
import { getCanvasBaseLayer } from '../../util';
import { CanvasMaskOptions } from './CanvasMaskOptions';
import { CanvasSettingsButtonPopover } from './CanvasSettingsButtonPopover';
import { CanvasRedoButton } from './CanvasRedoButton';
import { CanvasUndoButton } from './CanvasUndoButton';

// import {
// 	resetCanvas,
// 	resetCanvasView,
// 	resizeAndScaleCanvas,
// 	setIsMaskEnabled,
// 	setLayer,
// 	setTool,
// } from 'canvas/store/canvasSlice';
// import _ from 'lodash';
// import { mergeAndUploadCanvas } from 'canvas/store/thunks/mergeAndUploadCanvas';
// import { systemSelector } from 'system/store/systemSelectors';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';

// export const selector = createSelector(
// 	[systemSelector, canvasSelector, isStagingSelector],
// 	(system, canvas, isStaging) => {
// 		const { isProcessing } = system;
// 		const { tool, shouldCropToBoundingBoxOnSave, layer, isMaskEnabled } =
// 			canvas;

// 		return {
// 			isProcessing,
// 			isStaging,
// 			isMaskEnabled,
// 			tool,
// 			layer,
// 			shouldCropToBoundingBoxOnSave,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasOutpaintingControls: FC = () => {
  // const {
  // 	isProcessing,
  // 	isStaging,
  // 	isMaskEnabled,
  // 	layer,
  // 	tool,
  // 	shouldCropToBoundingBoxOnSave,
  // } = useAppSelector(selector);
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
    ['shift+m'],
    () => {
      handleMergeVisible();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    [canvasBaseLayer, isProcessing]
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

  useHotkeys(
    ['shift+d'],
    () => {
      handleDownloadAsImage();
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

  const handleMergeVisible = () => {
    alert('mergeAndUploadCanvas action placeholder');
    // dispatch(
    // 	mergeAndUploadCanvas({
    // 		cropVisible: false,
    // 		shouldSetAsInitialImage: true,
    // 	})
    // );
  };

  const handleSaveToGallery = () => {
    alert('mergeAndUploadCanvas action placeholder');
    // dispatch(
    // 	mergeAndUploadCanvas({
    // 		cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
    // 		cropToBoundingBox: shouldCropToBoundingBoxOnSave,
    // 		shouldSaveToGallery: true,
    // 	})
    // );
  };

  const handleCopyImageToClipboard = () => {
    alert('mergeAndUploadCanvas action placeholder');
    // dispatch(
    // 	mergeAndUploadCanvas({
    // 		cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
    // 		cropToBoundingBox: shouldCropToBoundingBoxOnSave,
    // 		shouldCopy: true,
    // 	})
    // );
  };

  const handleDownloadAsImage = () => {
    alert('mergeAndUploadCanvas action placeholder');
    // dispatch(
    // 	mergeAndUploadCanvas({
    // 		cropVisible: shouldCropToBoundingBoxOnSave ? false : true,
    // 		cropToBoundingBox: shouldCropToBoundingBoxOnSave,
    // 		shouldDownload: true,
    // 	})
    // );
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
          aria-label="Merge Visible (Shift+M)"
          tooltip="Merge Visible (Shift+M)"
          icon={<FaLayerGroup />}
          onClick={handleMergeVisible}
          isDisabled={isStaging}
        />
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
          aria-label="Download as Image (Shift+D)"
          tooltip="Download as Image (Shift+D)"
          icon={<FaDownload />}
          onClick={handleDownloadAsImage}
          isDisabled={isStaging}
        />
      </ButtonGroup>
      <ButtonGroup isAttached>
        <CanvasUndoButton />
        <CanvasRedoButton />
      </ButtonGroup>

      <ButtonGroup isAttached>
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
        <CanvasSettingsButtonPopover />
      </ButtonGroup>
    </div>
  );
};
