/* eslint-disable no-mixed-spaces-and-tabs */
import { atom } from 'recoil';
import { CanvasLayerState, CanvasTool } from './canvasTypes';
import _ from 'lodash';

export const initialLayerState: CanvasLayerState = {
  objects: [],
  stagingArea: {
    images: [],
    selectedImageIndex: -1,
  },
};

// Not used states from redux that not present here:
// boundingBoxPreviewFill, inpaintReplace, shouldLockBoundingBox, shouldShowBrushPreview, shouldShowCheckboardTransparency, shouldUseInpaintReplace

export const boundingBoxCoordinatesAtom = atom({
  key: 'canvas.boundingBoxCoordinates',
  default: { x: 0, y: 0 },
});

export const boundingBoxDimensionsAtom = atom({
  key: 'canvas.boundingBoxDimensions',
  default: { width: 512, height: 512 },
});

export const boundingBoxScaleMethodAtom = atom({
  key: 'canvas.boundingBoxScaleMethod',
  default: 'auto',
});

export const brushColorAtom = atom({
  key: 'canvas.brushColor',
  default: { r: 90, g: 90, b: 255, a: 1 },
});

export const brushSizeAtom = atom({
  key: 'canvas.brushSize',
  default: 50,
});

export const canvasContainerDimensionsAtom = atom({
  key: 'canvas.canvasContainerDimensions',
  default: { width: 0, height: 0 },
});

export const colorPickerColorAtom = atom({
  key: 'canvas.colorPickerColor',
  default: { r: 90, g: 90, b: 255, a: 1 },
});

export const cursorPositionAtom = atom<null | { x: number; y: number }>({
  key: 'canvas.cursorPosition',
  default: null,
});

export const doesCanvasNeedScalingAtom = atom({
  key: 'canvas.doesCanvasNeedScaling',
  default: false,
});

export const futureLayerStatesAtom = atom({
  key: 'canvas.futureLayerStates',
  default: [] as CanvasLayerState[],
});

export const isCanvasInitializedAtom = atom({
  key: 'canvas.isCanvasInitialized',
  default: false,
});

export const isDrawingAtom = atom({
  key: 'canvas.isDrawing',
  default: false,
});

export const isMaskEnabledAtom = atom({
  key: 'canvas.isMaskEnabled',
  default: false,
});

export const isMouseOverBoundingBoxAtom = atom({
  key: 'canvas.isMouseOverBoundingBox',
  default: false,
});

export const isMoveBoundingBoxKeyHeldAtom = atom({
  key: 'canvas.isMoveBoundingBoxKeyHeld',
  default: false,
});

export const isMoveStageKeyHeldAtom = atom({
  key: 'canvas.isMoveStageKeyHeld',
  default: false,
});

export const isMovingBoundingBoxAtom = atom({
  key: 'canvas.isMovingBoundingBox',
  default: false,
});

export const isMovingStageAtom = atom({
  key: 'canvas.isMovingStage',
  default: false,
});

export const isTransformingBoundingBoxAtom = atom({
  key: 'canvas.isTransformingBoundingBox',
  default: false,
});

export const layerAtom = atom({
  key: 'canvas.layer',
  default: 'mask',
});

export const layerStateAtom = atom({
  key: 'canvas.layerState',
  default: initialLayerState,
});

export const maskColorAtom = atom({
  key: 'canvas.maskColor',
  default: { r: 255, g: 90, b: 90, a: 1 },
});

export const maxHistoryAtom = atom({
  key: 'canvas.maxHistory',
  default: 128,
});

export const minimumStageScaleAtom = atom({
  key: 'canvas.minimumStageScale',
  default: 1,
});

export const pastLayerStatesAtom = atom({
  key: 'canvas.pastLayerStates',
  default: [] as CanvasLayerState[],
});

export const scaledBoundingBoxDimensionsAtom = atom({
  key: 'canvas.scaledBoundingBoxDimensions',
  default: { width: 512, height: 512 },
});

export const shouldAutoSaveAtom = atom({
  key: 'canvas.shouldAutoSave',
  default: false,
});

export const shouldCropToBoundingBoxOnSaveAtom = atom({
  key: 'canvas.shouldCropToBoundingBoxOnSave',
  default: false,
});

export const shouldDarkenOutsideBoundingBoxAtom = atom({
  key: 'canvas.shouldDarkenOutsideBoundingBox',
  default: false,
});

export const shouldPreserveMaskedAreaAtom = atom({
  key: 'canvas.shouldPreserveMaskedArea',
  default: false,
});

export const shouldRestrictStrokesToBoxAtom = atom({
  key: 'canvas.shouldRestrictStrokesToBox',
  default: false,
});

export const shouldShowBoundingBoxAtom = atom({
  key: 'canvas.shouldShowBoundingBox',
  default: true,
});

export const shouldShowBrushAtom = atom({
  key: 'canvas.shouldShowBrush',
  default: true,
});

export const shouldShowCanvasDebugInfoAtom = atom({
  key: 'canvas.shouldShowCanvasDebugInfo',
  default: false,
});

export const shouldShowGridAtom = atom({
  key: 'canvas.shouldShowGrid',
  default: true,
});

export const shouldShowIntermediatesAtom = atom({
  key: 'canvas.shouldShowIntermediates',
  default: true,
});

export const shouldShowStagingImageAtom = atom({
  key: 'canvas.shouldShowStagingImage',
  default: true,
});

export const shouldShowStagingOutlineAtom = atom({
  key: 'canvas.shouldShowStagingOutline',
  default: true,
});

export const shouldSnapToGridAtom = atom({
  key: 'canvas.shouldSnapToGrid',
  default: true,
});

export const stageCoordinatesAtom = atom({
  key: 'canvas.stageCoordinates',
  default: { x: 0, y: 0 },
});

export const stageDimensionsAtom = atom({
  key: 'canvas.stageDimensions',
  default: { width: 0, height: 0 },
});

export const stageScaleAtom = atom({
  key: 'canvas.stageScale',
  default: 1,
});

export const toolAtom = atom({
  key: 'canvas.tool',
  default: 'brush' as CanvasTool,
});

export * from './canvas.toolPreview.selectors';
export * from './canvas.statusText.selectors';
export * from './canvas.actions';
export * from './canvas.selectors';
