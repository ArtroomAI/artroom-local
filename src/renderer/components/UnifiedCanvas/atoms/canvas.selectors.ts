import { rgbaColorToString, roundToHundred } from '../util';
import * as atoms from './canvas.atoms';
import { selector } from 'recoil';
import { isProcessingAtom } from './system.atoms';
import { isCanvasBaseImage } from './canvasTypes';

export const brushColorStringSelector = selector({
  key: 'canvas.brushSizeString.selector',
  get: ({ get }) => {
    const brushColor = get(atoms.brushColorAtom);
    return rgbaColorToString(brushColor);
  },
});

export const toolSelector = selector({
  key: 'canvas.tool.selector',
  get: ({ get }) => {
    return get(atoms.toolAtom);
  },
  set: ({ set }, toolValue) => {
    set(atoms.toolAtom, toolValue);
    if (toolValue !== 'move') {
      set(atoms.isTransformingBoundingBoxAtom, false);
      set(atoms.isMouseOverBoundingBoxAtom, false);
      set(atoms.isMovingBoundingBoxAtom, false);
      set(atoms.isMovingStageAtom, false);
    }
  },
});

export const maskColorStringSelector = selector({
  key: 'canvas.maskColorString.selector',
  get: ({ get }) => {
    return rgbaColorToString(get(atoms.maskColorAtom));
  },
});

export const stageCursorSelector = selector({
  key: 'stageCursor.selector',
  get: ({ get }) => {
    const tool = get(atoms.toolAtom);
    const isMovingStage = get(atoms.isMovingStageAtom);
    const isTransformingBoundingBox = get(atoms.isTransformingBoundingBoxAtom);
    const shouldRestrictStrokesToBox = get(
      atoms.shouldRestrictStrokesToBoxAtom
    );
    const isMouseOverBoundingBox = get(atoms.isMouseOverBoundingBoxAtom);
    const isStaging = get(atoms.isStagingSelector);

    if (tool === 'move' || isStaging) {
      if (isMovingStage) {
        return 'grabbing';
      } else {
        return 'grab';
      }
    } else if (isTransformingBoundingBox) {
      return undefined;
    } else if (shouldRestrictStrokesToBox && !isMouseOverBoundingBox) {
      return 'default';
    }
  },
});

export const isModifyingBoundingBoxSelector = selector({
  key: 'isModifyingBoundingBox.selector',
  get: ({ get }) => {
    const isTransformingBoundingBox = get(atoms.isTransformingBoundingBoxAtom);
    const isMovingBoundingBox = get(atoms.isMovingBoundingBoxAtom);
    return isTransformingBoundingBox || isMovingBoundingBox;
  },
});

export const cursorCoordinatesStringSelector = selector({
  key: 'cursorCoordinatesString.selector',
  get: ({ get }) => {
    const cursorPosition = get(atoms.cursorPositionAtom);

    const { cursorX, cursorY } = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return `(${roundToHundred(cursorX)}, ${roundToHundred(cursorY)})`;
  },
});

export const canRedoSelector = selector({
  key: 'canRedo.selector',
  get: ({ get }) => {
    const futureLayerStates = get(atoms.futureLayerStatesAtom);
    const isProcessing = get(isProcessingAtom);
    return futureLayerStates.length > 0 && !isProcessing;
  },
});

export const canUndoSelector = selector({
  key: 'canUndo.selector',
  get: ({ get }) => {
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const isProcessing = get(isProcessingAtom);
    return pastLayerStates.length > 0 && !isProcessing;
  },
});

export const colorPickerColorStringSelector = selector({
  key: 'colorPickerColorString.selector',
  get: ({ get }) => {
    return rgbaColorToString(get(atoms.colorPickerColorAtom));
  },
});

export const isStagingSelector = selector({
  key: 'isStaging.selector',
  get: ({ get }) => {
    const layerState = get(atoms.layerStateAtom);
    const isProcessing = get(isProcessingAtom);
    return layerState.stagingArea.images.length > 0 || isProcessing;
  },
});

export const initialCanvasImageSelector = selector({
  key: 'initialCanvasImage.selector',
  get: ({ get }) => {
    const layerState = get(atoms.layerStateAtom);
    return layerState.objects.find(isCanvasBaseImage);
  },
});
