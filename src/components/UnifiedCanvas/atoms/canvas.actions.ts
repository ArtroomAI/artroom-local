import {
  CanvasBaseLine,
  CanvasImage,
  CanvasLayerState,
  CanvasMaskLine,
  CanvasObject,
  Dimensions,
  isCanvasAnyLine,
  isCanvasBaseImage,
  isCanvasMaskLine,
} from './canvasTypes';
import * as atoms from './canvas.atoms';
import { selector } from 'recoil';
import {
  calculateCoordinates,
  calculateScale,
  floorCoordinates,
  getScaledBoundingBoxDimensions,
  roundDimensionsTo64,
  roundDownToMultiple,
  roundToMultiple,
  STAGE_PADDING_PERCENTAGE,
} from '../util';
import _ from 'lodash';
import { IRect, Vector2d } from 'konva/lib/types';
import { Image } from '../painter';

export const setInitialCanvasImageAction = selector<Image | null>({
  key: 'canvas.setInitialCanvasImage.action',
  get: () => {
    return null;
  },
  set: ({ get, set }, newImage) => {
    const stageDimensions = get(atoms.stageDimensionsAtom);
    const boundingBoxScaleMethod = get(atoms.boundingBoxScaleMethodAtom);
    const newBoundingBoxDimensions = {
      width: roundDownToMultiple(
        _.clamp((newImage as Image).width, 64, 512),
        64
      ),
      height: roundDownToMultiple(
        _.clamp((newImage as Image).height, 64, 512),
        64
      ),
    };
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);

    const newBoundingBoxCoordinates = {
      x: roundToMultiple(
        (newImage as Image).width / 2 - newBoundingBoxDimensions.width / 2,
        64
      ),
      y: roundToMultiple(
        (newImage as Image).height / 2 - newBoundingBoxDimensions.height / 2,
        64
      ),
    };

    if (boundingBoxScaleMethod === 'auto') {
      const scaledDimensions = getScaledBoundingBoxDimensions(
        newBoundingBoxDimensions
      );
      set(atoms.scaledBoundingBoxDimensionsAtom, scaledDimensions);
    }

    set(atoms.boundingBoxDimensionsAtom, newBoundingBoxDimensions);
    set(atoms.boundingBoxCoordinatesAtom, newBoundingBoxCoordinates);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    set(atoms.layerStateAtom, {
      ...atoms.initialLayerState,
      objects: [
        {
          kind: 'image',
          layer: 'base',
          x: 0,
          y: 0,
          width: (newImage as Image).width,
          height: (newImage as Image).height,
          image: newImage as Image,
        },
      ],
    });

    set(atoms.futureLayerStatesAtom, []);

    set(atoms.isCanvasInitializedAtom, false);

    const newScale = calculateScale(
      stageDimensions.width,
      stageDimensions.height,
      (newImage as Image).width,
      (newImage as Image).height,
      STAGE_PADDING_PERCENTAGE
    );

    const newCoordinates = calculateCoordinates(
      stageDimensions.width,
      stageDimensions.height,
      0,
      0,
      (newImage as Image).width,
      (newImage as Image).height,
      newScale
    );

    set(atoms.stageScaleAtom, newScale);
    set(atoms.stageCoordinatesAtom, newCoordinates);
    set(atoms.doesCanvasNeedScalingAtom, true);
  },
});

type AddImagePayload = {
  boundingBox: IRect;
  image: Image;
};

export const addImageToStagingAreaAction = selector<AddImagePayload | void>({
  key: 'canvas.addImageToStagingArea.action',
  get: () => {
    return;
  },
  set: ({ set, get }, newImage) => {
    const { boundingBox, image } = newImage as AddImagePayload;

    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    if (!boundingBox || !image) return;

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: {
        ...layerState.stagingArea,
        images: [
          ...layerState.stagingArea.images,
          {
            kind: 'image',
            layer: 'base',
            ...boundingBox,
            image,
          },
        ],
      },
    } as CanvasLayerState);
    // TODO Questionable if will work the same way in recoil as in redux
    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: {
        ...layerState.stagingArea,
        selectedImageIndex: layerState.stagingArea.images.length - 1,
      },
    });

    set(atoms.futureLayerStatesAtom, []);
  },
});

export const setMergedCanvasAction = selector<CanvasImage | null>({
  key: 'canvas.setMergedCanvas.action',
  get: () => {
    return null;
  },
  set: ({ set, get }, newValue) => {
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    set(atoms.futureLayerStatesAtom, []);

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: [newValue as CanvasObject],
    });
  },
});

export const addLineAction = selector<number[]>({
  key: 'canvas.addLine.action',
  get: () => {
    return [];
  },
  set: ({ set, get }, payload) => {
    const tool = get(atoms.toolAtom);
    const layer = get(atoms.layerAtom);
    const brushColor = get(atoms.brushColorAtom);
    const brushSize = get(atoms.brushSizeAtom);
    const shouldRestrictStrokesToBox = get(
      atoms.shouldRestrictStrokesToBoxAtom
    );
    const layerState = get(atoms.layerStateAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const maxHistory = get(atoms.maxHistoryAtom);
    const boundingBoxCoordinates = get(atoms.boundingBoxCoordinatesAtom);
    const boundingBoxDimensions = get(atoms.boundingBoxDimensionsAtom);

    if (tool === 'move' || tool === 'colorPicker') return;

    const newStrokeWidth = brushSize / 2;

    // set & then spread this to only conditionally add the "color" key
    const newColor =
      layer === 'base' && tool === 'brush' ? { color: brushColor } : {};

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    const newLine: CanvasMaskLine | CanvasBaseLine = {
      kind: 'line',
      layer: layer as 'base' | 'mask',
      tool,
      strokeWidth: newStrokeWidth,
      points: payload as number[],
      ...newColor,
    };

    if (shouldRestrictStrokesToBox) {
      newLine.clip = {
        ...boundingBoxCoordinates,
        ...boundingBoxDimensions,
      };
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: [...layerState.objects, newLine],
    });
    set(atoms.futureLayerStatesAtom, []);
  },
});

export const clearMaskAction = selector<void>({
  key: 'canvas.clearMask.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const layerState = get(atoms.layerStateAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);
    set(atoms.layerStateAtom, {
      ...layerState,
      objects: layerState.objects.filter((obj) => !isCanvasMaskLine(obj)),
    });
    set(atoms.futureLayerStatesAtom, []);
    set(atoms.shouldPreserveMaskedAreaAtom, false);
  },
});

export const setIsMaskEnabledAction = selector<boolean>({
  key: 'canvas.setIsMaskEnabled.action',
  get: ({ get }) => {
    return get(atoms.isMaskEnabledAtom);
  },
  set: ({ set }, newValue) => {
    set(atoms.isMaskEnabledAtom, newValue);
    set(atoms.layerAtom, newValue ? 'mask' : 'sketch');
  },
});

export const setBoundingBoxDimensionsAction = selector<Dimensions>({
  key: 'canvas.setBoundingBoxDimensions.action',
  get: ({ get }) => {
    return get(atoms.boundingBoxDimensionsAtom);
  },
  set: ({ set, get }, newValue) => {
    const boundingBoxScaleMethod = get(atoms.boundingBoxScaleMethodAtom);

    const newDimensions = roundDimensionsTo64(newValue as Dimensions);
    set(atoms.boundingBoxDimensionsAtom, newDimensions);

    if (boundingBoxScaleMethod === 'auto') {
      const scaledDimensions = getScaledBoundingBoxDimensions(newDimensions);
      set(atoms.scaledBoundingBoxDimensionsAtom, scaledDimensions);
    }
  },
});

export const setBoundingBoxCoordinatesAction = selector<Vector2d>({
  key: 'canvas.setBoundingBoxCoordinates.action',
  get: ({ get }) => {
    return get(atoms.boundingBoxCoordinatesAtom);
  },
  set: ({ set }, newValue) => {
    set(
      atoms.boundingBoxCoordinatesAtom,
      floorCoordinates(newValue as Vector2d)
    );
  },
});

export const clearCanvasHistoryAction = selector<void>({
  key: 'canvas.clearCanvasHistory.action',
  get: () => {
    return;
  },
  set: ({ set }) => {
    set(atoms.pastLayerStatesAtom, []);
    set(atoms.futureLayerStatesAtom, []);
  },
});

export const discardStagedImagesAction = selector<void>({
  key: 'canvas.discardStagedImages.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: { ...atoms.initialLayerState.stagingArea },
    });

    set(atoms.futureLayerStatesAtom, []);
    set(atoms.shouldShowStagingOutlineAtom, true);
  },
});

export const addFillRectAction = selector<void>({
  key: 'canvas.addFillRect.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const boundingBoxCoordinates = get(atoms.boundingBoxCoordinatesAtom);
    const boundingBoxDimensions = get(atoms.boundingBoxDimensionsAtom);
    const brushColor = get(atoms.brushColorAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: [
        ...layerState.objects,
        {
          kind: 'fillRect',
          layer: 'base',
          ...boundingBoxCoordinates,
          ...boundingBoxDimensions,
          color: brushColor,
        },
      ],
    } as CanvasLayerState);

    set(atoms.futureLayerStatesAtom, []);
  },
});

export const addEraseRectAction = selector<void>({
  key: 'canvas.addEraseRect.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const boundingBoxCoordinates = get(atoms.boundingBoxCoordinatesAtom);
    const boundingBoxDimensions = get(atoms.boundingBoxDimensionsAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: [
        ...layerState.objects,
        {
          kind: 'eraseRect',
          layer: 'base',
          ...boundingBoxCoordinates,
          ...boundingBoxDimensions,
        },
      ],
    } as CanvasLayerState);

    set(atoms.futureLayerStatesAtom, []);
  },
});

export const mouseLeftCanvasAction = selector<void>({
  key: 'canvas.mouseLeftCanvas.action',
  get: () => {
    return;
  },
  set: ({ set }) => {
    set(atoms.cursorPositionAtom, null);
    set(atoms.isDrawingAtom, false);
    set(atoms.isMouseOverBoundingBoxAtom, false);
    set(atoms.isMovingBoundingBoxAtom, false);
    set(atoms.isTransformingBoundingBoxAtom, false);
  },
});

export const resetCanvasInteractionStateAction = selector<void>({
  key: 'canvas.resetCanvasInteractionState.action',
  get: () => {
    return;
  },
  set: ({ set }) => {
    set(atoms.cursorPositionAtom, null);
    set(atoms.isDrawingAtom, false);
    set(atoms.isMouseOverBoundingBoxAtom, false);
    set(atoms.isMoveBoundingBoxKeyHeldAtom, false);
    set(atoms.isMoveStageKeyHeldAtom, false);
    set(atoms.isMovingBoundingBoxAtom, false);
    set(atoms.isMovingStageAtom, false);
    set(atoms.isTransformingBoundingBoxAtom, false);
  },
});

export const addPointToCurrentLineAction = selector<number[]>({
  key: 'canvas.addPointToCurrentLine.action',
  get: () => {
    return [];
  },
  set: ({ set, get }, newValue) => {
    const layerState = get(atoms.layerStateAtom);

    const lastLine = layerState.objects.findLast(isCanvasAnyLine);
    const lastLineIndex = layerState.objects.findLastIndex(isCanvasAnyLine);

    if (!lastLine) return;

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: layerState.objects.map((elem, index) => {
        if (index === lastLineIndex) {
          return {
            ...elem,
            points: [...(elem as any).points, ...(newValue as number[])],
          };
        } else {
          return elem;
        }
      }),
    });
  },
});

export const commitColorPickerColorAction = selector<void>({
  key: 'canvas.commitColorPickerColor.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const colorPickerColor = get(atoms.colorPickerColorAtom);
    const brushColor = get(atoms.brushColorAtom);
    set(atoms.brushColorAtom, {
      ...colorPickerColor,
      a: brushColor.a,
    });
    set(atoms.toolAtom, 'brush');
  },
});

export const commitStagingAreaImageAction = selector<void>({
  key: 'canvas.commitStagingAreaImage.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const layerState = get(atoms.layerStateAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const maxHistory = get(atoms.maxHistoryAtom);
    const { images, selectedImageIndex } = layerState.stagingArea;

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, {
      ...layerState,
      objects: [...layerState.objects, { ...images[selectedImageIndex] }],
    });

    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: { ...atoms.initialLayerState.stagingArea },
    });

    set(atoms.futureLayerStatesAtom, []);
    set(atoms.shouldShowStagingOutlineAtom, true);
    set(atoms.shouldShowStagingImageAtom, true);
  },
});

export const nextStagingAreaImageAction = selector<void>({
  key: 'canvas.nextStagingAreaImage.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const layerState = get(atoms.layerStateAtom);

    const currentIndex = layerState.stagingArea.selectedImageIndex;
    const length = layerState.stagingArea.images.length;

    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: {
        ...layerState.stagingArea,
        selectedImageIndex: Math.min(currentIndex + 1, length - 1),
      },
    });
  },
});

export const prevStagingAreaImageAction = selector<void>({
  key: 'canvas.prevStagingAreaImage.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const layerState = get(atoms.layerStateAtom);

    const currentIndex = layerState.stagingArea.selectedImageIndex;

    set(atoms.layerStateAtom, {
      ...layerState,
      stagingArea: {
        ...layerState.stagingArea,
        selectedImageIndex: Math.max(currentIndex - 1, 0),
      },
    });
  },
});

export const resetCanvasAction = selector<void>({
  key: 'canvas.redoAction.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    set(atoms.layerStateAtom, atoms.initialLayerState);

    set(atoms.futureLayerStatesAtom, []);
  },
});

export const resetCanvasViewAction = selector<{
  contentRect: IRect;
}>({
  key: 'canvas.resetCanvasView.action',
  get: () => {
    return {
      contentRect: {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
      },
    };
  },
  set: ({ get, set }, newValue) => {
    const { contentRect } = newValue as {
      contentRect: IRect;
    };
    const stageDimensions = get(atoms.stageDimensionsAtom);
    const boundingBoxScaleMethod = get(atoms.boundingBoxScaleMethodAtom);

    const { width: stageWidth, height: stageHeight } = stageDimensions;

    const { x, y, width, height } = contentRect;

    if (width !== 0 && height !== 0) {
      const newScale = calculateScale(
        stageWidth,
        stageHeight,
        width,
        height,
        STAGE_PADDING_PERCENTAGE
      );

      const newCoordinates = calculateCoordinates(
        stageWidth,
        stageHeight,
        x,
        y,
        width,
        height,
        newScale
      );

      set(atoms.stageScaleAtom, newScale);
      set(atoms.stageCoordinatesAtom, newCoordinates);
    } else {
      const newScale = calculateScale(
        stageWidth,
        stageHeight,
        512,
        512,
        STAGE_PADDING_PERCENTAGE
      );

      const newCoordinates = calculateCoordinates(
        stageWidth,
        stageHeight,
        0,
        0,
        512,
        512,
        newScale
      );

      const newBoundingBoxDimensions = { width: 512, height: 512 };

      set(atoms.stageScaleAtom, newScale);
      set(atoms.stageCoordinatesAtom, newCoordinates);
      set(atoms.boundingBoxCoordinatesAtom, { x: 0, y: 0 });
      set(atoms.boundingBoxDimensionsAtom, newBoundingBoxDimensions);

      if (boundingBoxScaleMethod === 'auto') {
        const scaledDimensions = getScaledBoundingBoxDimensions(
          newBoundingBoxDimensions
        );
        set(atoms.scaledBoundingBoxDimensionsAtom, scaledDimensions);
      }
    }
  },
});

export const resizeAndScaleCanvasAction = selector<void>({
  key: 'canvas.resizeAndScaleCanvas.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const canvasContainerDimensions = get(atoms.canvasContainerDimensionsAtom);
    const layerState = get(atoms.layerStateAtom);
    const boundingBoxScaleMethod = get(atoms.boundingBoxScaleMethodAtom);

    const { width: containerWidth, height: containerHeight } =
      canvasContainerDimensions;

    const initialCanvasImage = layerState.objects.find(isCanvasBaseImage);

    const newStageDimensions = {
      width: Math.floor(containerWidth),
      height: Math.floor(containerHeight),
    };

    if (!initialCanvasImage) {
      const newScale = calculateScale(
        newStageDimensions.width,
        newStageDimensions.height,
        512,
        512,
        STAGE_PADDING_PERCENTAGE
      );

      const newCoordinates = calculateCoordinates(
        newStageDimensions.width,
        newStageDimensions.height,
        0,
        0,
        512,
        512,
        newScale
      );

      const newBoundingBoxDimensions = { width: 512, height: 512 };

      set(atoms.stageScaleAtom, newScale);
      set(atoms.stageCoordinatesAtom, newCoordinates);
      set(atoms.stageDimensionsAtom, newStageDimensions);
      set(atoms.boundingBoxCoordinatesAtom, { x: 0, y: 0 });
      set(atoms.boundingBoxDimensionsAtom, newBoundingBoxDimensions);

      if (boundingBoxScaleMethod === 'auto') {
        const scaledDimensions = getScaledBoundingBoxDimensions(
          newBoundingBoxDimensions
        );
        set(atoms.scaledBoundingBoxDimensionsAtom, scaledDimensions);
      }

      return;
    }

    const { width: imageWidth, height: imageHeight } = initialCanvasImage;

    const padding = 0.95;

    const newScale = calculateScale(
      containerWidth,
      containerHeight,
      imageWidth,
      imageHeight,
      padding
    );

    const newCoordinates = calculateCoordinates(
      newStageDimensions.width,
      newStageDimensions.height,
      0,
      0,
      imageWidth,
      imageHeight,
      newScale
    );

    set(atoms.minimumStageScaleAtom, newScale);
    set(atoms.stageScaleAtom, newScale);
    set(atoms.stageCoordinatesAtom, floorCoordinates(newCoordinates));
    set(atoms.stageDimensionsAtom, newStageDimensions);

    set(atoms.isCanvasInitializedAtom, true);
  },
});

export const resizeCanvasAction = selector<void>({
  key: 'canvas.resizeCanvas.action',
  get: () => {
    return;
  },
  set: ({ set, get }) => {
    const canvasContainerDimensions = get(atoms.canvasContainerDimensionsAtom);
    const layerState = get(atoms.layerStateAtom);
    const boundingBoxScaleMethod = get(atoms.boundingBoxScaleMethodAtom);

    const { width: containerWidth, height: containerHeight } =
      canvasContainerDimensions;

    const newStageDimensions = {
      width: Math.floor(containerWidth),
      height: Math.floor(containerHeight),
    };

    set(atoms.stageDimensionsAtom, newStageDimensions);

    if (!layerState.objects.find(isCanvasBaseImage)) {
      const newScale = calculateScale(
        newStageDimensions.width,
        newStageDimensions.height,
        512,
        512,
        STAGE_PADDING_PERCENTAGE
      );

      const newCoordinates = calculateCoordinates(
        newStageDimensions.width,
        newStageDimensions.height,
        0,
        0,
        512,
        512,
        newScale
      );

      const newBoundingBoxDimensions = { width: 512, height: 512 };

      set(atoms.stageScaleAtom, newScale);

      set(atoms.stageCoordinatesAtom, newCoordinates);
      set(atoms.boundingBoxCoordinatesAtom, { x: 0, y: 0 });
      set(atoms.boundingBoxDimensionsAtom, newBoundingBoxDimensions);

      if (boundingBoxScaleMethod === 'auto') {
        const scaledDimensions = getScaledBoundingBoxDimensions(
          newBoundingBoxDimensions
        );
        set(atoms.scaledBoundingBoxDimensionsAtom, scaledDimensions);
      }
    }
  },
});

export const redoAction = selector<void>({
  key: 'canvas.redo.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const futureLayerStates = get(atoms.futureLayerStatesAtom);
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    const futureLayerStatesCopy = [...futureLayerStates];

    const targetState = futureLayerStatesCopy.shift();
    // to imitate shift on redux state that would change object
    set(atoms.futureLayerStatesAtom, futureLayerStatesCopy);

    if (!targetState) return;

    set(atoms.pastLayerStatesAtom, [
      ...pastLayerStates,
      _.cloneDeep(layerState),
    ]);

    if (pastLayerStates.length > maxHistory) {
      set(atoms.pastLayerStatesAtom, pastLayerStates.slice(1));
    }

    set(atoms.layerStateAtom, targetState);
  },
});

export const undoAction = selector<void>({
  key: 'canvas.undo.action',
  get: () => {
    return;
  },
  set: ({ get, set }) => {
    const pastLayerStates = get(atoms.pastLayerStatesAtom);
    const layerState = get(atoms.layerStateAtom);
    const futureLayerStates = get(atoms.futureLayerStatesAtom);
    const maxHistory = get(atoms.maxHistoryAtom);

    // by default pop() operation in recoil not valid, because objects are configurable: false
    const pastLayerStatesCopy = [...pastLayerStates];

    const targetState = pastLayerStatesCopy.pop();

    // to imitate shift on redux state that would change object
    set(atoms.pastLayerStatesAtom, pastLayerStatesCopy);

    if (!targetState) return;

    set(atoms.futureLayerStatesAtom, [
      _.cloneDeep(layerState),
      ...futureLayerStates,
    ]);

    if (futureLayerStates.length > maxHistory) {
      set(
        atoms.futureLayerStatesAtom,
        futureLayerStates.slice(0, futureLayerStates.length - 1)
      );
    }

    set(atoms.layerStateAtom, targetState);
  },
});
