// import { createSlice, PayloadAction } from '@reduxjs/toolkit';
// import { Image } from 'painter';
// import {
//   roundDownToMultiple,
//   roundToMultiple,
// } from 'common/util/roundDownToMultiple';
// import { IRect, Vector2d } from 'konva/lib/types';
// import _ from 'lodash';
// import { RgbaColor } from 'react-colorful';
// import calculateCoordinates from '../util/calculateCoordinates';
// import calculateScale from '../util/calculateScale';
// import { STAGE_PADDING_PERCENTAGE } from '../util/constants';
// import floorCoordinates from '../util/floorCoordinates';
// import getScaledBoundingBoxDimensions from '../util/getScaledBoundingBoxDimensions';
// import roundDimensionsTo64 from '../util/roundDimensionsTo64';

import {
  CanvasState,
  CanvasLayerState,
  // CanvasImage,
  // CanvasLayer,
  // CanvasBaseLine,
  // CanvasMaskLine,
  // CanvasTool,
  // Dimensions,
  // isCanvasAnyLine,
  // isCanvasBaseImage,
  // isCanvasMaskLine,
} from '../../atoms/canvasTypes'

export const initialLayerState: CanvasLayerState = {
  objects: [],
  stagingArea: {
    images: [],
    selectedImageIndex: -1,
  },
}

const initialCanvasState: CanvasState = {
  boundingBoxCoordinates: { x: 0, y: 0 },
  boundingBoxDimensions: { width: 512, height: 512 },
  boundingBoxPreviewFill: { r: 0, g: 0, b: 0, a: 0.5 },
  boundingBoxScaleMethod: 'auto',
  brushColor: { r: 90, g: 90, b: 255, a: 1 },
  brushSize: 50,
  canvasContainerDimensions: { width: 0, height: 0 },
  colorPickerColor: { r: 90, g: 90, b: 255, a: 1 },
  cursorPosition: null,
  doesCanvasNeedScaling: false,
  futureLayerStates: [],
  inpaintReplace: 0.1,
  isCanvasInitialized: false,
  isDrawing: false,
  isMaskEnabled: true,
  isMouseOverBoundingBox: false,
  isMoveBoundingBoxKeyHeld: false,
  isMoveStageKeyHeld: false,
  isMovingBoundingBox: false,
  isMovingStage: false,
  isTransformingBoundingBox: false,
  layer: 'base',
  layerState: initialLayerState,
  maskColor: { r: 255, g: 90, b: 90, a: 1 },
  maxHistory: 128,
  minimumStageScale: 1,
  pastLayerStates: [],
  scaledBoundingBoxDimensions: { width: 512, height: 512 },
  shouldAutoSave: false,
  shouldCropToBoundingBoxOnSave: false,
  shouldDarkenOutsideBoundingBox: false,
  shouldLockBoundingBox: false,
  shouldPreserveMaskedArea: false,
  shouldRestrictStrokesToBox: true,
  shouldShowBoundingBox: true,
  shouldShowBrush: true,
  shouldShowBrushPreview: false,
  shouldShowCanvasDebugInfo: false,
  shouldShowCheckboardTransparency: false,
  shouldShowGrid: true,
  shouldShowIntermediates: true,
  shouldShowStagingImage: true,
  shouldShowStagingOutline: true,
  shouldSnapToGrid: true,
  shouldUseInpaintReplace: false,
  stageCoordinates: { x: 0, y: 0 },
  stageDimensions: { width: 0, height: 0 },
  stageScale: 1,
  tool: 'brush',
}

// export const canvasSlice = createSlice({
// 	name: 'canvas',
// 	initialState: initialCanvasState,
// 	reducers: {
// 		setTool: (state, action: PayloadAction<CanvasTool>) => {
// 			const tool = action.payload;
// 			state.tool = action.payload;
// 			if (tool !== 'move') {
// 				state.isTransformingBoundingBox = false;
// 				state.isMouseOverBoundingBox = false;
// 				state.isMovingBoundingBox = false;
// 				state.isMovingStage = false;
// 			}
// 		},
// 		setLayer: (state, action: PayloadAction<CanvasLayer>) => {
// 			state.layer = action.payload;
// 		},
// 		setMaskColor: (state, action: PayloadAction<RgbaColor>) => {
// 			state.maskColor = action.payload;
// 		},
// 		setBrushColor: (state, action: PayloadAction<RgbaColor>) => {
// 			state.brushColor = action.payload;
// 		},
// 		setBrushSize: (state, action: PayloadAction<number>) => {
// 			state.brushSize = action.payload;
// 		},
// 		clearMask: state => {
// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));
// 			state.layerState.objects = state.layerState.objects.filter(
// 				obj => !isCanvasMaskLine(obj),
// 			);
// 			state.futureLayerStates = [];
// 			state.shouldPreserveMaskedArea = false;
// 		},
// 		toggleShouldInvertMask: state => {
// 			state.shouldPreserveMaskedArea = !state.shouldPreserveMaskedArea;
// 		},
// 		toggleShouldShowMask: state => {
// 			state.isMaskEnabled = !state.isMaskEnabled;
// 		},
// 		setShouldPreserveMaskedArea: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldPreserveMaskedArea = action.payload;
// 		},
// 		setIsMaskEnabled: (state, action: PayloadAction<boolean>) => {
// 			state.isMaskEnabled = action.payload;
// 			state.layer = action.payload ? 'mask' : 'base';
// 		},
// 		setCursorPosition: (state, action: PayloadAction<Vector2d | null>) => {
// 			state.cursorPosition = action.payload;
// 		},
// 		setInitialCanvasImage: (state, action: PayloadAction<Image>) => {
// 			const image = action.payload;
// 			const { stageDimensions } = state;

// 			const newBoundingBoxDimensions = {
// 				width: roundDownToMultiple(_.clamp(image.width, 64, 512), 64),
// 				height: roundDownToMultiple(_.clamp(image.height, 64, 512), 64),
// 			};

// 			const newBoundingBoxCoordinates = {
// 				x: roundToMultiple(
// 					image.width / 2 - newBoundingBoxDimensions.width / 2,
// 					64,
// 				),
// 				y: roundToMultiple(
// 					image.height / 2 - newBoundingBoxDimensions.height / 2,
// 					64,
// 				),
// 			};

// 			if (state.boundingBoxScaleMethod === 'auto') {
// 				const scaledDimensions = getScaledBoundingBoxDimensions(
// 					newBoundingBoxDimensions,
// 				);
// 				state.scaledBoundingBoxDimensions = scaledDimensions;
// 			}

// 			state.boundingBoxDimensions = newBoundingBoxDimensions;
// 			state.boundingBoxCoordinates = newBoundingBoxCoordinates;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			state.layerState = {
// 				...initialLayerState,
// 				objects: [
// 					{
// 						kind: 'image',
// 						layer: 'base',
// 						x: 0,
// 						y: 0,
// 						width: image.width,
// 						height: image.height,
// 						image: image,
// 					},
// 				],
// 			};
// 			state.futureLayerStates = [];

// 			state.isCanvasInitialized = false;
// 			const newScale = calculateScale(
// 				stageDimensions.width,
// 				stageDimensions.height,
// 				image.width,
// 				image.height,
// 				STAGE_PADDING_PERCENTAGE,
// 			);

// 			const newCoordinates = calculateCoordinates(
// 				stageDimensions.width,
// 				stageDimensions.height,
// 				0,
// 				0,
// 				image.width,
// 				image.height,
// 				newScale,
// 			);
// 			state.stageScale = newScale;
// 			state.stageCoordinates = newCoordinates;
// 			state.doesCanvasNeedScaling = true;
// 		},
// 		setBoundingBoxDimensions: (
// 			state,
// 			action: PayloadAction<Dimensions>,
// 		) => {
// 			const newDimensions = roundDimensionsTo64(action.payload);
// 			state.boundingBoxDimensions = newDimensions;

// 			if (state.boundingBoxScaleMethod === 'auto') {
// 				const scaledDimensions =
// 					getScaledBoundingBoxDimensions(newDimensions);
// 				state.scaledBoundingBoxDimensions = scaledDimensions;
// 			}
// 		},
// 		setBoundingBoxCoordinates: (state, action: PayloadAction<Vector2d>) => {
// 			state.boundingBoxCoordinates = floorCoordinates(action.payload);
// 		},
// 		setStageCoordinates: (state, action: PayloadAction<Vector2d>) => {
// 			state.stageCoordinates = action.payload;
// 		},
// 		setDoesCanvasNeedScaling: (state, action: PayloadAction<boolean>) => {
// 			state.doesCanvasNeedScaling = action.payload;
// 		},
// 		setStageScale: (state, action: PayloadAction<number>) => {
// 			state.stageScale = action.payload;
// 		},
// 		setShouldDarkenOutsideBoundingBox: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldDarkenOutsideBoundingBox = action.payload;
// 		},
// 		setIsDrawing: (state, action: PayloadAction<boolean>) => {
// 			state.isDrawing = action.payload;
// 		},
// 		clearCanvasHistory: state => {
// 			state.pastLayerStates = [];
// 			state.futureLayerStates = [];
// 		},
// 		setShouldShowBoundingBox: (state, action: PayloadAction<boolean>) => {
// 			state.shouldShowBoundingBox = action.payload;
// 		},
// 		setIsTransformingBoundingBox: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.isTransformingBoundingBox = action.payload;
// 		},
// 		setIsMovingBoundingBox: (state, action: PayloadAction<boolean>) => {
// 			state.isMovingBoundingBox = action.payload;
// 		},
// 		setIsMouseOverBoundingBox: (state, action: PayloadAction<boolean>) => {
// 			state.isMouseOverBoundingBox = action.payload;
// 		},
// 		addImageToStagingArea: (
// 			state,
// 			action: PayloadAction<{
// 				boundingBox: IRect;
// 				image: Image;
// 			}>,
// 		) => {
// 			const { boundingBox, image } = action.payload;

// 			if (!boundingBox || !image) return;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState.stagingArea.images.push({
// 				kind: 'image',
// 				layer: 'base',
// 				...boundingBox,
// 				image,
// 			});

// 			state.layerState.stagingArea.selectedImageIndex =
// 				state.layerState.stagingArea.images.length - 1;

// 			state.futureLayerStates = [];
// 		},
// 		discardStagedImages: state => {
// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState.stagingArea = {
// 				...initialLayerState.stagingArea,
// 			};

// 			state.futureLayerStates = [];
// 			state.shouldShowStagingOutline = true;
// 			state.shouldShowStagingOutline = true;
// 		},
// 		addFillRect: state => {
// 			const {
// 				boundingBoxCoordinates,
// 				boundingBoxDimensions,
// 				brushColor,
// 			} = state;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState.objects.push({
// 				kind: 'fillRect',
// 				layer: 'base',
// 				...boundingBoxCoordinates,
// 				...boundingBoxDimensions,
// 				color: brushColor,
// 			});

// 			state.futureLayerStates = [];
// 		},
// 		addEraseRect: state => {
// 			const { boundingBoxCoordinates, boundingBoxDimensions } = state;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState.objects.push({
// 				kind: 'eraseRect',
// 				layer: 'base',
// 				...boundingBoxCoordinates,
// 				...boundingBoxDimensions,
// 			});

// 			state.futureLayerStates = [];
// 		},
// 		addLine: (state, action: PayloadAction<number[]>) => {
// 			const {
// 				tool,
// 				layer,
// 				brushColor,
// 				brushSize,
// 				shouldRestrictStrokesToBox,
// 			} = state;

// 			if (tool === 'move' || tool === 'colorPicker') return;

// 			const newStrokeWidth = brushSize / 2;

// 			// set & then spread this to only conditionally add the "color" key
// 			const newColor =
// 				layer === 'base' && tool === 'brush'
// 					? { color: brushColor }
// 					: {};

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			const newLine: CanvasMaskLine | CanvasBaseLine = {
// 				kind: 'line',
// 				layer,
// 				tool,
// 				strokeWidth: newStrokeWidth,
// 				points: action.payload,
// 				...newColor,
// 			};

// 			if (shouldRestrictStrokesToBox) {
// 				newLine.clip = {
// 					...state.boundingBoxCoordinates,
// 					...state.boundingBoxDimensions,
// 				};
// 			}

// 			state.layerState.objects.push(newLine);

// 			state.futureLayerStates = [];
// 		},
// 		addPointToCurrentLine: (state, action: PayloadAction<number[]>) => {
// 			const lastLine = state.layerState.objects.findLast(isCanvasAnyLine);

// 			if (!lastLine) return;

// 			lastLine.points.push(...action.payload);
// 		},
// 		undo: state => {
// 			const targetState = state.pastLayerStates.pop();

// 			if (!targetState) return;

// 			state.futureLayerStates.unshift(_.cloneDeep(state.layerState));

// 			if (state.futureLayerStates.length > state.maxHistory) {
// 				state.futureLayerStates.pop();
// 			}

// 			state.layerState = targetState;
// 		},
// 		redo: state => {
// 			const targetState = state.futureLayerStates.shift();

// 			if (!targetState) return;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState = targetState;
// 		},
// 		setShouldShowGrid: (state, action: PayloadAction<boolean>) => {
// 			state.shouldShowGrid = action.payload;
// 		},
// 		setIsMovingStage: (state, action: PayloadAction<boolean>) => {
// 			state.isMovingStage = action.payload;
// 		},
// 		setShouldSnapToGrid: (state, action: PayloadAction<boolean>) => {
// 			state.shouldSnapToGrid = action.payload;
// 		},
// 		setShouldAutoSave: (state, action: PayloadAction<boolean>) => {
// 			state.shouldAutoSave = action.payload;
// 		},
// 		setShouldShowIntermediates: (state, action: PayloadAction<boolean>) => {
// 			state.shouldShowIntermediates = action.payload;
// 		},
// 		resetCanvas: state => {
// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			state.layerState = initialLayerState;
// 			state.futureLayerStates = [];
// 		},
// 		setCanvasContainerDimensions: (
// 			state,
// 			action: PayloadAction<Dimensions>,
// 		) => {
// 			state.canvasContainerDimensions = action.payload;
// 		},
// 		resizeAndScaleCanvas: state => {
// 			const { width: containerWidth, height: containerHeight } =
// 				state.canvasContainerDimensions;

// 			const initialCanvasImage =
// 				state.layerState.objects.find(isCanvasBaseImage);

// 			const newStageDimensions = {
// 				width: Math.floor(containerWidth),
// 				height: Math.floor(containerHeight),
// 			};

// 			if (!initialCanvasImage) {
// 				const newScale = calculateScale(
// 					newStageDimensions.width,
// 					newStageDimensions.height,
// 					512,
// 					512,
// 					STAGE_PADDING_PERCENTAGE,
// 				);

// 				const newCoordinates = calculateCoordinates(
// 					newStageDimensions.width,
// 					newStageDimensions.height,
// 					0,
// 					0,
// 					512,
// 					512,
// 					newScale,
// 				);

// 				const newBoundingBoxDimensions = { width: 512, height: 512 };

// 				state.stageScale = newScale;
// 				state.stageCoordinates = newCoordinates;
// 				state.stageDimensions = newStageDimensions;
// 				state.boundingBoxCoordinates = { x: 0, y: 0 };
// 				state.boundingBoxDimensions = newBoundingBoxDimensions;

// 				if (state.boundingBoxScaleMethod === 'auto') {
// 					const scaledDimensions = getScaledBoundingBoxDimensions(
// 						newBoundingBoxDimensions,
// 					);
// 					state.scaledBoundingBoxDimensions = scaledDimensions;
// 				}

// 				return;
// 			}

// 			const { width: imageWidth, height: imageHeight } =
// 				initialCanvasImage;

// 			const padding = 0.95;

// 			const newScale = calculateScale(
// 				containerWidth,
// 				containerHeight,
// 				imageWidth,
// 				imageHeight,
// 				padding,
// 			);

// 			const newCoordinates = calculateCoordinates(
// 				newStageDimensions.width,
// 				newStageDimensions.height,
// 				0,
// 				0,
// 				imageWidth,
// 				imageHeight,
// 				newScale,
// 			);

// 			state.minimumStageScale = newScale;
// 			state.stageScale = newScale;
// 			state.stageCoordinates = floorCoordinates(newCoordinates);
// 			state.stageDimensions = newStageDimensions;

// 			state.isCanvasInitialized = true;
// 		},
// 		resizeCanvas: state => {
// 			const { width: containerWidth, height: containerHeight } =
// 				state.canvasContainerDimensions;

// 			const newStageDimensions = {
// 				width: Math.floor(containerWidth),
// 				height: Math.floor(containerHeight),
// 			};

// 			state.stageDimensions = newStageDimensions;

// 			if (!state.layerState.objects.find(isCanvasBaseImage)) {
// 				const newScale = calculateScale(
// 					newStageDimensions.width,
// 					newStageDimensions.height,
// 					512,
// 					512,
// 					STAGE_PADDING_PERCENTAGE,
// 				);

// 				const newCoordinates = calculateCoordinates(
// 					newStageDimensions.width,
// 					newStageDimensions.height,
// 					0,
// 					0,
// 					512,
// 					512,
// 					newScale,
// 				);

// 				const newBoundingBoxDimensions = { width: 512, height: 512 };

// 				state.stageScale = newScale;

// 				state.stageCoordinates = newCoordinates;
// 				state.boundingBoxCoordinates = { x: 0, y: 0 };
// 				state.boundingBoxDimensions = newBoundingBoxDimensions;

// 				if (state.boundingBoxScaleMethod === 'auto') {
// 					const scaledDimensions = getScaledBoundingBoxDimensions(
// 						newBoundingBoxDimensions,
// 					);
// 					state.scaledBoundingBoxDimensions = scaledDimensions;
// 				}
// 			}
// 		},
// 		resetCanvasView: (
// 			state,
// 			action: PayloadAction<{
// 				contentRect: IRect;
// 			}>,
// 		) => {
// 			const { contentRect } = action.payload;
// 			const {
// 				stageDimensions: { width: stageWidth, height: stageHeight },
// 			} = state;

// 			const { x, y, width, height } = contentRect;

// 			if (width !== 0 && height !== 0) {
// 				const newScale = calculateScale(
// 					stageWidth,
// 					stageHeight,
// 					width,
// 					height,
// 					STAGE_PADDING_PERCENTAGE,
// 				);

// 				const newCoordinates = calculateCoordinates(
// 					stageWidth,
// 					stageHeight,
// 					x,
// 					y,
// 					width,
// 					height,
// 					newScale,
// 				);

// 				state.stageScale = newScale;
// 				state.stageCoordinates = newCoordinates;
// 			} else {
// 				const newScale = calculateScale(
// 					stageWidth,
// 					stageHeight,
// 					512,
// 					512,
// 					STAGE_PADDING_PERCENTAGE,
// 				);

// 				const newCoordinates = calculateCoordinates(
// 					stageWidth,
// 					stageHeight,
// 					0,
// 					0,
// 					512,
// 					512,
// 					newScale,
// 				);

// 				const newBoundingBoxDimensions = { width: 512, height: 512 };

// 				state.stageScale = newScale;
// 				state.stageCoordinates = newCoordinates;
// 				state.boundingBoxCoordinates = { x: 0, y: 0 };
// 				state.boundingBoxDimensions = newBoundingBoxDimensions;

// 				if (state.boundingBoxScaleMethod === 'auto') {
// 					const scaledDimensions = getScaledBoundingBoxDimensions(
// 						newBoundingBoxDimensions,
// 					);
// 					state.scaledBoundingBoxDimensions = scaledDimensions;
// 				}
// 			}
// 		},
// 		nextStagingAreaImage: state => {
// 			const currentIndex =
// 				state.layerState.stagingArea.selectedImageIndex;
// 			const length = state.layerState.stagingArea.images.length;

// 			state.layerState.stagingArea.selectedImageIndex = Math.min(
// 				currentIndex + 1,
// 				length - 1,
// 			);
// 		},
// 		prevStagingAreaImage: state => {
// 			const currentIndex =
// 				state.layerState.stagingArea.selectedImageIndex;

// 			state.layerState.stagingArea.selectedImageIndex = Math.max(
// 				currentIndex - 1,
// 				0,
// 			);
// 		},
// 		commitStagingAreaImage: state => {
// 			const { images, selectedImageIndex } = state.layerState.stagingArea;

// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			if (state.pastLayerStates.length > state.maxHistory) {
// 				state.pastLayerStates.shift();
// 			}

// 			state.layerState.objects.push({
// 				...images[selectedImageIndex],
// 			});

// 			state.layerState.stagingArea = {
// 				...initialLayerState.stagingArea,
// 			};

// 			state.futureLayerStates = [];
// 			state.shouldShowStagingOutline = true;
// 			state.shouldShowStagingImage = true;
// 		},
// 		setShouldShowStagingImage: (state, action: PayloadAction<boolean>) => {
// 			state.shouldShowStagingImage = action.payload;
// 		},
// 		setShouldShowStagingOutline: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldShowStagingOutline = action.payload;
// 		},
// 		setShouldShowCanvasDebugInfo: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldShowCanvasDebugInfo = action.payload;
// 		},
// 		setShouldRestrictStrokesToBox: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldRestrictStrokesToBox = action.payload;
// 		},
// 		setShouldCropToBoundingBoxOnSave: (
// 			state,
// 			action: PayloadAction<boolean>,
// 		) => {
// 			state.shouldCropToBoundingBoxOnSave = action.payload;
// 		},
// 		setColorPickerColor: (state, action: PayloadAction<RgbaColor>) => {
// 			state.colorPickerColor = action.payload;
// 		},
// 		commitColorPickerColor: state => {
// 			state.brushColor = {
// 				...state.colorPickerColor,
// 				a: state.brushColor.a,
// 			};
// 			state.tool = 'brush';
// 		},
// 		setMergedCanvas: (state, action: PayloadAction<CanvasImage>) => {
// 			state.pastLayerStates.push(_.cloneDeep(state.layerState));

// 			state.futureLayerStates = [];

// 			state.layerState.objects = [action.payload];
// 		},
// 		resetCanvasInteractionState: state => {
// 			state.cursorPosition = null;
// 			state.isDrawing = false;
// 			state.isMouseOverBoundingBox = false;
// 			state.isMoveBoundingBoxKeyHeld = false;
// 			state.isMoveStageKeyHeld = false;
// 			state.isMovingBoundingBox = false;
// 			state.isMovingStage = false;
// 			state.isTransformingBoundingBox = false;
// 		},
// 		mouseLeftCanvas: state => {
// 			state.cursorPosition = null;
// 			state.isDrawing = false;
// 			state.isMouseOverBoundingBox = false;
// 			state.isMovingBoundingBox = false;
// 			state.isTransformingBoundingBox = false;
// 		},
// 	},
// });

// export const {
// 	addEraseRect,
// 	addFillRect,
// 	addImageToStagingArea, // SOCKET EXCLUSIVE
// 	addLine,
// 	addPointToCurrentLine,
// 	clearCanvasHistory,
// 	clearMask,
// 	commitColorPickerColor,
// 	commitStagingAreaImage,
// 	discardStagedImages,
// 	mouseLeftCanvas,
// 	nextStagingAreaImage,
// 	prevStagingAreaImage,
// 	redo,
// 	resetCanvas,
// 	resetCanvasInteractionState,
// 	resetCanvasView,
// 	resizeAndScaleCanvas,
// 	resizeCanvas,
// 	setBoundingBoxCoordinates,
// 	setBoundingBoxDimensions,
// 	setBrushColor,
// 	setBrushSize,
// 	setCanvasContainerDimensions,
// 	setColorPickerColor,
// 	setCursorPosition,
// 	setDoesCanvasNeedScaling,
// 	setInitialCanvasImage,
// 	setIsDrawing,
// 	setIsMaskEnabled,
// 	setIsMouseOverBoundingBox,
// 	setIsMovingBoundingBox,
// 	setIsMovingStage,
// 	setIsTransformingBoundingBox,
// 	setLayer,
// 	setMaskColor,
// 	setMergedCanvas, // THUNK EXCLUSIVE
// 	setShouldAutoSave,
// 	setShouldCropToBoundingBoxOnSave,
// 	setShouldDarkenOutsideBoundingBox,
// 	setShouldPreserveMaskedArea,
// 	setShouldShowBoundingBox,
// 	setShouldShowCanvasDebugInfo,
// 	setShouldShowGrid,
// 	setShouldShowIntermediates,
// 	setShouldShowStagingImage,
// 	setShouldShowStagingOutline,
// 	setShouldSnapToGrid,
// 	setStageCoordinates,
// 	setStageScale,
// 	setTool,
// 	undo,
// 	setShouldRestrictStrokesToBox,
// } = canvasSlice.actions;

// export default canvasSlice.reducer;
