import { selector } from 'recoil';
import { roundToHundred } from '../util';
import {
	boundingBoxCoordinatesAtom,
	boundingBoxDimensionsAtom,
	boundingBoxScaleMethodAtom,
	layerAtom,
	scaledBoundingBoxDimensionsAtom,
	stageCoordinatesAtom,
	stageDimensionsAtom,
	stageScaleAtom,
} from './canvas.atoms';

export const activeLayerColorSelector = selector({
	key: 'activeLayerColor.selector',
	get: ({ get }) => {
		const layer = get(layerAtom);
		return layer === 'mask' ? 'var(--status-working-color)' : 'inherit';
	},
});

export const activeLayerStringSelector = selector({
	key: 'activeLayerString.selector',
	get: ({ get }) => {
		const layer = get(layerAtom);
		return layer.charAt(0).toUpperCase() + layer.slice(1);
	},
});

export const boundingBoxColorSelector = selector({
	key: 'boundingBoxColor.selector',
	get: ({ get }) => {
		const boundingBoxScaleMethod = get(boundingBoxScaleMethodAtom);
		const boundingBoxDimensions = get(boundingBoxDimensionsAtom);
		const scaledBoundingBoxDimensions = get(
			scaledBoundingBoxDimensionsAtom,
		);

		const { width: boxWidth, height: boxHeight } = boundingBoxDimensions;
		const { width: scaledBoxWidth, height: scaledBoxHeight } =
			scaledBoundingBoxDimensions;

		if (
			(boundingBoxScaleMethod === 'none' &&
				(boxWidth < 512 || boxHeight < 512)) ||
			(boundingBoxScaleMethod === 'manual' &&
				scaledBoxWidth * scaledBoxHeight < 512 * 512)
		) {
			return 'var(--status-working-color)';
		}

		return 'inherit';
	},
});

export const boundingBoxCoordinatesStringSelector = selector({
	key: 'boundingBoxCoordinatesString.selector',
	get: ({ get }) => {
		const boundingBoxCoordinates = get(boundingBoxCoordinatesAtom);

		const { x: boxX, y: boxY } = boundingBoxCoordinates;

		return `(${roundToHundred(boxX)}, ${roundToHundred(boxY)})`;
	},
});

export const boundingBoxDimensionsStringSelector = selector({
	key: 'boundingBoxDimensionsString.selector',
	get: ({ get }) => {
		const boundingBoxDimensions = get(boundingBoxDimensionsAtom);

		const { width: boxWidth, height: boxHeight } = boundingBoxDimensions;

		return `${boxWidth}×${boxHeight}`;
	},
});

export const scaledBoundingBoxDimensionsStringSelector = selector({
	key: 'scaledBoundingBoxDimensionsString.selector',
	get: ({ get }) => {
		const scaledBoundingBoxDimensions = get(
			scaledBoundingBoxDimensionsAtom,
		);

		const { width: scaledBoxWidth, height: scaledBoxHeight } =
			scaledBoundingBoxDimensions;

		return `${scaledBoxWidth}×${scaledBoxHeight}`;
	},
});

export const shouldShowScaledBoundingBoxSelector = selector({
	key: 'shouldShowScaledBoundingBox.selector',
	get: ({ get }) => {
		return get(boundingBoxScaleMethodAtom) !== 'none';
	},
});

export const canvasCoordinatesStringSelector = selector({
	key: 'canvasCoordinatesString.selector',
	get: ({ get }) => {
		const stageCoordinates = get(stageCoordinatesAtom);

		const { x: stageX, y: stageY } = stageCoordinates;

		return `${roundToHundred(stageX)}×${roundToHundred(stageY)}`;
	},
});

export const canvasDimensionsStringSelector = selector({
	key: 'canvasDimensionsString.selector',
	get: ({ get }) => {
		const stageDimensions = get(stageDimensionsAtom);

		const { width: stageWidth, height: stageHeight } = stageDimensions;

		return `${stageWidth}×${stageHeight}`;
	},
});

export const canvasScaleStringSelector = selector({
	key: 'canvasScaleString.selector',
	get: ({ get }) => {
		const stageScale = get(stageScaleAtom);
		return Math.round(stageScale * 100);
	},
});

export const shouldShowBoundingBoxSelector = selector({
	key: 'shouldShowBoundingBox.selector',
	get: ({ get }) => {
		return get(boundingBoxScaleMethodAtom) !== 'auto';
	},
});
