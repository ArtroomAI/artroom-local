import React from 'react';
import { FC } from 'react';
import { CanvasStatusTextCursorPos } from './CanvasStatusText/CanvasStatusTextCursorPos';
import { useRecoilValue } from 'recoil';
import {
	activeLayerColorSelector,
	activeLayerStringSelector,
	boundingBoxColorSelector,
	boundingBoxCoordinatesStringSelector,
	boundingBoxDimensionsStringSelector,
	scaledBoundingBoxDimensionsStringSelector,
	shouldShowScaledBoundingBoxSelector,
	canvasCoordinatesStringSelector,
	canvasDimensionsStringSelector,
	shouldShowCanvasDebugInfoAtom,
	canvasScaleStringSelector,
	shouldShowBoundingBoxSelector,
} from '../atoms/canvas.atoms';

// import _ from 'lodash';
// import {canvasSelector} from 'canvas/store/canvasSelectors';
// import {roundToHundred} from '../util';

// const selector = createSelector(
// 	[canvasSelector],
// 	(canvas) => {
// 		const {
// 			stageDimensions: {width: stageWidth, height: stageHeight},
// 			stageCoordinates: {x: stageX, y: stageY},
// 			boundingBoxDimensions: {width: boxWidth, height: boxHeight},
// 			scaledBoundingBoxDimensions: {
// 				width: scaledBoxWidth,
// 				height: scaledBoxHeight,
// 			},
// 			boundingBoxCoordinates: {x: boxX, y: boxY},
// 			stageScale,
// 			shouldShowCanvasDebugInfo,
// 			layer,
// 			boundingBoxScaleMethod,
// 		} = canvas;

// 		let boundingBoxColor = 'inherit';

// 		if (
// 			(boundingBoxScaleMethod === 'none' &&
// 				(boxWidth < 512 || boxHeight < 512)) ||
// 			(boundingBoxScaleMethod === 'manual' &&
// 				scaledBoxWidth * scaledBoxHeight < 512 * 512)
// 		) {
// 			boundingBoxColor = 'var(--status-working-color)';
// 		}

// 		const activeLayerColor =
// 			layer === 'mask' ? 'var(--status-working-color)' : 'inherit';

// 		return {
// 			activeLayerColor,
// 			activeLayerString: layer.charAt(0).toUpperCase() + layer.slice(1),
// 			boundingBoxColor,
// 			boundingBoxCoordinatesString: `(${roundToHundred(
// 				boxX
// 			)}, ${roundToHundred(boxY)})`,
// 			boundingBoxDimensionsString: `${boxWidth}×${boxHeight}`,
// 			scaledBoundingBoxDimensionsString: `${scaledBoxWidth}×${scaledBoxHeight}`,
// 			canvasCoordinatesString: `${roundToHundred(stageX)}×${roundToHundred(
// 				stageY
// 			)}`,
// 			canvasDimensionsString: `${stageWidth}×${stageHeight}`,
// 			canvasScaleString: Math.round(stageScale * 100),
// 			shouldShowCanvasDebugInfo,
// 			shouldShowBoundingBox: boundingBoxScaleMethod !== 'auto',
// 			shouldShowScaledBoundingBox: boundingBoxScaleMethod !== 'none',
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	}
// );

export const CanvasStatusText: FC = () => {
	// const {
	// 	activeLayerColor,
	// 	activeLayerString,
	// 	boundingBoxColor,
	// 	boundingBoxCoordinatesString,
	// 	boundingBoxDimensionsString,
	// 	scaledBoundingBoxDimensionsString,
	// 	shouldShowScaledBoundingBox,
	// 	canvasCoordinatesString,
	// 	canvasDimensionsString,
	// 	canvasScaleString,
	// 	shouldShowCanvasDebugInfo,
	// 	shouldShowBoundingBox,
	// } = useAppSelector(selector);

	const activeLayerColor = useRecoilValue(activeLayerColorSelector);
	const activeLayerString = useRecoilValue(activeLayerStringSelector);
	const boundingBoxColor = useRecoilValue(boundingBoxColorSelector);
	const boundingBoxCoordinatesString = useRecoilValue(
		boundingBoxCoordinatesStringSelector,
	);
	const boundingBoxDimensionsString = useRecoilValue(
		boundingBoxDimensionsStringSelector,
	);
	const scaledBoundingBoxDimensionsString = useRecoilValue(
		scaledBoundingBoxDimensionsStringSelector,
	);
	const shouldShowScaledBoundingBox = useRecoilValue(
		shouldShowScaledBoundingBoxSelector,
	);
	const canvasCoordinatesString = useRecoilValue(
		canvasCoordinatesStringSelector,
	);
	const canvasDimensionsString = useRecoilValue(
		canvasDimensionsStringSelector,
	);
	const shouldShowCanvasDebugInfo = useRecoilValue(
		shouldShowCanvasDebugInfoAtom,
	);
	const canvasScaleString = useRecoilValue(canvasScaleStringSelector);
	const shouldShowBoundingBox = useRecoilValue(shouldShowBoundingBoxSelector);

	return (
		<div className="canvas-status-text">
			<div
				style={{
					color: activeLayerColor,
				}}>{`Active Layer: ${activeLayerString}`}</div>
			<div>{`Canvas Scale: ${canvasScaleString}%`}</div>
			{shouldShowBoundingBox && (
				<div
					style={{
						color: boundingBoxColor,
					}}>{`Bounding Box: ${boundingBoxDimensionsString}`}</div>
			)}
			{shouldShowScaledBoundingBox && (
				<div
					style={{
						color: boundingBoxColor,
					}}>{`Scaled Bounding Box: ${scaledBoundingBoxDimensionsString}`}</div>
			)}
			{shouldShowCanvasDebugInfo && (
				<>
					<div>{`Bounding Box Position: ${boundingBoxCoordinatesString}`}</div>
					<div>{`Canvas Dimensions: ${canvasDimensionsString}`}</div>
					<div>{`Canvas Position: ${canvasCoordinatesString}`}</div>
					<CanvasStatusTextCursorPos />
				</>
			)}
		</div>
	);
};
