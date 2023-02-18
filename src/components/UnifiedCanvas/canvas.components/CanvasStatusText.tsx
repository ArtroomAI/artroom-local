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

export const CanvasStatusText: FC = () => {

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
