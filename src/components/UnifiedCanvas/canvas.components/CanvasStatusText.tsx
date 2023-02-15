import { FC } from 'react';
import { Flex, Box } from '@chakra-ui/react';
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
	cursorCoordinatesStringSelector,
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

	const cursorCoordinatesString = useRecoilValue(
		cursorCoordinatesStringSelector,
	);

	return (
		<Flex
			position="absolute"
			top={0}
			left={0}
			opacity={0.65}
			direction="column"
			fontSize="0.8rem"
			padding="0.25rem"
			minWidth="12rem"
			borderRadius="0.25rem"
			margin="0.25rem"
			pointerEvents="none">
			<Box
				color={
					activeLayerColor
				}>{`Active Layer: ${activeLayerString}`}</Box>
			<Box>{`Canvas Scale: ${canvasScaleString}%`}</Box>
			{shouldShowBoundingBox && (
				<Box
					color={
						boundingBoxColor
					}>{`Bounding Box: ${boundingBoxDimensionsString}`}</Box>
			)}
			{shouldShowScaledBoundingBox && (
				<Box
					color={
						boundingBoxColor
					}>{`Scaled Bounding Box: ${scaledBoundingBoxDimensionsString}`}</Box>
			)}
			{shouldShowCanvasDebugInfo && (
				<>
					<Box>{`Bounding Box Position: ${boundingBoxCoordinatesString}`}</Box>
					<Box>{`Canvas Dimensions: ${canvasDimensionsString}`}</Box>
					<Box>{`Canvas Position: ${canvasCoordinatesString}`}</Box>
					<Box>{`Cursor Position: ${cursorCoordinatesString}`}</Box>
				</>
			)}
		</Flex>
	);
};
