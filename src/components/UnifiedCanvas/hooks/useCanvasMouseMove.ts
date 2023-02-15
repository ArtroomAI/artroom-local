import Konva from 'konva';
import { Vector2d } from 'konva/lib/types';
import { MutableRefObject, useCallback } from 'react';
import { getScaledCursorPosition } from '../util';
import { useColorPicker } from './useColorUnderCursor';
import { useRecoilValue, useSetRecoilState } from 'recoil';
import {
	cursorPositionAtom,
	addPointToCurrentLineAction,
	toolAtom,
	isDrawingAtom,
	isStagingSelector,
	layerAtom,
	layerStateAtom,
} from '../atoms/canvas.atoms';

export const useCanvasMouseMove = (
	stageRef: MutableRefObject<Konva.Stage | null>,
	didMouseMoveRef: MutableRefObject<boolean>,
	lastCursorPositionRef: MutableRefObject<Vector2d>,
) => {
	const { updateColorUnderCursor } = useColorPicker();
	const tool = useRecoilValue(toolAtom);
	const isDrawing = useRecoilValue(isDrawingAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	const setCursorPosition = useSetRecoilState(cursorPositionAtom);
	const addPointToCurrentLine = useSetRecoilState(
		addPointToCurrentLineAction,
	);
	const layer = useRecoilValue(layerAtom);
	const layerState = useRecoilValue(layerStateAtom);

	return useCallback(() => {
		if (!stageRef.current) return;

		const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

		if (!scaledCursorPosition) return;
		setCursorPosition(scaledCursorPosition);

		lastCursorPositionRef.current = scaledCursorPosition;

		if (tool === 'colorPicker') {
			updateColorUnderCursor();
			return;
		}

		if (!isDrawing || tool === 'moveBoundingBox' || isStaging) return;

		didMouseMoveRef.current = true;
		// Possible room for performance improvement
		const targetImageLayer = layerState.images.find(
			elem => elem.id === layer,
		);

		const imageLayerOffset =
			layer !== 'base' && layer !== 'mask' && targetImageLayer
				? targetImageLayer.picture
				: { x: 0, y: 0 };

		addPointToCurrentLine([
			// scaledCursorPosition.x,
			// scaledCursorPosition.y,
			scaledCursorPosition.x - imageLayerOffset.x,
			scaledCursorPosition.y - imageLayerOffset.y,
		]);
	}, [
		didMouseMoveRef,
		isDrawing,
		isStaging,
		lastCursorPositionRef,
		stageRef,
		tool,
		updateColorUnderCursor,
	]);
};
