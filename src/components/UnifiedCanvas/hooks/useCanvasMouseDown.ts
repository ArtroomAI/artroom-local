import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { MutableRefObject, useCallback } from 'react';
import { getScaledCursorPosition } from '../util';
import { useColorPicker } from './useColorUnderCursor';
import {
	addLineAction,
	isDrawingAtom,
	isMovingStageAtom,
	toolAtom,
	isStagingSelector,
	layerAtom,
	layerStateAtom,
} from '../atoms/canvas.atoms';
import { useRecoilValue, useSetRecoilState } from 'recoil';

export const useCanvasMouseDown = (
	stageRef: MutableRefObject<Konva.Stage | null>,
) => {
	const { commitColorUnderCursor } = useColorPicker();

	const addLine = useSetRecoilState(addLineAction);
	const setIsDrawing = useSetRecoilState(isDrawingAtom);
	const setIsMovingStage = useSetRecoilState(isMovingStageAtom);
	const tool = useRecoilValue(toolAtom);
	const isStaging = useRecoilValue(isStagingSelector);
	const layer = useRecoilValue(layerAtom);
	const layerState = useRecoilValue(layerStateAtom);

	return useCallback(
		(e: KonvaEventObject<MouseEvent | TouchEvent>) => {
			if (!stageRef.current) return;

			stageRef.current.container().focus();

			if (tool === 'moveBoundingBox' || isStaging) {
				setIsMovingStage(true);
				return;
			}

			if (tool === 'move' || tool === 'transform') {
				return;
			}

			if (tool === 'colorPicker') {
				commitColorUnderCursor();
				return;
			}

			const scaledCursorPosition = getScaledCursorPosition(
				stageRef.current,
			);

			if (!scaledCursorPosition) return;

			e.evt.preventDefault();

			setIsDrawing(true);
			// Possible room for performance improvement
			const targetImageLayer = layerState.images.find(
				elem => elem.id === layer,
			);

			const imageLayerOffset =
				layer !== 'base' && layer !== 'mask' && targetImageLayer
					? targetImageLayer.picture
					: { x: 0, y: 0 };

			// Add a new line starting from the current cursor position.
			// addLine([scaledCursorPosition.x, scaledCursorPosition.y]);

			addLine([
				scaledCursorPosition.x - imageLayerOffset.x,
				scaledCursorPosition.y - imageLayerOffset.y,
			]);
		},
		[stageRef, tool, isStaging, commitColorUnderCursor],
	);
};
