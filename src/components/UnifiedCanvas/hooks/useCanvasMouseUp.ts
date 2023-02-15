import Konva from 'konva';
import { MutableRefObject, useCallback } from 'react';
import { getScaledCursorPosition } from '../util';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import {
	isDrawingAtom,
	isMovingStageAtom,
	addPointToCurrentLineAction,
	toolAtom,
	isStagingSelector,
	layerAtom,
	layerStateAtom,
} from '../atoms/canvas.atoms';

export const useCanvasMouseUp = (
	stageRef: MutableRefObject<Konva.Stage | null>,
	didMouseMoveRef: MutableRefObject<boolean>,
) => {
	const [isDrawing, setIsDrawing] = useRecoilState(isDrawingAtom);
	const setIsMovingStage = useSetRecoilState(isMovingStageAtom);
	const addPointToCurrentLine = useSetRecoilState(
		addPointToCurrentLineAction,
	);
	const tool = useRecoilValue(toolAtom);
	const isStaging = useRecoilValue(isStagingSelector);
	const layer = useRecoilValue(layerAtom);
	const layerState = useRecoilValue(layerStateAtom);

	return useCallback(() => {
		if (tool === 'moveBoundingBox' || isStaging) {
			setIsMovingStage(false);
			return;
		}

		if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
			const scaledCursorPosition = getScaledCursorPosition(
				stageRef.current,
			);

			if (!scaledCursorPosition) return;

			// Possible room for performance improvement
			const targetImageLayer = layerState.images.find(
				elem => elem.id === layer,
			);

			const imageLayerOffset =
				layer !== 'base' && layer !== 'mask' && targetImageLayer
					? targetImageLayer.picture
					: { x: 0, y: 0 };
			/**
			 * Extend the current line.
			 * In this case, the mouse didn't move, so we append the same point to
			 * the line's existing points. This allows the line to render as a circle
			 * centered on that point.
			 */

			addPointToCurrentLine([
				// scaledCursorPosition.x,
				// scaledCursorPosition.y,
				scaledCursorPosition.x - imageLayerOffset.x,
				scaledCursorPosition.y - imageLayerOffset.y,
			]);
		} else {
			didMouseMoveRef.current = false;
		}
		setIsDrawing(false);
	}, [didMouseMoveRef, isDrawing, isStaging, stageRef, tool]);
};
