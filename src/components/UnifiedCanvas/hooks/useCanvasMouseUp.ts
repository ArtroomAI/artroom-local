import Konva from 'konva';
import _ from 'lodash';
import { MutableRefObject, useCallback } from 'react';
import { getScaledCursorPosition } from '../util';
import { useRecoilState, useRecoilValue, useSetRecoilState } from 'recoil';
import {
	isDrawingAtom,
	isMovingStageAtom,
	addPointToCurrentLineAction,
	toolAtom,
	isStagingSelector,
} from '../atoms/canvas.atoms';
import { KonvaEventObject } from 'konva/lib/Node';

// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import {
// 	canvasSelector,
// 	isStagingSelector,
// } from 'canvas/store/canvasSelectors';
// import {
// 	addPointToCurrentEraserLine,
// 	addPointToCurrentLine,
// 	setIsDrawing,
// 	setIsMovingStage,
// } from 'canvas/store/canvasSlice';

// const selector = createSelector(
// 	[activeTabNameSelector, canvasSelector, isStagingSelector],
// 	(activeTabName, canvas, isStaging) => {
// 		const { tool, isDrawing } = canvas;
// 		return {
// 			tool,
// 			isDrawing,
// 			activeTabName,
// 			isStaging,
// 		};
// 	},
// 	{ memoizeOptions: { resultEqualityCheck: _.isEqual } },
// );

export const useCanvasMouseUp = (
	stageRef: MutableRefObject<Konva.Stage | null>,
	didMouseMoveRef: MutableRefObject<boolean>,
) => {
	// const { tool, isDrawing, isStaging } = useAppSelector(selector);

	const [isDrawing, setIsDrawing] = useRecoilState(isDrawingAtom);
	const setIsMovingStage = useSetRecoilState(isMovingStageAtom);
	const addPointToCurrentLine = useSetRecoilState(
		addPointToCurrentLineAction,
	);
	const tool = useRecoilValue(toolAtom);
	const isStaging = useRecoilValue(isStagingSelector);

	return useCallback((e: KonvaEventObject<MouseEvent | TouchEvent>) => {
		if (tool === 'move' || isStaging || (e.evt as MouseEvent).button === 1) {
			setIsMovingStage(false);
			stageRef.current.stopDrag();
			return;
		}

		if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
			const scaledCursorPosition = getScaledCursorPosition(
				stageRef.current,
			);

			if (!scaledCursorPosition) return;

			/**
			 * Extend the current line.
			 * In this case, the mouse didn't move, so we append the same point to
			 * the line's existing points. This allows the line to render as a circle
			 * centered on that point.
			 */

			addPointToCurrentLine([
				scaledCursorPosition.x,
				scaledCursorPosition.y,
			]);
		} else {
			didMouseMoveRef.current = false;
		}
		setIsDrawing(false);
	}, [didMouseMoveRef, isDrawing, isStaging, stageRef, tool]);
};
