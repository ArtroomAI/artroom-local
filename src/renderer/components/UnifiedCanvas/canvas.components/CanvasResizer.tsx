import React from 'react';
import { Spinner } from '@chakra-ui/react';
import { useLayoutEffect, useRef, FC } from 'react';
import { useSetRecoilState, useRecoilValue, useRecoilState } from 'recoil';
import {
	doesCanvasNeedScalingAtom,
	resizeAndScaleCanvasAction,
	resizeCanvasAction,
	canvasContainerDimensionsAtom,
	isCanvasInitializedAtom,
	initialCanvasImageSelector,
} from '../atoms/canvas.atoms';

// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import {
// 	resizeAndScaleCanvas,
// 	resizeCanvas,
// 	setCanvasContainerDimensions,
// 	setDoesCanvasNeedScaling,
// } from 'canvas/store/canvasSlice';
// import {
// 	canvasSelector,
// 	initialCanvasImageSelector,
// } from 'canvas/store/canvasSelectors';

// const canvasResizerSelector = createSelector(
// 	canvasSelector,
// 	initialCanvasImageSelector,
// 	activeTabNameSelector,
// 	(canvas, initialCanvasImage, activeTabName) => {
// 		const { doesCanvasNeedScaling, isCanvasInitialized } = canvas;
// 		return {
// 			doesCanvasNeedScaling,
// 			activeTabName,
// 			initialCanvasImage,
// 			isCanvasInitialized,
// 		};
// 	},
// );

export const CanvasResizer: FC = () => {
	// const {
	// 	doesCanvasNeedScaling,
	// 	activeTabName,
	// 	initialCanvasImage,
	// 	isCanvasInitialized,
	// } = useAppSelector(canvasResizerSelector);

	const [doesCanvasNeedScaling, setDoesCanvasNeedScaling] = useRecoilState(
		doesCanvasNeedScalingAtom,
	);
	const resizeAndScaleCanvas = useSetRecoilState(resizeAndScaleCanvasAction);
	const resizeCanvas = useSetRecoilState(resizeCanvasAction);
	const setCanvasContainerDimensions = useSetRecoilState(
		canvasContainerDimensionsAtom,
	);
	const isCanvasInitialized = useRecoilValue(isCanvasInitializedAtom);
	const initialCanvasImage = useRecoilValue(initialCanvasImageSelector);

	const ref = useRef<HTMLDivElement>(null);

	useLayoutEffect(() => {
		window.setTimeout(() => {
			if (!ref.current) return;

			const { clientWidth, clientHeight } = ref.current;

			setCanvasContainerDimensions({
				width: clientWidth,
				height: clientHeight,
			});

			if (!isCanvasInitialized) {
				resizeAndScaleCanvas();
			} else {
				resizeCanvas();
			}

			setDoesCanvasNeedScaling(false);
		}, 0);
	}, [initialCanvasImage, doesCanvasNeedScaling, isCanvasInitialized]);

	return (
		<div ref={ref} className="inpainting-canvas-area">
			<Spinner thickness="2px" speed="1s" size="xl" />
		</div>
	);
};
