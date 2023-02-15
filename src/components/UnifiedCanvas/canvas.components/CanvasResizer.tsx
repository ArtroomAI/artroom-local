import { Spinner, Flex } from '@chakra-ui/react';
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

export const CanvasResizer: FC = () => {
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
		<Flex
			ref={ref}
			flexDirection="column"
			alignItems="center"
			justifyContent="center"
			rowGap="1rem"
			width="100%"
			height="100%">
			<Spinner thickness="2px" speed="1s" size="xl" />
		</Flex>
	);
};
