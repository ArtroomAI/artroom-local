import { useCallback } from 'react';
import { useSetRecoilState } from 'recoil';
import { mouseLeftCanvasAction } from '../atoms/canvas.atoms';

// import { mouseLeftCanvas } from 'canvas/store/canvasSlice';

export const useCanvasMouseOut = () => {
	const mouseLeftCanvas = useSetRecoilState(mouseLeftCanvasAction);

	return useCallback(() => {
		mouseLeftCanvas();
	}, [mouseLeftCanvas]);
};
