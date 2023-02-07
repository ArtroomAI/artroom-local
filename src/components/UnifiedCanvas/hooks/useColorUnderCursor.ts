import Konva from 'konva';
import { getCanvasBaseLayer, getCanvasStage } from '../util';
import { useSetRecoilState } from 'recoil';
import {
	colorPickerColorAtom,
	commitColorPickerColorAction,
} from '../atoms/canvas.atoms';

// import {
// 	commitColorPickerColor,
// 	setColorPickerColor,
// } from '../store/canvasSlice';

export const useColorPicker = () => {
	const canvasBaseLayer = getCanvasBaseLayer();
	const stage = getCanvasStage();

	const setColorPickerColor = useSetRecoilState(colorPickerColorAtom);
	const commitColorPickerColor = useSetRecoilState(
		commitColorPickerColorAction,
	);

	return {
		updateColorUnderCursor: () => {
			if (!stage || !canvasBaseLayer) return;

			const position = stage.getPointerPosition();

			if (!position) return;

			const pixelRatio = Konva.pixelRatio;

			const [r, g, b, a] = canvasBaseLayer
				.getContext()
				.getImageData(
					position.x * pixelRatio,
					position.y * pixelRatio,
					1,
					1,
				).data;

			setColorPickerColor({ r, g, b, a });
		},
		commitColorUnderCursor: () => {
			commitColorPickerColor();
		},
	};
};
