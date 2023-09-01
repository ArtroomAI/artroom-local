export {}
// import { RootState } from 'app/store';
// import { activeTabNameSelector } from 'options/store/optionsSelectors';
// import { systemSelector } from 'system/store/systemSelectors';
// import { CanvasImage, CanvasState, isCanvasBaseImage } from './canvasTypes';

// export const canvasSelector = (state: RootState): CanvasState => state.canvas;

// export const isStagingSelector = createSelector(
// 	[canvasSelector, systemSelector],
// 	(canvas, system) =>
// 		canvas.layerState.stagingArea.images.length > 0 || system.isProcessing,
// );

// export const initialCanvasImageSelector = (
// 	state: RootState,
// ): CanvasImage | undefined =>
// 	state.canvas.layerState.objects.find(isCanvasBaseImage);
