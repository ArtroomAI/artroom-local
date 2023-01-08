export {};
// import { AnyAction, ThunkAction } from '@reduxjs/toolkit';
// import { RootState } from 'app/store';
// import { ImageUploadResponse, Image } from 'painter';
// import { v4 as uuidv4 } from 'uuid';
// import { copyImage, downloadFile, layerToDataURL } from '../../util';
// import { getCanvasBaseLayer } from '../../util/konvaInstanceProvider';
// import {
// 	addToast,
// 	setCurrentStatus,
// 	setIsCancelable,
// 	setIsProcessing,
// 	setProcessingIndeterminateTask,
// } from 'system/store/systemSlice';
// import { addImage } from 'gallery/store/gallerySlice';
// import { setMergedCanvas } from '../canvasSlice';
// import { CanvasState } from '../canvasTypes';

// type MergeAndUploadCanvasConfig = {
// 	cropVisible?: boolean;
// 	cropToBoundingBox?: boolean;
// 	shouldSaveToGallery?: boolean;
// 	shouldDownload?: boolean;
// 	shouldCopy?: boolean;
// 	shouldSetAsInitialImage?: boolean;
// };

// const defaultConfig: MergeAndUploadCanvasConfig = {
// 	cropVisible: false,
// 	cropToBoundingBox: false,
// 	shouldSaveToGallery: false,
// 	shouldDownload: false,
// 	shouldCopy: false,
// 	shouldSetAsInitialImage: true,
// };

// export const mergeAndUploadCanvas =
// 	(
// 		config = defaultConfig,
// 	): ThunkAction<void, RootState, unknown, AnyAction> =>
// 	async (dispatch, getState) => {
// 		const {
// 			cropVisible,
// 			cropToBoundingBox,
// 			shouldSaveToGallery,
// 			shouldDownload,
// 			shouldCopy,
// 			shouldSetAsInitialImage,
// 		} = config;

// 		dispatch(setProcessingIndeterminateTask('Exporting Image'));
// 		dispatch(setIsCancelable(false));

// 		const state = getState() as RootState;

// 		const {
// 			stageScale,
// 			boundingBoxCoordinates,
// 			boundingBoxDimensions,
// 			stageCoordinates,
// 		} = state.canvas as CanvasState;

// 		const canvasBaseLayer = getCanvasBaseLayer();

// 		if (!canvasBaseLayer) {
// 			dispatch(setIsProcessing(false));
// 			dispatch(setIsCancelable(true));

// 			return;
// 		}

// 		const { dataURL, boundingBox: originalBoundingBox } = layerToDataURL(
// 			canvasBaseLayer,
// 			stageScale,
// 			stageCoordinates,
// 			cropToBoundingBox
// 				? { ...boundingBoxCoordinates, ...boundingBoxDimensions }
// 				: undefined,
// 		);

// 		if (!dataURL) {
// 			dispatch(setIsProcessing(false));
// 			dispatch(setIsCancelable(true));
// 			return;
// 		}

// 		const formData = new FormData();

// 		formData.append(
// 			'data',
// 			JSON.stringify({
// 				dataURL,
// 				filename: 'merged_canvas.png',
// 				kind: shouldSaveToGallery ? 'result' : 'temp',
// 				cropVisible,
// 			}),
// 		);

// 		const response = await fetch(window.location.origin + '/upload', {
// 			method: 'POST',
// 			body: formData,
// 		});

// 		const image = (await response.json()) as ImageUploadResponse;

// 		const { url, width, height } = image;

// 		const newImage: Image = {
// 			uuid: uuidv4(),
// 			category: shouldSaveToGallery ? 'result' : 'user',
// 			...image,
// 		};

// 		if (shouldDownload) {
// 			downloadFile(url);
// 			dispatch(
// 				addToast({
// 					title: 'Image Download Started',
// 					status: 'success',
// 					duration: 2500,
// 					isClosable: true,
// 				}),
// 			);
// 		}

// 		if (shouldCopy) {
// 			copyImage(url, width, height);
// 			dispatch(
// 				addToast({
// 					title: 'Image Copied',
// 					status: 'success',
// 					duration: 2500,
// 					isClosable: true,
// 				}),
// 			);
// 		}

// 		if (shouldSetAsInitialImage) {
// 			dispatch(
// 				setMergedCanvas({
// 					kind: 'image',
// 					layer: 'base',
// 					...originalBoundingBox,
// 					image: newImage,
// 				}),
// 			);
// 			dispatch(
// 				addToast({
// 					title: 'Canvas Merged',
// 					status: 'success',
// 					duration: 2500,
// 					isClosable: true,
// 				}),
// 			);
// 		}
// 		dispatch(setIsProcessing(false));
// 		dispatch(setCurrentStatus('Connected'));
// 		dispatch(setIsCancelable(true));
// 	};
