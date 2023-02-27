export {};
// import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
// import { Socket } from 'socket.io-client';
// import { Image } from 'painter';
// import type { RootState } from 'app/store';
// import {
// 	GalleryCategory,
// 	GalleryState,
// 	removeImage,
// } from 'gallery/store/gallerySlice';
// import {
// 	frontendToBackendParameters,
// 	FrontendToBackendParametersConfig,
// } from 'common/util/parameterTranslation';

// import dateFormat from 'dateformat';
// import { OptionsState } from 'options/store/optionsSlice';
// import {
// 	addLogEntry,
// 	generationRequested,
// 	modelChangeRequested,
// 	setIsProcessing,
// } from 'system/store/systemSlice';

// /**
//  * Returns an object containing all functions which use `socketio.emit()`.
//  * i.e. those which make server requests.
//  */
// const makeSocketIOEmitters = (
// 	store: MiddlewareAPI<Dispatch<AnyAction>, RootState>,
// 	socketio: Socket,
// ) => {
// 	// We need to dispatch actions to redux and get pieces of state from the store.
// 	const { dispatch, getState } = store;

// 	return {
// 		emitGenerateImage: (generationMode: 'unifiedCanvas') => {
// 			dispatch(setIsProcessing(true));

// 			const state: RootState = getState();

// 			const {
// 				options: optionsState,
// 				system: systemState,
// 				canvas: canvasState,
// 			} = state;

// 			const frontendToBackendParametersConfig: FrontendToBackendParametersConfig =
// 				{
// 					generationMode,
// 					optionsState,
// 					canvasState,
// 					systemState,
// 				};

// 			dispatch(generationRequested());

// 			const {
// 				generationParameters,
// 				esrganParameters,
// 				facetoolParameters,
// 			} = frontendToBackendParameters(frontendToBackendParametersConfig);

// 			socketio.emit(
// 				'generateImage',
// 				generationParameters,
// 				esrganParameters,
// 				facetoolParameters,
// 			);

// 			// we need to truncate the init_mask base64 else it takes up the whole log
// 			// TODO: handle maintaining masks for reproducibility in future
// 			if (generationParameters.init_mask) {
// 				generationParameters.init_mask = generationParameters.init_mask
// 					.substr(0, 64)
// 					.concat('...');
// 			}
// 			if (generationParameters.init_img) {
// 				generationParameters.init_img = generationParameters.init_img
// 					.substr(0, 64)
// 					.concat('...');
// 			}

// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Image generation requested: ${JSON.stringify({
// 						...generationParameters,
// 						...esrganParameters,
// 						...facetoolParameters,
// 					})}`,
// 				}),
// 			);
// 		},
// 		emitRunESRGAN: (imageToProcess: Image) => {
// 			dispatch(setIsProcessing(true));
// 			const options: OptionsState = getState().options;
// 			const { upscalingLevel, upscalingStrength } = options;
// 			const esrganParameters = {
// 				upscale: [upscalingLevel, upscalingStrength],
// 			};
// 			socketio.emit('runPostprocessing', imageToProcess, {
// 				type: 'esrgan',
// 				...esrganParameters,
// 			});
// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `ESRGAN upscale requested: ${JSON.stringify({
// 						file: imageToProcess.url,
// 						...esrganParameters,
// 					})}`,
// 				}),
// 			);
// 		},
// 		emitRunFacetool: (imageToProcess: Image) => {
// 			dispatch(setIsProcessing(true));
// 			const options: OptionsState = getState().options;
// 			const { facetoolType, facetoolStrength, codeformerFidelity } =
// 				options;
// 			const facetoolParameters: Record<string, unknown> = {
// 				facetool_strength: facetoolStrength,
// 			};
// 			if (facetoolType === 'codeformer') {
// 				facetoolParameters.codeformer_fidelity = codeformerFidelity;
// 			}
// 			socketio.emit('runPostprocessing', imageToProcess, {
// 				type: facetoolType,
// 				...facetoolParameters,
// 			});
// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Face restoration (${facetoolType}) requested: ${JSON.stringify(
// 						{
// 							file: imageToProcess.url,
// 							...facetoolParameters,
// 						},
// 					)}`,
// 				}),
// 			);
// 		},
// 		emitDeleteImage: (imageToDelete: Image) => {
// 			const { url, uuid, category, thumbnail } = imageToDelete;
// 			dispatch(removeImage(imageToDelete));
// 			socketio.emit('deleteImage', url, thumbnail, uuid, category);
// 		},
// 		emitRequestImages: (category: GalleryCategory) => {
// 			const gallery: GalleryState = getState().gallery;
// 			const { earliest_mtime } = gallery.categories[category];
// 			socketio.emit('requestImages', category, earliest_mtime);
// 		},
// 		emitRequestNewImages: (category: GalleryCategory) => {
// 			const gallery: GalleryState = getState().gallery;
// 			const { latest_mtime } = gallery.categories[category];
// 			socketio.emit('requestLatestImages', category, latest_mtime);
// 		},
// 		emitCancelProcessing: () => {
// 			socketio.emit('cancel');
// 		},
// 		emitRequestSystemConfig: () => {
// 			socketio.emit('requestSystemConfig');
// 		},
// 		emitRequestModelChange: (modelName: string) => {
// 			dispatch(modelChangeRequested());
// 			socketio.emit('requestModelChange', modelName);
// 		},
// 		emitSaveStagingAreaImageToGallery: (url: string) => {
// 			socketio.emit('requestSaveStagingAreaImageToGallery', url);
// 		},
// 		emitRequestEmptyTempFolder: () => {
// 			socketio.emit('requestEmptyTempFolder');
// 		},
// 	};
// };

// export default makeSocketIOEmitters;
