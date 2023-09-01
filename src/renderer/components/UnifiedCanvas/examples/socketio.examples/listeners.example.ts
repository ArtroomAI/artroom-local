export {}
// import { AnyAction, MiddlewareAPI, Dispatch } from '@reduxjs/toolkit';
// import { v4 as uuidv4 } from 'uuid';
// import {
// 	ImageResultResponse,
// 	ErrorResponse,
// 	SystemStatus,
// 	GalleryImagesResponse,
// 	ImageDeletedResponse,
// 	Image,
// 	SystemConfig,
// 	ModelChangeResponse,
// } from 'painter';
// import dateFormat from 'dateformat';
// import {
// 	addLogEntry,
// 	setIsConnected,
// 	setIsProcessing,
// 	setSystemStatus,
// 	setCurrentStatus,
// 	setSystemConfig,
// 	processingCanceled,
// 	errorOccurred,
// 	setModelList,
// 	setIsCancelable,
// 	addToast,
// } from 'system/store/systemSlice';

// import {
// 	addGalleryImages,
// 	addImage,
// 	clearIntermediateImage,
// 	GalleryState,
// 	removeImage,
// 	setIntermediateImage,
// } from 'gallery/store/gallerySlice';

// import {
// 	clearInitialImage,
// 	setInfillMethod,
// 	setInitialImage,
// 	setMaskPath,
// } from 'options/store/optionsSlice';
// import {
// 	requestImages,
// 	requestNewImages,
// 	requestSystemConfig,
// } from './actions';
// import { addImageToStagingArea } from 'canvas/store/canvasSlice';
// import { tabMap } from 'tabs/tabMap';
// import type { RootState } from 'app/store';
// import { useSetRecoilState } from 'recoil';
// import { addImageToStagingAreaAction } from '../../canvas/store/atoms';

// /**
//  * Returns an object containing listener callbacks for socketio events.
//  * TODO: This file is large, but simple. Should it be split up further?
//  */
// const makeSocketIOListeners = (
// 	store: MiddlewareAPI<Dispatch<AnyAction>, RootState>,
// ) => {
// 	const { dispatch, getState } = store;

// 	return {
// 		/**
// 		 * Callback to run when we receive a 'connect' event.
// 		 */
// 		onConnect: () => {
// 			try {
// 				dispatch(setIsConnected(true));
// 				dispatch(setCurrentStatus('Connected'));
// 				dispatch(requestSystemConfig());
// 				const gallery: GalleryState = getState().gallery;

// 				if (gallery.categories.result.latest_mtime) {
// 					dispatch(requestNewImages('result'));
// 				} else {
// 					dispatch(requestImages('result'));
// 				}

// 				if (gallery.categories.user.latest_mtime) {
// 					dispatch(requestNewImages('user'));
// 				} else {
// 					dispatch(requestImages('user'));
// 				}
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'disconnect' event.
// 		 */
// 		onDisconnect: () => {
// 			try {
// 				dispatch(setIsConnected(false));
// 				dispatch(setCurrentStatus('Disconnected'));
// 				dispatch(
// 					addLogEntry({
// 						timestamp: dateFormat(new Date(), 'isoDateTime'),
// 						message: `Disconnected from server`,
// 						level: 'warning',
// 					}),
// 				);
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'generationResult' event.
// 		 */
// 		onGenerationResult: (data: ImageResultResponse) => {
// 			try {
// 				const state: RootState = getState();
// 				const { shouldLoopback, activeTab } = state.options;
// 				const { boundingBox: _, generationMode, ...rest } = data;
// 				const newImage = {
// 					uuid: uuidv4(),
// 					...rest,
// 				};
// 				if (['txt2img', 'img2img'].includes(generationMode)) {
// 					dispatch(
// 						addImage({
// 							category: 'result',
// 							image: { ...newImage, category: 'result' },
// 						}),
// 					);
// 				}
// 				if (generationMode === 'unifiedCanvas' && data.boundingBox) {
// 					const { boundingBox } = data;
// 					dispatch(
// 						addImageToStagingArea({
// 							image: { ...newImage, category: 'temp' },
// 							boundingBox,
// 						}),
// 					);
// 					if (state.canvas.shouldAutoSave) {
// 						dispatch(
// 							addImage({
// 								image: { ...newImage, category: 'result' },
// 								category: 'result',
// 							}),
// 						);
// 					}
// 				}
// 				if (shouldLoopback) {
// 					const activeTabName = tabMap[activeTab];
// 					switch (activeTabName) {
// 						case 'img2img': {
// 							dispatch(setInitialImage(newImage));
// 							break;
// 						}
// 					}
// 				}
// 				dispatch(clearIntermediateImage());
// 				dispatch(
// 					addLogEntry({
// 						timestamp: dateFormat(new Date(), 'isoDateTime'),
// 						message: `Image generated: ${data.url}`,
// 					}),
// 				);
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'intermediateResult' event.
// 		 */
// 		onIntermediateResult: (data: ImageResultResponse) => {
// 			try {
// 				dispatch(
// 					setIntermediateImage({
// 						uuid: uuidv4(),
// 						...data,
// 						category: 'result',
// 					}),
// 				);
// 				if (!data.isBase64) {
// 					dispatch(
// 						addLogEntry({
// 							timestamp: dateFormat(new Date(), 'isoDateTime'),
// 							message: `Intermediate image generated: ${data.url}`,
// 						}),
// 					);
// 				}
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive an 'esrganResult' event.
// 		 */
// 		onPostprocessingResult: (data: ImageResultResponse) => {
// 			try {
// 				dispatch(
// 					addImage({
// 						category: 'result',
// 						image: {
// 							uuid: uuidv4(),
// 							...data,
// 							category: 'result',
// 						},
// 					}),
// 				);
// 				dispatch(
// 					addLogEntry({
// 						timestamp: dateFormat(new Date(), 'isoDateTime'),
// 						message: `Postprocessed: ${data.url}`,
// 					}),
// 				);
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'progressUpdate' event.
// 		 * TODO: Add additional progress phases
// 		 */
// 		onProgressUpdate: (data: SystemStatus) => {
// 			try {
// 				dispatch(setIsProcessing(true));
// 				dispatch(setSystemStatus(data));
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'progressUpdate' event.
// 		 */
// 		onError: (data: ErrorResponse) => {
// 			const { message, additionalData } = data;

// 			if (additionalData) {
// 				// TODO: handle more data than short message
// 			}

// 			try {
// 				dispatch(
// 					addLogEntry({
// 						timestamp: dateFormat(new Date(), 'isoDateTime'),
// 						message: `Server error: ${message}`,
// 						level: 'error',
// 					}),
// 				);
// 				dispatch(errorOccurred());
// 				dispatch(clearIntermediateImage());
// 			} catch (e) {
// 				console.error(e);
// 			}
// 		},
// 		/**
// 		 * Callback to run when we receive a 'galleryImages' event.
// 		 */
// 		onGalleryImages: (data: GalleryImagesResponse) => {
// 			const { images, areMoreImagesAvailable, category } = data;

// 			/**
// 			 * the logic here ideally would be in the reducer but we have a side effect:
// 			 * generating a uuid. so the logic needs to be here, outside redux.
// 			 */

// 			// Generate a UUID for each image
// 			const preparedImages = images.map((image): Image => {
// 				return {
// 					uuid: uuidv4(),
// 					...image,
// 				};
// 			});

// 			dispatch(
// 				addGalleryImages({
// 					images: preparedImages,
// 					areMoreImagesAvailable,
// 					category,
// 				}),
// 			);

// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Loaded ${images.length} images`,
// 				}),
// 			);
// 		},
// 		/**
// 		 * Callback to run when we receive a 'processingCanceled' event.
// 		 */
// 		onProcessingCanceled: () => {
// 			dispatch(processingCanceled());
// 			const { intermediateImage } = getState().gallery;
// 			if (intermediateImage) {
// 				if (!intermediateImage.isBase64) {
// 					dispatch(
// 						addImage({
// 							category: 'result',
// 							image: intermediateImage,
// 						}),
// 					);
// 					dispatch(
// 						addLogEntry({
// 							timestamp: dateFormat(new Date(), 'isoDateTime'),
// 							message: `Intermediate image saved: ${intermediateImage.url}`,
// 						}),
// 					);
// 				}
// 				dispatch(clearIntermediateImage());
// 			}
// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Processing canceled`,
// 					level: 'warning',
// 				}),
// 			);
// 		},
// 		/**
// 		 * Callback to run when we receive a 'imageDeleted' event.
// 		 */
// 		onImageDeleted: (data: ImageDeletedResponse) => {
// 			const { url } = data;

// 			// remove image from gallery
// 			dispatch(removeImage(data));

// 			// remove references to image in options
// 			const { initialImage, maskPath } = getState().options;

// 			if (initialImage === url || (initialImage as Image)?.url === url) {
// 				dispatch(clearInitialImage());
// 			}

// 			if (maskPath === url) {
// 				dispatch(setMaskPath(''));
// 			}

// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Image deleted: ${url}`,
// 				}),
// 			);
// 		},
// 		onSystemConfig: (data: SystemConfig) => {
// 			dispatch(setSystemConfig(data));
// 			if (!data.infill_methods.includes('patchmatch')) {
// 				dispatch(setInfillMethod(data.infill_methods[0]));
// 			}
// 		},
// 		onModelChanged: (data: ModelChangeResponse) => {
// 			const { model_name, model_list } = data;
// 			dispatch(setModelList(model_list));
// 			dispatch(setCurrentStatus('Model Changed'));
// 			dispatch(setIsProcessing(false));
// 			dispatch(setIsCancelable(true));
// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Model changed: ${model_name}`,
// 					level: 'info',
// 				}),
// 			);
// 		},
// 		onModelChangeFailed: (data: ModelChangeResponse) => {
// 			const { model_name, model_list } = data;
// 			dispatch(setModelList(model_list));
// 			dispatch(setIsProcessing(false));
// 			dispatch(setIsCancelable(true));
// 			dispatch(errorOccurred());
// 			dispatch(
// 				addLogEntry({
// 					timestamp: dateFormat(new Date(), 'isoDateTime'),
// 					message: `Model change failed: ${model_name}`,
// 					level: 'error',
// 				}),
// 			);
// 		},
// 		onTempFolderEmptied: () => {
// 			dispatch(
// 				addToast({
// 					title: 'Temp Folder Emptied',
// 					status: 'success',
// 					duration: 2500,
// 					isClosable: true,
// 				}),
// 			);
// 		},
// 	};
// };

// export default makeSocketIOListeners;
