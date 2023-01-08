export {};
// import { Middleware } from '@reduxjs/toolkit';
// import { io } from 'socket.io-client';
//
// import makeSocketIOListeners from './listeners';
// import makeSocketIOEmitters from './emitters';
//
// import {
// 	ErrorResponse,
// 	ImageResultResponse,
// 	SystemStatus,
// 	GalleryImagesResponse,
// 	ImageDeletedResponse,
// 	SystemConfig,
// 	ModelChangeResponse,
// } from 'painter';
//
// /**
//  * Creates a socketio middleware to handle communication with server.
//  *
//  * Special `socketio/actionName` actions are created in actions.ts and
//  * exported for use by the application, which treats them like any old
//  * action, using `dispatch` to dispatch them.
//  *
//  * These actions are intercepted here, where `socketio.emit()` calls are
//  * made on their behalf - see `emitters.ts`. The emitter functions
//  * are the outbound communication to the server.
//  *
//  * Listeners are also established here - see `listeners.ts`. The listener
//  * functions receive communication from the server and usually dispatch
//  * some new action to handle whatever data was sent from the server.
//  */
// export const socketioMiddleware = () => {
// 	const { origin } = new URL(window.location.href);
//
// 	const socketio = io(origin, {
// 		timeout: 60000,
// 		path: window.location.pathname + 'socket.io',
// 	});
//
// 	let areListenersSet = false;
//
// 	const middleware: Middleware = store => next => action => {
// 		const {
// 			onConnect,
// 			onDisconnect,
// 			onError,
// 			onPostprocessingResult,
// 			onGenerationResult,
// 			onIntermediateResult,
// 			onProgressUpdate,
// 			onGalleryImages,
// 			onProcessingCanceled,
// 			onImageDeleted,
// 			onSystemConfig,
// 			onModelChanged,
// 			onModelChangeFailed,
// 			onTempFolderEmptied,
// 		} = makeSocketIOListeners(store);
//
// 		const {
// 			emitGenerateImage,
// 			emitRunESRGAN,
// 			emitRunFacetool,
// 			emitDeleteImage,
// 			emitRequestImages,
// 			emitRequestNewImages,
// 			emitCancelProcessing,
// 			emitRequestSystemConfig,
// 			emitRequestModelChange,
// 			emitSaveStagingAreaImageToGallery,
// 			emitRequestEmptyTempFolder,
// 		} = makeSocketIOEmitters(store, socketio);
//
// 		/**
// 		 * If this is the first time the middleware has been called (e.g. during store setup),
// 		 * initialize all our socket.io listeners.
// 		 */
// 		if (!areListenersSet) {
// 			socketio.on('connect', () => onConnect());
//
// 			socketio.on('disconnect', () => onDisconnect());
//
// 			socketio.on('error', (data: ErrorResponse) => onError(data));
//
// 			socketio.on('generationResult', (data: ImageResultResponse) =>
// 				onGenerationResult(data),
// 			);
//
// 			socketio.on('postprocessingResult', (data: ImageResultResponse) =>
// 				onPostprocessingResult(data),
// 			);
//
// 			socketio.on('intermediateResult', (data: ImageResultResponse) =>
// 				onIntermediateResult(data),
// 			);
//
// 			socketio.on('progressUpdate', (data: SystemStatus) =>
// 				onProgressUpdate(data),
// 			);
//
// 			socketio.on('galleryImages', (data: GalleryImagesResponse) =>
// 				onGalleryImages(data),
// 			);
//
// 			socketio.on('processingCanceled', () => {
// 				onProcessingCanceled();
// 			});
//
// 			socketio.on('imageDeleted', (data: ImageDeletedResponse) => {
// 				onImageDeleted(data);
// 			});
//
// 			socketio.on('systemConfig', (data: SystemConfig) => {
// 				onSystemConfig(data);
// 			});
//
// 			socketio.on('modelChanged', (data: ModelChangeResponse) => {
// 				onModelChanged(data);
// 			});
//
// 			socketio.on('modelChangeFailed', (data: ModelChangeResponse) => {
// 				onModelChangeFailed(data);
// 			});
//
// 			socketio.on('tempFolderEmptied', () => {
// 				onTempFolderEmptied();
// 			});
//
// 			areListenersSet = true;
// 		}
//
// 		/**
// 		 * Handle redux actions caught by middleware.
// 		 */
// 		switch (action.type) {
// 			case 'socketio/generateImage': {
// 				emitGenerateImage(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/runESRGAN': {
// 				emitRunESRGAN(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/runFacetool': {
// 				emitRunFacetool(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/deleteImage': {
// 				emitDeleteImage(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/requestImages': {
// 				emitRequestImages(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/requestNewImages': {
// 				emitRequestNewImages(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/cancelProcessing': {
// 				emitCancelProcessing();
// 				break;
// 			}
//
// 			case 'socketio/requestSystemConfig': {
// 				emitRequestSystemConfig();
// 				break;
// 			}
//
// 			case 'socketio/requestModelChange': {
// 				emitRequestModelChange(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/saveStagingAreaImageToGallery': {
// 				emitSaveStagingAreaImageToGallery(action.payload);
// 				break;
// 			}
//
// 			case 'socketio/requestEmptyTempFolder': {
// 				emitRequestEmptyTempFolder();
// 				break;
// 			}
// 		}
//
// 		next(action);
// 	};
//
// 	return middleware;
// };
