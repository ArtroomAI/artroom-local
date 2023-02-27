export {};
// import { createAction } from '@reduxjs/toolkit';
// import { GalleryCategory } from 'gallery/store/gallerySlice';
// import { InvokeTabName } from 'tabs/tabMap';
// import { Image } from 'painter';

/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

// export const generateImage = createAction<InvokeTabName>(
// 	'socketio/generateImage',
// );
// export const runESRGAN = createAction<Image>('socketio/runESRGAN');
// export const runFacetool = createAction<Image>('socketio/runFacetool');
// export const deleteImage = createAction<Image>('socketio/deleteImage');
// export const requestImages = createAction<GalleryCategory>(
// 	'socketio/requestImages',
// );
// export const requestNewImages = createAction<GalleryCategory>(
// 	'socketio/requestNewImages',
// );
// export const cancelProcessing = createAction<undefined>(
// 	'socketio/cancelProcessing',
// );

// export const requestSystemConfig = createAction<undefined>(
// 	'socketio/requestSystemConfig',
// );

// export const requestModelChange = createAction<string>(
// 	'socketio/requestModelChange',
// );

// export const saveStagingAreaImageToGallery = createAction<string>(
// 	'socketio/saveStagingAreaImageToGallery',
// );
//
// export const emptyTempFolder = createAction<undefined>(
// 	'socketio/requestEmptyTempFolder',
// );
