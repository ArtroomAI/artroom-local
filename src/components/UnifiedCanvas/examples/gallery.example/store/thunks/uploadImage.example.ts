export {};
// import { AnyAction, ThunkAction } from '@reduxjs/toolkit';
// import {RootState} from 'app/store';
// import {Image, ImageUploadResponse} from 'painter';
// import {v4 as uuidv4} from 'uuid';
// import {setInitialCanvasImageAction} from '../../../canvas/store/atoms';
// import {useSetRecoilState} from 'recoil';
// import { setInitialCanvasImage } from 'canvas/store/canvasSlice';
// import {addImage} from '../gallerySlice';

// type UploadImageConfig = {
// 	imageFile: File;
// };

// export const uploadImage =
// 	(
// 		config: UploadImageConfig,
// 	): ThunkAction<void, RootState, unknown, AnyAction> =>
// 	async (dispatch, getState) => {
// 		const { imageFile } = config;
// 		const setInitialCanvasImage = useSetRecoilState(
// 			setInitialCanvasImageAction,
// 		);
//
// 		const state = getState() as RootState;
//
// 		const formData = new FormData();
//
// 		formData.append('file', imageFile, imageFile.name);
// 		formData.append(
// 			'data',
// 			JSON.stringify({
// 				kind: 'init',
// 			}),
// 		);
//
// 		const response = await fetch(window.location.origin + '/upload', {
// 			method: 'POST',
// 			body: formData,
// 		});
//
// 		const image = (await response.json()) as ImageUploadResponse;
// 		console.log(image);
// 		const newImage: Image = {
// 			uuid: uuidv4(),
// 			category: 'user',
// 			...image,
// 		};
//
// 		setInitialCanvasImage(newImage);
// 	};
