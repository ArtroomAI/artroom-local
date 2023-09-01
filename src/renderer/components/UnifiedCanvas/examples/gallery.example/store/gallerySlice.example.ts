import { Image } from 'painter'
import { IRect } from 'konva/lib/types'

// import { createSlice, PayloadAction } from '@reduxjs/toolkit';
// import { clamp } from 'lodash';

export type GalleryCategory = 'user' | 'result'

// export type AddImagesPayload = {
// 	images: Array<Image>;
// 	areMoreImagesAvailable: boolean;
// 	category: GalleryCategory;
// };

type Gallery = {
  images: Image[]
  latest_mtime?: number
  earliest_mtime?: number
  areMoreImagesAvailable: boolean
}

interface GalleryState {
  currentImage?: Image
  currentImageUuid: string
  intermediateImage?: Image & {
    boundingBox?: IRect
    generationMode?: 'unifiedCanvas'
  }
  shouldHoldGalleryOpen: boolean
  shouldAutoSwitchToNewImages: boolean
  categories: {
    user: Gallery
    result: Gallery
  }
  currentCategory: GalleryCategory
}

const initialState: GalleryState = {
  currentImageUuid: '',
  shouldHoldGalleryOpen: false,
  shouldAutoSwitchToNewImages: true,
  currentCategory: 'result',
  categories: {
    user: {
      images: [],
      latest_mtime: undefined,
      earliest_mtime: undefined,
      areMoreImagesAvailable: true,
    },
    result: {
      images: [],
      latest_mtime: undefined,
      earliest_mtime: undefined,
      areMoreImagesAvailable: true,
    },
  },
}

// export const gallerySlice = createSlice({
// 	name: 'gallery',
// 	initialState,
// 	reducers: {
// 		removeImage: (state, action: PayloadAction<ImageDeletedResponse>) => {
// 			const { uuid, category } = action.payload;

// 			const tempImages =
// 				state.categories[category as GalleryCategory].images;

// 			const newImages = tempImages.filter(image => image.uuid !== uuid);

// 			if (uuid === state.currentImageUuid) {
// 				/**
// 				 * We are deleting the currently selected image.
// 				 *
// 				 * We want the new currentl selected image to be under the cursor in the
// 				 * gallery, so we need to do some fanagling. The currently selected image
// 				 * is set by its UUID, not its index in the image list.
// 				 *
// 				 * Get the currently selected image's index.
// 				 */
// 				const imageToDeleteIndex = tempImages.findIndex(
// 					image => image.uuid === uuid,
// 				);

// 				/**
// 				 * New current image needs to be in the same spot, but because the gallery
// 				 * is sorted in reverse order, the new current image's index will actuall be
// 				 * one less than the deleted image's index.
// 				 *
// 				 * Clamp the new index to ensure it is valid..
// 				 */
// 				const newCurrentImageIndex = clamp(
// 					imageToDeleteIndex,
// 					0,
// 					newImages.length - 1,
// 				);

// 				state.currentImage = newImages.length
// 					? newImages[newCurrentImageIndex]
// 					: undefined;

// 				state.currentImageUuid = newImages.length
// 					? newImages[newCurrentImageIndex].uuid
// 					: '';
// 			}

// 			state.categories[category as GalleryCategory].images = newImages;
// 		},
// 		addImage: (
// 			state,
// 			action: PayloadAction<{
// 				image: Image;
// 				category: GalleryCategory;
// 			}>,
// 		) => {
// 			const { image: newImage, category } = action.payload;
// 			const { uuid, url, mtime } = newImage;

// 			const tempCategory = state.categories[category as GalleryCategory];

// 			// Do not add duplicate images
// 			if (
// 				tempCategory.images.find(
// 					i => i.url === url && i.mtime === mtime,
// 				)
// 			) {
// 				return;
// 			}

// 			tempCategory.images.unshift(newImage);
// 			if (state.shouldAutoSwitchToNewImages) {
// 				state.currentImageUuid = uuid;
// 				state.currentImage = newImage;
// 				state.currentCategory = category;
// 			}
// 			state.intermediateImage = undefined;
// 			tempCategory.latest_mtime = mtime;
// 		},
// 		setIntermediateImage: (
// 			state,
// 			action: PayloadAction<
// 				Image & { boundingBox?: IRect; generationMode?: InvokeTabName }
// 			>,
// 		) => {
// 			state.intermediateImage = action.payload;
// 		},
// 		clearIntermediateImage: state => {
// 			state.intermediateImage = undefined;
// 		},
// 		addGalleryImages: (state, action: PayloadAction<AddImagesPayload>) => {
// 			const { images, areMoreImagesAvailable, category } = action.payload;
// 			const tempImages = state.categories[category].images;

// 			// const prevImages = category === 'user' ? state.userImages : state.resultImages

// 			if (images.length > 0) {
// 				// Filter images that already exist in the gallery
// 				const newImages = images.filter(
// 					newImage =>
// 						!tempImages.find(
// 							i =>
// 								i.url === newImage.url &&
// 								i.mtime === newImage.mtime,
// 						),
// 				);
// 				state.categories[category].images = tempImages
// 					.concat(newImages)
// 					.sort((a, b) => b.mtime - a.mtime);

// 				if (!state.currentImage) {
// 					const newCurrentImage = images[0];
// 					state.currentImage = newCurrentImage;
// 					state.currentImageUuid = newCurrentImage.uuid;
// 				}

// 				// keep track of the timestamps of latest and earliest images received
// 				state.categories[category].latest_mtime = images[0].mtime;
// 				state.categories[category].earliest_mtime =
// 					images[images.length - 1].mtime;
// 			}

// 			if (areMoreImagesAvailable !== undefined) {
// 				state.categories[category].areMoreImagesAvailable =
// 					areMoreImagesAvailable;
// 			}
// 		},
// 	},
// });

// export const {
// 	addImage,
// 	clearIntermediateImage,
// 	removeImage,
// 	addGalleryImages,
// 	setIntermediateImage,
// } = gallerySlice.actions;

// export default gallerySlice.reducer;
