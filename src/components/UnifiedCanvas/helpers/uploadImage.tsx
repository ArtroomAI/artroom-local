import _ from 'lodash';
import { v4 as uuidv4 } from 'uuid';
import { LAYER_REG_EXP, getNameForLayer } from '../../../utils';
import { Vector2d } from 'konva/lib/types';
import { Dimensions, CanvasLayerState } from '../atoms/canvasTypes';
import { SetterOrUpdater } from 'recoil';

// const LOCAL_URL = process.env._LOCAL_URL;
// const ARTROOM_URL = import.meta.env.VITE_ARTROOM_URL;

function getImageSize(url: string) {
	const img = document.createElement('img');

	const promise = new Promise<Dimensions>((resolve, reject) => {
		img.onload = () => {
			// Natural size is the actual image size regardless of rendering.
			// The 'normal' `width`/`height` are for the **rendered** size.
			const width = img.naturalWidth;
			const height = img.naturalHeight;
			// Resolve promise with the width and height
			resolve({ width, height });
		};
		// Reject promise on error
		img.onerror = reject;
	});
	// Setting the source makes it start downloading and eventually call `onload`
	img.src = url;

	return promise;
}

interface UploadImageArgument {
	imageFile: File;
	boundingBoxCoordinates: Vector2d;
	boundingBoxDimensions: Dimensions;
	setPastLayerStates: SetterOrUpdater<CanvasLayerState[]>;
	pastLayerStates: CanvasLayerState[];
	layerState: CanvasLayerState;
	maxHistory: number;
	setLayerState: SetterOrUpdater<CanvasLayerState>;
	setFutureLayerStates: SetterOrUpdater<CanvasLayerState[]>;
	setLayer: SetterOrUpdater<string>;
	setPastLayer: SetterOrUpdater<string>;
	setFutureLayer: SetterOrUpdater<string>;
	layer: string;
}

export const uploadImage = async ({
	imageFile,
	boundingBoxCoordinates,
	boundingBoxDimensions,
	setPastLayerStates,
	pastLayerStates,
	layerState,
	maxHistory,
	setLayerState,
	setFutureLayerStates,
	// setInitialCanvasImage,
	setLayer,
	setPastLayer,
	setFutureLayer,
	layer,
}: UploadImageArgument) => {
	const boundingBox = {
		...boundingBoxCoordinates,
		...boundingBoxDimensions,
	};

	const imageURL = URL.createObjectURL(imageFile);

	const imageDimensions = await getImageSize(imageURL);

	// if (layerState.objects.length === 0) {
	// if (layerState.images.length === 0) {
	// setInitialCanvasImage(newImage);

	setPastLayerStates([...pastLayerStates, _.cloneDeep(layerState)]);
	setPastLayer(layer);

	if (pastLayerStates.length > maxHistory) {
		setPastLayerStates(pastLayerStates.slice(1));
	}

	const newLayerId = uuidv4();
	setLayerState({
		...layerState,
		images: [
			...layerState.images,
			{
				name: getNameForLayer(LAYER_REG_EXP, layerState.images),
				id: newLayerId,
				picture: {
					kind: 'image',
					uuid: newLayerId,
					// layer: 'base',
					...boundingBox,
					width: imageDimensions.width,
					height: imageDimensions.height,
					// image: newImage,
					url: URL.createObjectURL(imageFile),
				},
				opacity: 1,
				isVisible: true,
			},
		],
	});
	setFutureLayerStates([]);
	setFutureLayer('');
	setLayer(newLayerId);
};
