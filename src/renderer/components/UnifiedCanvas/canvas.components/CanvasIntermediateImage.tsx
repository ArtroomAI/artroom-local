import React from 'react';
import { FC } from 'react';
import { ImageConfig } from 'konva/lib/shapes/Image';

// import { RootState } from 'app/store';
// import { GalleryState } from 'gallery/store/gallerySlice';
// import _ from 'lodash';
// import { Image as KonvaImage } from 'react-konva';

// const selector = createSelector(
//   [(state: RootState) => state.gallery],
//   (gallery: GalleryState) => {
//     return gallery.intermediateImage ? gallery.intermediateImage : null;
//   },
//   {
//     memoizeOptions: {
//       resultEqualityCheck: _.isEqual,
//     },
//   }
// );

type Props = Omit<ImageConfig, 'image'>;

export const CanvasIntermediateImage: FC<Props> = props => {
	const { ...rest } = props;
	// const intermediateImage = useAppSelector(selector);

	// const [loadedImageElement, setLoadedImageElement] =
	//   useState<HTMLImageElement | null>(null);
	//
	// useEffect(() => {
	//   if (!intermediateImage) return;
	//   const tempImage = new Image();
	//
	//   tempImage.onload = () => {
	//     setLoadedImageElement(tempImage);
	//   };
	//   tempImage.src = intermediateImage.url;
	// }, [intermediateImage]);
	//
	// if (!intermediateImage?.boundingBox) return null;
	//
	// const {
	//   boundingBox: { x, y, width, height },
	// } = intermediateImage;

	return null;
	// return loadedImageElement ? (
	//   <KonvaImage
	//     x={x}
	//     y={y}
	//     width={width}
	//     height={height}
	//     image={loadedImageElement}
	//     listening={false}
	//     {...rest}
	//   />
	// ) : null;
};
