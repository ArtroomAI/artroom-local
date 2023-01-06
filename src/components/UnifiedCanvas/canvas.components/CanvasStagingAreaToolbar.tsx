import React from 'react';
import { ButtonGroup, Flex } from '@chakra-ui/react';
import { useCallback, FC } from 'react';
import {
  FaArrowLeft,
  FaArrowRight,
  FaCheck,
  FaEye,
  FaEyeSlash,
  FaPlus,
  // FaSave,
} from 'react-icons/fa';
import { useHotkeys } from 'react-hotkeys-hook';
import { useSetRecoilState, useRecoilState, useRecoilValue } from 'recoil';
import {
  discardStagedImagesAction,
  shouldShowStagingImageAtom,
  shouldShowStagingOutlineAtom,
  commitStagingAreaImageAction,
  nextStagingAreaImageAction,
  prevStagingAreaImageAction,
  layerStateAtom,
} from '../atoms/canvas.atoms';
import { IconButton } from '../components';

// import _ from 'lodash';
// import { canvasSelector } from 'canvas/store/canvasSelectors';
// import {
// 	commitStagingAreaImage,
// 	discardStagedImages,
// 	nextStagingAreaImage,
// 	prevStagingAreaImage,
// 	setShouldShowStagingImage,
// 	setShouldShowStagingOutline,
// } from 'canvas/store/canvasSlice';
// import { saveStagingAreaImageToGallery } from 'app/socketio/actions';

// const selector = createSelector(
// 	[canvasSelector],
// 	canvas => {
// 		const {
// 			layerState: {
// 				stagingArea: { images, selectedImageIndex },
// 			},
// 			shouldShowStagingOutline,
// 			shouldShowStagingImage,
// 		} = canvas;

// 		return {
// 			currentStagingAreaImage:
// 				images.length > 0 ? images[selectedImageIndex] : undefined,
// 			isOnFirstImage: selectedImageIndex === 0,
// 			isOnLastImage: selectedImageIndex === images.length - 1,
// 			shouldShowStagingImage,
// 			shouldShowStagingOutline,
// 		};
// 	},
// 	{
// 		memoizeOptions: {
// 			resultEqualityCheck: _.isEqual,
// 		},
// 	},
// );

export const CanvasStagingAreaToolbar: FC = () => {
  // const {
  // 	isOnFirstImage,
  // 	isOnLastImage,
  // 	currentStagingAreaImage,
  // 	shouldShowStagingImage,
  // } = useAppSelector(selector);

  const discardStagedImages = useSetRecoilState(discardStagedImagesAction);
  const setShouldShowStagingOutline = useSetRecoilState(
    shouldShowStagingOutlineAtom
  );
  const [shouldShowStagingImage, setShouldShowStagingImage] = useRecoilState(
    shouldShowStagingImageAtom
  );
  const commitStagingAreaImage = useSetRecoilState(
    commitStagingAreaImageAction
  );
  const nextStagingAreaImage = useSetRecoilState(nextStagingAreaImageAction);
  const prevStagingAreaImage = useSetRecoilState(prevStagingAreaImageAction);
  const layerState = useRecoilValue(layerStateAtom);
  const {
    stagingArea: { images, selectedImageIndex },
  } = layerState;

  const currentStagingAreaImage =
    images.length > 0 ? images[selectedImageIndex] : undefined;
  const isOnFirstImage = selectedImageIndex === 0;
  const isOnLastImage = selectedImageIndex === images.length - 1;

  const handleMouseOver = useCallback(() => {
    setShouldShowStagingOutline(true);
  }, []);

  const handleMouseOut = useCallback(() => {
    setShouldShowStagingOutline(false);
  }, []);

  useHotkeys(
    ['left'],
    () => {
      handlePrevImage();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['right'],
    () => {
      handleNextImage();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  useHotkeys(
    ['enter'],
    () => {
      handleAccept();
    },
    {
      enabled: () => true,
      preventDefault: true,
    }
  );

  const handlePrevImage = () => prevStagingAreaImage();
  const handleNextImage = () => nextStagingAreaImage();
  const handleAccept = () => commitStagingAreaImage();

  if (!currentStagingAreaImage) return null;

  return (
    <Flex
      pos="absolute"
      bottom="1rem"
      w="100%"
      align="center"
      justify="center"
      filter="drop-shadow(0 0.5rem 1rem rgba(0,0,0))"
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <ButtonGroup isAttached>
        <IconButton
          tooltip="Previous (Left)"
          aria-label="Previous (Left)"
          icon={<FaArrowLeft />}
          onClick={handlePrevImage}
          data-selected
          isDisabled={isOnFirstImage}
        />
        <IconButton
          tooltip="Next (Right)"
          aria-label="Next (Right)"
          icon={<FaArrowRight />}
          onClick={handleNextImage}
          data-selected
          isDisabled={isOnLastImage}
        />
        <IconButton
          tooltip="Accept (Enter)"
          aria-label="Accept (Enter)"
          icon={<FaCheck />}
          onClick={handleAccept}
          data-selected
        />
        <IconButton
          tooltip="Show/Hide"
          aria-label="Show/Hide"
          data-alert={!shouldShowStagingImage}
          icon={shouldShowStagingImage ? <FaEye /> : <FaEyeSlash />}
          onClick={() => setShouldShowStagingImage(!shouldShowStagingImage)}
          data-selected
        />
        {/* <IconButton
					tooltip="Save to Gallery"
					aria-label="Save to Gallery"
					icon={<FaSave/>}
					onClick={() =>
						dispatch(
							saveStagingAreaImageToGallery(currentStagingAreaImage.image.url)
						)
					}
					data-selected={true}
				/> */}
        <IconButton
          tooltip="Discard All"
          aria-label="Discard All"
          icon={<FaPlus style={{ transform: 'rotate(45deg)' }} />}
          onClick={() => discardStagedImages()}
          data-selected
          style={{ backgroundColor: 'var(--btn-delete-image)' }}
          fontSize={20}
        />
      </ButtonGroup>
    </Flex>
  );
};
