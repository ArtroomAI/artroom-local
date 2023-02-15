import { FC, useLayoutEffect } from 'react';
import { Box, Flex } from '@chakra-ui/react';
import _ from 'lodash';
import { useRecoilState } from 'recoil';
import { ImageUploader } from './components';
import {
	CanvasResizer,
	Canvas,
	CanvasOutpaintingControls,
} from './canvas.components';
import { doesCanvasNeedScalingAtom } from './atoms/canvas.atoms';

export const UnifiedCanvas: FC = () => {
	const [doesCanvasNeedScaling, setDoesCanvasNeedScaling] = useRecoilState(
		doesCanvasNeedScalingAtom,
	);

	useLayoutEffect(() => {
		setDoesCanvasNeedScaling(true);

		const resizeCallback = _.debounce(() => {
			setDoesCanvasNeedScaling(true);
		}, 250);

		window.addEventListener('resize', resizeCallback);

		return () => window.removeEventListener('resize', resizeCallback);
	}, []);

	// const getImageByUuid = useGetImageByUuid();
	// const setInitialCanvasImage = useSetRecoilState(setInitialCanvasImageAction)

	return (
		<Box>
			<ImageUploader>
				<Box height="65vh">
					<Flex
						direction="column"
						alignItems="center"
						rowGap="1rem"
						width="100%"
						height="100%">
						<CanvasOutpaintingControls />
						<Flex
							flexDirection="column"
							alignItems="center"
							justifyContent="center"
							rowGap="1rem"
							width="100%"
							height="100%">
							{doesCanvasNeedScaling ? (
								<CanvasResizer />
							) : (
								<Canvas />
							)}
						</Flex>
					</Flex>
				</Box>
			</ImageUploader>
		</Box>
	);
};
