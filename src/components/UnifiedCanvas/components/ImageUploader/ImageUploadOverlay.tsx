import { FC } from 'react';
import { Heading, Box, Flex, useColorModeValue } from '@chakra-ui/react';
import { SystemProps } from '@chakra-ui/system';
import { useHotkeys } from 'react-hotkeys-hook';

type ImageUploadOverlayProps = {
	isDragAccept: boolean;
	isDragReject: boolean;
	overlaySecondaryText: string;
	setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

export const ImageUploadOverlay: FC<ImageUploadOverlayProps> = props => {
	const {
		isDragAccept,
		isDragReject,
		overlaySecondaryText,
		setIsHandlingUpload,
	} = props;

	useHotkeys('esc', () => {
		setIsHandlingUpload(false);
	});

	const dropzoneOverlayStyles = {
		opacity: 0.5,
		width: '100%',
		height: '100%',
		direction: 'column' as SystemProps['flexDirection'],
		rowGap: '1rem',
		alignItems: 'center',
		justifyContent: 'center',
		bg: useColorModeValue('#dcdee0', '#1a1a20'),
	};

	return (
		<Box
			position="absolute"
			top={0}
			left={0}
			width="100vw"
			height="100vh"
			zIndex={1999}
			backdropFilter="blur(20px)">
			{isDragAccept && (
				<Flex
					{...dropzoneOverlayStyles}
					boxShadow="inset 0 0 20rem 1rem #ebb905">
					<Heading size={'lg'}>
						Upload Image{overlaySecondaryText}
					</Heading>
				</Flex>
			)}
			{isDragReject && (
				<Flex
					{...dropzoneOverlayStyles}
					boxShadow={`inset 0 0 20rem 1rem ${useColorModeValue(
						'#ca0000',
						'#ff5a5a',
					)}`}>
					<Heading size={'lg'}>Invalid Upload</Heading>
					<Heading size={'md'}>
						Must be single JPEG or PNG image
					</Heading>
				</Flex>
			)}
		</Box>
	);
};
