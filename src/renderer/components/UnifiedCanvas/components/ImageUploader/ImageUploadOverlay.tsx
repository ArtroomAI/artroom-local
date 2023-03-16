import React, { FC } from 'react';
import { Heading } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';

type ImageUploadOverlayProps = {
	isDragAccept: boolean;
	isDragReject: boolean;
	setIsHandlingUpload: (isHandlingUpload: boolean) => void;
};

export const ImageUploadOverlay: FC<ImageUploadOverlayProps> = props => {
	const {
		isDragAccept,
		isDragReject,
		setIsHandlingUpload,
	} = props;

	useHotkeys('esc', () => {
		setIsHandlingUpload(false);
	});

	return (
		<div className="dropzone-container">
			{isDragAccept && (
				<div className="dropzone-overlay is-drag-accept">
					<Heading size={'lg'}>
						Upload Image
					</Heading>
				</div>
			)}
			{isDragReject && (
				<div className="dropzone-overlay is-drag-reject">
					<Heading size={'lg'}>Invalid Upload</Heading>
					<Heading size={'md'}>
						Must be single JPEG or PNG image
					</Heading>
				</div>
			)}
		</div>
	);
};
