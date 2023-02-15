import React from 'react';
import { Button } from '@chakra-ui/react';
import { AlertDialog } from '../AlertDialog/AlertDialog';

export const UploadResultButton: React.FC = () => {
	return (
		<AlertDialog
			triggerComponent={<Button>Upload to ArtRoom</Button>}
			title="Upload to ArtRoom"
			acceptCallback={() => alert('Upload')}
			acceptButtonText="Upload"
			acceptButtonProps={{ colorScheme: undefined, _hover: undefined }}>
			Would you like to upload this image from your name to ArtRoom for
			public viewing?
		</AlertDialog>
	);
};
