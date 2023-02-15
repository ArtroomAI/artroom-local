import React from 'react';
import { Box, Text, Link as ChakraLink, Button } from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import { PromotedPictureType } from '../../types';
import { onSendAnalyticEvent } from '../../utils';

interface IArtCredentialsProps {
	pictureData: PromotedPictureType;
}

export const ArtCredentials: React.FC<IArtCredentialsProps> = ({
	pictureData,
}) => (
	<Box zIndex={2} bottom={3} right={3} position={'absolute'}>
		{pictureData?.userName ? (
			<Text color="white">
				AI Art by{' '}
				<Button
					_hover={{
						textDecoration: 'none',
						color: 'white',
					}}
					color={'#D5D4D6'}
					as={Link}
					fontWeight={400}
					variant={'link'}
					to={`/profile/${pictureData.userName}`}>
					{`@${pictureData.userName}`}
				</Button>
			</Text>
		) : null}
		{pictureData?.modelType ? (
			<Text color={'white'}>
				Made with {''}
				<ChakraLink
					onClick={() => {
						if (pictureData.modelType?.name) {
							onSendAnalyticEvent(
								pictureData.modelType.name,
								'model_type',
								'model_type',
							);
						}
					}}
					textDecoration={'none'}
					_hover={{
						color: 'white',
					}}
					color={'#D5D4D6'}
					target={'_blank'}
					href={pictureData.modelType.link}>
					{pictureData.modelType.name}
				</ChakraLink>
			</Text>
		) : null}
	</Box>
);
