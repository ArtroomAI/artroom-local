import React from 'react';
import { Center, Spinner, CenterProps } from '@chakra-ui/react';

interface ILoaderProps {
	spinnerSize?: string;
	centerHeight?: string;
	centerProps?: CenterProps;
}

export const Loader: React.FC<ILoaderProps> = ({
	spinnerSize = 'xl',
	centerHeight,
	centerProps,
}) => (
	<Center h={centerHeight} {...centerProps}>
		<Spinner
			thickness="4px"
			speed="1s"
			emptyColor="gray.200"
			color="darkPrimary.500"
			size={spinnerSize}
		/>
	</Center>
);
