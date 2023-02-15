import React from 'react';
import { Skeleton } from '@chakra-ui/react';

interface IImageSkeletonProps {
	height?: number;
	width?: number;
	marginBottom?: string;
}

export const ImageSkeleton: React.FC<IImageSkeletonProps> = ({
	height = 0,
	width = 0,
	marginBottom,
}) => {
	return (
		<Skeleton
			height={'auto'}
			marginBottom={marginBottom}
			width="fill-available"
			paddingBottom={`${(height / width) * 100}%`}
		/>
	);
};
