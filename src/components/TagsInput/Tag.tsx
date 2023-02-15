import React from 'react';
import { Tag as ChakraTag, TagLabel, TagCloseButton } from '@chakra-ui/react';

interface ITagProps {
	tag: {
		id: string;
		name: string;
	};
	onDelete: React.MouseEventHandler<HTMLButtonElement>;
}

export const Tag: React.FC<ITagProps> = ({ tag, onDelete }) => (
	<ChakraTag
		size="md"
		borderRadius="full"
		variant="solid"
		colorScheme="lightPrimary"
		mr={1}
		mb={1}>
		<TagLabel>{tag.name}</TagLabel>
		<TagCloseButton onClick={onDelete} />
	</ChakraTag>
);
