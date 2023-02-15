import React, { useRef } from 'react';
import { TagType } from '../../types';
import { List, ListItem, Text, useColorModeValue } from '@chakra-ui/react';

interface ISuggestionsProps {
	query: string;
	selectedIndex: number;
	suggestions: TagType[];
	handleClick: (value: number) => void;
	handleHover: (value: number) => void;
	minQueryLength?: number;
	shouldRenderSuggestions?: (value: string) => boolean;
	isFocused: boolean;
	labelField: string;
}

export const Suggestions: React.FC<ISuggestionsProps> = ({
	minQueryLength = 2,
	suggestions,
	handleClick,
	handleHover,
	isFocused,
	labelField,
	query,
	selectedIndex,
	shouldRenderSuggestions,
}) => {
	const suggestionsContainerRef = useRef<HTMLUListElement | null>(null);

	const renderSuggestion = (item: TagType) => (
		<Text>{`${item.id === '0' ? 'Add ' : ''}${item.name}`}</Text>
	);

	const mappedSuggestions = suggestions.map((item, i) => (
		<ListItem
			key={i}
			cursor="pointer"
			onMouseDown={() => handleClick(i)}
			onTouchStart={() => handleClick(i)}
			onMouseOver={() => handleHover(i)}
			bg={i === selectedIndex ? 'lightPrimary.500' : ''}
			color={
				i === selectedIndex
					? useColorModeValue('white', 'black')
					: useColorModeValue('black', 'white')
			}
			borderRadius={5}
			padding={3}>
			{renderSuggestion(item)}
		</ListItem>
	));

	const shouldRenderSuggestionsFunc = (query: string) =>
		query.length >= minQueryLength && isFocused;

	const shouldRenderSuggestionsVar =
		shouldRenderSuggestions || shouldRenderSuggestionsFunc;
	if (suggestions.length === 0 || !shouldRenderSuggestionsVar(query)) {
		return null;
	}

	return (
		<List
			ref={suggestionsContainerRef}
			position="absolute"
			w="100%"
			bg={useColorModeValue('white', 'gray.800')}
			p={2}
			boxShadow="xl"
			zIndex={100}>
			{mappedSuggestions}
		</List>
	);
};
