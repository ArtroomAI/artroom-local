import React, { useState, useRef, useEffect } from 'react';
import { TagType } from '../../types';
import {
	KEYS,
	DEFAULT_PLACEHOLDER,
	DEFAULT_LABEL_FIELD,
	INPUT_FIELD_POSITIONS,
} from './constants';
import {
	InputProps,
	InputGroup,
	Input,
	InputRightElement,
	Spinner,
	Box,
} from '@chakra-ui/react';
import { Suggestions } from './Suggestions';
import { buildRegExpFromDelimiters } from './utils';
// import isEqual from 'lodash/isEqual';
import uniq from 'lodash/uniq';
import { Tag } from './Tag';

interface IReactTagsProps {
	placeholder?: string;
	labelField?: string;
	suggestions?: TagType[];
	delimiters?: number[];
	autofocus?: boolean;
	inline?: boolean; // TODO: Remove in v7.x.x
	inputFieldPosition?: 'inline' | 'top' | 'bottom';
	// INPUT_FIELD_POSITIONS.INLINE |
	// INPUT_FIELD_POSITIONS.TOP |
	// INPUT_FIELD_POSITIONS.BOTTOM,
	handleDelete?: (index: number, ev: any) => void;
	handleAddition?: (value: TagType) => void;
	handleFilterSuggestions?: (
		query: string,
		suggestions: TagType[],
	) => TagType[];
	allowDeleteFromEmptyInput?: boolean;
	allowAdditionFromPaste?: boolean;
	handleInputChange?: (value: string) => void;
	handleInputFocus?: (value: string) => void;
	handleInputBlur?: (value: string) => void;
	minQueryLength?: number;
	shouldRenderSuggestions?: (value: string) => boolean;
	autocomplete?: number | boolean;
	readOnly?: boolean;
	name?: string;
	id?: string;
	maxLength?: number;
	inputValue?: string;
	tags?: TagType[];
	allowUnique?: boolean;
	inputProps?: InputProps;
	editable?: boolean;
	loading?: boolean;
	placeholderColor?: string;
}

export const ReactTags: React.FC<IReactTagsProps> = ({
	allowAdditionFromPaste = true,
	allowDeleteFromEmptyInput = true,
	allowUnique = true,
	autocomplete = false,
	autofocus = true,
	delimiters = [...KEYS.ENTER, KEYS.TAB],
	editable = false,
	handleAddition,
	handleDelete,
	handleFilterSuggestions,
	handleInputChange,
	id,
	inline = true,
	inputFieldPosition = 'inline',
	inputProps,
	inputValue,
	labelField = DEFAULT_LABEL_FIELD,
	loading = false,
	maxLength,
	minQueryLength,
	name,
	placeholder = DEFAULT_PLACEHOLDER,
	placeholderColor,
	readOnly = false,
	shouldRenderSuggestions,
	suggestions = [],
	tags = [],
	handleInputBlur,
	handleInputFocus,
}) => {
	const [ariaLiveStatus, setAriaLiveStatus] = useState('');
	const [isFocused, setIsFocused] = useState(false);
	const [selectedIndex, setSelectedIndex] = useState(-1);
	const [query, setQuery] = useState('');
	const [suggestionsState, setSuggestions] = useState(suggestions);
	const [selectionMode, setSelectionMode] = useState(false);

	const textInputRef = useRef<HTMLInputElement | null>(null);
	const reactTagsRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		if (autofocus && !readOnly) {
			resetAndFocusInput();
		}
	}, []);

	useEffect(() => {
		updateSuggestions();
	}, [suggestions]);

	const handleFocus: React.FocusEventHandler<HTMLInputElement> = event => {
		const value = event.target.value;
		if (handleInputFocus) {
			handleInputFocus(value);
		}
		setIsFocused(true);
	};

	const handleBlur: React.FocusEventHandler<HTMLInputElement> = event => {
		const value = event.target.value;
		if (handleInputBlur) {
			handleInputBlur(value);
			if (textInputRef.current) {
				textInputRef.current.value = '';
			}
		}
		setIsFocused(false);
	};

	const handleChange: React.ChangeEventHandler<HTMLInputElement> = e => {
		if (handleInputChange) {
			handleInputChange(e.target.value);
		}

		const queryFromEv = e.target.value.trim();
		setQuery(queryFromEv);
		updateSuggestions();
	};

	const getQueryIndex = (query: string, item: TagType) =>
		item.name.toLowerCase().indexOf(query.toLowerCase());

	const filteredSuggestions = (query: string) => {
		if (allowUnique) {
			const existingTags = tags.map(tag => tag.name.toLowerCase());
			suggestions = suggestions.filter(
				suggestion =>
					!existingTags.includes(suggestion.name.toLowerCase()),
			);
		}
		if (handleFilterSuggestions) {
			return handleFilterSuggestions(query, suggestions);
		}

		const exactSuggestions = suggestions.filter(item => {
			return getQueryIndex(query, item) === 0;
		});
		const partialSuggestions = suggestions.filter(item => {
			return getQueryIndex(query, item) > 0;
		});
		return exactSuggestions.concat(partialSuggestions);
	};

	const updateSuggestions = () => {
		const suggestions = filteredSuggestions(query);

		setSuggestions(suggestions);
		setSelectedIndex(
			selectedIndex >= suggestions.length
				? suggestions.length - 1
				: selectedIndex,
		);
	};

	const resetAndFocusInput = () => {
		setQuery('');
		if (textInputRef.current) {
			textInputRef.current.value = '';
			textInputRef.current.focus();
		}
	};

	const addTag = (tag: TagType) => {
		if (!tag.id || !tag.name) {
			return;
		}
		const existingKeys = tags.map(tag => tag.id.toLowerCase());

		// Return if tag has been already added
		if (allowUnique && existingKeys.indexOf(tag.name.toLowerCase()) >= 0) {
			return;
		}
		if (autocomplete) {
			const possibleMatches = filteredSuggestions(tag.name);

			if (
				(autocomplete === 1 && possibleMatches.length === 1) ||
				(autocomplete === true && possibleMatches.length)
			) {
				// tag = possibleMatches[0];
				tag = suggestionsState[selectedIndex];
			}
		}

		handleAddition?.(tag);

		// reset the state
		setQuery('');
		setSelectionMode(false);
		setSelectedIndex(-1);

		resetAndFocusInput();
	};

	const handleKeyDown: React.KeyboardEventHandler<HTMLInputElement> = e => {
		// hide suggestions menu on escape
		if (e.keyCode === KEYS.ESCAPE) {
			e.preventDefault();
			e.stopPropagation();
			setSelectedIndex(-1);
			setSelectionMode(false);
			setSuggestions([]);
		}

		// When one of the terminating keys is pressed, add current query to the tags.
		// If no text is typed in so far, ignore the action - so we don't end up with a terminating
		// character typed in.
		if (delimiters.indexOf(e.keyCode) !== -1 && !e.shiftKey) {
			if (e.keyCode !== KEYS.TAB || query !== '') {
				e.preventDefault();
			}

			const selectedQuery =
				selectionMode && selectedIndex !== -1
					? suggestionsState[selectedIndex]
					: '';
			// : { id: '0', [this.props.labelField]: query };

			if (selectedQuery !== '') {
				addTag(selectedQuery);
			}
		}

		// when backspace key is pressed and query is blank, delete tag
		if (
			e.keyCode === KEYS.BACKSPACE &&
			query === '' &&
			allowDeleteFromEmptyInput
		) {
			handleDeleteFunc(tags.length - 1, e);
		}

		// up arrow
		if (e.keyCode === KEYS.UP_ARROW) {
			e.preventDefault();
			setSelectedIndex(
				selectedIndex <= 0
					? suggestionsState.length - 1
					: selectedIndex - 1,
			);
			setSelectionMode(true);
		}

		// down arrow
		if (e.keyCode === KEYS.DOWN_ARROW) {
			e.preventDefault();
			setSelectedIndex(
				suggestionsState.length === 0
					? -1
					: (selectedIndex + 1) % suggestionsState.length,
			);
			setSelectionMode(true);
		}
	};

	const handleSuggestionClick = (i: number) => {
		addTag(suggestionsState[i]);
	};

	const handleSuggestionHover = (i: number) => {
		setSelectedIndex(i);
		setSelectionMode(true);
	};

	const handlePaste: React.ClipboardEventHandler<HTMLInputElement> = e => {
		if (!allowAdditionFromPaste) {
			return;
		}

		e.preventDefault();

		const clipboardData = e.clipboardData || '';
		const clipboardText = clipboardData.getData('text');

		// const { maxLength = clipboardText.length } = this.props;

		const maxTextLength = Math.min(
			clipboardText.length,
			clipboardText.length,
		);
		const pastedText = clipboardData
			.getData('text')
			.substr(0, maxTextLength);

		// Used to determine how the pasted content is split.
		const delimiterRegExp = buildRegExpFromDelimiters(delimiters);
		const tags = pastedText.split(delimiterRegExp);

		// Only add unique tags
		uniq(tags).forEach(tag => addTag({ id: tag, name: tag }));
	};

	const getTagItems = () => {
		return tags.map((tag, index) => (
			<React.Fragment key={index}>
				<Tag tag={tag} onDelete={ev => handleDeleteFunc(index, ev)} />
			</React.Fragment>
		));
	};

	const handleDeleteFunc = (index: number, event: any) => {
		event.preventDefault();
		event.stopPropagation();
		const currentTags = tags.slice();
		// Early exit from the function if the array
		// is already empty
		if (currentTags.length === 0) {
			return;
		}
		let ariaLiveStatus = `Tag at index ${index} with value ${currentTags[index].id} deleted.`;
		handleDelete?.(index, event);
		const allTags =
			reactTagsRef.current?.querySelectorAll('.ReactTags__remove');
		let nextElementToFocus, nextIndex, nextTag;
		if (allTags) {
			if (index === 0 && currentTags.length > 1) {
				nextElementToFocus = allTags[0];
				nextIndex = 0;
				nextTag = currentTags[1];
			} else {
				nextElementToFocus = allTags[index - 1];
				nextIndex = index - 1;
				nextTag = currentTags[nextIndex];
			}
			if (!nextElementToFocus) {
				nextIndex = -1;
				nextElementToFocus = textInputRef.current;
			}
			if (nextIndex >= 0) {
				ariaLiveStatus += ` Tag at index ${nextIndex} with value ${nextTag.id} focussed. Press backspace to remove`;
			} else {
				ariaLiveStatus +=
					'Input focussed. Press enter to add a new tag';
			}
		}

		// @ts-ignore
		nextElementToFocus?.focus();
		setAriaLiveStatus(ariaLiveStatus);
	};

	const position = !inline
		? INPUT_FIELD_POSITIONS.BOTTOM
		: inputFieldPosition;

	const tagInput = !readOnly ? (
		<Box display="inline-block" width="100%">
			<InputGroup>
				<Input
					{...inputProps}
					ref={textInputRef}
					type="text"
					placeholder={placeholder}
					aria-label={placeholder}
					onFocus={handleFocus}
					onBlur={handleBlur}
					onChange={handleChange}
					onKeyDown={handleKeyDown}
					onPaste={handlePaste}
					name={name}
					id={id}
					maxLength={maxLength}
					value={inputValue}
					data-automation="input"
					data-testid="input"
					_placeholder={{
						color: placeholderColor,
					}}
				/>
				{loading ? (
					<InputRightElement>
						<Spinner />
					</InputRightElement>
				) : null}
			</InputGroup>

			<Suggestions
				query={query}
				suggestions={suggestionsState}
				labelField={labelField}
				selectedIndex={selectedIndex}
				handleClick={handleSuggestionClick}
				handleHover={handleSuggestionHover}
				minQueryLength={minQueryLength}
				shouldRenderSuggestions={shouldRenderSuggestions}
				isFocused={isFocused}
			/>
		</Box>
	) : null;

	return (
		<Box position="relative" ref={reactTagsRef}>
			<p
				role="alert"
				className="sr-only"
				style={{
					position: 'absolute',
					overflow: 'hidden',
					clip: 'rect(0 0 0 0)',
					margin: '-1px',
					padding: 0,
					width: '1px',
					height: '1px',
					border: 0,
				}}>
				{ariaLiveStatus}
			</p>
			{position === INPUT_FIELD_POSITIONS.TOP && tagInput}
			<Box mt="10px">
				{getTagItems()}
				{position === INPUT_FIELD_POSITIONS.INLINE && tagInput}
			</Box>
			{position === INPUT_FIELD_POSITIONS.BOTTOM && tagInput}
		</Box>
	);
};
