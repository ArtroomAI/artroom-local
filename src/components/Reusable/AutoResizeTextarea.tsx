import { forwardRef } from 'react';
import { Textarea, TextareaProps } from '@chakra-ui/react';
import ResizeTextarea from 'react-textarea-autosize';

export const AutoResizeTextarea = forwardRef<
	HTMLTextAreaElement,
	TextareaProps
>((props, ref) => (
	<Textarea
		as={ResizeTextarea}
		minH="unset"
		minRows={1}
		overflow="hidden"
		ref={ref}
		resize="none"
		spellCheck="false"
		style={{
			borderWidth: '1px',
			borderStyle: 'solid',
		}}
		w="100%"
		{...props}
	/>
));

AutoResizeTextarea.displayName = 'AutoResizeTextarea';
