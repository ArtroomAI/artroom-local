import React from 'react';
import {
	Textarea,
	FormControl,
	FormLabel,
	FormErrorMessage,
	TextareaProps,
	useColorModeValue,
} from '@chakra-ui/react';

interface ITextareaProps extends TextareaProps {
	label: string;
	errorMsg?: string;
}

export const InputTextArea = React.forwardRef<null, ITextareaProps>(
	({ label, errorMsg, ...rest }, ref) => {
		const placeholderColor = useColorModeValue('gray.500', 'gray.100');
		return (
			<FormControl isInvalid={!!errorMsg} mb={2}>
				<FormLabel mb={0}>{label}</FormLabel>
				<Textarea
					focusBorderColor="lightPrimary.400"
					ref={ref}
					_placeholder={{ color: placeholderColor }}
					{...rest}
				/>
				<FormErrorMessage mt={0}>{errorMsg}</FormErrorMessage>
			</FormControl>
		);
	},
);

InputTextArea.displayName = 'InputTextArea';
