import React from 'react';
import {
	FormControl,
	FormLabel,
	Input,
	InputGroup,
	InputLeftElement,
	FormErrorMessage,
	InputProps,
	FormControlProps,
	useColorModeValue,
} from '@chakra-ui/react';

interface IInputProps extends InputProps {
	errorMsg?: string;
	label: string;
	formControlProps?: FormControlProps;
	leftIcon?: JSX.Element;
}

export const InputText = React.forwardRef<null, IInputProps>(
	({ label, errorMsg, formControlProps, leftIcon, ...rest }, ref) => {
		const placeholderColor = useColorModeValue('gray.500', 'gray.100');
		return (
			<FormControl isInvalid={!!errorMsg} mb={2} {...formControlProps}>
				<FormLabel mb={0}>{label}</FormLabel>
				<InputGroup>
					{leftIcon ? (
						<InputLeftElement>{leftIcon}</InputLeftElement>
					) : null}
					<Input
						ref={ref}
						_placeholder={{ color: placeholderColor }}
						{...rest}
					/>
				</InputGroup>
				<FormErrorMessage mt={0}>{errorMsg}</FormErrorMessage>
			</FormControl>
		);
	},
);
InputText.displayName = 'InputText';
