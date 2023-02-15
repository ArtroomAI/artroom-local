import React from 'react';
import {
	Input,
	InputGroup,
	InputLeftElement,
	Box,
	IconButton,
	Popover,
	PopoverTrigger,
	PopoverContent,
	PopoverArrow,
	PopoverCloseButton,
	PopoverBody,
	InputProps,
	useColorModeValue,
} from '@chakra-ui/react';
import { RiSearchLine } from 'react-icons/ri';

interface ISearchProps extends InputProps {
	variant: 'header' | 'main';
	onSearch?: () => void;
}

export const Search = React.forwardRef<null, ISearchProps>(
	({ variant, onSearch, ...rest }, ref) => {
		const placeholderColor = useColorModeValue('gray.500', 'gray.100');
		return (
			<Box>
				{variant === 'header' ? (
					<Box display={{ base: 'block', md: 'none' }}>
						<Popover>
							<PopoverTrigger>
								<IconButton aria-label="search" variant="ghost">
									<RiSearchLine size={20} />
								</IconButton>
							</PopoverTrigger>
							<PopoverContent bg="darkPrimary.50" w="100vw">
								<PopoverArrow bg="darkPrimary.50" />
								<PopoverCloseButton />
								<PopoverBody>
									<Input
										ref={ref}
										placeholder="Search..."
										// bg="white"
										height={'40px'}
										fontSize={'16px'}
										{...rest}
									/>
								</PopoverBody>
							</PopoverContent>
						</Popover>
					</Box>
				) : null}

				<InputGroup
					display={{
						base: variant === 'header' ? 'none' : 'block',
						md: 'block',
					}}>
					<InputLeftElement
						h="100%"
						w={variant === 'header' ? 'auto' : '60px'}>
						<IconButton
							variant="unstyled"
							display="flex"
							alignItems="center"
							aria-label="search"
							h="100%"
							w={variant === 'header' ? 'auto' : '60px'}
							onClick={onSearch}>
							<RiSearchLine />
						</IconButton>
					</InputLeftElement>
					<Input
						ref={ref}
						placeholder="Search..."
						// bg="white"
						_placeholder={{
							color: placeholderColor,
						}}
						height={variant === 'header' ? '40px' : '60px'}
						pl={variant === 'header' ? 'auto' : '70px'}
						fontSize={variant === 'header' ? '16px' : '20px'}
						{...rest}
					/>
				</InputGroup>
			</Box>
		);
	},
);
Search.displayName = 'SearchInput';
