import {
	Popover as ChakraPopover,
	PopoverArrow,
	PopoverContent,
	PopoverTrigger,
	BoxProps,
	PopoverProps,
} from '@chakra-ui/react';
import { ReactNode, FC } from 'react';

type IPopoverProps = PopoverProps & {
	triggerComponent: ReactNode;
	triggerContainerProps?: BoxProps;
	children: ReactNode;
	hasArrow?: boolean;
};

export const Popover: FC<IPopoverProps> = props => {
	const { triggerComponent, children, hasArrow = true, ...rest } = props;

	return (
		<ChakraPopover {...rest}>
			<PopoverTrigger>{triggerComponent}</PopoverTrigger>
			<PopoverContent p="1rem">
				{hasArrow && <PopoverArrow />}
				{children}
			</PopoverContent>
		</ChakraPopover>
	);
};
