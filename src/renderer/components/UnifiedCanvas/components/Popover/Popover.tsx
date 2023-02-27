import React from 'react';
import {
	Popover as ChakraPopover,
	PopoverArrow,
	PopoverContent,
	PopoverTrigger,
	BoxProps,
} from '@chakra-ui/react';
import { PopoverProps } from '@chakra-ui/react';
import { ReactNode, FC } from 'react';

type IPopoverProps = PopoverProps & {
	triggerComponent: ReactNode;
	triggerContainerProps?: BoxProps;
	children: ReactNode;
	styleClass?: string;
	hasArrow?: boolean;
};

export const Popover: FC<IPopoverProps> = props => {
	const {
		triggerComponent,
		children,
		styleClass,
		hasArrow = true,
		...rest
	} = props;

	return (
		<ChakraPopover {...rest}>
			<PopoverTrigger>{triggerComponent}</PopoverTrigger>
			<PopoverContent
				className={`painter__popover-content ${styleClass}`}>
				{hasArrow && (
					<PopoverArrow className="painter__popover-arrow" />
				)}
				{children}
			</PopoverContent>
		</ChakraPopover>
	);
};
