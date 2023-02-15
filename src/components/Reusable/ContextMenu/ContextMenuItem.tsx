import React, { useContext } from 'react';
import { Button, useColorModeValue } from '@chakra-ui/react';
import { ContextMenuContext } from './ContextMenu';

interface IContextMenuItemProps {
	children: React.ReactNode;
	onClick: (e: React.MouseEvent<HTMLButtonElement, MouseEvent>) => void;
	disabled: boolean;
}

export const ContextMenuItem: React.FC<IContextMenuItemProps> = ({
	onClick,
	disabled = false,
	children,
}) => {
	const { closeMenu } = useContext(ContextMenuContext);
	return (
		<Button
			borderRadius={0}
			disabled={disabled}
			justifyContent="left"
			onClick={e => {
				e.preventDefault();
				onClick(e);
				closeMenu();
			}}
			overflow="hidden"
			size="sm"
			textOverflow="ellipsis"
			variant="ghost"
			color={useColorModeValue('black', 'white')}
			w="100%">
			{children}
		</Button>
	);
};
