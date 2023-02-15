import { Box } from '@chakra-ui/react';
import React, { useContext } from 'react';
import { ContextMenuContext } from './ContextMenu';

interface IContextMenuTriggerProps {
	children: React.ReactNode;
}

export const ContextMenuTrigger: React.FC<IContextMenuTriggerProps> = ({
	children,
}) => {
	const { openMenu, setPosition } = useContext(ContextMenuContext);
	return (
		<Box
			onContextMenu={event => {
				event.preventDefault();
				setPosition({ x: event.clientX, y: event.clientY });
				openMenu();
			}}>
			{children}
		</Box>
	);
};
