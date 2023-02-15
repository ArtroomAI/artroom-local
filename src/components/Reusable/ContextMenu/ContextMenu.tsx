import { useDisclosure } from '@chakra-ui/react';
import React, { useRef, useState } from 'react';

export const ContextMenuContext = React.createContext({
	isOpen: false,
	closeMenu: () => {},
	openMenu: () => {},
	menuRef: null,
	position: { x: 0, y: 0 },
	setPosition: ({ x, y }: { x: number; y: number }) => {},
});

export const ContextMenu: React.FC<{ children: React.ReactNode }> = ({
	children,
}) => {
	const { isOpen, onClose: closeMenu, onOpen: openMenu } = useDisclosure();
	const [position, setPosition] = useState({ x: 0, y: 0 });
	const menuRef = useRef<HTMLElement | null>(null);
	return (
		<ContextMenuContext.Provider
			value={{
				isOpen,
				closeMenu,
				openMenu,
				// @ts-ignore
				menuRef,
				position,
				setPosition,
			}}>
			{children}
		</ContextMenuContext.Provider>
	);
};
