import React, { useContext, useEffect, useState } from 'react';
import { ContextMenuContext } from './ContextMenu';
import { MotionBox } from './MotionBox';
import { useOutsideClick } from '@chakra-ui/react';
import { useColorModeValue } from '@chakra-ui/react';

const motionVariants = {
	enter: {
		visibility: 'visible',
		opacity: 1,
		scale: 1,
		transition: {
			duration: 0.2,
			ease: [0.4, 0, 0.2, 1],
		},
	},
	exit: {
		transitionEnd: {
			visibility: 'hidden',
		},
		opacity: 0,
		scale: 0.8,
		transition: {
			duration: 0.1,
			easings: 'easeOut',
		},
	},
};

/*
 * Type Position = {
 *     left?: number | string;
 *     right?: number | string;
 *     top?: number | string;
 *     bottom?: number | string;
 * };
 */

interface IContextMenuListProps {
	children: React.ReactNode;
}

export const ContextMenuList: React.FC<IContextMenuListProps> = ({
	children,
}) => {
	const {
		closeMenu,
		isOpen,
		menuRef,
		position: { x, y },
	} = useContext(ContextMenuContext);

	const [position, setPosition] = useState({});

	// TODO: Any less manual way to do this
	useEffect(() => {
		let left;
		let right;
		let top;
		let bottom;
		// @ts-ignore
		const menuHeight = menuRef?.current?.clientHeight;
		// @ts-ignore
		const menuWidth = menuRef?.current?.clientWidth;
		const windowWidth = window.innerWidth;
		const windowHeight = window.innerHeight;
		if (!menuHeight || !menuWidth) {
			return;
		}
		if (x + menuWidth > windowWidth) {
			right = windowWidth - x;
		} else {
			left = x;
		}
		if (y + menuHeight > windowHeight) {
			bottom = windowHeight - y;
		} else {
			top = y;
		}
		setPosition({
			top: `${top}px`,
			bottom: `${bottom}px`,
			left: `${left}px`,
			right: `${right}px`,
		});
	}, [menuRef, x, y]);

	useOutsideClick({
		// @ts-ignore
		ref: menuRef,
		handler: () => {
			if (isOpen) {
				closeMenu();
			}
		},
	});

	return (
		<MotionBox
			animate={isOpen ? 'enter' : 'exit'}
			bg={useColorModeValue('white', 'background')}
			borderRadius={5}
			borderWidth={2}
			display="flex"
			flexDirection="column"
			maxW={96}
			minW={40}
			position="fixed"
			py={1}
			ref={menuRef}
			variants={motionVariants}
			zIndex={1000}
			{...position}>
			{children}
		</MotionBox>
	);
};
