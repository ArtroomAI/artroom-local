import React, { useState, useEffect } from 'react';
import { Box, BoxProps } from '@chakra-ui/react';

interface IProps {
	horizontal?: boolean;
	onReachBottom?: () => void;
	onReachTop?: () => void;
	onReachLeft?: () => void;
	onReachRight?: () => void;
	onScroll?: any;
	position?: number;
	height?: string;
	boxProps?: BoxProps;
	children?: React.ReactNode;
}

export const InfiniteScroll: React.FC<IProps> = ({
	horizontal = false,
	onReachBottom,
	onReachLeft,
	onReachRight,
	onReachTop,
	onScroll,
	position = 0,
	children,
	height = 'inherit',
	boxProps,
}) => {
	const [scroller, setScroller] = useState<any>(null);
	const [prevScroll, setPrevScroll] = useState<any>(0);

	const handleScrollerRef = (reference: any) => {
		setScroller(reference);
	};

	useEffect(() => {
		if (position) {
			setScrollPosition(position);
		}
	}, []);

	const setScrollPosition = (position = 0) => {
		if (horizontal) {
			setScroller((prev: any) => ({ ...prev, scrollLeft: position }));
		} else {
			setScroller((prev: any) => ({ ...prev, scrollTop: position }));
		}

		setPrevScroll(position);
	};

	const handleHorizontalScroll = () => {
		const { firstChild, lastChild, scrollLeft, offsetLeft, offsetWidth } =
			scroller;

		const leftEdge = firstChild.offsetLeft;
		const rightEdge = lastChild.offsetLeft + lastChild.offsetWidth;
		const scrolledLeft = scrollLeft + offsetLeft;
		const scrolledRight = scrolledLeft + offsetWidth;

		if (scrolledRight >= rightEdge) {
			onReachRight?.();
		} else if (scrolledLeft <= leftEdge) {
			onReachLeft?.();
		}
	};

	const handleVerticalScroll = () => {
		const { firstChild, lastChild, scrollTop, offsetTop, offsetHeight } =
			scroller;

		const topEdge = firstChild.offsetTop;
		const bottomEdge = lastChild.offsetTop + lastChild.offsetHeight;
		const scrolledUp = scrollTop + offsetTop;
		const scrolledDown = scrolledUp + offsetHeight;

		if (scrolledDown >= bottomEdge) {
			onReachBottom?.();
		} else if (scrolledUp <= topEdge) {
			onReachTop?.();
		}
	};

	const handleScroll = () => {
		let scrolledTo = 0;

		if (horizontal) {
			handleHorizontalScroll();
			scrolledTo = scroller.scrollLeft;
		} else {
			handleVerticalScroll();
			scrolledTo = scroller.scrollTop;
		}

		onScroll?.(scrolledTo, prevScroll);
		setPrevScroll(scrolledTo);
	};

	const whiteSpace = horizontal ? 'nowrap' : 'normal';

	return (
		<Box
			ref={handleScrollerRef}
			style={{
				overflow: 'auto',
				height,
				width: 'inherit',
				WebkitOverflowScrolling: 'inherit',
				whiteSpace,
				display: horizontal ? 'flex' : 'block',
				...boxProps?.style,
			}}
			onScroll={handleScroll}
			{...boxProps}>
			{children}
		</Box>
	);
};
