import React from 'react';
import { usePagination, DOTS } from '../../utils';
import { RiArrowRightLine, RiArrowLeftLine } from 'react-icons/ri';
import { HStack, IconButton, Button, Text } from '@chakra-ui/react';

interface IPaginationProps {
	totalCount: number;
	pageSize: number;
	currentPage: number;
	siblingCount?: number;
	onPageChange: (value: number) => void;
}

export const Pagination: React.FC<IPaginationProps> = ({
	pageSize,
	totalCount,
	currentPage,
	siblingCount = 1,
	onPageChange,
}) => {
	const paginationRange = usePagination({
		currentPage,
		totalCount,
		siblingCount,
		pageSize,
	});

	if (currentPage === 0 || paginationRange.length < 2) {
		return null;
	}

	const onNext = () => {
		onPageChange(currentPage + 1);
	};

	const onPrevious = () => {
		onPageChange(currentPage - 1);
	};

	const lastPage = paginationRange[paginationRange.length - 1];

	return (
		<HStack>
			<IconButton
				aria-label="Previous page"
				isDisabled={currentPage === 1}
				onClick={onPrevious}
				variant="outline">
				<RiArrowLeftLine />
			</IconButton>
			{paginationRange.map(
				(pageNumber: number | string, index: number) => {
					const isSelected = pageNumber === currentPage;
					if (pageNumber === DOTS) {
						return <Text key={index}>&#8230;</Text>;
					}

					return (
						<Button
							key={index}
							variant={isSelected ? 'solid' : 'outline'}
							colorScheme="darkPrimary"
							onClick={() => onPageChange(+pageNumber)}>
							{pageNumber}
						</Button>
					);
				},
			)}
			<IconButton
				aria-label="Next page"
				variant="outline"
				isDisabled={currentPage === lastPage}
				onClick={onNext}>
				<RiArrowRightLine />
			</IconButton>
		</HStack>
	);
};
