import React from 'react';
import { useRecoilValue } from 'recoil';
import * as atom from '../../atoms/atoms';
import { IconType } from 'react-icons';
import {
	HStack,
	Icon,
	Text,
	Flex,
	Link,
	Tooltip,
	Image,
	Center,
} from '@chakra-ui/react';
import { Link as RouterLink, useLocation } from 'react-router-dom';

interface INavItemProps {
	url?: string;
	icon: IconType | string;
	title: string;
	onClick?: () => void;
	elementClass?: string;
	isExternal?: boolean;
}

export const NavItem: React.FC<INavItemProps> = ({
	url,
	icon,
	title,
	onClick,
	elementClass,
	isExternal,
}) => {
	const navSize = useRecoilValue(atom.navSizeState);
	const location = useLocation();
	return (
		<>
			<Flex
				flexDir="column"
				mt={15}
				w="100%"
				onClick={onClick || undefined}
				as={onClick ? 'button' : undefined}
				transition="all .3s">
				<Link
					as={isExternal || onClick ? undefined : RouterLink}
					href={onClick ? undefined : url}
					// @ts-ignore
					to={isExternal || onClick ? undefined : url}
					target={onClick || !isExternal ? undefined : 'blank'}
					borderRadius={8}
					color={
						url && location.pathname.includes(url)
							? 'white'
							: undefined
					}
					bg={
						url && location.pathname.includes(url)
							? 'buttonPrimary'
							: undefined
					}
					_hover={{
						textDecor: 'none',
						backgroundColor:
							url && location.pathname.includes(url)
								? 'darkPrimary.400'
								: 'buttonPrimary',
						color: 'white',
					}}
					p={2.5}>
					<HStack className={elementClass}>
						<Tooltip
							fontSize="md"
							label={navSize === 'small' ? title : ' '}
							placement="bottom"
							shouldWrapChildren>
							<Center minW="20px">
								{typeof icon === 'string' ? (
									<Image src={icon} width="20px" />
								) : (
									<Icon
										as={icon}
										fontSize="xl"
										justifyContent="center"
									/>
								)}
							</Center>
						</Tooltip>

						<Text
							align="center"
							fontSize="md"
							pl={5}
							whiteSpace="nowrap"
							pr={10}>
							{title}
						</Text>
					</HStack>
				</Link>
			</Flex>
		</>
	);
};
