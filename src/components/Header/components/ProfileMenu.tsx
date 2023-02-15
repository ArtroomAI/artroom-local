import React, { useMemo } from 'react';
import {
	Menu,
	MenuButton,
	Avatar,
	MenuList,
	Button,
	MenuDivider,
	MenuItem,
	MenuGroup,
	Text,
	HStack,
	Box,
	useBreakpointValue,
	chakra,
} from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import { getPictureUrl } from '../../../utils';
import { LoginUserResponse } from '../../../types';

interface IProfileMenuProps {
	onLogout: () => void;
	user: LoginUserResponse;
	pictureItemHash: string | null;
	firstName?: string;
	lastName?: string;
}

export const ProfileMenu: React.FC<IProfileMenuProps> = ({
	onLogout,
	user,
	pictureItemHash,
	firstName,
	lastName,
}) => {
	const showProfileLabel = useBreakpointValue({ base: true, lg: false });

	return (
		<Menu>
			<MenuButton
				as={Button}
				rounded={'full'}
				variant={'link'}
				cursor={'pointer'}
				minW={0}>
				<Avatar
					src={getPictureUrl(pictureItemHash)}
					ignoreFallback
					size={{ base: 'sm', md: 'md' }}
				/>
			</MenuButton>
			<MenuList zIndex={2000} alignItems={'center'}>
				<Link to={`/profile/${user.username}`}>
					<HStack px={3}>
						<Avatar src={getPictureUrl(pictureItemHash)} />
						<Box>
							<Text fontSize="md">{`${
								firstName || user.firstName
							} ${lastName || user.lastName}`}</Text>
							<Text fontSize="sm" color="gray.400">
								{user.email}
							</Text>
						</Box>
					</HStack>
				</Link>
				<MenuDivider />
				<Box px={3}>
					<Text>
						<chakra.span fontWeight="bold" fontSize="lg">
							2000
						</chakra.span>{' '}
						shards
					</Text>
				</Box>
				<MenuDivider />
				<MenuGroup title={showProfileLabel ? 'Profile' : ''}>
					<MenuItem as={Link} to={`/profile/${user.username}`}>
						Public Profile
					</MenuItem>
					<MenuItem as={Link} to="/account-settings/profile">
						Settings
					</MenuItem>
					<MenuItem as={Link} to="/account-settings/billing">
						Billing
					</MenuItem>
				</MenuGroup>
				<MenuDivider display={{ base: 'block', lg: 'none' }} />
				<MenuGroup
					title="Other"
					display={{ base: 'block', lg: 'none' }}>
					<MenuItem
						as={Link}
						to="/blog"
						display={{ base: 'block', lg: 'none' }}>
						Blog
					</MenuItem>
					<MenuItem
						as={Link}
						to="/featured"
						display={{ base: 'block', lg: 'none' }}>
						Featured
					</MenuItem>
					<MenuItem
						as={Link}
						to="/download-app"
						display={{ base: 'block', lg: 'none' }}>
						Download App
					</MenuItem>
					<MenuItem
						as={Link}
						to="/pricing"
						display={{ base: 'block', lg: 'none' }}>
						Pricing
					</MenuItem>
				</MenuGroup>
				<MenuDivider />
				<MenuItem onClick={onLogout}>Log out</MenuItem>
			</MenuList>
		</Menu>
	);
};
