import React, { useState } from 'react';
import {
	Box,
	Flex,
	Button,
	Stack,
	useColorModeValue,
	Image,
	IconButton,
} from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { Search } from '../Search/Search';
import { logoutRequest } from '../../redux/reducers/auth.reducer';
import { RootStore } from '../../redux';
import logoImage from '../../assets/images/Artroom FF-01.png';
import { History } from '../../constants';
import { ProfileMenu, AuthLinks } from './components';
import { RiImageEditLine } from 'react-icons/ri';

interface Header {
	hideSearch?: boolean;
}

export const Header: React.FC<Header> = ({ hideSearch = false }) => {
	const dispatch = useDispatch();
	const [searchValue, setSearchValue] = useState('');
	const { token, user } = useSelector((state: RootStore) => state.auth);
	const { firstName, lastName, pictureItemHash } = useSelector(
		(state: RootStore) => state.accountSettings,
	);

	const headerLinkColor = useColorModeValue('darkPrimary.500', 'white');

	return (
		<Box>
			<Flex
				h={'60px'}
				py={{ base: 2 }}
				px={{ base: 4 }}
				borderBottom={1}
				borderStyle="solid"
				borderColor={useColorModeValue('gray.200', 'gray.900')}
				align="center"
				justify="space-between">
				<Flex flexShrink={0}>
					<RouterLink to="/" style={{ flexShrink: 0 }}>
						<Image
							w={'160px'}
							src={logoImage}
							alt={'Logo image'}
							mr={3}
						/>
					</RouterLink>
					<Button
						as={RouterLink}
						minW="auto"
						fontWeight={400}
						variant={'link'}
						to="/"
						color={headerLinkColor}
						display={{ base: 'none', lg: 'flex' }}
						mr={3}>
						Home
					</Button>
					<Button
						as={RouterLink}
						minW="auto"
						fontWeight={400}
						variant={'link'}
						color={headerLinkColor}
						display={{ base: 'none', lg: 'flex' }}
						to="/featured"
						mr={3}>
						Featured
					</Button>
					<Button
						as={RouterLink}
						minW="auto"
						fontWeight={400}
						variant={'link'}
						color={headerLinkColor}
						display={{ base: 'none', lg: 'flex' }}
						to="/blog"
						mr={3}>
						Blog
					</Button>
					<Button
						as={RouterLink}
						minW="auto"
						fontWeight={400}
						variant={'link'}
						color={headerLinkColor}
						display={{ base: 'none', lg: 'flex' }}
						to="/download-app"
						mr={3}>
						Download
					</Button>
					<Button
						as={RouterLink}
						to="/pricing"
						minW="auto"
						color={headerLinkColor}
						fontWeight={400}
						variant={'link'}
						display={{ base: 'none', lg: 'flex' }}
						mr={3}>
						Pricing
					</Button>
				</Flex>

				<Stack
					justify={'flex-end'}
					align="center"
					direction={'row'}
					spacing={3}>
					{!hideSearch ? (
						<Flex justify={{ base: 'flex-end', md: 'center' }}>
							<Search
								onSearch={() => {
									if (searchValue) {
										History.push(`/results/${searchValue}`);
										setSearchValue('');
									}
								}}
								value={searchValue}
								onChange={e => {
									setSearchValue(e.target.value);
								}}
								onKeyPress={e => {
									if (e.key === 'Enter' && searchValue) {
										History.push(`/results/${searchValue}`);
										setSearchValue('');
									}
								}}
								variant="header"
							/>
						</Flex>
					) : null}
					<IconButton
						display={{ base: 'flex', lg: 'none' }}
						aria-label="Upload image"
						as={RouterLink}
						to="/creator/body">
						<RiImageEditLine size={20} />
					</IconButton>
					<Button
						variant={'solid'}
						display={{ base: 'none', lg: 'flex' }}
						mr={4}
						as={RouterLink}
						to="/creator/body"
						leftIcon={<RiImageEditLine size={20} />}>
						Creator
					</Button>
					{!token.accessToken ? (
						<AuthLinks />
					) : (
						<>
							<ProfileMenu
								onLogout={() => dispatch(logoutRequest())}
								user={user}
								pictureItemHash={pictureItemHash}
								firstName={firstName}
								lastName={lastName}
							/>
						</>
					)}
				</Stack>
			</Flex>
		</Box>
	);
};
