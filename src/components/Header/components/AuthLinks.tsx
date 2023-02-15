import React from 'react';
import {
	Button,
	useBreakpointValue,
	Menu,
	MenuItem,
	MenuList,
	MenuButton,
	MenuDivider,
	MenuGroup,
} from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import { RiMenuFill } from 'react-icons/ri';

export const AuthLinks: React.FC = () => {
	const shouldShowHamburger = useBreakpointValue({ base: true, lg: false });
	return (
		<>
			{shouldShowHamburger ? (
				<Menu>
					<MenuButton
						as={Button}
						rounded={'full'}
						variant={'link'}
						cursor={'pointer'}
						minW={0}>
						<RiMenuFill size={30} />
					</MenuButton>
					<MenuList zIndex={20} alignItems={'center'}>
						<MenuGroup title="Profile">
							<MenuItem as={Link} to="/sign-in">
								Sign In
							</MenuItem>
							<MenuItem as={Link} to="/sign-up">
								Sign Up
							</MenuItem>
						</MenuGroup>
						<MenuDivider />
						<MenuGroup title="Other">
							<MenuItem as={Link} to="/blog">
								Blog
							</MenuItem>
							<MenuItem as={Link} to="/featured">
								Featured
							</MenuItem>
							<MenuItem as={Link} to="/download-app">
								Download App
							</MenuItem>
							<MenuItem as={Link} to="/pricing">
								Pricing
							</MenuItem>
						</MenuGroup>
					</MenuList>
				</Menu>
			) : (
				<>
					<Button
						as={Link}
						fontWeight={400}
						variant={'link'}
						to="/sign-in">
						Sign In
					</Button>
				</>
			)}
		</>
	);
};
