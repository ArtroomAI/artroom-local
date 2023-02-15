import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../../atoms/atoms';
import { Flex, IconButton, useColorModeValue } from '@chakra-ui/react';
import {
	FaPaintBrush,
	FaMagic,
	FaFileImage,
	FaChevronLeft,
	FaChevronRight,
} from 'react-icons/fa';
import { FiMenu, FiSettings } from 'react-icons/fi';
import { Tour } from './ProjectTour/Tour';
import { NavItem } from './NavItem';
import EquilibriumLogo from '../../assets/images/equilibriumai.png';
import { FaDiscord } from 'react-icons/fa';

export const Sidebar: React.FC = () => {
	const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

	return (
		<Flex
			alignItems="center"
			top="60px"
			h="calc(100vh - 150px)"
			m="15px"
			zIndex={50}
			position="absolute"
			transition="all .3s"
			w={navSize === 'large' ? '250px' : '75px'}>
			<Flex
				backgroundColor={useColorModeValue('gray.200', '#182138')}
				overflow="hidden"
				borderRadius="30px"
				boxShadow="0 4px 12px 0 rgba(0, 0, 0, 0.05)"
				flexDir="column"
				h="100%"
				justifyContent="space-between"
				// opacity={0.6}
				bg={useColorModeValue('#EEF1F6', '#0F192C')}
				w="100%">
				<Flex as="nav" flexDir="column" mr="20px" p="10px" w="100%">
					<NavItem
						icon={FaMagic}
						title="Create"
						elementClass="home-nav"
						url="body"
					/>
					<NavItem
						icon={FaPaintBrush}
						title="Paint"
						url="paint"
						elementClass="paint-nav"
					/>
					<NavItem
						icon={FiMenu}
						title="Queue"
						elementClass="queue-nav"
						url="queue"
					/>

					{/* TODO: REDO UPSCALER TO BE MODIFIED FOR CLOUD  */}
					{/* <NavItem
                        className="upscale-nav"
                        icon={GiResize}
                        linkTo="#/upscale"
						navSize={navSize}
                        title="Upscale" /> */}

					<NavItem
						icon={FaFileImage}
						title="Image Viewer"
						elementClass="image-viewer"
						url="imageviewer"
					/>
				</Flex>

				<Flex
					as="nav"
					flexDir="column"
					mr="20px"
					// p="5%"
					p="10px"
					pb="20px"
					w="100%">
					<NavItem
						icon={EquilibriumLogo}
						title="Learn More"
						url="https://equilibriumai.com/"
						isExternal={true}
					/>

					<NavItem
						icon={FaDiscord}
						title="Join Discord"
						url="https://discord.gg/XNEmesgTFy"
						isExternal={true}
					/>

					<Tour />

					<NavItem
						icon={FiSettings}
						title="Settings"
						elementClass="settings-nav"
						url="settings"
					/>
				</Flex>
			</Flex>

			<Flex
				alignItems="center"
				backgroundColor={useColorModeValue('gray.200', '#182138')}
				borderBottomRightRadius="15px"
				borderTopRightRadius="15px"
				cursor="pointer"
				h="95%"
				onClick={() => {
					changeNavSize(navSize === 'small' ? 'large' : 'small');
				}}
				width="15px">
				<IconButton
					_hover={{
						textDecor: 'none',
						backgroundColor: '#0760cb',
					}}
					background={useColorModeValue('gray.200', '#182138')}
					icon={
						navSize === 'small' ? (
							<FaChevronRight />
						) : (
							<FaChevronLeft />
						)
					}
					aria-label="menu-expander"
				/>
			</Flex>
		</Flex>
	);
};
