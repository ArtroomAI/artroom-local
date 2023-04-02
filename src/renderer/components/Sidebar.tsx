import React from 'react';
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    IconButton
} from '@chakra-ui/react';
import {
    FaPaintBrush,
    FaMagic,
    FaFileImage,
    FaChevronLeft,
    FaChevronRight
} from 'react-icons/fa';
import {
    FiGitMerge,
    FiMenu,
} from 'react-icons/fi';
import {IoSettingsSharp} from 'react-icons/io5'
import {
    GiResize
} from 'react-icons/gi';
import NavItem from './NavItem';
import Tour from './ProjectTour/Tour';
import Discord from './Discord';
import EquilibriumAI from './EquilibriumAI';

export default function Sidebar () {
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);

    return (
        <Flex
            alignItems="center"
            h="95%"
            m="15px"
            pos="fixed"
            w={navSize === 'large'
                ? '250px'
                : '75px'}
        >
            <Flex
                backgroundColor="#182138"
                borderRadius="30px"
                boxShadow="0 4px 12px 0 rgba(0, 0, 0, 0.05)"
                flexDir="column"
                h="100%"
                justifyContent="space-between"
                opacity={0.6}
                w="100%"
            >
                <Flex
                    as="nav"
                    flexDir="column"
                    mr="20px"
                    p="5%"
                    w="100%"
                >
                    <NavItem
                        className="home-nav"
                        icon={FaMagic}
                        linkTo="#/"
                        navSize={navSize}
                        title="Create" />

                    <NavItem
                        className="paint-nav"
                        icon={FaPaintBrush}
                        linkTo="#/paint"
                        navSize={navSize}
                        title="Paint" />

                    <NavItem
                        className="queue-nav"
                        icon={FiMenu}
                        linkTo="#/queue"
                        navSize={navSize}
                        title="Queue" />

                    <NavItem
                        className="image-editor-nav"
                        icon={GiResize}
                        linkTo="#/image-editor"
                        navSize={navSize}
                        title="Image Editor" />

                    <NavItem
                        className="merge-nav"
                        icon={FiGitMerge}
                        linkTo="#/merge"
                        navSize={navSize}
                        title="Merge models" />
                    <NavItem
                        className="image-viewer"
                        icon={FaFileImage}
                        linkTo="#/imageviewer"
                        navSize={navSize}
                        title="Image Viewer" />

                </Flex>

                <Flex
                    as="nav"
                    flexDir="column"
                    mr="20px"
                    p="5%"
                    pb="20px"
                    w="100%">
                    <EquilibriumAI />

                    <Discord />

                    {/* <Tour /> */}

                    <NavItem
                        className="settings-nav"
                        icon={IoSettingsSharp}
                        linkTo="#/settings"
                        navSize={navSize}
                        title="Settings" />
                </Flex>
            </Flex>

            <Flex
                alignItems="center"
                backgroundColor="#182138"
                borderBottomRightRadius="15px"
                borderTopRightRadius="15px"
                cursor="pointer"
                h="95%"
                onClick={() => {
                    changeNavSize(navSize === 'small' ? 'large' : 'small');
                }}
                width="15px"
            >
                <IconButton
                    _hover={{
                        textDecor: 'none',
                        backgroundColor: '#AEC8CA'
                    }}
                    background="#182138"
                    icon={navSize === 'small'
                        ? <FaChevronRight />
                        : <FaChevronLeft />} aria-label='menu-expander'/>
            </Flex>
        </Flex>

    );
}
