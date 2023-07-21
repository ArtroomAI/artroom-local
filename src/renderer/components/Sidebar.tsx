import React from 'react';
import { Flex, IconButton } from '@chakra-ui/react';
import {
    FaPaintBrush,
    FaMagic,
    FaFileImage,
    FaChevronLeft,
    FaChevronRight,
    FaDiscord
} from 'react-icons/fa';
import {
    GiGraduateCap
} from 'react-icons/gi'
import {
    FiGitMerge,
    FiMenu,
} from 'react-icons/fi';
import {
    AiOutlineExperiment
} from 'react-icons/ai'
import {IoSettingsSharp} from 'react-icons/io5'
import {
    GiResize
} from 'react-icons/gi';
import NavItem from './NavItem';
import CivitaiLogo from '../images/civitai.png';
import ArtroomLogo from '../images/ArtroomLogo.png';

interface SidebarProps {
    navSize: 'large' | 'small';
    setNavSize: React.Dispatch<React.SetStateAction<"large" | "small">>;
}

export default function Sidebar ({ navSize, setNavSize } : SidebarProps) {
    return (
        <Flex
            alignItems="center"
            pos="fixed"
            top="45px"
            left="15px"
            bottom="15px"
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
                    <NavItem
                        className="trainer"
                        icon={AiOutlineExperiment}
                        linkTo="#/trainer"
                        navSize={navSize}
                        title="Trainer" />
                </Flex>

                <Flex
                    as="nav"
                    flexDir="column"
                    mr="20px"
                    p="5%"
                    pb="20px"
                    w="100%">

                    <NavItem
                        className="artroom-website-link"
                        title="Go to Website"
                        icon={ArtroomLogo}
                        navSize={navSize}
                        onClick={window.api.openWebsite} />

                    <NavItem
                        className="discord-link"
                        title="Join Discord"
                        icon={FaDiscord}
                        navSize={navSize}
                        onClick={window.api.openDiscord} />
                    
                    <NavItem
                        className="equilibrium-link"
                        title="Learn More"
                        icon={GiGraduateCap}
                        navSize={navSize}
                        onClick={window.api.openTutorial} />
                        
                    <NavItem
                        className="civitai-link"
                        title="Get Models"
                        icon={CivitaiLogo}
                        navSize={navSize}
                        onClick={window.api.openCivitai} />

                    {/* <Tour navSize={navSize} /> */}

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
                    setNavSize(navSize === 'small' ? 'large' : 'small');
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
                        : <FaChevronLeft />} aria-label='menu-expander' />
            </Flex>
        </Flex>

    );
}
