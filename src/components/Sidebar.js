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
    FaChevronLeft,
    FaChevronRight
} from 'react-icons/fa';
import {
    FiMenu,
    FiSettings
} from 'react-icons/fi';
import {
    GiResize
} from 'react-icons/gi';
import EquilibriumAILogo from '../images/equilibriumai.png';
import NavItem from '../components/NavItem';
import Tour from './ProjectTour/Tour';
import Discord from './Discord';
import Viewer from './Viewer';
import EquilibriumAI from './EquilibriumAI';

export default function Sidebar () {
    const [
        navSize,
        changeNavSize
    ] = useRecoilState(atom.navSizeState);

    return (
        <Flex
            backgroundColor="#182138"
            borderRadius="30px"
            boxShadow="0 4px 12px 0 rgba(0, 0, 0, 0.05)"
            flexDir="column"
            h="95%"
            justifyContent="space-between"
            m="15px"
            opacity={0.6}
            pos="fixed"
            transition="all .25s ease"
            w={navSize === 'small'
                ? '75px'
                : '180px'}
        >
            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                as="nav"
                flexDir="column"
                mr="20px"
                p="7.5%"
                w="100%"
            >
                <IconButton
                    _hover={{ textDecor: 'none',
                        backgroundColor: '#AEC8CA' }}
                    background="none"
                    icon={navSize === 'small'
                        ? <FaChevronRight />
                        : <FaChevronLeft />}
                    mt={5}
                    onClick={() => {
                        if (navSize === 'small') {
                            changeNavSize('large');
                        } else {
                            changeNavSize('small');
                        }
                    }}
                />

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
                    className="upscale-nav"
                    icon={GiResize}
                    linkTo="#/upscale"
                    navSize={navSize}
                    title="Upscale" />

                <Viewer />
            </Flex>

            <Flex
                alignItems={navSize === 'small'
                    ? 'center'
                    : 'flex-start'}
                as="nav"
                flexDir="column"
                mr="20px"
                p="7.5%"
                pb="20px"
                w="100%">
                <EquilibriumAI />

                <Discord />

                <Tour />

                <NavItem
                    className="settings-nav"
                    icon={FiSettings}
                    linkTo="#/settings"
                    navSize={navSize}
                    title="Settings" />
            </Flex>
        </Flex>
    );
}
