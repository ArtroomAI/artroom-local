import React from 'react'
import { useRecoilState } from 'recoil';
import * as atom from '../atoms/atoms';
import {
    Flex,
    IconButton,
} from '@chakra-ui/react'
import {
    FaPaintBrush,
    FaMagic,
    FaChevronLeft,
    FaChevronRight
} from 'react-icons/fa'
import {
    FiMenu,
    FiSettings,
} from 'react-icons/fi'
import{
    GiResize
} from 'react-icons/gi'
import EquilibriumAILogo from '../images/equilibriumai.png'
import NavItem from '../components/NavItem'
import Tour from './ProjectTour/Tour';
import Discord from './Discord';
import Viewer from './Viewer';
import EquilibriumAI from './EquilibriumAI';

export default function Sidebar() {
    const [navSize, changeNavSize] = useRecoilState(atom.navSizeState);
    
    return (
        <Flex
            pos='fixed'
            h='95%'
            boxShadow='0 4px 12px 0 rgba(0, 0, 0, 0.05)'
            opacity={0.6}
            m='15px'
            borderRadius='30px'  
            w = {navSize === 'small' ? '75px' : '180px'}
            flexDir='column'
            justifyContent='space-between'
            backgroundColor = '#182138'
            transition='all .25s ease'
        >
            <Flex
                p='7.5%'
                mr = '20px'
                flexDir='column'
                w='100%'
                alignItems={navSize === 'small' ? 'center' : 'flex-start'}
                as='nav'
                >
                <IconButton
                    background='none'
                    mt={5}
                    _hover={{ textDecor: 'none', backgroundColor: '#AEC8CA' }}
                    icon={navSize === 'small' ? <FaChevronRight/> : <FaChevronLeft/>}
                    onClick={() => {
                        if (navSize === 'small')
                            changeNavSize('large')
                        else
                            changeNavSize('small')
                    }}  
                ></IconButton>
                <NavItem navSize={navSize} icon={FaMagic} title='Create' linkTo='#/' className='home-nav'/>
                <NavItem navSize={navSize} icon={FaPaintBrush} title='Paint' linkTo='#/paint' className='paint-nav'/>
                <NavItem navSize={navSize} icon={FiMenu} title='Queue' linkTo='#/queue' className='queue-nav'/>
                <NavItem navSize={navSize} icon={GiResize} title='Upscale' linkTo='#/upscale' className='upscale-nav'/>
                <Viewer></Viewer>
            </Flex>
            <Flex  
                p='7.5%'
                mr = '20px'
                flexDir='column'
                w='100%'
                alignItems={navSize === 'small' ? 'center' : 'flex-start'}
                as='nav'
                pb='20px'>
                <EquilibriumAI></EquilibriumAI>
                <Discord></Discord>
                <Tour></Tour>
                <NavItem navSize={navSize} icon={FiSettings} title='Settings' linkTo='#/settings' className='settings-nav'/>
            </Flex>
        </Flex>
    )
}